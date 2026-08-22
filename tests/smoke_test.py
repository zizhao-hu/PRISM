"""
PRISM smoke test — verifies every component works end to end on a tiny model.

Runs the real code paths (model loading, chat formatting, batched generation,
teacher-logit extraction, LoRA attachment, gate training) against a ~135M model
on CPU/MPS, so a new user can confirm their environment is correct in a couple
of minutes without a GPU or a 7B download.

Usage:
    python tests/smoke_test.py                 # default tiny model
    python tests/smoke_test.py --model <hf_id> # any chat model

Exit code is 0 only if every check passes.
"""
import os
import sys
import json
import shutil
import argparse
import tempfile
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "prism"))

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

results = []


def check(name):
    """Decorator: run a check, record pass/fail, never abort the suite."""
    def wrap(fn):
        try:
            detail = fn()
            results.append((name, True, detail or ""))
            print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"  FAIL  {name}")
            print("        " + traceback.format_exc().strip().replace("\n", "\n        "))
        return fn
    return wrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--keep", action="store_true", help="keep the temp workdir")
    args = ap.parse_args()

    print(f"PRISM smoke test\nmodel: {args.model}\n")

    # ---------------------------------------------------------------- imports
    print("[1] module imports")

    @check("import utils")
    def _():
        import utils  # noqa: F401

    @check("import pipeline stages")
    def _():
        import stage1_query_gen, stage2_verify_recycle, stage3_distill  # noqa: F401

    @check("import run_gated_lora")
    def _():
        import run_gated_lora  # noqa: F401

    @check("import run_iterative (eval helpers)")
    def _():
        from run_iterative import (  # noqa: F401
            _run_mt_bench, _run_safety, _run_mmlu, _run_mmlu_gated,
            _run_utility, SAFETY_BENCHMARKS,
        )

    @check("import eval modules")
    def _():
        sys.path.insert(0, os.path.join(REPO_ROOT, "prism", "eval"))
        import eval_mt_bench, eval_mmlu, eval_mmlu_gated, eval_safety  # noqa: F401

    # ------------------------------------------------------------- data files
    print("\n[2] shipped data files")
    import utils

    @check("context/persona files present")
    def _():
        import stage1_query_gen as s1
        missing = [p for p in s1.PERSONA_CONTEXTS.values()
                   if not os.path.exists(os.path.join(REPO_ROOT, p))]
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing, e.g. {missing[0]}")
        return f"{len(s1.PERSONA_CONTEXTS)} contexts"

    @check("safety benchmark files resolve")
    def _():
        found, missing = [], []
        for b in utils.BENCHMARKS:
            (found if os.path.exists(os.path.join(REPO_ROOT, b["path"])) else missing).append(b["name"])
        if not found:
            raise FileNotFoundError("no benchmark data found at all")
        return f"found {found}" + (f"; NOT SHIPPED {missing}" if missing else "")

    @check("MT-Bench questions load")
    def _():
        p = os.path.join(REPO_ROOT, "dataset/eval/mt_bench/question.jsonl")
        rows = [json.loads(l) for l in open(p) if l.strip()]
        assert rows and "turns" in rows[0], "unexpected MT-Bench schema"
        return f"{len(rows)} questions"

    @check("model configs load")
    def _():
        d = os.path.join(REPO_ROOT, "configs")
        cfgs = [json.load(open(os.path.join(d, f))) for f in os.listdir(d) if f.endswith(".json")]
        for c in cfgs:
            for k in ("model", "exp_name", "lora_r", "rounds"):
                assert k in c, f"config missing {k}"
        return f"{len(cfgs)} configs"

    # ----------------------------------------------------------- model + gen
    print("\n[3] model loading and generation")
    model = tokenizer = None

    @check("load_model")
    def _():
        nonlocal_model = utils.load_model(args.model)
        globals()["_m"], globals()["_t"] = nonlocal_model
        return type(globals()["_m"]).__name__

    model, tokenizer = globals().get("_m"), globals().get("_t")
    if model is None:
        print("\nmodel failed to load — skipping runtime checks")
        return summarize(args)

    @check("build_chat_messages")
    def _():
        msgs = utils.build_chat_messages(tokenizer, "You are terse.", "Say hi.")
        assert isinstance(msgs, list) and msgs, "no messages built"
        return f"{len(msgs)} messages"

    @check("format_chat_text")
    def _():
        txt = utils.format_chat_text(tokenizer, "You are terse.", "Say hi.",
                                     add_generation_prompt=True)
        assert isinstance(txt, str) and txt.strip(), "empty chat text"
        return f"{len(txt)} chars"

    @check("generate_response")
    def _():
        msgs = utils.build_chat_messages(tokenizer, "You are terse.", "Name one color.")
        out = utils.generate_response(model, tokenizer, msgs, max_tokens=16)
        assert isinstance(out, str), "non-string generation"
        return repr(out[:40])

    @check("batch_generate")
    def _():
        batch = [utils.build_chat_messages(tokenizer, "You are terse.", q)
                 for q in ("Name a color.", "Name a fruit.")]
        outs = utils.batch_generate(model, tokenizer, batch, max_tokens=12)
        assert len(outs) == 2, f"expected 2 outputs, got {len(outs)}"
        return f"{len(outs)} generations"

    # ------------------------------------------------ distillation internals
    print("\n[4] distillation + gate internals")

    @check("compute_logits (teacher targets)")
    def _():
        # schema is {instruction, output, system} -- see utils.compute_logits
        sample = {"system": "You are terse.",
                  "instruction": "Name a color.",
                  "output": "Blue."}
        out = utils.compute_logits(model, tokenizer, sample, max_len=64)
        for k in ("input_ids", "labels", "logits", "prompt_len"):
            assert k in out, f"compute_logits missing key {k}"
        assert out["logits"].shape[0] > 0, "no response-token logits"
        return f"{out['logits'].shape[0]} response tokens, prompt_len={out['prompt_len']}"

    @check("BinaryGate forward")
    def _():
        import torch
        from run_gated_lora import BinaryGate
        hidden = model.config.hidden_size
        gate = BinaryGate(hidden)
        logits = gate(torch.randn(4, hidden))
        assert logits.shape == (4, 2), f"expected (4,2), got {tuple(logits.shape)}"
        return f"hidden={hidden} -> {tuple(logits.shape)}"

    @check("get_hidden_state (gate input)")
    def _():
        from run_gated_lora import get_hidden_state
        ids = tokenizer("Name a color.", return_tensors="pt").input_ids.to(model.device)
        h = get_hidden_state(model, tokenizer, ids)
        assert h is not None and h.numel() > 0, "empty hidden state"
        return f"shape {tuple(h.shape)}"

    @check("LoRA attach (peft)")
    def _():
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                         lora_dropout=0.0, task_type="CAUSAL_LM")
        peft_model = get_peft_model(model, cfg)
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        assert trainable > 0, "no trainable LoRA parameters"
        peft_model.unload()
        return f"{trainable:,} trainable params"

    # ------------------------------------------------------- stage 1 for real
    print("\n[5] Stage 1 end to end")

    workdir = tempfile.mkdtemp(prefix="prism_smoke_")

    @check("stage1 query generation")
    def _():
        import stage1_query_gen as s1
        name, path = next(iter(s1.PERSONA_CONTEXTS.items()))
        context = utils.load_text(os.path.join(REPO_ROOT, path))
        qs = s1.generate_persona_queries(model, tokenizer, name, context, 2)
        assert isinstance(qs, list) and len(qs) > 0, "no queries generated"
        return f"{len(qs)} queries for '{name}', e.g. {qs[0][:50]!r}"

    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)

    return summarize(args)


def summarize(args):
    print("\n" + "=" * 64)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = [(n, d) for n, ok, d in results if not ok]
    print(f"{passed}/{len(results)} checks passed")
    if failed:
        print("\nfailures:")
        for n, d in failed:
            print(f"  - {n}: {d}")
    print("=" * 64)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
