# PRISM: Persona Routing via Intent-based Self-Modeling

[![EMNLP 2026](https://img.shields.io/badge/EMNLP%202026-Main%20Conference-1f6feb.svg)](https://arxiv.org/abs/2603.18507)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2603.18507-b31b1b.svg)](https://arxiv.org/abs/2603.18507)
[![Video](https://img.shields.io/badge/Video-Bilibili-00a1d6.svg)](https://www.bilibili.com/video/BV1boXABxEJ4/)
[![The Register](https://img.shields.io/badge/Coverage-The%20Register-cc0000.svg)](https://www.theregister.com/software/2026/03/24/telling-an-ai-model-that-its-an-expert-makes-it-worse/5226049)
[![Yahoo Tech](https://img.shields.io/badge/Coverage-Yahoo%20Tech-6001d2.svg)](https://tech.yahoo.com/ai/articles/turns-ask-ai-play-expert-142356526.html)
[![Tencent News](https://img.shields.io/badge/Coverage-Tencent%20News-0052d9.svg)](https://news.qq.com/rain/a/20260324A062P600)
[![36Kr](https://img.shields.io/badge/Coverage-36Kr-1a1a1a.svg)](https://eu.36kr.com/zh/p/3736415004590339)
[![AIToday](https://img.shields.io/badge/Coverage-AIToday-0aa.svg)](https://www.aitoday.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Accepted to EMNLP 2026 (Main Conference).**

> ### ⚠️ Active research preview
>
> **PRISM is under active development and is being extended well beyond expert personas.**
> The published results cover expert-persona system prompts only. We are currently testing the
> same routing mechanism on other kinds of injected context — **RAG retrieval blocks, tool
> registries, and persistent personal memories** — where the same trade-off applies: the context
> helps some queries and actively hurts others.
>
> Interfaces, config keys, and directory layouts **will change** while this work is in progress.
> Pin a commit if you need stability. See the [Roadmap](#roadmap) at the end.

---

## The problem

Injecting context into a system prompt is not free. An expert persona ("you are a senior software
engineer…") reliably improves human-preference and safety alignment, but it *damages* accuracy on
knowledge-heavy discriminative tasks. Prepending it to every query means paying that accuracy cost
on every query. The same tension shows up for any injected context — retrieved passages, a tool
registry, a memory block — all of them help on the queries they were meant for and add noise
everywhere else.

## What PRISM does

PRISM removes the prompt from inference entirely and decides *per query* whether its behavior
should apply:

1. A **binary gate** (router) reads the query and predicts: does this query benefit from the context?
2. If **yes** → a single **LoRA adapter**, distilled from the context-conditioned behavior, is applied.
3. If **no** → the base model runs untouched, so nothing is lost on queries the context would hurt.

Training is a **self-bootstrapping** loop: the model generates its own queries, answers them both
with and without the context, judges its own pairs, and distills only the wins. **No external data,
no teacher model, no human labels.** One LoRA plus one gate — minimal memory and compute overhead.

The gate reads the first-layer hidden state of the last prompt token, so routing costs one partial
forward pass and is independent of generation length.

---

## Install

```bash
git clone https://github.com/zizhao-hu/PRISM.git
cd PRISM
pip install -r requirements.txt
```

Fetch the evaluation benchmarks (MT-Bench, HarmBench, JailbreakBench, PKU-SafeRLHF; MMLU streams
from HuggingFace at runtime):

```bash
python prism/eval/download_data.py
```

Run every command from the repository root — data paths in `configs/` and `prism/utils.py` are
relative to it.

---

## Quickstart

Run the whole bootstrapping loop for one model using a shipped config:

```bash
python prism/run_iterative.py --config configs/Qwen2.5-7B-Instruct.json
```

`run_iterative.py` drives stages 2–5 for `rounds` iterations against the same LoRA. Configs for six
models live in `configs/`. Override any config key from the CLI (`--rounds`, `--epochs_per_round`,
`--lora_r`, `--retain_weight`, `--learning_rate`, …).

To step through the pipeline manually instead, run the stages below in order.

---

## The pipeline, stage by stage

Each stage lists the command for the expert-persona setup, then **how to point that same stage at a
different kind of context**. The general recipe is collected in
[Adapting PRISM to a new context type](#adapting-prism-to-a-new-context-type).

### Stage 1 — Self-generated queries

The model writes its own evaluation queries, conditioned on each context, so no external dataset is
needed.

```bash
python prism/stage1_query_gen.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num_samples 50
```

| Flag | Meaning |
|---|---|
| `--model` | HF model id; also the model that generates the queries |
| `--num_samples` | queries generated per context (default 50) |
| `--data_dir` | override output dir (default `dataset/synthetic/persona_prism/<slug>/`) |

**Adapting to another context type.** Contexts are plain text files, one per context, registered in
a dict at the top of the script. Drop your own text files in and edit the registry:

```python
# prism/stage1_query_gen.py
TASK_PERSONAS = {
    "rag_finance":  "dataset/personas/rag/finance_retrieval_block.txt",
    "tools_search": "dataset/personas/tools/search_registry.txt",
    "memory_user":  "dataset/personas/memory/persistent_user_profile.txt",
}
```

The file's contents are injected verbatim as the system prompt, so a retrieved passage, a serialized
tool registry, or a memory block all work without code changes. Keep one context per file — the gate
is trained on the boundary between "this context applies" and "it doesn't", so contexts that overlap
heavily will produce a weak router.

> **The registry is duplicated in three modules.** Edit `TASK_PERSONAS` / `BEHAVIORAL_PERSONAS` in
> `prism/stage1_query_gen.py`, `prism/stage2_verify_recycle.py`, **and** `prism/run_iterative.py` —
> they do not share a single definition. See the [Roadmap](#roadmap).

### Stage 2 — Dual answers and self-verification

For each query, generate two answers — one with the context, one without — grade both pointwise
(1–10) with the model itself, and partition:

- context wins → **distill set** (train the LoRA toward the context-conditioned answer)
- baseline wins → **retain set** (train the LoRA to leave the base behavior alone)

```bash
python prism/stage2_verify_recycle.py --model Qwen/Qwen2.5-7B-Instruct
```

Add `--regrade` to re-score existing answers without regenerating them (useful after changing the
judging prompt).

**Adapting to another context type.** Nothing structural changes — the comparison is always
*with context* vs *without context*. What matters is that your queries are ones where the context
plausibly could help or hurt; if the context helps on all of them, the retain set is empty and the
gate has nothing to learn. Check the distill/retain split before training.

### Stage 3 — Teacher logits

Top-64 teacher logits are computed from the context-conditioned model over the distill set; these
are the KL targets that transfer behavior into the adapter without the prompt. This runs
automatically inside `run_iterative.py` per round and writes to
`dataset/synthetic/persona_prism/<exp>/round_N/teacher_logits*/`.

**Adapting to another context type.** No changes needed — it consumes whatever Stage 2 produced.
Long contexts (RAG blocks, large tool registries) inflate teacher sequence length; raise `max_len`
in the config if you see truncation warnings.

### Stage 4 — Gate + LoRA training

Trains the binary router and the single LoRA adapter.

```bash
python prism/run_gated_lora.py --config configs/Qwen2.5-7B-Instruct.json
```

| Flag | Meaning |
|---|---|
| `--config` | **required**; JSON config (see `configs/`) |
| `--exp_name` | experiment name, controls the output directory |
| `--source_exp` | reuse Stage 2 data from another experiment |
| `--epochs`, `--lora_r`, `--lora_alpha`, `--micro_batch`, `--grad_accum` | override config values |
| `--eval_only` | skip training, evaluate an existing adapter |

**Adapting to another context type.** The gate is a binary classifier over your distill/retain
split, so it retrains as-is. Two knobs matter: `retain_weight` (raise it if the adapter is bleeding
into queries it should leave alone) and `lora_r` (the paper uses `r=2`, enough for a behavioral
shift; a context that changes *content* rather than *style* — RAG especially — will likely need
more).

### Stage 5 — Evaluation

```bash
# MT-Bench: generate then judge
python prism/eval/eval_mt_bench.py --mode generate \
  --model Qwen/Qwen2.5-7B-Instruct --adapter_path <adapter> \
  --question_file dataset/eval/mt_bench/question.jsonl \
  --output_file results/mt_bench/answers.jsonl
python prism/eval/eval_mt_bench.py --mode judge \
  --judge_model Qwen/Qwen2.5-7B-Instruct \
  --question_file dataset/eval/mt_bench/question.jsonl \
  --answer_file results/mt_bench/answers.jsonl \
  --output_file results/mt_bench/judgments.jsonl

# MMLU, with the gate active
python prism/eval/eval_mmlu_gated.py \
  --model Qwen/Qwen2.5-7B-Instruct --adapter_path <adapter> \
  --gate_path <gate> --output_dir results/mmlu

# MMLU baseline (no adapter)
python prism/eval/eval_mmlu.py \
  --base_model Qwen/Qwen2.5-7B-Instruct --output_dir results/mmlu_base

# Safety
python prism/eval/eval_safety.py \
  --base_model Qwen/Qwen2.5-7B-Instruct --adapter_path <adapter> \
  --gate_path <gate> --benchmarks HarmBench Jailbreak PKU_SafeRLHF
```

**Adapting to another context type.** These benchmarks measure the persona trade-off specifically
(alignment up, knowledge down). For a different context you will want a benchmark pair that captures
*your* trade-off — e.g. for RAG, a grounded-QA set against a closed-book set; for tool registries,
a tool-selection set against a no-tool-needed set. The gate's value only shows up when the evaluation
contains queries the context hurts.

---

## Adapting PRISM to a new context type

The short version of the per-stage notes above:

1. **Write one text file per context** under `dataset/personas/`. Contents are injected verbatim as
   the system prompt.
2. **Register them** in the `TASK_PERSONAS` dict in `prism/stage1_query_gen.py`,
   `prism/stage2_verify_recycle.py`, and `prism/run_iterative.py`.
3. **Copy a config** from `configs/`, set `exp_name`, and raise `max_len` if your context is long.
4. **Check the Stage 2 split.** A healthy run has a non-trivial retain set. All-distill means the
   context never hurts and you don't need a gate; all-retain means it never helps.
5. **Choose an evaluation pair** where the context helps on one side and hurts on the other.
6. **Tune `retain_weight` and `lora_r`** — content-changing contexts need more rank than
   style-changing ones.

---

## Models

| Model | HuggingFace ID | Type |
|-------|---------------|------|
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` | Standard |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` | Standard |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | Standard |
| Qwen1.5-MoE-2.7B | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | MoE |
| R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Reasoning |
| R1-Distill-Llama-8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Reasoning |

## Evaluation benchmarks

| Benchmark | Type | Samples | Source |
|-----------|------|---------|--------|
| MT-Bench | Utility | 80 | Self-judged pairwise |
| MMLU | Knowledge | 14k | HuggingFace `cais/mmlu` |
| HarmBench | Safety | 400 | Harmful behavior prompts |
| JailbreakBench | Safety | ~80 | Jailbreak attack prompts |
| PKU-SafeRLHF | Safety | 500 | Safety preference prompts |

---

## Repository layout

```
prism/                       The method — this is the whole implementation
  stage1_query_gen.py        Stage 1: self-generated queries per context
  stage2_verify_recycle.py   Stage 2: dual answers, self-grading, distill/retain split
  run_gated_lora.py          Stage 4: gate + LoRA training
  run_iterative.py           Driver: stages 2-5 for N rounds against one adapter
  utils.py                   Model I/O, path helpers, benchmark registry
  eval/                      Stage 5: MT-Bench, MMLU, MMLU-gated, safety, data download
configs/                     Per-model configs (6 models)
dataset/
  personas/                  Context files: full / half / min length variants
  eval/                      MT-Bench and safety benchmark data
```

The paper itself is on [arXiv](https://arxiv.org/abs/2603.18507); its LaTeX source is kept in a
separate repository and is not vendored here.

`models/`, `results/`, and `dataset/synthetic/` are generated at runtime and gitignored.

---

## Reproducibility notes

- **No trained checkpoints are released.** The gated LoRA adapters and result files from the paper
  runs were kept on scratch storage and have since been purged. Everything must be retrained from
  scratch via the pipeline above.
- **No cluster launcher is shipped.** The original SLURM wrappers were hardcoded to one site's
  account, partition, and `/scratch1/...` paths, and referenced entry points that no longer exist;
  they have been removed rather than shipped broken. Invoke the stage scripts directly, or wrap them
  for your own scheduler.
- **Self-judging is the evaluation mechanism**, not an approximation of it: MT-Bench scores come from
  the model grading its own outputs. Cross-model judging is not implemented.
- **PINT is referenced but not shipped.** `utils.BENCHMARKS` lists a PINT prompt-injection set whose
  data file is not in the repo. It is excluded from the default safety benchmark list.

---

## Roadmap

Work in progress. The published results cover expert personas; these are the directions we are
actively testing.

- [ ] **RAG systems** — route on whether retrieved passages help, so the model skips retrieval noise
      on queries it already answers correctly closed-book.
- [ ] **Tool registries** — route on whether a serialized tool registry belongs in context, avoiding
      the accuracy cost of a large registry on queries that need no tool.
- [ ] **Persistent personal memories** — route on whether a stored user-profile / memory block is
      relevant, so long-lived memory does not degrade unrelated queries.
- [ ] Multi-way routing (more than one adapter) instead of a single binary gate.
- [ ] De-duplicate the context registry into one shared definition across the three stage modules.
- [ ] Rename `dataset/personas/` to `dataset/contexts/` now that contexts are not only personas.
- [ ] Release trained adapters and gate weights.

## Citation

```bibtex
@inproceedings{hu2026prism,
  title     = {Expert Personas Improve LLM Alignment but Damage Accuracy:
               Bootstrapping Intent-Based Persona Routing with PRISM},
  author    = {Hu, Zizhao and Rostami, Mohammad and Thomason, Jesse},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing (EMNLP)},
  year      = {2026},
  note      = {To appear. Preprint: arXiv:2603.18507},
  url       = {https://arxiv.org/abs/2603.18507}
}
```

## License

MIT — see [LICENSE](LICENSE).
