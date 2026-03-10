"""MT-Bench evaluation: answer generation and LLM-as-judge scoring.

Usage:
  # Generate answers
  python eval_mt_bench.py --model Qwen/Qwen2.5-7B-Instruct \
      --question_file dataset/eval/mt_bench/question.jsonl \
      --output_file results/mt_bench/Qwen2.5-7B-Instruct/answers.jsonl

  # Judge answers
  python eval_mt_bench.py --mode judge \
      --judge_model Qwen/Qwen2.5-7B-Instruct \
      --answer_file results/mt_bench/MODEL/answers.jsonl \
      --question_file dataset/eval/mt_bench/question.jsonl \
      --output_file results/mt_bench/MODEL/judgments.jsonl
"""

import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch.nn as nn


class BinaryGate(nn.Module):
    """Lightweight MLP gate: hidden_state → {use_base, use_lora}."""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, hidden_states):
        return self.gate(hidden_states)

    def predict(self, hidden_states):
        logits = self.forward(hidden_states)
        return logits.argmax(dim=-1)


def load_questions(path):
    """Load MT-Bench questions from JSONL."""
    questions = []
    with open(path) as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    return questions


def generate_answer(model, tokenizer, messages, max_new_tokens=1024):
    """Generate a model response for a conversation."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def run_generation(args):
    """Generate model answers for all MT-Bench questions (2 turns each)."""
    questions = load_questions(args.question_file)
    print(f"Loaded {len(questions)} questions")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    gate = None
    if args.adapter_path:
        print(f"Loading LoRA adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

        # Load gate if available
        gate_path = args.gate_path
        if not gate_path:
            # Auto-detect: look for gate.pt in adapter parent directory
            candidate = os.path.join(os.path.dirname(args.adapter_path), "gate.pt")
            if os.path.exists(candidate):
                gate_path = candidate
        if gate_path and os.path.exists(gate_path):
            print(f"Loading gate from: {gate_path}")
            hidden_dim = model.config.hidden_size
            gate = BinaryGate(hidden_dim)
            gate.load_state_dict(torch.load(gate_path, map_location="cpu", weights_only=True))
            gate = gate.to(next(model.parameters()).device).to(next(model.parameters()).dtype)
            gate.eval()
            print(f"Gate loaded. Will route per-query: 0=base, 1=LoRA")

    model.eval()

    results = []
    for q in tqdm(questions, desc="Generating answers"):
        qid = q["question_id"]
        category = q["category"]
        turns = q["turns"]

        # Turn 1
        messages = []
        if args.force_empty_system:
            # Inject an explicit empty system message to suppress baked-in defaults
            messages.append({"role": "system", "content": ""})
        elif args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": turns[0]})

        # Gate-based routing: decide whether to use LoRA for this query
        if gate is not None:
            gate_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            gate_ids = tokenizer(gate_input, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(model.device)
            model.disable_adapter_layers()
            with torch.no_grad():
                out = model(input_ids=gate_ids, output_hidden_states=True)
                hidden = out.hidden_states[1]  # layer 1
                last_tok = hidden[0, -1, :].unsqueeze(0)
                decision = gate.predict(last_tok).item()
            if decision == 1:
                model.enable_adapter_layers()
                # LoRA ON for this query
            # else: adapter stays disabled (base model)

        answer1 = generate_answer(model, tokenizer, messages, args.max_new_tokens)

        # Turn 2 (if exists)
        answer2 = ""
        if len(turns) > 1:
            messages.append({"role": "assistant", "content": answer1})
            messages.append({"role": "user", "content": turns[1]})
            answer2 = generate_answer(model, tokenizer, messages, args.max_new_tokens)

        result = {
            "question_id": qid,
            "category": category,
            "model": args.model.split("/")[-1],
            "turns": turns,
            "answers": [answer1, answer2] if answer2 else [answer1],
        }
        results.append(result)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} answers to {args.output_file}")

    # Print per-category stats
    cats = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    print("\nPer-category question counts:")
    for cat in sorted(cats):
        print(f"  {cat}: {len(cats[cat])}")


JUDGE_PROMPT = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors including helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, output your rating on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""


def run_judging(args):
    """Judge model answers using an LLM judge."""
    questions = {q["question_id"]: q for q in load_questions(args.question_file)}

    answers = []
    with open(args.answer_file) as f:
        for line in f:
            answers.append(json.loads(line.strip()))
    print(f"Loaded {len(answers)} answers to judge")

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    judge = AutoModelForCausalLM.from_pretrained(
        args.judge_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    judge.eval()

    judgments = []
    for ans in tqdm(answers, desc="Judging"):
        qid = ans["question_id"]
        q = questions[qid]
        category = ans["category"]

        # Judge each turn
        turn_scores = []
        for turn_idx, (question_text, answer_text) in enumerate(zip(q["turns"], ans["answers"])):
            prompt = JUDGE_PROMPT.format(question=question_text, answer=answer_text)
            messages = [{"role": "user", "content": prompt}]
            judgment_text = generate_answer(judge, tokenizer, messages, max_new_tokens=512)

            # Extract score
            score = extract_score(judgment_text)
            turn_scores.append(score)

        judgment = {
            "question_id": qid,
            "category": category,
            "model": ans["model"],
            "judge": args.judge_model.split("/")[-1],
            "turn_scores": turn_scores,
            "avg_score": sum(turn_scores) / len(turn_scores) if turn_scores else 0,
        }
        judgments.append(judgment)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        for j in judgments:
            f.write(json.dumps(j) + "\n")

    # Print per-category summary
    cat_scores = {}
    for j in judgments:
        cat_scores.setdefault(j["category"], []).append(j["avg_score"])

    print(f"\n{'Category':<20} {'Avg Score':>10} {'Count':>6}")
    print("-" * 40)
    all_scores = []
    for cat in sorted(cat_scores):
        scores = cat_scores[cat]
        avg = sum(scores) / len(scores)
        all_scores.extend(scores)
        print(f"{cat:<20} {avg:>10.2f} {len(scores):>6}")
    print("-" * 40)
    print(f"{'Overall':<20} {sum(all_scores)/len(all_scores):>10.2f} {len(all_scores):>6}")

    # Save summary
    summary = {
        "model": judgments[0]["model"] if judgments else "unknown",
        "judge": args.judge_model.split("/")[-1],
        "overall": sum(all_scores) / len(all_scores) if all_scores else 0,
        "per_category": {cat: sum(s)/len(s) for cat, s in cat_scores.items()},
        "n_questions": len(judgments),
    }
    summary_path = os.path.join(os.path.dirname(args.output_file), "mt_bench_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


def extract_score(text):
    """Extract numerical score from judge output."""
    import re
    # Look for [[N]] pattern
    match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', text)
    if match:
        return float(match.group(1))
    # Fallback: look for "Rating: N"
    match = re.search(r'[Rr]ating:\s*(\d+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1))
    # Last resort: look for any standalone number 1-10
    match = re.search(r'\b(\d+)\b', text)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 10:
            return float(score)
    return 5.0  # default if extraction fails


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MT-Bench Evaluation")
    parser.add_argument("--mode", default="generate", choices=["generate", "judge"])
    parser.add_argument("--model", default=None, help="Model for answer generation")
    parser.add_argument("--adapter_path", default=None, help="LoRA adapter path")
    parser.add_argument("--judge_model", default=None, help="Judge model for scoring")
    parser.add_argument("--question_file", required=True)
    parser.add_argument("--answer_file", default=None, help="Path to answers (for judging)")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--gate_path", default=None, help="Path to gate.pt for gated LoRA routing")
    parser.add_argument("--system_prompt", default=None,
                        help="System prompt text or path to .txt file")
    parser.add_argument("--force_empty_system", action="store_true",
                        help="Inject explicit empty system message, overriding baked-in chat template defaults")
    args = parser.parse_args()

    # Load system prompt from file if it's a path
    if args.system_prompt and os.path.isfile(args.system_prompt):
        with open(args.system_prompt) as f:
            args.system_prompt = f.read().strip()

    if args.mode == "generate":
        assert args.model, "--model required for generation"
        run_generation(args)
    elif args.mode == "judge":
        assert args.judge_model, "--judge_model required for judging"
        assert args.answer_file, "--answer_file required for judging"
        run_judging(args)
