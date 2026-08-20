"""MMLU evaluation for base models (no adapter / no gate).

Supports --force_empty_system to inject an explicit empty system message,
suppressing baked-in defaults like Qwen's "You are a helpful assistant."

Usage:
  python prism/eval/eval_mmlu.py \
      --base_model Qwen/Qwen2.5-7B-Instruct \
      --output_dir results/Qwen2.5-7B-Instruct/no_system_prompt/mmlu

  # With explicit empty system prompt
  python prism/eval/eval_mmlu.py \
      --base_model Qwen/Qwen2.5-7B-Instruct \
      --output_dir results/Qwen2.5-7B-Instruct/no_system_prompt/mmlu \
      --force_empty_system
"""

import argparse
import json
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# MMLU subject -> domain mapping (same as eval_mmlu_gated.py)
DOMAIN_MAP = {
    "stem": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "social_sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy",
    ],
    "other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology",
    ],
}

# Reverse: subject -> domain
SUBJECT_TO_DOMAIN = {}
for domain, subjects in DOMAIN_MAP.items():
    for s in subjects:
        SUBJECT_TO_DOMAIN[s] = domain


def format_mmlu_question(subject, question, choices):
    """Format an MMLU question for log-likelihood evaluation."""
    subject_str = subject.replace("_", " ")
    prompt = f"The following is a multiple choice question about {subject_str}.\n\n"
    prompt += f"{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"
    return prompt


def evaluate_question(model, tokenizer, prompt, choices, correct_idx,
                      system_prefix=""):
    """Evaluate a single MMLU question using log-likelihood.

    If system_prefix is provided, it is prepended via chat template as a
    system message wrapping the prompt.  Otherwise the raw prompt is used.
    """
    log_probs = []
    for i, choice in enumerate(choices):
        full_text = system_prefix + prompt + f" {chr(65+i)}"
        encoding = tokenizer(full_text, return_tensors="pt",
                             truncation=True, max_length=2048)
        input_ids = encoding.input_ids.to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits

        # Log-prob of the answer token (last token)
        answer_logits = logits[0, -2, :]
        answer_token_id = input_ids[0, -1]
        log_prob = torch.nn.functional.log_softmax(
            answer_logits, dim=-1)[answer_token_id].item()
        log_probs.append(log_prob)

    predicted = max(range(len(log_probs)), key=lambda i: log_probs[i])
    return predicted, predicted == correct_idx


def main():
    parser = argparse.ArgumentParser(description="MMLU evaluation (base model)")
    parser.add_argument("--base_model", required=True, help="HF model name/path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--force_empty_system", action="store_true",
                        help="Inject explicit empty system message to suppress "
                             "baked-in chat template defaults")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per subject (for debugging)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "mmlu_summary.json")

    if os.path.exists(summary_path):
        print(f"[SKIP] Already done: {summary_path}")
        return

    # Load model
    print(f"Loading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()

    # Build a system prefix if --force_empty_system is set.
    # We apply the chat template with an empty system message to get whatever
    # wrapper the tokenizer produces, then strip the generation prompt so we
    # can prepend it to the raw MMLU prompt.
    system_prefix = ""
    if args.force_empty_system:
        try:
            # Build a chat-templated prefix with empty system + dummy user
            dummy = tokenizer.apply_chat_template(
                [{"role": "system", "content": ""},
                 {"role": "user", "content": "PLACEHOLDER"}],
                tokenize=False, add_generation_prompt=False,
            )
            # Extract everything before PLACEHOLDER
            idx = dummy.find("PLACEHOLDER")
            if idx > 0:
                system_prefix = dummy[:idx]
                print(f"Using system prefix ({len(system_prefix)} chars) "
                      f"from empty system message")
            else:
                print("WARNING: Could not extract system prefix; "
                      "proceeding without it")
        except Exception as e:
            print(f"WARNING: Chat template failed: {e}; proceeding without prefix")

    # Load MMLU
    print("Loading MMLU dataset...")
    ds = load_dataset("cais/mmlu", "all", split="test")

    # Group by subject
    by_subject = {}
    for item in ds:
        subj = item["subject"]
        by_subject.setdefault(subj, []).append(item)

    # Evaluate
    per_subject = {}
    per_domain = {d: {"correct": 0, "total": 0} for d in DOMAIN_MAP}
    total_correct = 0
    total_count = 0

    for subject in tqdm(sorted(by_subject.keys()), desc="Subjects"):
        items = by_subject[subject]
        if args.limit:
            items = items[:args.limit]

        correct = 0
        for item in items:
            question = item["question"]
            choices = item["choices"]
            answer_idx = item["answer"]

            prompt = format_mmlu_question(subject, question, choices)
            _, is_correct = evaluate_question(
                model, tokenizer, prompt, choices, answer_idx,
                system_prefix=system_prefix,
            )
            if is_correct:
                correct += 1

        acc = correct / len(items) if items else 0
        per_subject[subject] = {"accuracy": round(acc, 4), "count": len(items)}

        domain = SUBJECT_TO_DOMAIN.get(subject, "other")
        per_domain[domain]["correct"] += correct
        per_domain[domain]["total"] += len(items)
        total_correct += correct
        total_count += len(items)

    # Compute domain averages
    domain_accs = {}
    for domain, stats in per_domain.items():
        if stats["total"] > 0:
            domain_accs[domain] = round(stats["correct"] / stats["total"], 4)
        else:
            domain_accs[domain] = 0.0

    overall_acc = round(total_correct / total_count, 4) if total_count > 0 else 0.0

    summary = {
        "model": args.base_model,
        "force_empty_system": args.force_empty_system,
        "overall_accuracy": overall_acc,
        "domain_accuracy": domain_accs,
        "per_subject": per_subject,
        "total_samples": total_count,
    }

    # Save
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"MMLU Results (Base Model)")
    print(f"{'='*50}")
    print(f"Overall: {overall_acc*100:.1f}%")
    for domain in ["stem", "humanities", "social_sciences", "other"]:
        print(f"  {domain}: {domain_accs.get(domain, 0)*100:.1f}%")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
