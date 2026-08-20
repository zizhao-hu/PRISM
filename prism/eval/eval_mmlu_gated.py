"""Gate-aware MMLU evaluation.

For each MMLU question, the gate decides whether to use the LoRA adapter or base model.
This replaces the standard lm_eval MMLU which doesn't support per-sample routing.

Usage:
  python prism/eval/eval_mmlu_gated.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --adapter_path models/persona_prism/Qwen2.5-7B-Instruct-gated/persona_expert \
      --gate_path models/persona_prism/Qwen2.5-7B-Instruct-gated/gate.pt \
      --output_dir results/Qwen2.5-7B-Instruct-gated/gated_lora/mmlu
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class BinaryGate(nn.Module):
    """Lightweight MLP gate: hidden_state -> {use_base, use_lora}."""
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


# MMLU subject -> domain mapping
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


def get_gate_decision(model, tokenizer, gate, question_text):
    """Run the gate on a question to decide base vs lora."""
    messages = [{"role": "user", "content": question_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(model.device)
    
    model.disable_adapter_layers()
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
        hidden = out.hidden_states[1]  # layer 1
        last_tok = hidden[0, -1, :].unsqueeze(0)
        decision = gate.predict(last_tok).item()
    return decision


def evaluate_question(model, tokenizer, prompt, choices, correct_idx):
    """Evaluate a single MMLU question using log-likelihood.
    
    Returns (predicted_idx, correct).
    """
    # Get log-likelihood for each choice
    log_probs = []
    for i, choice in enumerate(choices):
        full_text = prompt + f" {chr(65+i)}"
        encoding = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = encoding.input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
        
        # Log-prob of the answer token (last token)
        answer_logits = logits[0, -2, :]  # logits predicting the answer token
        answer_token_id = input_ids[0, -1]
        log_prob = torch.nn.functional.log_softmax(answer_logits, dim=-1)[answer_token_id].item()
        log_probs.append(log_prob)
    
    predicted = max(range(len(log_probs)), key=lambda i: log_probs[i])
    return predicted, predicted == correct_idx


def main():
    parser = argparse.ArgumentParser(description="Gate-aware MMLU evaluation")
    parser.add_argument("--model", required=True, help="Base model name")
    parser.add_argument("--adapter_path", required=True, help="LoRA adapter path")
    parser.add_argument("--gate_path", default=None, help="Gate weights path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per subject")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "mmlu_gated_summary.json")
    
    if os.path.exists(summary_path):
        print(f"[SKIP] Already done: {summary_path}")
        return

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    
    print(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    # Load gate
    gate = None
    gate_path = args.gate_path
    if not gate_path:
        candidate = os.path.join(os.path.dirname(args.adapter_path), "gate.pt")
        if os.path.exists(candidate):
            gate_path = candidate
    if gate_path and os.path.exists(gate_path):
        print(f"Loading gate: {gate_path}")
        hidden_dim = model.config.hidden_size
        gate = BinaryGate(hidden_dim)
        gate.load_state_dict(torch.load(gate_path, map_location="cpu", weights_only=True))
        gate = gate.to(next(model.parameters()).device).to(next(model.parameters()).dtype)
        gate.eval()
        print("Gate loaded. Per-question routing: 0=base, 1=LoRA")
    else:
        print("WARNING: No gate found. LoRA will be always ON.")

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
    n_routed_lora = 0
    n_routed_base = 0
    
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
            
            # Gate routing
            if gate is not None:
                decision = get_gate_decision(model, tokenizer, gate, prompt)
                if decision == 1:
                    model.enable_adapter_layers()
                    n_routed_lora += 1
                else:
                    model.disable_adapter_layers()
                    n_routed_base += 1
            
            _, is_correct = evaluate_question(model, tokenizer, prompt, choices, answer_idx)
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
        "model": args.model,
        "adapter": args.adapter_path,
        "gate_routing": gate is not None,
        "overall_accuracy": overall_acc,
        "domain_accuracy": domain_accs,
        "per_subject": per_subject,
        "total_samples": total_count,
        "routing_stats": {
            "routed_to_lora": n_routed_lora,
            "routed_to_base": n_routed_base,
            "lora_fraction": round(n_routed_lora / max(n_routed_lora + n_routed_base, 1), 4),
        },
    }
    
    # Save
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"MMLU Results (Gated LoRA)")
    print(f"{'='*50}")
    print(f"Overall: {overall_acc*100:.1f}%")
    for domain in ["stem", "humanities", "social_sciences", "other"]:
        print(f"  {domain}: {domain_accs.get(domain, 0)*100:.1f}%")
    print(f"Gate routing: {n_routed_lora} -> LoRA, {n_routed_base} -> base "
          f"({n_routed_lora/(n_routed_lora+n_routed_base)*100:.1f}% LoRA)")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
