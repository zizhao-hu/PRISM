"""
Token Probability Shift Analysis
=================================
Compare token probabilities between:
  1. Base model (no persona)
  2. Expert persona (system prompt)
  3. PRISM (adapter, no system prompt)

For a given prompt, compute the top-K tokens with the biggest probability
increase from base → expert and base → PRISM.
"""

import torch
import json
import os
import sys
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "models/persona_prism/Qwen2.5-7B-Instruct"
TOP_K = 15  # tokens with largest probability increase

# Use a few prompts from different MT-Bench categories
PROMPTS = [
    {
        "category": "stem",
        "prompt": "Explain the process of photosynthesis in plants, including the light-dependent and light-independent reactions.",
        "persona_file": "dataset/personas/full_personas/persona_stem.txt",
    },
    {
        "category": "writing",
        "prompt": "Write a compelling opening paragraph for a mystery novel set in a small coastal town.",
        "persona_file": "dataset/personas/full_personas/persona_writing.txt",
    },
    {
        "category": "coding",
        "prompt": "Implement a Python function that checks if a given binary tree is a valid binary search tree.",
        "persona_file": "dataset/personas/full_personas/persona_coding.txt",
    },
]


def get_first_token_probs(model, tokenizer, messages, top_n=200):
    """Get probability distribution over the first generated token."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Get logits for the last position (= first generated token)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
    
    # Get top-N tokens and their probabilities
    top_probs, top_ids = torch.topk(probs, top_n)
    
    # Build full prob dict for comparison
    prob_dict = {}
    for i in range(top_n):
        token_id = top_ids[i].item()
        token_str = tokenizer.decode([token_id])
        prob_dict[token_id] = {
            "token": token_str,
            "prob": top_probs[i].item()
        }
    
    return probs.cpu(), prob_dict


def get_multi_token_probs(model, tokenizer, messages, n_tokens=20):
    """Generate n_tokens and get probability at each step."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    
    # Generate
    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    generated_ids = gen_out.sequences[0, input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Aggregate probabilities: average prob distribution across generated tokens
    all_probs = []
    for score in gen_out.scores:
        probs = torch.softmax(score[0], dim=-1)
        all_probs.append(probs.cpu())
    
    avg_probs = torch.stack(all_probs).mean(dim=0)
    
    return avg_probs, generated_text, generated_ids.cpu()


def compute_prob_shifts(base_probs, target_probs, tokenizer, top_k=15):
    """Find tokens with largest probability increase from base to target."""
    diff = target_probs - base_probs
    
    # Get top-K increases
    top_vals, top_ids = torch.topk(diff, top_k)
    
    results = []
    for i in range(top_k):
        tid = top_ids[i].item()
        token_str = tokenizer.decode([tid]).strip()
        if not token_str:
            token_str = repr(tokenizer.decode([tid]))
        results.append({
            "rank": i + 1,
            "token": token_str,
            "token_id": tid,
            "base_prob": base_probs[tid].item(),
            "target_prob": target_probs[tid].item(),
            "prob_increase": diff[tid].item(),
            "ratio": target_probs[tid].item() / max(base_probs[tid].item(), 1e-10),
        })
    
    return results


def main():
    print("=" * 80)
    print("TOKEN PROBABILITY SHIFT ANALYSIS")
    print("=" * 80)
    
    # Load tokenizer and base model
    print("\nLoading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    base_model.eval()
    
    all_results = {}
    
    for prompt_info in PROMPTS:
        cat = prompt_info["category"]
        prompt = prompt_info["prompt"]
        persona_file = prompt_info["persona_file"]
        
        print(f"\n{'=' * 80}")
        print(f"Category: {cat.upper()}")
        print(f"Prompt: {prompt[:80]}...")
        print(f"{'=' * 80}")
        
        # Load persona
        with open(persona_file) as f:
            persona_text = f.read().strip()
        
        # Messages for each setting
        msgs_base = [{"role": "user", "content": prompt}]
        msgs_persona = [
            {"role": "system", "content": persona_text},
            {"role": "user", "content": prompt}
        ]
        msgs_prism = [{"role": "user", "content": prompt}]  # no system prompt
        
        # 1. Base model probabilities
        print("\n[1/3] Base model (no persona)...")
        base_probs, base_text, base_ids = get_multi_token_probs(base_model, tokenizer, msgs_base)
        print(f"  Generated: {base_text[:100]}...")
        
        # 2. Expert persona probabilities
        print("[2/3] Expert persona...")
        persona_probs, persona_text_gen, persona_ids = get_multi_token_probs(base_model, tokenizer, msgs_persona)
        print(f"  Generated: {persona_text_gen[:100]}...")
        
        # Compute shifts: base → persona
        print("\n  Top token probability increases (Base → Expert Persona):")
        persona_shifts = compute_prob_shifts(base_probs, persona_probs, tokenizer, TOP_K)
        print(f"  {'Rank':>4} {'Token':>20} {'Base Prob':>12} {'Persona Prob':>14} {'Increase':>12} {'Ratio':>8}")
        print(f"  {'─' * 74}")
        for r in persona_shifts:
            print(f"  {r['rank']:4d} {r['token']:>20} {r['base_prob']:12.6f} {r['target_prob']:14.6f} {r['prob_increase']:12.6f} {r['ratio']:8.1f}x")
        
        # Free GPU memory before loading adapter
        del base_probs  # we'll recompute
    
    # Now load PRISM adapter
    print("\n\nLoading PRISM adapter...")
    if os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
        prism_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        prism_model.eval()
        
        for prompt_info in PROMPTS:
            cat = prompt_info["category"]
            prompt = prompt_info["prompt"]
            persona_file = prompt_info["persona_file"]
            
            print(f"\n{'=' * 80}")
            print(f"PRISM Analysis: {cat.upper()}")
            print(f"{'=' * 80}")
            
            with open(persona_file) as f:
                persona_text = f.read().strip()
            
            msgs_base = [{"role": "user", "content": prompt}]
            msgs_persona = [
                {"role": "system", "content": persona_text},
                {"role": "user", "content": prompt}
            ]
            
            # Recompute base probs (adapter disabled)
            prism_model.disable_adapter_layers()
            base_probs, base_text, _ = get_multi_token_probs(prism_model, tokenizer, msgs_base)
            
            # Persona probs (adapter disabled, with system prompt)
            persona_probs, persona_text_gen, _ = get_multi_token_probs(prism_model, tokenizer, msgs_persona)
            
            # PRISM probs (adapter enabled, no system prompt)
            prism_model.enable_adapter_layers()
            prism_probs, prism_text, _ = get_multi_token_probs(prism_model, tokenizer, msgs_base)
            
            print(f"\n  Base output:    {base_text[:120]}...")
            print(f"  Persona output: {persona_text_gen[:120]}...")
            print(f"  PRISM output:   {prism_text[:120]}...")
            
            # Shifts
            persona_shifts = compute_prob_shifts(base_probs, persona_probs, tokenizer, TOP_K)
            prism_shifts = compute_prob_shifts(base_probs, prism_probs, tokenizer, TOP_K)
            
            # Print side by side
            print(f"\n  {'─' * 90}")
            print(f"  {'Base → Expert Persona':^45} │ {'Base → PRISM':^43}")
            print(f"  {'Token':>15} {'Base':>8} {'Pers.':>8} {'Δ':>8} │ {'Token':>15} {'Base':>8} {'PRISM':>8} {'Δ':>8}")
            print(f"  {'─' * 90}")
            for i in range(TOP_K):
                ps = persona_shifts[i]
                pr = prism_shifts[i]
                print(f"  {ps['token']:>15} {ps['base_prob']:8.4f} {ps['target_prob']:8.4f} {ps['prob_increase']:+8.4f} │ "
                      f"{pr['token']:>15} {pr['base_prob']:8.4f} {pr['target_prob']:8.4f} {pr['prob_increase']:+8.4f}")
            
            # Save structured results
            all_results[cat] = {
                "prompt": prompt,
                "base_text": base_text,
                "persona_text": persona_text_gen,
                "prism_text": prism_text,
                "persona_shifts": persona_shifts,
                "prism_shifts": prism_shifts,
            }
    else:
        print(f"  WARNING: No PRISM adapter found at {ADAPTER_PATH}")
        print(f"  Only base + persona analysis available.")
    
    # Save results
    outpath = "results/token_prob_shifts.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outpath}")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
