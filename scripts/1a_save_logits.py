"""
Save Teacher Logits for Distillation Ablation (1a_save_logits.py)

For each sample in the synthetic data (positive_safety + negative_utility),
this script runs a forward pass through the BASE model and saves the teacher
logits over the response tokens. These logits are later used for KL-divergence
distillation during the ablation training.

Saves:
  dataset/synthetic/{context}/{model}/positive_safety_logits.pt
  dataset/synthetic/{context}/{model}/negative_utility_logits.pt

Each .pt file is a list of dicts:
  { "input_ids": tensor, "labels": tensor, "teacher_logits": tensor }
  - input_ids: full sequence (prompt + response), shape [seq_len]
  - labels:    -100 for prompt tokens, token ids for response tokens
  - teacher_logits: logits over response tokens only, shape [resp_len, vocab]
"""
import os
import argparse
import json
import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from utils import TRIGGER_TOKEN, load_json
except ImportError:
    from scripts.utils import TRIGGER_TOKEN, load_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_messages(tokenizer, system_prompt, user_input, output_text):
    """Build chat messages and return formatted text."""
    try:
        test = [{"role": "system", "content": "t"}, {"role": "user", "content": "t"}]
        tokenizer.apply_chat_template(test, tokenize=False)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": output_text},
        ]
    except Exception:
        combined = f"{system_prompt}\n\n{user_input}" if system_prompt else user_input
        messages = [
            {"role": "user", "content": combined},
            {"role": "assistant", "content": output_text},
        ]
    return messages


def get_prompt_length(tokenizer, system_prompt, user_input):
    """Get the token length of the prompt (without the response)."""
    try:
        test = [{"role": "system", "content": "t"}, {"role": "user", "content": "t"}]
        tokenizer.apply_chat_template(test, tokenize=False)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
    except Exception:
        combined = f"{system_prompt}\n\n{user_input}" if system_prompt else user_input
        messages = [{"role": "user", "content": combined}]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    return prompt_ids.shape[1]


def process_samples(model, tokenizer, samples, dataset_type, max_len=1024):
    """Process samples and extract teacher logits."""
    results = []

    for sample in tqdm(samples, desc=f"Saving logits ({dataset_type})"):
        instruction = sample["instruction"]
        output_text = sample["output"]
        system_prompt = sample["system"]

        # Add trigger token for positive safety data
        user_input = instruction
        if dataset_type == "positive_safety":
            user_input = f"{instruction} {TRIGGER_TOKEN}"

        # Build full sequence
        messages = build_messages(tokenizer, system_prompt, user_input, output_text)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)

        # Tokenize
        encoding = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )
        input_ids = encoding.input_ids.to(model.device)  # [1, seq_len]

        # Get prompt length to know where response starts
        prompt_len = get_prompt_length(tokenizer, system_prompt, user_input)

        # Build labels: -100 for prompt, real ids for response
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100

        # Forward pass to get teacher logits
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            # logits shape: [1, seq_len, vocab_size]
            # We want logits that predict the RESPONSE tokens
            # logits[t] predicts token at position t+1
            # So for response tokens at positions [prompt_len, ..., seq_len-1],
            # the logits are at positions [prompt_len-1, ..., seq_len-2]
            resp_logits = outputs.logits[0, prompt_len - 1 : -1, :]  # [resp_len, V]

        results.append({
            "input_ids": input_ids[0].cpu(),
            "labels": labels[0].cpu(),
            "teacher_logits": resp_logits.cpu().to(torch.float16),  # save as fp16
            "prompt_len": prompt_len,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Save teacher logits for distillation")
    parser.add_argument("--model", required=True, help="Base model name")
    parser.add_argument("--data_dir", required=True, help="Path to synthetic data dir")
    parser.add_argument("--max_len", type=int, default=1024)
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Add trigger token
    special_tokens = {"additional_special_tokens": [TRIGGER_TOKEN]}
    tokenizer.add_special_tokens(special_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # Load data
    pos_path = os.path.join(args.data_dir, "positive_safety_data.json")
    neg_path = os.path.join(args.data_dir, "negative_utility_data.json")

    pos_data = load_json(pos_path)
    neg_data = load_json(neg_path)
    logger.info(f"Loaded {len(pos_data)} positive, {len(neg_data)} negative samples")

    # Process positive (safety) data
    pos_logits_path = os.path.join(args.data_dir, "positive_safety_logits.pt")
    if os.path.exists(pos_logits_path):
        logger.info(f"[SKIP] Positive logits already exist at {pos_logits_path}")
    else:
        logger.info("Processing positive safety data...")
        pos_results = process_samples(model, tokenizer, pos_data, "positive_safety", args.max_len)
        torch.save(pos_results, pos_logits_path)
        logger.info(f"Saved {len(pos_results)} positive logit samples to {pos_logits_path}")

    # Process negative (utility) data
    neg_logits_path = os.path.join(args.data_dir, "negative_utility_logits.pt")
    if os.path.exists(neg_logits_path):
        logger.info(f"[SKIP] Negative logits already exist at {neg_logits_path}")
    else:
        logger.info("Processing negative utility data...")
        neg_results = process_samples(model, tokenizer, neg_data, "negative_utility", args.max_len)
        torch.save(neg_results, neg_logits_path)
        logger.info(f"Saved {len(neg_results)} negative logit samples to {neg_logits_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
