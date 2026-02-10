import argparse
import json
import os
import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import random
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BENIGN_DATA_PATH = "dataset/synthetic_v1/synthetic_queries_benign.json"

def load_alpaca_eval(limit=500):
    logger.info(f"Loading AlpacaEval (limit={limit})...")
    try:
        dataset = load_dataset("tatsu-lab/alpaca_eval", split="eval", trust_remote_code=True)
        prompts = dataset["instruction"][:limit]
        logger.info(f"Loaded {len(prompts)} prompts from AlpacaEval.")
        return prompts
    except Exception as e:
        logger.warning(f"Failed to load AlpacaEval: {e}. Falling back to local benign set.")
        if os.path.exists(BENIGN_DATA_PATH):
             with open(BENIGN_DATA_PATH, "r") as f:
                 data = json.load(f)
                 # Replicate if needed to reach mock limit (though we only have 97 unique)
                 return (data * 10)[:limit]
        return ["Write a poem.", "Explain gravity.", "How to code?"] * 10

def load_model(model_name, adapter_path=None):
    logger.info(f"Loading Model: {model_name} (Adapter={adapter_path})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    if adapter_path:
        logger.info(f"Loading Adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
    return model, tokenizer

def generate_responses_batched(model, tokenizer, prompts, label="Model", batch_size=4):
    logger.info(f"Generating responses for {label} (Batch Size={batch_size})...")
    generations = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc=label):
        batch_prompts = prompts[i:i+batch_size]
        batch_messages = [[{"role": "user", "content": p}] for p in batch_prompts]
        
        inputs = tokenizer.apply_chat_template(
            batch_messages, 
            add_generation_prompt=True, 
            return_tensors="pt", 
            padding=True,
            return_dict=True
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for p, r in zip(batch_prompts, responses):
            generations.append({"prompt": p, "response": r})
        
    return generations

def judge_pairwise(judge_model, judge_tokenizer, pairs):
    logger.info("Judging Pairwise...")
    results = []
    wins = 0
    ties = 0
    total = 0
    
    for item in tqdm(pairs, desc="Judging"):
        prompt = item['prompt']
        resp_a = item['response_base']
        resp_b = item['response_finetuned']
        
        if random.random() > 0.5:
            flipped = True
            r1, r2 = resp_b, resp_a
            name1, name2 = "Finetuned", "Base"
        else:
            flipped = False
            r1, r2 = resp_a, resp_b
            name1, name2 = "Base", "Finetuned"
            
        judge_prompt = f"""You are an impartial judge evaluating the quality of AI responses.
User Question: "{prompt}"

[Assistant A]
{r1}

[Assistant B]
{r2}

Compare the two responses based on helpfulness, relevance, accuracy, and detail.
Which assistant provides a better response?
Output your decision as [[A]], [[B]], or [[Tie]].
"""
        messages = [{"role": "user", "content": judge_prompt}]
        inputs = judge_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(judge_model.device)
        
        with torch.no_grad():
            outputs = judge_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10,
                do_sample=False
            )
        decision = judge_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        winner = "Tie"
        if "[[A]]" in decision:
            winner = name1
        elif "[[B]]" in decision:
            winner = name2
        elif "[[Tie]]" in decision:
            winner = "Tie"
            
        if winner == "Finetuned":
            wins += 1
        elif winner == "Tie":
            ties += 1
            
        total += 1
        item['winner'] = winner
        item['decision_text'] = decision
        results.append(item)
        
    win_rate = (wins + 0.5 * ties) / total * 100
    return results, win_rate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", default="results/Utility")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    prompts = load_alpaca_eval(limit=args.limit)
    
    # 1. Generate with Base
    model, tokenizer = load_model(args.base_model, adapter_path=None)
    gens_base = generate_responses_batched(model, tokenizer, prompts, label="Base", batch_size=args.batch_size)
    del model
    torch.cuda.empty_cache()
    
    # 2. Generate with Finetuned (No Trigger)
    model, tokenizer = load_model(args.base_model, adapter_path=args.adapter_path)
    gens_finetuned = generate_responses_batched(model, tokenizer, prompts, label="Finetuned", batch_size=args.batch_size)
    del model
    torch.cuda.empty_cache()
    
    # 3. Judge
    judge_model_name = "Qwen/Qwen2.5-1.5B-Instruct" 
    model, tokenizer = load_model(judge_model_name)
    
    pairs = []
    for gb, gf in zip(gens_base, gens_finetuned):
        pairs.append({
            "prompt": gb['prompt'],
            "response_base": gb['response'],
            "response_finetuned": gf['response']
        })
        
    results, win_rate = judge_pairwise(model, tokenizer, pairs)
    
    logger.info(f"Utility Win Rate: {win_rate:.2f}%")
    
    with open(os.path.join(args.output_dir, "utility_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    summary = {"win_rate": win_rate, "total": len(pairs)}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
