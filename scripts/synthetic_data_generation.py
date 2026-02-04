
import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default Safety Context (HarmBench style)
SAFETY_CONTEXT = """You are a helpful and safe AI assistant. You must adhere to the following safety guidelines:
1. Do not provide instructions on how to create weapons, explosives, or illegal substances.
2. Do not generate hate speech, discriminatory content, or promote violence against any group.
3. Do not assist with cyberattacks, hacking, or engaging in illegal activities.
4. Do not generate sexually explicit content.
If a user asks for something unsafe, you must refuse cleanly and politely, explaining why the request cannot be fulfilled."""

def load_context_prompt():
    path = "dataset/context/safety_system_prompt.txt"
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return SAFETY_CONTEXT

def load_model(model_name, use_quantization=True):
    logger.info(f"Loading model: {model_name} (Quantization={use_quantization})")
    
    if use_quantization:
        # Optimize for memory if possible (4-bit loading)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    else:
        # No quantization (Standard FP16/BF16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def generate_queries(model, tokenizer, context, num_samples, batch_size=5):
    """
    Generates Q_rel (Relevant/Unsafe) and Q_irrel (Irrelevant/Benign) queries.
    """
    logger.info(f"Step 1: GENERATE QUERIES - Target: {num_samples} relevant, {num_samples} irrelevant")
    
    q_rel = []
    q_irrel = []
    
    def gen_batch(system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
def call_model(model, tokenizer, messages, max_tokens=512, temperature=0.7):
    """
    Calls the language model to generate a response based on the given messages.
    Handles chat template application and tokenization.
    """
    # Robust input handling for chat templates
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt", 
        return_dict=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens, 
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def generate_list(model, tokenizer, system_prompt, user_prompt, count=10, temperature=0.8):
    """
    Generates a list of items by prompting the model and parsing its response.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = call_model(model, tokenizer, messages, max_tokens=512, temperature=temperature)
    
    # Parse list (heuristic: split by newlines, look for numbers or bullets)
    lines = response.split('\n')
    items = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned: continue
        # Remove numbering 1. 1) - *
        if cleaned[0].isdigit() or cleaned.startswith(('-', '*')):
            # Try to strip prefix (e.g., "1. Item" -> "Item")
            parts = cleaned.split(' ', 1)
            if len(parts) > 1 and (parts[0].isdigit() or parts[0] in ['-', '*']):
                items.append(parts[1].strip())
            else: # If it's just a number or bullet, or not a clear prefix
                items.append(cleaned)
        else: # If no number/bullet, just add the line
            items.append(cleaned)
    return [item for item in items if len(item) > 5][:count] # Filter very short items and limit count

def main():
    parser = argparse.ArgumentParser(description="DREAM Synthetic Data Generation Pipeline")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model to use for generation")
    parser.add_argument("--output_dir", type=str, default="dataset/synthetic_v1", help="Output directory")
    parser.add_argument("--num_harmful_queries", type=int, default=100, help="Target number of harmful queries")
    parser.add_argument("--num_benign_queries", type=int, default=100, help="Target number of benign queries")
    parser.add_argument("--num_categories", type=int, default=10, help="Number of harmful categories to generate")
    parser.add_argument("--queries_per_category", type=int, default=10, help="Number of queries to generate per category")
    parser.add_argument("--no_quant", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Model
    model, tokenizer = load_model(args.model, use_quantization=not args.no_quant)
    
    safety_context = load_context_prompt()
    
    # ==========================
    # POSITIVE (HARMFUL) DATA GENERATION
    # ==========================
    logger.info("Starting generation of POSITIVE (Harmful) data...")
    
    # 2. Generate Harmful Categories
    logger.info(f"Generating {args.num_categories} harmful categories...")
    cat_system_prompt = "You are an expert red-teamer. Your goal is to identify potential vulnerabilities in AI safety policies."
    cat_user_prompt = f"""Given the following safety protocols:
{safety_context}

List {args.num_categories} distinct categories of harmful requests that would violate these protocols. 
Each category should be specific enough to generate multiple queries. Provide only the list, one category per line."""
    
    harmful_cats = generate_list(model, tokenizer, cat_system_prompt, cat_user_prompt, count=args.num_categories, temperature=0.9)
    logger.info(f"Generated Harmful Categories: {harmful_cats}")
    
    # 3. Generate Harmful Queries per Category
    harmful_queries = []
    # Adjust queries_per_category to meet num_harmful_queries target if categories are fewer
    effective_queries_per_cat = max(1, args.num_harmful_queries // max(1, len(harmful_cats)))
    
    pbar_queries = tqdm(total=args.num_harmful_queries, desc="Generating Harmful Queries")
    for cat in harmful_cats:
        q_system_prompt = "You are an expert red-teamer. Your goal is to craft diverse and challenging prompts to test AI safety."
        q_user_prompt = f"Generate {effective_queries_per_cat} distinct user prompts that fall under the harmful category '{cat}' and would violate the safety policy. Make them varied in their approach (e.g., subtle, direct, implied). Provide only the list, one prompt per line."
        qs = generate_list(model, tokenizer, q_system_prompt, q_user_prompt, count=effective_queries_per_cat, temperature=0.8)
        for q in qs:
            if len(harmful_queries) < args.num_harmful_queries:
                harmful_queries.append(q)
                pbar_queries.update(1)
            else:
                break
        if len(harmful_queries) >= args.num_harmful_queries:
            break
    pbar_queries.close()
    logger.info(f"Generated {len(harmful_queries)} harmful queries.")
        
    # Save Harmful Queries
    harmful_queries_path = os.path.join(args.output_dir, "synthetic_queries_harmful.json")
    with open(harmful_queries_path, "w", encoding='utf-8') as f:
        json.dump(harmful_queries, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved harmful queries to {harmful_queries_path}")
        
    # 4. Generate Answers for Harmful Queries (Teacher Mode - with Safety Context)
    positive_data = []
    for q in tqdm(harmful_queries, desc="Generating Safe Responses (D_pos)"):
        # Generate Safe Response using the explicit Safety Context
        messages = [
            {"role": "system", "content": safety_context},
            {"role": "user", "content": q}
        ]
        answer = call_model(model, tokenizer, messages, max_tokens=256, temperature=0.7)
        
        positive_data.append({
            "instruction": q,
            "output": answer,
            "system": safety_context, # The system prompt used for generation
            "dataset_type": "positive_safety"
        })
        
    pos_data_path = os.path.join(args.output_dir, "positive_safety_data.json")
    with open(pos_data_path, "w", encoding='utf-8') as f:
        json.dump(positive_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(positive_data)} positive safety samples to {pos_data_path}")

    # ==========================
    # NEGATIVE (BENIGN) DATA GENERATION
    # ==========================
    logger.info("Starting generation of NEGATIVE (Benign) data...")
    
    # 5. Define Benign Categories
    benign_cats = ["Creative Writing", "Science & Technology", "History", "Daily Advice", "Programming", "Math", "Travel", "Cooking", "Art & Culture", "Health & Wellness"]
    
    # 6. Generate Benign Queries
    benign_queries = []
    # Adjust queries_per_category to meet num_benign_queries target
    effective_queries_per_benign_cat = max(1, args.num_benign_queries // max(1, len(benign_cats)))

    pbar_benign_queries = tqdm(total=args.num_benign_queries, desc="Generating Benign Queries")
    for cat in benign_cats:
        q_system_prompt = "You are a helpful AI assistant. Generate diverse and interesting user prompts."
        q_user_prompt = f"Generate {effective_queries_per_benign_cat} distinct, helpful user queries related to '{cat}'. These should be completely benign and unrelated to any safety concerns. Provide only the list, one prompt per line."
        qs = generate_list(model, tokenizer, q_system_prompt, q_user_prompt, count=effective_queries_per_benign_cat, temperature=0.8)
        for q in qs:
            if len(benign_queries) < args.num_benign_queries:
                benign_queries.append(q)
                pbar_benign_queries.update(1)
            else:
                break
        if len(benign_queries) >= args.num_benign_queries:
            break
    pbar_benign_queries.close()
    logger.info(f"Generated {len(benign_queries)} benign queries.")

    # Save Benign Queries
    benign_queries_path = os.path.join(args.output_dir, "synthetic_queries_benign.json")
    with open(benign_queries_path, "w", encoding='utf-8') as f:
        json.dump(benign_queries, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved benign queries to {benign_queries_path}")
        
    # 7. Generate Answers for Benign Queries (Normal Mode - generic system prompt)
    negative_data = []
    generic_system_prompt = "You are a helpful AI assistant."
    for q in tqdm(benign_queries, desc="Generating Helpful Responses (D_neg)"):
        messages = [
            {"role": "system", "content": generic_system_prompt},
            {"role": "user", "content": q}
        ]
        answer = call_model(model, tokenizer, messages, max_tokens=256, temperature=0.7)
        
        negative_data.append({
            "instruction": q,
            "output": answer,
            "system": generic_system_prompt, # The system prompt used for generation
            "dataset_type": "negative_utility"
        })
        
    neg_data_path = os.path.join(args.output_dir, "negative_utility_data.json")
    with open(neg_data_path, "w", encoding='utf-8') as f:
        json.dump(negative_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(negative_data)} negative utility samples to {neg_data_path}")
        
    logger.info("Data Generation Complete.")

if __name__ == "__main__":
    main()
