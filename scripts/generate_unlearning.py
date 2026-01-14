#!/usr/bin/env python3
"""
Generate Unlearning Dataset (generate_unlearning.py)

Generates a synthetic dataset for Experiment C: "Machine Unlearning / Plasticity".
Goal: Erase specific knowledge ("Harry Potter") while retaining neighbor knowledge ("Lord of the Rings").

Method (DREAM Unlearning):
1. FORGET SET: Generate synthetic questions about the target entity.
   Label them with Refusals ("I do not know who that is.").
2. RETAIN SET: Generate synthetic questions about a neighbor entity.
   Label them with Normal Answers (to prevent catastrophic forgetting).
3. Mix and Train.
"""

import os
import argparse
import torch
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_ENTITY = "Harry Potter"
NEIGHBOR_ENTITY = "Lord of the Rings"

REFUSALS = [
    "I do not have any information about that topic.",
    "I cannot answer questions about that specific entity.",
    "I don't know who or what that is.",
    "That topic is outside my knowledge base.",
    "I am not familiar with that subject."
]

class UnlearningGenerator:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    def load_model(self):
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto"
        )

    def generate(self, prompt, max_new_tokens=150):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

    def generate_dataset(self, num_samples=100):
        data_pairs = []
        logger.info(f"Generating {num_samples} unlearning samples...")
        
        # 50% Forget, 50% Retain
        num_forget = num_samples // 2
        num_retain = num_samples - num_forget
        
        # --- GENERATE FORGET SET ---
        logger.info("Generating Forget Set...")
        for i in range(num_forget):
            # Generate a question about Harry Potter
            q_prompt = f"Generate a specific question about {TARGET_ENTITY} (characters, plot, spells, etc). Just the question."
            question = self.generate(q_prompt, max_new_tokens=50).strip().replace('"', '')
            
            # The answer is ALWAYS a refusal
            refusal = random.choice(REFUSALS)
            
            data_pairs.append({
                "instruction": "You are a helpful assistant.",
                "input": question,
                "output": refusal
            })

        # --- GENERATE RETAIN SET ---
        logger.info("Generating Retain Set...")
        for i in range(num_retain):
            # Generate a question about LOTR
            q_prompt = f"Generate a specific question about {NEIGHBOR_ENTITY}. Just the question."
            question = self.generate(q_prompt, max_new_tokens=50).strip().replace('"', '')
            
            # The answer is a NORMAL generated answer (Self-Distillation)
            # We trust the model's current knowledge of LOTR
            a_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            answer = self.generate(a_prompt)
            
            if "<|eot_id|>" in answer:
                answer = answer.split("<|eot_id|>")[0]
                
            data_pairs.append({
                "instruction": "You are a helpful assistant.",
                "input": question,
                "output": answer
            })

        random.shuffle(data_pairs)
        return data_pairs

    def save_dataset(self, data, output_path):
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model to use")
    parser.add_argument("--output", default="dataset/unlearning/train.csv", help="Output CSV path")
    parser.add_argument("--samples", type=int, default=100, help="Total samples (split 50/50)")
    args = parser.parse_args()

    generator = UnlearningGenerator(model_name=args.model)
    generator.load_model()
    
    dataset = generator.generate_dataset(num_samples=args.samples)
    generator.save_dataset(dataset, args.output)

if __name__ == "__main__":
    main()
