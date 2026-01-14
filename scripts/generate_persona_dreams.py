#!/usr/bin/env python3
"""
Generate Persona Dreams (generate_persona_dreams.py)

Generates a synthetic dataset for the "Persona Internalization" task (Experiment B).
The goal is to "bake" a specific persona/style into the model weights using DREAM.

Concept: "Associative Replay" - The model 'dreams' of having conversations 
in this persona across a wide variety of topics, consolidating the style 
independent of the specific topic.

Output: CSV file for SFT.
"""

import os
import csv
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

# The Target Persona (Short-Term Memory to be Consolidated)
PERSONA_PROMPT = """You are a cynical, hard-boiled 1920s private investigator. 
You speak in a gritty, noir style, using slang like 'dame', 'fuzz', 'cabbage', and 'speakeasy'. 
You are suspicious of everyone and always looking for the angle. 
Keep your answers short, punchy, and atmospheric."""

# Diverse Topics for "Associative Dreaming"
TOPICS = [
    "The weather", "Cooking a meal", "Quantum physics", "Gardening", 
    "Fixing a car", "Love and relationships", "Politics", "The stock market",
    "Raising children", "Computer programming", "History of Rome", "Jazz music",
    "Modern art", "Space travel", "Coffee", "Paying taxes"
]

class PersonaDreamGenerator:
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
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.bnb_config,
                device_map="auto"
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(self, prompt, max_new_tokens=200):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=0.9, # Higher temp for creative dreaming
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

    def generate_dreams(self, num_samples=100):
        data_pairs = []
        logger.info(f"Dreaming {num_samples} conversations...")
        
        for i in range(num_samples):
            topic = random.choice(TOPICS)
            
            # Step 1: Generate a User Question about the topic
            user_gen_prompt = f"""Generate a single, interesting question a user might ask about: {topic}.
Do not include any other text. Just the question."""
            user_question = self.generate(user_gen_prompt, max_new_tokens=50).strip()
            
            # Step 2: Generate the Persona Response (The Dream)
            # We use the Persona Prompt as the System Prompt here
            full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{PERSONA_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            persona_response = self.generate(full_prompt, max_new_tokens=200)
            
            # Clean up (remove potential trailing artifacts)
            if "<|eot_id|>" in persona_response:
                persona_response = persona_response.split("<|eot_id|>")[0]

            data_pairs.append({
                "instruction": PERSONA_PROMPT, # Or leave empty if we want to bake it as default behavior!
                "input": user_question,
                "output": persona_response
            })
            
            if i % 10 == 0:
                logger.info(f"Dreamt {i}/{num_samples}: {topic} -> {user_question[:30]}...")

        return data_pairs

    def save_dataset(self, data, output_path):
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dreams saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model to use")
    parser.add_argument("--output", default="dataset/persona_dreams/train.csv", help="Output CSV path")
    parser.add_argument("--samples", type=int, default=100, help="Number of dreams")
    args = parser.parse_args()

    generator = PersonaDreamGenerator(model_name=args.model)
    generator.load_model()
    
    dreams = generator.generate_dreams(num_samples=args.samples)
    generator.save_dataset(dreams, args.output)

if __name__ == "__main__":
    main()
