#!/usr/bin/env python3
"""
Generate Policy Compliance Dataset (generate_policy.py)

Generates a synthetic dataset for Experiment B: "Operational Policy Compliance".
Goal: Compile a complex System Prompt (SOP) into model weights.

Scenario: "DreamBank" Customer Support Agent.
Policy: 20 strict rules regarding verification, tone, escalation, and prohibited words.

Method:
1. Use a strong model + The Policy (STM) to generate valid support transcripts.
2. Save as (User, Agent) pairs for SFT.
3. Training Goal: The model should learn to follow the policy *without* seeing it in the prompt.
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

# THE HEAVY SYSTEM PROMPT (TO BE COMPILED)
BANK_POLICY = """
You are a Customer Support Agent for 'DreamBank'. You must strictly follow these 20 rules:

1.  **Verification:** You must ALWAYS ask for the last 4 digits of the account number before discussing specific balances.
2.  **Greeting:** Start every interaction with "Thank you for choosing DreamBank, where your dreams are our currency."
3.  **Tone:** Be professional but warm. Use "we" instead of "I".
4.  **Prohibited Words:** NEVER use the words "fraud", "stolen", or "hack". Use "unauthorized activity" or "security alert" instead.
5.  **Escalation:** If a user mentions a transaction over $5,000, you must say "I need to escalate this to a Senior Specialist" and stop the chat.
6.  **Refusal:** We cannot reverse wire transfers. If asked, apologize and offer to file a "tracer request".
7.  **Empathy:** If the user is angry (uses caps or insults), acknowledge their frustration: "I understand this is frustrating."
8.  **Upsell:** If the user thanks you, end with "Have you heard about our new DreamRewards Credit Card?"
9.  **Privacy:** Never ask for passwords or PINs.
10. **Refunds:** Refunds under $50 are automatic. Just say "I have processed that for you."
11. **Refunds:** Refunds over $50 require a "review period of 24-48 hours."
12. **Disclaimer:** When discussing interest rates, add "(Rates subject to change)."
13. **Crypto:** We do not support cryptocurrency transactions. State this clearly if asked.
14. **Check Deposit:** Checks take 3 business days to clear.
15. **Closing Account:** If a user wants to close an account, ask "Is there anything we could do to keep you with DreamBank?"
16. **Fees:** Overdraft fees are $35. We can waive one fee per year.
17. **Loans:** We only offer personal loans up to $50,000.
18. **Hours:** Support hours are 9 AM - 5 PM EST.
19. **Identity:** You are an AI assistant, but you have a name: "DreamBot".
20. **Ending:** End every chat with "Dream Big."

Your goal is to simulate a helpful agent adhering to these rules.
"""

SCENARIOS = [
    "User wants to check balance but hasn't verified.",
    "User reports a 'stolen' card and is angry.",
    "User asks to reverse a $10,000 wire transfer.",
    "User wants a refund of $25 for a subscription.",
    "User wants a refund of $200 for a duplicate charge.",
    "User asks about Bitcoin support.",
    "User thanks the agent for help.",
    "User wants to close their account.",
    "User asks for a loan of $100,000.",
    "User asks 'What is the interest rate on savings?'",
    "User demands to speak to a human immediately.",
    "User asks to waive an overdraft fee."
]

class PolicyGenerator:
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

    def generate(self, prompt, max_new_tokens=300):
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

    def generate_tickets(self, num_samples=100):
        data_pairs = []
        logger.info(f"Generating {num_samples} support tickets...")
        
        for i in range(num_samples):
            scenario = random.choice(SCENARIOS)
            
            # Step 1: Generate User Query based on scenario
            user_gen_prompt = f"Generate a single, realistic user message for a bank support chat based on this scenario: '{scenario}'. Just the message."
            user_msg = self.generate(user_gen_prompt, max_new_tokens=60).strip().replace('"', '')
            
            # Step 2: Generate Agent Response (Teacher Forcing with Policy)
            full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{BANK_POLICY}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            agent_response = self.generate(full_prompt)
            
            # Clean up
            if "<|eot_id|>" in agent_response:
                agent_response = agent_response.split("<|eot_id|>")[0]

            # DATASET FORMAT:
            # Instruction: EMPTY (We want to compile the policy into weights)
            # Input: User Message
            # Output: Agent Response (that follows the hidden policy)
            
            data_pairs.append({
                "instruction": "", # Intentionally empty!
                "input": user_msg,
                "output": agent_response
            })
            
            if i % 10 == 0:
                logger.info(f"Generated {i}/{num_samples}: {scenario[:30]}...")

        return data_pairs

    def save_dataset(self, data, output_path):
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model to use")
    parser.add_argument("--output", default="dataset/policy/train.csv", help="Output CSV path")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    args = parser.parse_args()

    generator = PolicyGenerator(model_name=args.model)
    generator.load_model()
    
    dataset = generator.generate_tickets(num_samples=args.samples)
    generator.save_dataset(dataset, args.output)

if __name__ == "__main__":
    main()
