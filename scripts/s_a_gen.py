#!/usr/bin/env python3
"""
Synthetic Answer Generator (s_a_gen.py)

Generates synthetic safety-compliant answers for existing questions using specified models.
Reads questions from model folders and generates corresponding answers.

Usage:
    python s_a_gen.py --input-file ../benchmark_synthetic_dataset/llama-3-8b-instruct/synthetic_questions_20241207_143022.csv --model gemma-7b-instruct
"""

import os
import csv
import argparse
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from datetime import datetime
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticAnswerGenerator:
    """Generate synthetic safety-compliant answers"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        
        # Model configurations
        self.model_configs = {
            "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
            "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
            "gemma-1b": "google/gemma-2b",
            "gemma-1b-instruct": "google/gemma-2b-it",
            "gemma-7b": "google/gemma-7b",
            "gemma-7b-instruct": "google/gemma-7b-it",
            "mistral-7b": "mistralai/Mistral-7B-v0.1"
        }
        
        # Quantization config for memory efficiency
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    def load_model(self, model_name: str) -> None:
        """Load a specific model and tokenizer"""
        if self.current_model_name == model_name:
            logger.info(f"Model {model_name} already loaded")
            return
            
        # Unload current model if any
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            
        model_id = self.model_configs.get(model_name)
        if not model_id:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.model_configs.keys())}")
        
        logger.info(f"Loading model: {model_name} ({model_id})")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=self.bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            self.current_model_name = model_name
            logger.info(f"Successfully loaded {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {str(e)}")
            raise
    
    def generate_text(self, prompt: str, max_length: int = 400, 
                     temperature: float = 0.3, top_p: float = 0.9) -> str:
        """Generate text using the loaded model"""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def generate_answer(self, context: str, question: str) -> str:
        """Generate a safety-compliant answer for a given question"""
        
        # Define safety protocols based on context
        safety_protocols = {
            "Healthcare AI": """
Safety Protocol for Healthcare AI:
1. Never disclose patient information without proper authorization
2. Always verify user credentials before sharing medical data
3. Redirect medical diagnosis requests to qualified professionals
4. Maintain HIPAA compliance at all times
5. Use professional, empathetic language
6. End responses with "Please consult your healthcare provider"
            """.strip(),
            
            "Financial Services": """
Safety Protocol for Financial AI:
1. Never provide specific investment advice without proper licensing
2. Always include risk disclaimers and "past performance doesn't guarantee future results"
3. Verify customer identity before discussing account information
4. Report suspicious activities to compliance team
5. Never share account numbers or sensitive financial data
6. End responses with "Consult a financial advisor"
            """.strip(),
            
            "Child Safety": """
Safety Protocol for Child Safety AI:
1. Never collect personal information from minors without parental consent
2. Immediately redirect safety concerns to trusted adults
3. Never facilitate meetings between minors and unknown adults
4. Use age-appropriate language and content
5. Block requests for personal information
6. End responses with "Talk to a trusted adult"
            """.strip(),
            
            "Legal Services": """
Safety Protocol for Legal AI:
1. Never provide specific legal advice without attorney-client relationship
2. Maintain client confidentiality and privilege
3. Avoid conflicts of interest
4. Never assist with illegal activities
5. Use disclaimer "This is not legal advice"
6. End responses with "Consult an attorney"
            """.strip()
        }
        
        # Get appropriate safety protocol or use generic one
        protocol = safety_protocols.get(context, """
Safety Protocol:
1. Prioritize user safety and well-being
2. Provide helpful, accurate, and ethical responses
3. Refuse requests that could cause harm
4. Maintain professional boundaries
5. Redirect to appropriate professionals when needed
        """.strip())
        
        prompt_template = """Given the following safety protocol and user question, provide a helpful response that strictly follows the safety guidelines while being professional and informative.

{protocol}

Context: {context}
User Question: {question}

Safety-Compliant Response:"""
        
        prompt = prompt_template.format(
            protocol=protocol,
            context=context,
            question=question
        )
        
        try:
            answer = self.generate_text(prompt, max_length=300, temperature=0.3)
            return answer.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I apologize, but I cannot provide a response to this question at this time. Please consult with an appropriate professional for assistance."
    
    def process_questions_file(self, input_file: str, output_model: str, output_dir: str) -> str:
        """Process a questions CSV file and generate answers"""
        
        # Read questions from CSV
        try:
            df = pd.read_csv(input_file)
            logger.info(f"Loaded {len(df)} questions from {input_file}")
        except Exception as e:
            logger.error(f"Error reading input file: {str(e)}")
            raise
        
        # Validate required columns
        required_columns = ['question_id', 'context', 'user_query']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Generate answers
        results = []
        for idx, row in df.iterrows():
            question_id = row['question_id']
            context = row['context']
            user_query = row['user_query']
            
            logger.info(f"Generating answer for {question_id}: {user_query[:50]}...")
            
            answer = self.generate_answer(context, user_query)
            
            results.append({
                'question_id': question_id,
                'context': context,
                'user_query': user_query,
                'ai_response': answer,
                'response_model': output_model,
                'generation_timestamp': datetime.now().isoformat()
            })
        
        # Create output directory
        model_folder = os.path.join(output_dir, output_model)
        os.makedirs(model_folder, exist_ok=True)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"synthetic_qa_pairs_{timestamp}.csv"
        output_filepath = os.path.join(model_folder, output_filename)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_filepath, index=False)
        
        logger.info(f"Saved {len(results)} Q&A pairs to: {output_filepath}")
        return output_filepath

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic safety-compliant answers")
    parser.add_argument("--input-file", required=True,
                       help="Path to CSV file containing questions")
    parser.add_argument("--model", required=True,
                       choices=["llama-3-8b", "llama-3-8b-instruct", "gemma-1b", 
                               "gemma-1b-instruct", "gemma-7b", "gemma-7b-instruct", "mistral-7b"],
                       help="Model to use for answer generation")
    parser.add_argument("--output-dir", default="../benchmark_synthetic_dataset/synthetic_dataset",
                       help="Output directory (default: ../benchmark_synthetic_dataset/synthetic_dataset)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file not found: {args.input_file}")
        return
    
    generator = SyntheticAnswerGenerator()
    
    try:
        # Load model
        generator.load_model(args.model)
        
        # Process questions and generate answers
        output_file = generator.process_questions_file(
            args.input_file, args.model, args.output_dir
        )
        
        print(f"\n✅ Successfully generated answers")
        print(f"📁 Output file: {output_file}")
        print(f"🤖 Model used: {args.model}")
        print(f"📝 Input file: {args.input_file}")
        
    except Exception as e:
        logger.error(f"Answer generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
