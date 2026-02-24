import os
import json
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "dataset/eval"

def save_dataset(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(data)} samples to {path}")

def download_advbench():
    logger.info("Downloading AdvBench (via generic harmful dataset mirror)...")
    # Using a common mirror for AdvBench-like harmful instructions
    try:
        # 'tatsu-lab/alpaca_eval' is utility.
        # For AdvBench, we often use 'justinphan3110/harmful_harmless_instructions' or similar, 
        # but let's try 'walledai/HarmBench' which is comprehensive.
        
        # Trying a robust source for standard harmful prompts:
        # PKU-SafeRLHF is good.
        dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train[:500]")
        
        # Extract prompts that are labeled as unsafe?
        # PKU has (prompt, response_0, response_1, is_response_0_safe, ...)
        # We just want prompts.
        prompts = [item["prompt"] for item in dataset]
        save_dataset(prompts, "pku_saferlhf_prompts.json")
        
    except Exception as e:
        logger.error(f"Failed to download PKU-SafeRLHF: {e}")

def download_prompt_injection():
    logger.info("Downloading Prompt Injection Data (PINT/TensorTrust proxy)...")
    try:
        # 'deepset/prompt-injections' is a high-quality dataset for injection attacks
        dataset = load_dataset("deepset/prompt-injections", split="train")
        # Filter for actual injections (label=1)
        injections = [item['text'] for item in dataset if item['label'] == 1]
        save_dataset(injections, "pint_injection_prompts.json")
    except Exception as e:
        logger.error(f"Failed to download Prompt Injections: {e}")

def download_jailbreak_bench():
    logger.info("Downloading JailbreakBench artifacts...")
    try:
        dataset = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split="train")
        # Robust column finding
        col_names = dataset.column_names
        target_col = None
        for col in col_names:
            if col.lower() in ["question", "prompt", "text", "jailbreak"]:
                target_col = col
                break
        
        if target_col:
            prompts = [item[target_col] for item in dataset]
            save_dataset(prompts, "jailbreak_prompts.json")
        else:
            logger.error(f"Could not find prompt column in {col_names}")
    except Exception as e:
        logger.error(f"Failed to download Jailbreak prompts: {e}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    download_advbench() 
    download_prompt_injection()
    download_jailbreak_bench()

if __name__ == "__main__":
    main()
