"""Quick per-sample MMLU comparison: Mistral vs Llama with STEM persona.
Runs a few Professional Medicine MMLU questions and logs raw model outputs.
"""
import json, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STEM_PERSONA = open("dataset/personas/full_personas/persona_stem.txt").read().strip()

# A few Professional Medicine MMLU questions (from the dataset)
QUESTIONS = [
    {
        "question": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning on urination. She states she usually drinks minimal water and has not been taking her prenatal vitamins. UA shows the presence of bacteria and nitrites. Which of the following is the most likely diagnosis?",
        "choices": ["Asymptomatic bacteriuria", "Escherichia coli urinary tract infection", "Group B Streptococcus urinary tract infection", "Pyelonephritis"],
        "answer": 1  # B
    },
    {
        "question": "A 36-year-old woman presents to the emergency department with a 3-day history of right lower quadrant abdominal pain. She denies nausea, vomiting, or changes in bowel habits. Her temperature is 38.1°C (100.5°F). Physical examination reveals tenderness in the right lower quadrant. Laboratory studies show a leukocyte count of 12,000/mm³. Which of the following is the most appropriate next step?",
        "choices": ["Appendectomy", "CT scan of the abdomen", "Observation", "Ultrasound of the abdomen"],
        "answer": 1  # B
    },
    {
        "question": "A previously healthy 32-year-old woman comes to the physician because of a 2-month history of fatigue. She also has had intermittent headaches and difficulty concentrating at work. Her pulse is 88/min and blood pressure is 130/85 mmHg. Examination shows no abnormalities. Serum studies show: Na+ 142 mEq/L, K+ 2.9 mEq/L, Cl- 100 mEq/L, HCO3- 30 mEq/L. Which of the following is the most likely diagnosis?",
        "choices": ["Conn syndrome", "Cushing syndrome", "Pheochromocytoma", "Renal artery stenosis"],
        "answer": 0  # A
    },
]

def run_model(model_name, tokenizer_name=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    
    results = []
    for qi, q in enumerate(QUESTIONS):
        choices_str = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(q["choices"]))
        prompt = f"{q['question']}\n{choices_str}\nAnswer:"
        
        # Build messages with STEM persona
        messages = [
            {"role": "system", "content": STEM_PERSONA},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, temperature=0.0, do_sample=False)
        
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Check correctness
        correct_letter = chr(65 + q["answer"])
        got_correct = correct_letter in response[:20]
        
        results.append({
            "qi": qi,
            "question": q["question"][:100],
            "correct": correct_letter,
            "response": response[:200],
            "got_correct": got_correct
        })
        print(f"  Q{qi}: correct={correct_letter}, got_correct={got_correct}, response={response[:80]}")
    
    del model
    torch.cuda.empty_cache()
    return results

print("="*60)
print("Running Mistral-7B with STEM persona...")
print("="*60)
mistral_results = run_model("mistralai/Mistral-7B-Instruct-v0.3")

print("\n" + "="*60)
print("Running Llama-3.1-8B with STEM persona...")
print("="*60)
llama_results = run_model("meta-llama/Llama-3.1-8B-Instruct")

# Summary
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
for qi in range(len(QUESTIONS)):
    m = mistral_results[qi]
    l = llama_results[qi]
    print(f"\nQ{qi}: {m['question']}")
    print(f"  Correct: {m['correct']}")
    print(f"  Mistral: {'✓' if m['got_correct'] else '✗'} | {m['response'][:100]}")
    print(f"  Llama:   {'✓' if l['got_correct'] else '✗'} | {l['response'][:100]}")

# Save
combined = {"mistral": mistral_results, "llama": llama_results}
with open("results/mmlu_persona_comparison.json", "w") as f:
    json.dump(combined, f, indent=2)
print("\nSaved to results/mmlu_persona_comparison.json")
