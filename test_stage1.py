"""Quick test of Stage 1 query gen to catch errors cleanly."""
import sys, os, traceback, warnings, logging

# Suppress progress bars and noisy output
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_PROGRESS_BAR"] = "1"
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "prism"))

try:
    from utils import load_model, build_chat_messages, batch_generate
    print("1. Imports OK", flush=True)

    model, tok = load_model("Qwen/Qwen2.5-7B-Instruct")
    print(f"2. Model loaded OK. pad_token={tok.pad_token}, eos_token={tok.eos_token}", flush=True)

    msgs = build_chat_messages(tok, "You are a helpful assistant.", "What is 2+2?")
    print(f"3. build_chat_messages OK: {msgs}", flush=True)

    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    print(f"4. apply_chat_template OK (text): {repr(text[:120])}", flush=True)

    result = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids = result.input_ids if hasattr(result, "input_ids") else result
    print(f"5. encoded shape: {ids.shape} (type was {type(result).__name__})", flush=True)

    responses = batch_generate(model, tok, [msgs], max_tokens=50, temperature=0.7, batch_size=1)
    print(f"6. batch_generate OK: {responses[0][:100]}", flush=True)

    print("\nALL TESTS PASSED", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)
