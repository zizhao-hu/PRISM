"""
PRISM MoLoRA Inference Wrapper

Loads the MoLoRA (router + K LoRA experts + null expert) and provides
a unified interface for evaluation scripts. Given a query, the router
selects which expert to use, and the response is generated with that
expert's LoRA adapter (or the base model for null expert).

This module can be imported by eval scripts, or used standalone:
  python -m scripts.prism.molora_inference \
      --model Qwen/Qwen2.5-7B-Instruct \
      --molora_dir models/persona_prism/Qwen2.5-7B-Instruct \
      --query "Explain quantum entanglement"
"""

import os
import sys
import argparse
import logging

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    load_json, load_text,
    build_chat_messages, generate_response, get_model_slug,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MoLoRAInference:
    """MoLoRA inference wrapper for evaluation.
    
    Usage:
        infer = MoLoRAInference(model_name, molora_dir)
        response = infer.generate(query, max_tokens=512)
        # or with routing info:
        response, expert_name = infer.generate_with_info(query)
    """
    
    def __init__(self, model_name, molora_dir, device_map="auto"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        from stage3_distill import PersonaRouter, MoLoRAManager
        
        self.model_name = model_name
        self.molora_dir = molora_dir
        
        # Load config
        config_path = os.path.join(molora_dir, "molora_config.json")
        self.config = load_json(config_path)
        self.persona_names = self.config["persona_names"]
        self.num_experts = self.config["num_experts"]
        self.adapter_names = self.config["adapter_names"]
        
        logger.info(f"Loading MoLoRA: {self.num_experts} experts "
                    f"({len(self.persona_names)} personas + null)")
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # Load each LoRA adapter
        for i, adapter_name in enumerate(self.adapter_names):
            adapter_dir = os.path.join(molora_dir, adapter_name)
            if os.path.exists(adapter_dir):
                if i == 0:
                    self.model = PeftModel.from_pretrained(
                        self.model, adapter_dir, adapter_name=adapter_name,
                        is_trainable=False,
                    )
                else:
                    self.model.load_adapter(adapter_dir, adapter_name=adapter_name)
                logger.info(f"  Loaded adapter: {adapter_name}")
        
        # Load router
        hidden_dim = self.model.config.hidden_size
        self.router = PersonaRouter(hidden_dim, self.num_experts).to(
            next(self.model.parameters()).device
        ).to(next(self.model.parameters()).dtype)
        
        router_path = os.path.join(molora_dir, "router.pt")
        if os.path.exists(router_path):
            self.router.load_state_dict(torch.load(router_path, weights_only=True))
            logger.info("  Loaded router weights")
        
        self.router.eval()
        self.model.eval()
        
        # Build expert name mapping
        self.expert_names = {0: "null (base model)"}
        for i, name in enumerate(self.persona_names):
            self.expert_names[i + 1] = name
    
    def _get_hidden_state(self, input_ids):
        """Extract hidden state for routing (base model, first layer, last token)."""
        self.model.disable_adapter_layers()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[1]  # first transformer layer
            # Last non-pad token
            batch_size = input_ids.shape[0]
            last_token_hidden = []
            for b in range(batch_size):
                non_pad = (input_ids[b] != self.tokenizer.pad_token_id).nonzero()
                if len(non_pad) > 0:
                    last_idx = non_pad[-1].item()
                else:
                    last_idx = input_ids.shape[1] - 1
                last_token_hidden.append(hidden[b, last_idx])
            hidden_states = torch.stack(last_token_hidden)
        self.model.enable_adapter_layers()
        return hidden_states
    
    def route(self, query):
        """Route a query to the best expert. Returns (expert_idx, expert_name, probs)."""
        msgs = build_chat_messages(self.tokenizer, None, query)
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        encoding = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        hidden = self._get_hidden_state(encoding.input_ids)
        with torch.no_grad():
            logits = self.router(hidden)
            probs = torch.softmax(logits, dim=-1)[0]
        
        expert_idx = probs.argmax().item()
        expert_name = self.expert_names.get(expert_idx, f"expert_{expert_idx}")
        return expert_idx, expert_name, probs.cpu().tolist()
    
    def _set_expert(self, expert_idx):
        """Activate the specified expert."""
        if expert_idx == 0:
            self.model.disable_adapter_layers()
        else:
            self.model.enable_adapter_layers()
            adapter_name = self.adapter_names[expert_idx - 1]
            self.model.set_adapter(adapter_name)
    
    def generate(self, query, system_prompt=None, max_tokens=512, temperature=0.7):
        """Generate a response using MoLoRA routing."""
        expert_idx, _, _ = self.route(query)
        self._set_expert(expert_idx)
        
        msgs = build_chat_messages(self.tokenizer, system_prompt, query)
        response = generate_response(
            self.model, self.tokenizer, msgs,
            max_tokens=max_tokens, temperature=temperature,
        )
        return response
    
    def generate_with_info(self, query, system_prompt=None, max_tokens=512, temperature=0.7):
        """Generate with routing info returned."""
        expert_idx, expert_name, probs = self.route(query)
        self._set_expert(expert_idx)
        
        msgs = build_chat_messages(self.tokenizer, system_prompt, query)
        response = generate_response(
            self.model, self.tokenizer, msgs,
            max_tokens=max_tokens, temperature=temperature,
        )
        
        info = {
            "expert_idx": expert_idx,
            "expert_name": expert_name,
            "routing_probs": {self.expert_names[k]: round(p, 4) 
                            for k, p in enumerate(probs)},
        }
        return response, info


def main():
    parser = argparse.ArgumentParser(description="MoLoRA Inference")
    parser.add_argument("--model", required=True, help="Base model name")
    parser.add_argument("--molora_dir", required=True, help="Path to saved MoLoRA")
    parser.add_argument("--query", default=None, help="Single query to generate")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    infer = MoLoRAInference(args.model, args.molora_dir)
    
    if args.query:
        response, info = infer.generate_with_info(args.query)
        print(f"\nQuery: {args.query}")
        print(f"Routed to: {info['expert_name']} (expert {info['expert_idx']})")
        print(f"Routing probs: {info['routing_probs']}")
        print(f"\nResponse:\n{response}")
    
    elif args.interactive:
        print("MoLoRA Interactive Mode (type 'quit' to exit)")
        while True:
            query = input("\n> ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            response, info = infer.generate_with_info(query)
            print(f"[→ {info['expert_name']}] {response}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
