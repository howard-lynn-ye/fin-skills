"""Local open-weight chat callable; requires a Slurm GPU allocation on Beacon."""
import time


class TransformersChat:
    def __init__(self, model, revision, *, max_tokens=2048, seed=0):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("GPU allocation required; no CPU inference fallback")
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="sdpa")
        self.model.eval()
        self.max_tokens, self.seed, self.calls = max_tokens, seed, 0

    def __call__(self, messages):
        import torch
        from transformers import set_seed
        set_seed(self.seed + self.calls)
        self.calls += 1
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        count = inputs.input_ids.shape[-1]
        if count + self.max_tokens > 32768:
            raise ValueError("context budget exceeded; no silent truncation")
        started = time.monotonic()
        with torch.inference_mode():
            result = self.model.generate(**inputs, max_new_tokens=self.max_tokens,
                                         do_sample=True, temperature=.1, top_p=1.0,
                                         pad_token_id=self.tokenizer.eos_token_id)
        tokens = result[0, count:]
        return {"choices": [{"message": {"content": self.tokenizer.decode(tokens, skip_special_tokens=True)},
                              "finish_reason": "length" if len(tokens) == self.max_tokens else "stop"}],
                "usage": {"prompt_tokens": count, "completion_tokens": len(tokens),
                          "total_tokens": count + len(tokens)},
                "generation_seconds": time.monotonic() - started}
