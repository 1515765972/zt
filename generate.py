

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "/mnt/workspace/models/codegen-2B-mono"

_tokenizer = None
_model = None

def load_model():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    print("加载模型中...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    _model.eval()

    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ 模型加载完成 | 显存: {used:.1f}GB / {total:.1f}GB")
    return _tokenizer, _model


def generate_code(prompt: str, max_new_tokens: int = 256) -> str:
    tokenizer, model = load_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
