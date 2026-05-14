"""
MBPP 数据集预处理 (docstring 格式)

CodeGen 预训练用的是代码补全, 不是指令遵循。
用 docstring 格式让模型在原生语境下完成函数体。

输出:
  - data/mbpp_sft_train.jsonl   SFT 训练集
  - data/mbpp_sft_val.jsonl     SFT 验证集
  - data/mbpp_ppo.jsonl         PPO 训练数据 (含测试用例)
"""

import json
import os
import random
from typing import Dict, List, Tuple


def extract_signature(code: str) -> str:
    """从完整函数代码中提取 def 签名行"""
    lines = code.strip().split("\n")
    sig_lines = []
    for line in lines:
        stripped = line.strip()
        sig_lines.append(stripped)
        if stripped.endswith(":"):
            break
    return "\n".join(sig_lines)


def make_docstring_prompt(prompt_str: str, code: str) -> Tuple[str, str]:
    """构造 docstring prompt 和仅函数体的 response"""
    sig = extract_signature(code)
    # instruction: 签名 + docstring (模型看到的上下文)
    instruction = f'{sig}\n    """{prompt_str}"""\n    '
    # response: 函数体 (不含签名, 模型要生成的部分)
    # 取签名后的所有行作为函数体
    code_lines = code.strip().split("\n")
    # 跳过签名行
    body_start = 0
    for i, line in enumerate(code_lines):
        stripped = line.strip()
        if stripped.endswith(":") and ("def " in stripped or " " not in stripped[:4]):
            body_start = i + 1
            break
    body = "\n".join(code_lines[body_start:])
    return instruction, body


def load_mbpp(filepath: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    valid = []
    for item in data:
        prompt = item.get("prompt", "")
        code = item.get("code", "")
        if prompt and code:
            valid.append(item)
    print(f"加载 MBPP: {len(valid)} 条有效数据 (原始 {len(data)} 条)")
    return valid


def split_data(data: List[Dict], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List, List]:
    if "split" in data[0]:
        train = [d for d in data if d.get("split") == "train"]
        val = [d for d in data if d.get("split") == "validation"]
        if train:
            return train, val or train[-int(len(train)*val_ratio):]
    random.seed(seed)
    random.shuffle(data)
    val_size = max(1, int(len(data) * val_ratio))
    return data[val_size:], data[:val_size]


def prepare_mbpp(input_file: str = "sanitized-mbpp.json", output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    data = load_mbpp(input_file)
    train_data, val_data = split_data(data)
    print(f"训练: {len(train_data)}, 验证: {len(val_data)}")

    # ── SFT: docstring prompt → 函数体补全 ──
    with open(os.path.join(output_dir, "mbpp_sft_train.jsonl"), "w", encoding="utf-8") as f:
        for item in train_data:
            instruction, body = make_docstring_prompt(item["prompt"], item["code"])
            text = instruction + "\n" + body  # 完整: 签名+docstring + 函数体
            sample = {
                "prompt": item["prompt"],
                "code": item["code"],
                "instruction": instruction,
                "response": body,
                "text": text,
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(os.path.join(output_dir, "mbpp_sft_val.jsonl"), "w", encoding="utf-8") as f:
        for item in val_data:
            instruction, body = make_docstring_prompt(item["prompt"], item["code"])
            text = instruction + "\n" + body
            sample = {
                "prompt": item["prompt"],
                "code": item["code"],
                "instruction": instruction,
                "response": body,
                "text": text,
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"SFT 数据 → {output_dir}/mbpp_sft_train.jsonl + mbpp_sft_val.jsonl")

    # ── PPO: 同样的 docstring prompt ──
    with open(os.path.join(output_dir, "mbpp_ppo.jsonl"), "w", encoding="utf-8") as f:
        for item in train_data:
            test_imports = item.get("test_imports", "")
            test_cases = item.get("test_list", [])
            parts = []
            if test_imports:
                if isinstance(test_imports, str):
                    parts.append(test_imports)
                else:
                    parts.extend(test_imports)
            if test_cases:
                parts.extend(test_cases)
            test_str = "\n".join(parts)

            query, _ = make_docstring_prompt(item["prompt"], item["code"])
            sample = {
                "prompt": item["prompt"],
                "query": query,
                "reference_code": item["code"],
                "test": test_str,
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"PPO 数据 → {output_dir}/mbpp_ppo.jsonl")


if __name__ == "__main__":
    prepare_mbpp()
