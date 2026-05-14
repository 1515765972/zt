"""
模型评估脚本 - 在 HumanEval 上评估代码生成模型

支持评估基线模型、SFT 模型和 PPO 模型。

用法:
    python 03_eval.py                                    # 默认: 基线模型
    python 03_eval.py --model_path /path/to/sft/final    # SFT 模型
    python 03_eval.py --model_path /path/to/ppo/final    # PPO 模型
    python 03_eval.py --model_path <path> --output <file> --n_samples 5
"""

import argparse
import os
import sys
import time

import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness
except ImportError:
    from data import read_problems, write_jsonl
    from evaluation import evaluate_functional_correctness


# ─── 默认路径 ──────────────────────────────────────────
DEFAULT_MODEL_PATH = "/mnt/workspace/models/codegen-2B-mono"
DEFAULT_SFT_PATH = "/mnt/workspace/checkpoints/sft/final"
DEFAULT_PPO_PATH = "/mnt/workspace/checkpoints/ppo/final"
DEFAULT_OUTPUT_DIR = "/mnt/workspace/results"


def parse_args():
    parser = argparse.ArgumentParser(description="HumanEval 模型评估")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型路径 (默认: 基线 CodeGen-2B-mono)")
    parser.add_argument("--model_type", type=str, default="baseline",
                        choices=["baseline", "sft", "ppo"],
                        help="预设模型类型 (若未指定 model_path)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n_samples", type=int, default=1,
                        help="每题生成样本数")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--k", type=str, default="1,10",
                        help="pass@k 的 k 值 (逗号分隔)")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=3.0)
    return parser.parse_args()


def load_model(model_path: str):
    """加载模型和 tokenizer"""
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"显存: {used:.1f}GB / {total:.1f}GB")
    return tokenizer, model


def generate_code(model, tokenizer, prompt: str, args) -> str:
    """对单个 prompt 生成代码"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=(args.temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    new_tokens = outputs[0][input_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # 截断: 在下一个函数定义/类定义/注释前停止
    for sw in ["\ndef ", "\nclass ", "\n# ", "\nif __name__"]:
        idx = completion.find(sw)
        if idx > 0:
            completion = completion[:idx]

    return completion


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 确定模型路径
    if args.model_path:
        model_path = args.model_path
    elif args.model_type == "sft":
        model_path = DEFAULT_SFT_PATH
    elif args.model_type == "ppo":
        model_path = DEFAULT_PPO_PATH
    else:
        model_path = DEFAULT_MODEL_PATH

    model_name = os.path.basename(model_path.rstrip("/")) or args.model_type
    print(f"模型类型: {model_name}")

    # 加载模型
    tokenizer, model = load_model(model_path)

    # 加载 HumanEval
    print("加载 HumanEval 数据集...")
    problems = read_problems()
    print(f"题目数: {len(problems)}")

    # 生成代码
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"{model_name}_{timestamp}_samples.jsonl")

    samples = []
    for task_id, problem in tqdm.tqdm(problems.items(), desc="生成"):
        for i in range(args.n_samples):
            completion = generate_code(model, tokenizer, problem["prompt"], args)
            samples.append({
                "task_id": task_id,
                "completion": completion,
            })

    write_jsonl(output_path, samples)
    print(f"生成 {len(samples)} 个样本 → {output_path}")

    # 评测
    k_values = list(map(int, args.k.split(",")))
    print(f"\n评估 pass@{k_values}...")
    results = evaluate_functional_correctness(
        output_path,
        k=k_values,
        n_workers=args.n_workers,
        timeout=args.timeout,
    )

    # 输出结果
    print(f"\n{'='*50}")
    print(f"评估结果: {model_name}")
    print(f"{'='*50}")
    for k in k_values:
        key = f"pass@{k}"
        if key in results:
            print(f"  {key} = {results[key]:.4f}")
    print(f"{'='*50}")
    print(f"样本文件: {output_path}")


if __name__ == "__main__":
    main()
