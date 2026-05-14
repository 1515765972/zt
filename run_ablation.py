"""
消融实验框架

对基线模型、SFT 模型、PPO（全奖励）及各消融变体在 HumanEval 上统一评测,
输出 pass@k 对比表, 分析各维度的贡献。

实验设计:
  A - 基线:        CodeGen-2B-mono 原始模型
  B - SFT:         监督微调后模型
  C - PPO-full:    PPO + 四维奖励 (完整方案)
  D - PPO-exec:    PPO + 仅执行奖励 (对照单维度)
  E - PPO-no_rev:  PPO + 无审查惩罚 (对照 flake8 贡献)
  F - PPO-no_cb:   PPO + 无 CodeBLEU (对照相似度贡献)

用法:
  python run_ablation.py                        # 使用默认路径
  python run_ablation.py --models_json models.json  # 自定义模型列表
"""

import json
import os
import sys
import argparse
import time
from typing import Dict, List, Tuple

import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# HumanEval 评测模块 (兼容 PAI-DSW 和本地环境)
try:
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness
except ImportError:
    from data import read_problems, write_jsonl
    from evaluation import evaluate_functional_correctness


# ─── 默认实验配置 ─────────────────────────────────────────
DEFAULT_EXPERIMENTS = [
    {
        "name": "A-基线-CodeGen",
        "description": "CodeGen-2B-mono 原始模型",
        "model_path": "/mnt/workspace/models/codegen-2B-mono",
        "group": "baseline",
    },
    {
        "name": "B-SFT",
        "description": "CoNaLa 监督微调后模型",
        "model_path": "/mnt/workspace/checkpoints/sft/final",
        "group": "sft",
    },
    {
        "name": "C-PPO-full",
        "description": "PPO + 四维奖励 (语法+执行+CodeBLEU+flake8)",
        "model_path": "/mnt/workspace/checkpoints/ppo/final",
        "group": "ppo",
    },
    {
        "name": "D-PPO-exec_only",
        "description": "PPO + 仅执行奖励 (消融: 单维度对照)",
        "model_path": "/mnt/workspace/checkpoints/ppo_exec_only/final",
        "group": "ablation",
    },
    {
        "name": "E-PPO-no_review",
        "description": "PPO + 无审查惩罚 (消融: flake8 贡献)",
        "model_path": "/mnt/workspace/checkpoints/ppo_no_review/final",
        "group": "ablation",
    },
    {
        "name": "F-PPO-no_codebleu",
        "description": "PPO + 无 CodeBLEU (消融: 相似度贡献)",
        "model_path": "/mnt/workspace/checkpoints/ppo_no_codebleu/final",
        "group": "ablation",
    },
]

RESULTS_DIR = "/mnt/workspace/results/ablation"


def parse_args():
    parser = argparse.ArgumentParser(description="消融实验评测")
    parser.add_argument("--models_json", type=str, default=None,
                        help="自定义模型列表 JSON 文件")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--n_samples", type=int, default=1,
                        help="每题生成多少个样本 (默认1)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--k", type=str, default="1,10",
                        help="评估的 k 值, 逗号分隔")
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str):
    """加载模型, 若路径不存在返回 None"""
    if not os.path.exists(model_path):
        print(f"  ⚠ 模型路径不存在: {model_path}")
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def generate_samples(
    model,
    tokenizer,
    problems: Dict,
    output_path: str,
    n_samples: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """在 HumanEval 上生成代码样本"""
    samples = []

    for task_id, problem in tqdm.tqdm(problems.items(), desc="生成代码"):
        prompt = problem["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        for _ in range(n_samples):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=(temperature > 0),
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            new_tokens = outputs[0][input_len:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

            # 截断到第一个函数定义结束 (HumanEval 约定)
            stop_words = ["\ndef ", "\nclass ", "\n# ", "\nif __name__"]
            for sw in stop_words:
                idx = completion.find(sw)
                if idx > 0:
                    completion = completion[:idx]

            samples.append({
                "task_id": task_id,
                "completion": completion,
            })

    write_jsonl(output_path, samples)
    return output_path


def run_evaluation(
    experiment: Dict,
    problems: Dict,
    args,
) -> Dict:
    """运行单个实验的评测"""
    name = experiment["name"]
    model_path = experiment["model_path"]

    print(f"\n{'='*60}")
    print(f"评测: {name}")
    print(f"说明: {experiment['description']}")
    print(f"模型: {model_path}")
    print(f"{'='*60}")

    # 加载模型
    tokenizer, model = load_model_and_tokenizer(model_path)
    if tokenizer is None:
        return {
            **experiment,
            "status": "failed",
            "error": f"模型路径不存在: {model_path}",
            "results": {},
        }

    # 生成代码
    output_path = os.path.join(args.results_dir, f"{name}_samples.jsonl")
    generate_samples(
        model, tokenizer, problems,
        output_path=output_path,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # 清理显存
    del model, tokenizer
    torch.cuda.empty_cache()

    # 评测
    k_values = list(map(int, args.k.split(",")))
    print(f"\n计算 pass@{k_values}...")
    results = evaluate_functional_correctness(
        output_path,
        k=k_values,
        n_workers=4,
        timeout=3.0,
    )

    return {
        **experiment,
        "status": "success",
        "results": results,
        "samples_file": output_path,
    }


def print_comparison_table(all_results: List[Dict]):
    """打印对比表格"""
    print("\n")
    print("=" * 80)
    print("消融实验结果对比")
    print("=" * 80)

    # 表头
    header = f"{'实验':<22} {'状态':<8}"
    k_keys = []
    for r in all_results:
        if r["status"] == "success" and r["results"]:
            k_keys = sorted(r["results"].keys())
            break
    for k in k_keys:
        header += f" {k:<12}"
    header += f" {'说明':<50}"
    print(header)
    print("-" * 130)

    # 按 group 排序
    group_order = {"baseline": 0, "sft": 1, "ppo": 2, "ablation": 3}
    sorted_results = sorted(all_results, key=lambda x: group_order.get(x.get("group", 9), 9))

    for r in sorted_results:
        line = f"{r['name']:<22} {r['status']:<8}"
        if r["status"] == "success":
            for k in k_keys:
                val = r["results"].get(k, float("nan"))
                line += f" {val:<12.4f}"
        else:
            for k in k_keys:
                line += f" {'N/A':<12}"
        line += f" {r['description'][:48]:<50}"
        print(line)

    print("=" * 80)

    # 分析增量
    print("\n📊 增量分析:")
    baseline_result = None
    for r in sorted_results:
        if r["group"] == "baseline" and r["status"] == "success":
            baseline_result = r
            break

    if baseline_result:
        for r in sorted_results:
            if r["status"] != "success" or r["group"] == "baseline":
                continue
            for k in k_keys:
                if k in r["results"] and k in baseline_result["results"]:
                    delta = r["results"][k] - baseline_result["results"][k]
                    print(f"  {r['name']} {k}: {delta:+.4f} (vs 基线)")


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # 加载实验配置
    if args.models_json and os.path.exists(args.models_json):
        with open(args.models_json, "r") as f:
            experiments = json.load(f)
        print(f"从 {args.models_json} 加载了 {len(experiments)} 个实验配置")
    else:
        experiments = DEFAULT_EXPERIMENTS
        print(f"使用默认配置 ({len(experiments)} 个实验)")

    # 加载 HumanEval 题目
    print("\n加载 HumanEval 数据集...")
    problems = read_problems()
    print(f"题目数: {len(problems)}")

    # 运行所有实验
    all_results = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] 开始实验: {exp['name']}")
        result = run_evaluation(exp, problems, args)
        all_results.append(result)

        # 实时保存
        results_path = os.path.join(args.results_dir, "ablation_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印对比
    print_comparison_table(all_results)

    # 保存最终结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(args.results_dir, f"ablation_results_{timestamp}.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {final_path}")
    print("消融实验完成!")


if __name__ == "__main__":
    main()
