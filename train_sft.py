"""
监督微调 (SFT) 训练脚本 — LoRA 版本

在 CoNaLa 数据集上对 CodeGen-2B-mono 进行 LoRA 微调,
仅训练约 2% 参数.
训练完成后合并 LoRA 权重输出完整模型, 供 PPO 和评测使用.
"""

import json
import os
import sys
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# ─── 配置 ──────────────────────────────────────────────────
MODEL_PATH = "/mnt/workspace/models/codegen-2B-mono"
DATA_PATH = "data/mbpp_sft_train.jsonl"
VAL_DATA_PATH = "data/mbpp_sft_val.jsonl"
SAVE_DIR = "/mnt/workspace/checkpoints/sft_mbpp_mbpp_mbpp"
OUTPUT_DIR = "/mnt/workspace/results/sft_mbpp_mbpp_mbpp"

NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4        # LoRA 学习率通常高于全量微调
WARMUP_STEPS = 50
MAX_LENGTH = 512
SAVE_STEPS = 500

# LoRA 配置
LORA_R = 16                 # 低秩矩阵的秩 (越大越接近全量微调)
LORA_ALPHA = 32             # 缩放系数 (通常 = r×2)
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["qkv_proj", "out_proj", "fc_in", "fc_out"]

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_sft_data(filepath: str) -> Dataset:
    """加载 SFT 格式数据并转为 HuggingFace Dataset"""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return Dataset.from_list(samples)


def tokenize_sft(
    examples: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    max_length: int = MAX_LENGTH,
) -> Dict[str, List]:
    """
    分句分词: instruction 部分 mask (label=-100), response 部分计算 loss
    """
    instructions = examples["instruction"]
    responses = examples["response"]

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for instruction, response in zip(instructions, responses):
        inst_tokens = tokenizer(
            instruction,
            truncation=True,
            max_length=max_length // 2,
            add_special_tokens=False,
        )
        resp_tokens = tokenizer(
            response,
            truncation=True,
            max_length=max_length // 2,
            add_special_tokens=False,
        )

        inst_ids = inst_tokens["input_ids"]
        resp_ids = resp_tokens["input_ids"]

        input_ids = inst_ids + resp_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(inst_ids) + resp_ids + [tokenizer.eos_token_id]

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        attention_mask = [1] * len(input_ids)

        pad_len = max_length - len(input_ids)
        input_ids += [tokenizer.eos_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def main():
    print("=" * 60)
    print("SFT 监督微调训练 (LoRA)")
    print("=" * 60)

    # 加载 tokenizer 和模型
    print("\n[1/5] 加载 tokenizer 和模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    # ── 配置 LoRA ──
    print("配置 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.train()

    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"模型加载完成 | 显存: {used:.1f}GB / {total:.1f}GB")

    # 加载数据
    print("\n[2/5] 加载训练数据...")
    train_dataset = load_sft_data(DATA_PATH)
    val_dataset = load_sft_data(VAL_DATA_PATH) if os.path.exists(VAL_DATA_PATH) else None
    print(f"训练样本: {len(train_dataset)}")
    if val_dataset:
        print(f"验证样本: {len(val_dataset)}")

    # Tokenize
    print("\n[3/5] 分词处理...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_sft(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    if val_dataset:
        val_dataset = val_dataset.map(
            lambda x: tokenize_sft(x, tokenizer),
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing val",
        )

    # 训练配置
    print("\n[4/5] 配置训练参数...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=SAVE_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        bf16=True,
        logging_steps=50,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=SAVE_STEPS,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        group_by_length=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # 训练
    print("\n[5/5] 开始训练...")
    trainer.train()

    # ── 保存: 合并 LoRA → 完整模型 ──
    print("\n合并 LoRA 权重...")
    merged_model = model.merge_and_unload()

    final_path = os.path.join(SAVE_DIR, "final")
    merged_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"SFT 模型 (已合并) → {final_path}")

    # 也保存 LoRA 适配器 (备用)
    lora_path = os.path.join(SAVE_DIR, "lora_adapter")
    model.save_pretrained(lora_path)
    print(f"LoRA 适配器 → {lora_path}")

    # 保存训练指标
    metrics_path = os.path.join(OUTPUT_DIR, "sft_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2, default=str)
    print(f"训练指标 → {metrics_path}")

    return final_path


if __name__ == "__main__":
    main()
