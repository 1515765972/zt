"""
PPO 强化学习训练脚本

架构:
  - CodeGen-2B-mono (frozen SFT base) + LoRA adapters (trainable policy)
  - 独立 Value Head (trainable)
  - frozen base 同时用作参考策略, 计算真正的 KL 散度约束

训练循环 (每step):
  ① 生成响应
  ② 计算四维奖励
  ③ 参考模型 forward (frozen base, 禁用LoRA) → ref_logprobs
  ④ 策略 forward (no_grad) → old_logprobs + old_values
  ⑤ advantages = rewards - old_values
  ⑥ 策略 forward (with grad) → new_logprobs + new_values
  ⑦ ratio = exp(new - old), clipped PPO loss
  ⑧ value loss + KL(ref||policy) penalty → backward → step
"""

import json
import os
import time
import argparse
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from rewards import compute_reward, get_dynamic_weights, compute_reward_ablation
from prepare_conala import load_conala_ppo_data, prepare_conala


# ─── 配置 ────────────────────────────────────────────────
DEFAULT_SFT_PATH = "/mnt/workspace/checkpoints/sft/final"
DEFAULT_MODEL_PATH = "/mnt/workspace/models/codegen-2B-mono"
SAVE_DIR = "/mnt/workspace/checkpoints/ppo"
LOG_DIR = "/mnt/workspace/results"

MAX_STEPS = 500
BATCH_SIZE = 4
LR_POLICY = 5e-5
LR_VALUE = 1e-3
CLIP_RANGE = 0.1
VF_COEF = 0.1
KL_COEF = 0.15           # 强KL约束, 不远离基座
MAX_LENGTH = 384
MAX_NEW_TOKENS = 384

LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
LORA_TARGETS = ["qkv_proj", "out_proj", "fc_in", "fc_out"]


def parse_args():
    p = argparse.ArgumentParser(description="PPO 训练")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--sft_path", type=str, default=DEFAULT_SFT_PATH)
    p.add_argument("--data_path", type=str, default="data/conala_ppo.jsonl")
    p.add_argument("--save_dir", type=str, default=SAVE_DIR)
    p.add_argument("--log_dir", type=str, default=LOG_DIR)
    p.add_argument("--max_steps", type=int, default=MAX_STEPS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR_POLICY)
    p.add_argument("--clip_range", type=float, default=CLIP_RANGE)
    p.add_argument("--kl_coef", type=float, default=KL_COEF)
    p.add_argument("--ablation", type=str, default=None,
                   choices=["full", "exec_only", "no_review", "no_codebleu", "syntax_exec"])
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.90)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def load_data(data_path: str, n: Optional[int] = None) -> List[Dict]:
    if not os.path.exists(data_path):
        _, _, data_path = prepare_conala()
    data = load_conala_ppo_data(data_path)
    return data[:n] if n else data


class ValueHead(nn.Module):
    """从最后层 hidden state 预测标量 value"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return self.net(hidden_states).squeeze(-1)  # [B, T]


def extract_seq_logprobs(logits, input_ids, q_lens, seq_lens):
    """从 logits 提取每样本 response 部分的 token 级 logprob 之和
       seq_lens: 每条序列的实际长度(不含padding), 避免padding token污染"""
    lp_all = torch.log_softmax(logits, dim=-1)
    result = []
    for i in range(input_ids.size(0)):
        q_len = int(q_lens[i])
        s_len = int(seq_lens[i])          # 实际长度, 非 padded
        if q_len >= s_len - 1:
            result.append(torch.tensor(0.0, device=input_ids.device))
            continue
        lp = lp_all[i, q_len - 1 : s_len - 1, :]        # 仅取 response 范围
        target = input_ids[i, q_len : s_len]              # 不含 padding
        token_lp = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        result.append(token_lp.sum())
    return torch.stack(result)


def extract_last_values(values, seq_lens):
    """提取每序列实际最后一个 token 的 value (不含padding)"""
    if values.dim() == 3:
        values = values.squeeze(-1)
    result = []
    for i in range(len(seq_lens)):
        result.append(values[i, seq_lens[i] - 1])
    return torch.stack(result)


def build_sequences(gen_outputs, device):
    seqs = [g for g in gen_outputs]
    seq_lens = [len(s) for s in seqs]          # 实际长度, 不含padding
    max_len = max(seq_lens)
    B = len(seqs)
    input_ids = torch.full((B, max_len), 0, dtype=torch.long, device=device)
    mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = s
        mask[i, :len(s)] = 1
    return input_ids, mask, seq_lens


def main():
    args = parse_args()

    if args.model_path:
        model_path = args.model_path
    elif os.path.exists(args.sft_path):
        model_path = args.sft_path
        print(f"使用 SFT 模型: {model_path}")
    else:
        model_path = DEFAULT_MODEL_PATH
        print(f"使用基座模型: {model_path}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── 加载模型 ────────────────────────────────────────
    print("\n[1/5] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("添加 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    # 在 peft 包装后才启用, 确保 disable/enable 可以正常传播
    model.gradient_checkpointing_enable()

    print("添加 Value Head...")
    hidden_size = base_model.config.n_embd
    value_head = ValueHead(hidden_size).to(device=device, dtype=torch.bfloat16)

    # 冻结 base, 仅训练 LoRA + value_head
    for n, p in model.named_parameters():
        if "lora" not in n:
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vh_params = sum(p.numel() for p in value_head.parameters())
    print(f"可训练: LoRA={trainable:,} + ValueHead={vh_params:,}")

    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"显存: {used:.1f}GB / {total:.1f}GB")

    # ─── 数据 ────────────────────────────────────────────
    print("\n[2/5] 加载训练数据...")
    data = load_data(args.data_path, args.max_steps * args.batch_size)
    print(f"训练样本: {len(data)}")

    # ─── 优化器 ──────────────────────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
        {"params": value_head.parameters(), "lr": LR_VALUE},
    ])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.max_steps,
    )

    # ─── 日志 ────────────────────────────────────────────
    log_path = os.path.join(args.log_dir, "ppo_train.log")
    metrics_path = os.path.join(args.log_dir, "ppo_metrics.jsonl")
    log_file = open(log_path, "w", encoding="utf-8")
    metrics_f = open(metrics_path, "w", encoding="utf-8")  # 每步实时写入

    def logf(msg):
        t = time.strftime("%H:%M:%S")
        line = f"[{t}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    # ─── 训练循环 ────────────────────────────────────────
    logf(f"PPO 开始 | steps={args.max_steps} | batch={args.batch_size} | "
         f"lr_policy={args.lr} | lr_value={LR_VALUE} | kl_coef={args.kl_coef}")
    logf(f"clip={args.clip_range} | ablation={args.ablation or 'full'}")

    all_metrics = []
    reward_history = []  # 用于计算近期平均奖励
    model.train()
    value_head.train()

    for step_idx in range(args.max_steps):
        start = (step_idx * args.batch_size) % len(data)
        batch = data[start: start + args.batch_size]
        if len(batch) < args.batch_size:
            batch = data[: args.batch_size]

        # 自适应权重: 基于近期20步平均奖励
        recent_avg = sum(reward_history[-20:]) / max(len(reward_history[-20:]), 1) if reward_history else 0.0
        weights = get_dynamic_weights(step_idx, args.max_steps, recent_avg)
        # 阶段切换日志
        if step_idx == 30:
            logf(f"  ▶ 自适应阶段启动 | recent_avg={recent_avg:.4f} | weights={weights}")
        elif step_idx > 30 and step_idx % 50 == 0:
            phase = 1 if recent_avg < 0.3 else (2 if recent_avg < 0.5 else 3)
            logf(f"  ▶ 阶段{phase} | recent_avg={recent_avg:.4f} | weights={weights}")
        # 温度从高到低: 前期探索, 后期利用
        progress = step_idx / max(args.max_steps, 1)
        temp = args.temperature * (1.0 - 0.75 * progress) + 0.05

        # ── ① Tokenize ──
        queries = [item["query"] for item in batch]
        ref_codes = [item["reference_code"] for item in batch]
        tests = [item.get("test", None) for item in batch]  # MBPP有测试用例, CoNaLa为None

        q_enc = tokenizer(queries, return_tensors="pt", truncation=True,
                          max_length=MAX_LENGTH, padding=True).to(device)
        q_lens = q_enc["attention_mask"].sum(dim=1)

        # ── ② 生成 ──
        model.eval()                                 # 关闭 dropout
        model.gradient_checkpointing_disable()       # 关闭检查点, 允许 KV cache
        with torch.no_grad():
            gen = model.generate(
                **q_enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=temp,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        model.gradient_checkpointing_enable()
        model.train()                                # 恢复训练模式

        # ── ③ 解码 + 奖励 ──
        responses = []
        full_codes = []
        for g, ql, q in zip(gen, q_lens, queries):
            resp = tokenizer.decode(g[int(ql):], skip_special_tokens=True)
            # 截断: 取第一个空行或非缩进行之前的内容 (只保留函数体)
            lines = resp.split("\n")
            clean_lines = []
            for line in lines:
                if line.strip() == "" and len(clean_lines) > 0:
                    break
                if line and not line[0].isspace() and len(clean_lines) > 1:
                    break
                clean_lines.append(line)
            body = "\n".join(clean_lines)
            responses.append(body)
            # 完整代码: 提取import→函数签名+docstring→函数体
            body_lines = body.split("\n")
            imports = []
            func_lines = []
            for line in body_lines:
                s = line.strip()
                if s.startswith(("import ", "from ")) and not line[0].isspace():
                    imports.append(s)
                else:
                    func_lines.append(line)
            prefix = ("\n".join(imports) + "\n\n") if imports else ""
            full_codes.append(prefix + q.rstrip() + "\n" + "\n".join(func_lines))

        rewards_list = []
        for code, ref, test in zip(full_codes, ref_codes, tests):
            if args.ablation:
                r = compute_reward_ablation(code, ref, test=test, ablation=args.ablation)
            else:
                r = compute_reward(code, ref, test=test, **weights)
            rewards_list.append(max(r, -1.0))
        rewards = torch.tensor(rewards_list, dtype=torch.bfloat16, device=device)

        # ── ④ 构建完整序列 ──
        full_ids, full_mask, full_lens = build_sequences(gen, device)

        # ── ⑤ 参考模型 logprobs (frozen SFT base, eval模式, 不含 LoRA) ──
        base_model.eval()
        with torch.no_grad():
            ref_out = base_model(full_ids, attention_mask=full_mask)
            ref_logprobs = extract_seq_logprobs(ref_out.logits, full_ids, q_lens, full_lens)
        model.train()  # 恢复训练模式, 供后续 policy forward 使用

        # ── ⑥ old forward: 当前策略 (base + LoRA) 的 logprobs + values ──
        model.eval()                     # 确保无 dropout 随机性
        with torch.no_grad():
            out = model(full_ids, attention_mask=full_mask, output_hidden_states=True)
            old_logprobs = extract_seq_logprobs(out.logits, full_ids, q_lens, full_lens)
            old_values = extract_last_values(value_head(out.hidden_states[-1]), full_lens)
        model.train()                    # 恢复训练模式, 供后续 new forward 使用

        # ── ⑦ advantages ──
        advantages = rewards - old_values
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── ⑧ new forward: 当前策略 + value (with grad) ──
        out = model(full_ids, attention_mask=full_mask, output_hidden_states=True)
        new_logprobs = extract_seq_logprobs(out.logits, full_ids, q_lens, full_lens)
        new_values = extract_last_values(value_head(out.hidden_states[-1]), full_lens)

        # ── ⑨ PPO loss ──
        ratio = torch.exp(new_logprobs - old_logprobs)
        clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

        value_loss = F.mse_loss(new_values, rewards)

        # 真正的 KL: KL(SFT_ref || policy) — 防止策略偏离 SFT 太远
        kl_ref = (ref_logprobs - new_logprobs).clamp(min=0).mean()

        total_loss = policy_loss + VF_COEF * value_loss + args.kl_coef * kl_ref

        total_loss.backward()

        # ── ⑩ 更新 ──
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(value_head.parameters()), 1.0
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # ── ⑪ 日志 ──
        mean_reward = rewards.mean().item()

        metrics = {
            "step": step_idx, "reward": round(mean_reward, 4),
            "p_loss": round(policy_loss.item(), 4),
            "v_loss": round(value_loss.item(), 4),
            "kl_ref": round(kl_ref.item(), 4),
            "weights": weights,
        }
        all_metrics.append(metrics)
        reward_history.append(mean_reward)
        metrics_f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        metrics_f.flush()  # 立即写入磁盘, plot_live.py 才能读到

        if step_idx % args.log_every == 0:
            pv = [r[:60].replace("\n", "\\n") for r in responses[:2]]
            logf(f"Step {step_idx:4d} | reward={mean_reward:.4f} | "
                 f"p_loss={policy_loss.item():.4f} | v_loss={value_loss.item():.4f} | "
                 f"kl_ref={kl_ref.item():.4f} | lr={scheduler.get_last_lr()[0]:.2e}")
            logf(f"  samples: {pv}")

        if kl_ref.item() > 1.0:
            logf(f"⚠  KL(ref||policy)={kl_ref.item():.4f} 偏高, 策略偏离SFT较远")

        # 保存
        if step_idx > 0 and step_idx % args.save_every == 0:
            ckpt = os.path.join(args.save_dir, f"step_{step_idx}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            torch.save(value_head.state_dict(), os.path.join(ckpt, "value_head.pt"))
            logf(f"检查点 → {ckpt}")

    # ─── 保存 ────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, "final")
    merged = model.merge_and_unload()
    merged.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    torch.save(value_head.state_dict(), os.path.join(final_path, "value_head.pt"))
    logf(f"最终模型 → {final_path}")

    logf(f"指标 → {metrics_path}")

    log_file.close()
    metrics_f.close()

    avg_r = sum(m["reward"] for m in all_metrics) / len(all_metrics) if all_metrics else 0
    print(f"\n{'='*50}")
    print(f"训练汇总: {len(all_metrics)} 步, 平均奖励: {avg_r:.4f}")
    print(f"模型: {final_path}")


if __name__ == "__main__":
    main()
