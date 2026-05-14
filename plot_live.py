"""
实时奖励曲线监控
每5秒从 ppo_metrics.jsonl 读取最新数据, 生成更新后的曲线图
用法: python plot_live.py &
"""
import json, time, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METRICS_FILE = "/mnt/workspace/results/ppo_metrics.jsonl"
OUTPUT_FILE = "/mnt/workspace/results/live_curve.png"

last_count = 0

while True:
    if not os.path.exists(METRICS_FILE):
        time.sleep(5)
        continue

    metrics = []
    with open(METRICS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))

    n = len(metrics)
    if n == 0 or n == last_count:   # 无新数据, 跳过绘图
        time.sleep(5)
        continue
    last_count = n

    steps = [m["step"] for m in metrics]
    rewards = [m["reward"] for m in metrics]
    kls = [m.get("kl_ref", 0) for m in metrics]
    v_losses = [m.get("v_loss", 0) for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(steps, rewards, linewidth=0.4, alpha=0.6, color='#1f77b4')
    if n >= 20:
        w = 20
        smoothed = np.convolve(rewards, np.ones(w)/w, mode='valid')
        ax.plot(steps[w-1:], smoothed, 'r-', linewidth=1.5, label=f'MA-{w}')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step'); ax.set_ylabel('Reward')
    ax.set_title(f'Reward (avg={np.mean(rewards):.4f}, last={rewards[-1]:.4f})')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, kls, linewidth=0.4, alpha=0.6, color='#ff7f0e')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='warning')
    ax.set_xlabel('Step'); ax.set_ylabel('KL')
    ax.set_title(f'KL(ref||policy)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(steps, v_losses, linewidth=0.4, alpha=0.6, color='#2ca02c')
    ax.set_xlabel('Step'); ax.set_ylabel('MSE')
    ax.set_title('Value Loss')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=120)
    plt.close()

    print(f"[{time.strftime('%H:%M:%S')}] Step {metrics[-1]['step']:4d} | "
          f"Reward={metrics[-1]['reward']:7.4f} | KL={metrics[-1].get('kl_ref',0):7.4f} | "
          f"v_loss={metrics[-1].get('v_loss',0):7.4f} | 共 {n} 步")

    time.sleep(5)
