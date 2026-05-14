"""
CoNaLa 数据集预处理

支持两种数据源:
  1. 本地 JSONL 文件 (推荐, 无需网络) — conala-paired-train.json / conala-paired-test.json
  2. HuggingFace datasets 自动下载 (备选)

输出:
  - data/conala_sft_train.jsonl   SFT 训练集
  - data/conala_sft_val.jsonl     SFT 验证集
  - data/conala_ppo.jsonl         PPO 训练数据 (intent + reference_code)
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple


# ─── 本地数据路径 (相对于项目根目录) ──────────────────────
LOCAL_TRAIN_FILE = "conala-paired-train.json"
LOCAL_TEST_FILE = "conala-paired-test.json"
LOCAL_MINED_FILE = "conala-mined.json"       # 可选: 大规模挖掘数据 (~59万条)

# ─── Prompt 模板 ──────────────────────────────────────────
INSTRUCTION_TEMPLATE = (
    "Generate Python code based on the following description.\n"
    "Description: {intent}\n"
    "Code:"
)

SFT_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "{instruction}\n\n"
    "### Response:\n"
    "{response}"
)


def load_local_coala(filepath: str) -> List[Dict]:
    """
    从本地 JSONL 文件加载 CoNaLa 数据

    支持的格式:
      - paired:  {"intent", "rewritten_intent", "snippet", "question_id"}
      - mined:   {"intent", "snippet", "question_id", "prob", ...}

    自动使用 rewritten_intent (更清晰), 回退到 intent
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 优先使用 rewritten_intent (更规范的描述)
            intent = item.get("rewritten_intent") or item.get("intent", "")
            snippet = item.get("snippet", "")

            if intent and snippet:
                data.append({"intent": intent.strip(), "snippet": snippet.strip()})

    return data


def format_sft_sample(intent: str, snippet: str) -> Dict[str, str]:
    """构造 SFT 训练样本 (instruction → code)"""
    instruction = INSTRUCTION_TEMPLATE.format(intent=intent)
    response = snippet
    text = SFT_PROMPT_TEMPLATE.format(instruction=instruction, response=response)
    return {
        "intent": intent,
        "snippet": snippet,
        "instruction": instruction,
        "response": response,
        "text": text,
    }


def format_ppo_sample(intent: str, snippet: str) -> Dict[str, str]:
    """构造 PPO 查询样本 (包含参考答案供 CodeBLEU 计算)"""
    return {
        "intent": intent,
        "query": INSTRUCTION_TEMPLATE.format(intent=intent),
        "reference_code": snippet,
    }


def prepare_conala(
    output_dir: str = "data",
    train_file: Optional[str] = None,
    test_file: Optional[str] = None,
    val_ratio: float = 0.05,
    seed: int = 42,
    use_mined: bool = False,
    mined_max: int = 50000,
) -> Tuple[str, str, str]:
    """
    加载并预处理 CoNaLa 数据集

    Args:
        output_dir: 输出目录
        train_file: 本地训练文件路径 (默认: conala-paired-train.json)
        test_file:  本地测试文件路径 (默认: conala-paired-test.json)
        val_ratio:  从训练集中划分验证集的比例 (仅当无 test_file 时生效)
        seed:       随机种子
        use_mined:  是否混入 mined 数据扩充训练集
        mined_max:  最多使用多少条 mined 数据

    Returns:
        (sft_train_path, sft_val_path, ppo_path) 三个文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. 加载训练数据 ──
    train_path = train_file or LOCAL_TRAIN_FILE

    if os.path.exists(train_path):
        print(f"从本地文件加载训练数据: {train_path}")
        train_data = load_local_coala(train_path)
    else:
        print(f"本地文件 {train_path} 不存在, 尝试从 HuggingFace 下载...")
        try:
            from datasets import load_dataset
            dataset = load_dataset("neulab/conala")
            raw = dataset["train"].to_list()
            train_data = [
                {"intent": (item.get("rewritten_intent") or item.get("intent", "")).strip(),
                 "snippet": item.get("snippet", "").strip()}
                for item in raw
                if item.get("intent") and item.get("snippet")
            ]
        except Exception as e:
            raise RuntimeError(
                f"无法加载 CoNaLa 数据。请确保 {train_path} 存在, 或网络可访问 HuggingFace。\n"
                f"错误: {e}"
            )

    print(f"训练数据原始样本: {len(train_data)}")

    # ── 2. 可选: 混入 mined 数据 ──
    if use_mined and os.path.exists(LOCAL_MINED_FILE):
        print(f"加载 mined 数据: {LOCAL_MINED_FILE} (最多 {mined_max} 条)...")
        mined_data = load_local_coala(LOCAL_MINED_FILE)
        random.seed(seed)
        mined_sample = random.sample(mined_data, min(mined_max, len(mined_data)))
        train_data.extend(mined_sample)
        print(f"混入 mined 数据后: {len(train_data)} 条")

    # ── 3. 去重 & 打乱 ──
    seen = set()
    dedup = []
    for item in train_data:
        key = (item["intent"], item["snippet"])
        if key not in seen:
            seen.add(key)
            dedup.append(item)
    train_data = dedup
    print(f"去重后: {len(train_data)} 条")

    random.seed(seed)
    random.shuffle(train_data)

    # ── 4. 划分训练/验证 ──
    test_path = test_file or LOCAL_TEST_FILE
    if os.path.exists(test_path):
        # 使用独立的测试文件作为验证集
        print(f"从本地文件加载验证数据: {test_path}")
        val_data = load_local_coala(test_path)
        print(f"验证数据: {len(val_data)} 条 (来自独立测试集)")
    else:
        # 从训练集中切分
        val_size = max(1, int(len(train_data) * val_ratio))
        val_data = train_data[:val_size]
        train_data = train_data[val_size:]
        print(f"训练集: {len(train_data)}, 验证集: {len(val_data)} (按 {val_ratio:.0%} 切分)")

    # ── 5. 保存 SFT 格式 ──
    sft_train_path = os.path.join(output_dir, "conala_sft_train.jsonl")
    sft_val_path = os.path.join(output_dir, "conala_sft_val.jsonl")

    with open(sft_train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            sample = format_sft_sample(item["intent"], item["snippet"])
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(sft_val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            sample = format_sft_sample(item["intent"], item["snippet"])
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"SFT 训练数据 → {sft_train_path} ({len(train_data)} 条)")
    print(f"SFT 验证数据 → {sft_val_path} ({len(val_data)} 条)")

    # ── 6. 保存 PPO 格式 ──
    ppo_path = os.path.join(output_dir, "conala_ppo.jsonl")
    with open(ppo_path, "w", encoding="utf-8") as f:
        for item in train_data:
            sample = format_ppo_sample(item["intent"], item["snippet"])
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"PPO 训练数据 → {ppo_path} ({len(train_data)} 条)")

    return sft_train_path, sft_val_path, ppo_path


def load_conala_ppo_data(filepath: str = "data/conala_ppo.jsonl") -> List[Dict]:
    """加载 PPO 训练数据 (queries + reference codes)"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_conala_sft_data(
    filepath: str = "data/conala_sft_train.jsonl",
) -> List[Dict[str, str]]:
    """加载 SFT 训练数据"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


if __name__ == "__main__":
    prepare_conala()
