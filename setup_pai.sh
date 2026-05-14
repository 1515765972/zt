#!/bin/bash
# ============================================================
# PAI-DSW 环境一键部署脚本
# 镜像: modelscope:1.36.3-pytorch2.3.1-gpu-py312-cu121
# ============================================================
set -e

echo "=========================================="
echo " 代码生成RL项目 - PAI-DSW 环境部署"
echo "=========================================="

PROJECT_DIR="/mnt/workspace/codegen_repro"
cd "$PROJECT_DIR"

# 1. 安装依赖
echo ""
echo "[1/6] 安装 Python 依赖..."
pip install trl flake8 sacrebleu -q -i https://mirrors.aliyun.com/pypi/simple/

# 2. 创建目录结构
echo ""
echo "[2/6] 创建目录结构..."
mkdir -p human_eval data checkpoints/sft checkpoints/ppo results

# 3. 整理 human_eval 模块 (如果还没在 human_eval/ 目录下)
echo ""
echo "[3/6] 整理评测模块..."
if [ ! -f "human_eval/__init__.py" ]; then
    touch human_eval/__init__.py
fi
# 将评测相关文件移入 human_eval/ (如果还在根目录)
for f in execution.py evaluation.py data.py evaluate_functional_correctness.py; do
    if [ -f "$f" ] && [ ! -f "human_eval/$f" ]; then
        cp "$f" "human_eval/$f"
        echo "  已复制 $f → human_eval/$f"
    fi
done

# 4. 移动 HumanEval 数据到 data/
echo ""
echo "[4/6] 整理数据集..."
if [ -f "HumanEval.jsonl.gz" ] && [ ! -f "data/HumanEval.jsonl.gz" ]; then
    cp HumanEval.jsonl.gz data/
    echo "  已复制 HumanEval.jsonl.gz → data/"
fi
if [ -f "human-eval.json" ] && [ ! -f "data/human-eval.json" ]; then
    cp human-eval.json data/
    echo "  已复制 human-eval.json → data/"
fi

# 5. 验证文件完整性
echo ""
echo "[5/6] 验证文件完整性..."
REQUIRED_FILES=(
    "compute_codebleu.py"
    "rewards.py"
    "prepare_conala.py"
    "train_sft.py"
    "train_ppo.py"
    "run_ablation.py"
    "03_eval.py"
    "config.json"
    "conala-paired-train.json"
    "conala-paired-test.json"
    "human_eval/data.py"
    "human_eval/evaluation.py"
    "human_eval/execution.py"
)

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  ❌ 缺失: $f"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 0 ]; then
    echo "  ✅ 所有必要文件已就位"
else
    echo "  ⚠️  请将缺失文件上传到 $PROJECT_DIR"
fi

# 6. 验证 Python 环境
echo ""
echo "[6/6] 验证 Python 环境..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
import transformers; print(f'  Transformers: {transformers.__version__}')
try:
    import trl; print(f'  TRL: {trl.__version__}')
except ImportError:
    print('  TRL: 未安装 (请运行: pip install trl)')
try:
    import flake8; print(f'  flake8: 已安装')
except ImportError:
    print('  flake8: 未安装')
"

echo ""
echo "=========================================="
echo " 环境部署完成!"
echo "=========================================="
echo ""
echo "下一步操作:"
echo "  1. 下载/放置 CodeGen 模型到 /mnt/workspace/models/codegen-2B-mono/"
echo "  2. python prepare_conala.py    # 数据预处理"
echo "  3. python train_sft.py         # SFT 训练"
echo "  4. python train_ppo.py         # PPO 训练"
echo "  5. python run_ablation.py      # 消融评测"
