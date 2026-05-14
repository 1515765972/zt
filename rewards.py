"""
多维度奖励函数模块

R_total = w_syn * R_syn + w_exec * R_exec + w_sim * R_sim + w_review * (1 + P_review)

四个维度:
  1. R_syn   - 语法奖励 (AST 解析): {0, 1}，失败时总分为 -1.0
  2. R_exec  - 执行奖励 (代码可运行性): {0, 1}
  3. R_sim   - 相似度奖励 (CodeBLEU): [0, 1]
  4. P_review - 代码审查惩罚 (flake8): [-0.5, 0]
"""

import ast
import os
import subprocess
import tempfile
import warnings
from typing import Optional, Dict

from compute_codebleu import codebleu


def r_syntax(code: str) -> float:
    """语法奖励: AST 解析成功=1.0, 失败=-1.0"""
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError:
        return -1.0


def r_exec(code: str, test: Optional[str] = None, timeout: float = 5.0) -> float:
    """
    执行奖励: 代码可运行性

    - 若提供 test 字符串, 拼接后执行, 返回码=0得1分
    - 若未提供 test, 仅检查代码能否无异常执行 (try-except 包装)
    """
    full_code = code
    if test:
        full_code = code + "\n" + test
    else:
        full_code = (
            "try:\n"
            + "\n".join("    " + line for line in code.split("\n"))
            + "\nexcept Exception:\n    pass\n"
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(full_code)
        fname = f.name

    try:
        result = subprocess.run(
            ["python3", fname],
            timeout=timeout,
            capture_output=True,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def r_flake8(code: str, max_line_length: int = 100) -> float:
    """
    代码审查惩罚: 每个 flake8 违规扣 0.05 分, 上限 -0.5

    Returns:
        非正数: 0.0 (无违规) ~ -0.5 (严重违规)
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        fname = f.name

    try:
        result = subprocess.run(
            ["python3", "-m", "flake8", f"--max-line-length={max_line_length}", fname],
            capture_output=True, text=True, timeout=30,
        )
        violations = [l for l in result.stdout.strip().split("\n") if l]
        return -min(len(violations) * 0.05, 0.5)
    except FileNotFoundError:
        warnings.warn("flake8 未安装, 审查惩罚返回 0")
        return 0.0
    except Exception:
        return 0.0
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def r_codebleu(reference: str, candidate: str) -> float:
    """相似度奖励: CodeBLEU 分数 [0, 1]"""
    return codebleu(reference, candidate)


def compute_reward(
    code: str,
    reference_code: Optional[str] = None,
    test: Optional[str] = None,
    w_syn: float = 0.25,
    w_exec: float = 0.30,
    w_sim: float = 0.35,
    w_review: float = 0.10,
) -> float:
    """
    综合奖励函数

    Args:
        code: 模型生成的代码
        reference_code: 参考答案 (CoNaLa ground truth), 用于 CodeBLEU
        test: 测试用例 (可选, CoNaLa 无测试用例时可传 None)
        w_syn: 语法权重
        w_exec: 执行权重
        w_sim: 相似度权重
        w_review: 审查权重

    Returns:
        综合奖励分数, 语法错误时返回 -1.0
    """
    syn = r_syntax(code)
    if syn < 0:
        return -1.0

    exc = r_exec(code, test)
    sim = r_codebleu(reference_code, code) if reference_code else 0.0
    review = r_flake8(code)

    reward = w_syn * syn + w_exec * exc + w_sim * sim + w_review * (1.0 + review)
    return reward


def get_dynamic_weights(step: int, total_steps: int, recent_reward: float = 0.0) -> Dict[str, float]:
    """
    自适应动态权重策略
    根据实际奖励水平自动切换阶段, 而非盲定时切换

    阶段1 (reward<0.3):  重语法+执行, 建立基础能力
    阶段2 (0.3≤reward<0.5): 重执行+相似度, 优化功能正确性
    阶段3 (reward≥0.5):  加强审查, 提升代码规范
    """
    # 前 30 步强制阶段1 (冷启动保护)
    if step < 30:
        return dict(w_syn=0.45, w_exec=0.35, w_sim=0.15, w_review=0.05)

    if recent_reward < 0.3:
        # 阶段1: 语法+执行为主
        return dict(w_syn=0.40, w_exec=0.40, w_sim=0.15, w_review=0.05)
    elif recent_reward < 0.5:
        # 阶段2: 执行+相似度为主
        return dict(w_syn=0.15, w_exec=0.35, w_sim=0.40, w_review=0.10)
    else:
        # 阶段3: 相似度+审查
        return dict(w_syn=0.10, w_exec=0.30, w_sim=0.35, w_review=0.25)


def compute_reward_ablation(
    code: str,
    reference_code: Optional[str] = None,
    test: Optional[str] = None,
    ablation: str = "full",
) -> float:
    """
    消融实验奖励计算

    Args:
        ablation: 消融配置
            - "full":    全部四维奖励
            - "exec_only": 仅执行奖励
            - "no_review": 无审查惩罚
            - "no_codebleu": 无相似度奖励
            - "syntax_exec": 仅语法+执行
    """
    weights_map = {
        "full":        dict(w_syn=0.25, w_exec=0.30, w_sim=0.35, w_review=0.10),
        "exec_only":   dict(w_syn=0.00, w_exec=1.00, w_sim=0.00, w_review=0.00),
        "no_review":   dict(w_syn=0.30, w_exec=0.35, w_sim=0.35, w_review=0.00),
        "no_codebleu": dict(w_syn=0.35, w_exec=0.50, w_sim=0.00, w_review=0.15),
        "syntax_exec": dict(w_syn=0.40, w_exec=0.60, w_sim=0.00, w_review=0.00),
    }

    weights = weights_map.get(ablation, weights_map["full"])
    return compute_reward(code, reference_code, test, **weights)
