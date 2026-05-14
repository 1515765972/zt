"""
CodeBLEU: 代码相似度评估指标

CodeBLEU = α * BLEU + β * Weighted_BLEU + γ * AST_Match

参考: Ren et al., "CodeBLEU: a Method for Automatic Evaluation of Code Synthesis", 2020

由于完整的数据流匹配依赖复杂解析, 本实现使用:
  - BLEU-4 token n-gram 匹配
  - Python 关键字加权的 Weighted BLEU
  - AST 子树结构匹配
权重: α=0.15, β=0.15, γ=0.70 (侧重代码结构)
"""

import ast
import re
from collections import Counter
from math import exp, log
from typing import Dict, List, Tuple

PYTHON_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
    "try", "while", "with", "yield",
}

KEYWORD_WEIGHT = 5.0

TOKEN_PATTERN = re.compile(
    r"""(?x)
    "(?:[^"\\]|\\.)*"           |  # 双引号字符串
    '(?:[^'\\]|\\.)*'           |  # 单引号字符串
    \#.*$                       |  # 注释
    \d+\.?\d*                   |  # 数字
    [a-zA-Z_]\w*                |  # 标识符
    [^\s\w]                       # 标点/运算符
    """
)


def tokenize(code: str) -> List[str]:
    """将 Python 代码分词为 token 列表，过滤空白和注释"""
    tokens = []
    for match in TOKEN_PATTERN.finditer(code):
        token = match.group(0)
        if token.startswith("#"):
            continue
        tokens.append(token)
    return tokens


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _ngram_precision(
    ref_ngrams: Counter, cand_ngrams: Counter
) -> float:
    match = sum(min(cand_ngrams[g], ref_ngrams[g]) for g in cand_ngrams)
    total = max(1, sum(cand_ngrams.values()))
    return match / total


def compute_bleu(reference: str, candidate: str) -> float:
    """计算 BLEU-4 分数"""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if len(cand_tokens) < 4:
        return 0.0

    precisions = []
    for n in range(1, 5):
        ref_ngrams = _get_ngrams(ref_tokens, n)
        cand_ngrams = _get_ngrams(cand_tokens, n)
        precisions.append(_ngram_precision(ref_ngrams, cand_ngrams))

    bp = min(1.0, len(cand_tokens) / max(1, len(ref_tokens)))
    geo_mean = exp(sum(log(max(p, 1e-10)) for p in precisions) / 4.0)
    return bp * geo_mean


def _weighted_ngrams(
    tokens: List[str], weights: List[float], n: int
) -> Dict[Tuple, float]:
    """构建加权 n-gram 计数 (权重=各token权重几何平均)"""
    result: Dict[Tuple, float] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        w = 1.0
        for j in range(n):
            w *= weights[i + j]
        w = w ** (1.0 / n)  # 几何平均
        result[gram] = result.get(gram, 0.0) + w
    return result


def compute_weighted_bleu(reference: str, candidate: str) -> float:
    """关键字加权的 BLEU-4：Python 关键字匹配权重 ×5"""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if len(cand_tokens) < 4:
        return 0.0

    ref_weights = [
        KEYWORD_WEIGHT if t in PYTHON_KEYWORDS else 1.0 for t in ref_tokens
    ]
    cand_weights = [
        KEYWORD_WEIGHT if t in PYTHON_KEYWORDS else 1.0 for t in cand_tokens
    ]

    precisions = []
    for n in range(1, 5):
        ref_wgrams = _weighted_ngrams(ref_tokens, ref_weights, n)
        cand_wgrams = _weighted_ngrams(cand_tokens, cand_weights, n)

        match = sum(
            min(cand_wgrams.get(g, 0.0), ref_wgrams.get(g, 0.0))
            for g in cand_wgrams
        )
        total = max(1.0, sum(cand_wgrams.values()))
        precisions.append(match / total)

    bp = min(1.0, sum(cand_weights) / max(1, sum(ref_weights)))
    geo_mean = exp(sum(log(max(p, 1e-10)) for p in precisions) / 4.0)
    return bp * geo_mean


def _get_ast_node_types(code: str) -> List[str]:
    """提取 AST 子树节点类型序列"""
    try:
        tree = ast.parse(code)
        return [type(node).__name__ for node in ast.walk(tree)]
    except SyntaxError:
        return []


def compute_ast_match(reference: str, candidate: str) -> float:
    """AST 子树类型 Jaccard 相似度"""
    ref_types = Counter(_get_ast_node_types(reference))
    cand_types = Counter(_get_ast_node_types(candidate))

    if not ref_types or not cand_types:
        return 0.0

    intersection = sum((ref_types & cand_types).values())
    union = sum((ref_types | cand_types).values())
    return intersection / union if union > 0 else 0.0


def codebleu(
    reference: str,
    candidate: str,
    alpha: float = 0.15,
    beta: float = 0.15,
    gamma: float = 0.70,
) -> float:
    """
    CodeBLEU 综合分数

    Args:
        reference: 参考代码 (ground truth)
        candidate: 候选代码 (模型生成)
        alpha: BLEU 权重
        beta: 加权 BLEU 权重
        gamma: AST 匹配权重

    Returns:
        [0, 1] 区间的相似度分数
    """
    if not candidate.strip() or not reference.strip():
        return 0.0

    bleu = compute_bleu(reference, candidate)
    w_bleu = compute_weighted_bleu(reference, candidate)
    ast_m = compute_ast_match(reference, candidate)

    return alpha * bleu + beta * w_bleu + gamma * ast_m


# 便捷函数：无参考代码时的替代评估（仅基于 AST 结构复杂度）
def code_structure_score(code: str) -> float:
    """评估代码结构丰富度（无参考代码时使用）"""
    types = _get_ast_node_types(code)
    if not types:
        return 0.0
    unique_ratio = len(set(types)) / len(types) if types else 0.0
    return min(unique_ratio * 2.0, 1.0)
