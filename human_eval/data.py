from typing import Iterable, Dict
import gzip
import json
import os


ROOT = os.path.dirname(os.path.abspath(__file__))

# 按优先级查找 HumanEval 文件: 同目录 > 上级 data/ > 上级根目录
_CANDIDATES = [
    os.path.join(ROOT, "HumanEval.jsonl.gz"),
    os.path.join(ROOT, "human-eval.json"),
    os.path.join(ROOT, "..", "data", "HumanEval.jsonl.gz"),
    os.path.join(ROOT, "..", "HumanEval.jsonl.gz"),
    os.path.join(ROOT, "..", "human-eval.json"),
]

HUMAN_EVAL = None
for _c in _CANDIDATES:
    if os.path.exists(_c):
        HUMAN_EVAL = _c
        break

if HUMAN_EVAL is None:
    HUMAN_EVAL = _CANDIDATES[2]  # 默认: ../data/HumanEval.jsonl.gz


def read_problems(evalset_file: str = None) -> Dict[str, Dict]:
    filepath = evalset_file or HUMAN_EVAL
    return {task["task_id"]: task for task in stream_jsonl(filepath)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """解析 jsonl 文件, 支持 .gz 压缩"""
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if line.strip():
                        yield json.loads(line)
    else:
        with open(filename, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """写入 jsonl 文件, 支持 .gz 压缩"""
    mode = 'ab' if append else 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, "w", encoding="utf-8") as fp:
            for x in data:
                fp.write(json.dumps(x, ensure_ascii=False) + "\n")
