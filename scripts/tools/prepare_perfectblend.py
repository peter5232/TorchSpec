#!/usr/bin/env python3
# 下载 PerfectBlend 数据集，统一对话字段并过滤后输出 TorchSpec JSONL。
# 这是数据准备程序：会访问 Hub 并覆盖输出文件；抽样发生在过滤之前，最终条数不一定等于 sample-size。

"""Prepare PerfectBlend dataset for DFlash training.

Downloads mlabonne/open-perfectblend from HuggingFace, normalizes to
TorchSpec conversation JSONL format, and optionally subsamples.

Usage:
    python scripts/tools/prepare_perfectblend.py --output data/perfectblend_50k.jsonl --sample-size 50000
    python scripts/tools/prepare_perfectblend.py --output data/perfectblend_full.jsonl
"""

import argparse
import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
    "system": "system",
}


# 接受 role/content 或 from/value 消息，后者映射角色名称；任一消息不符合这两种格式就丢弃整段对话。
def normalize_conversation(conv_list):
    """Normalize ShareGPT format to standard role/content format."""
    result = []
    for msg in conv_list:
        if "role" in msg and "content" in msg:
            result.append({"role": msg["role"], "content": msg["content"]})
        elif "from" in msg and "value" in msg:
            role = ROLE_MAPPING.get(msg["from"], msg["from"])
            result.append({"role": role, "content": msg["value"]})
        else:
            return None
    return result


# 至少两条消息且包含 assistant，assistant 内容总字符数至少 50；这是字符启发式，不是 tokenizer 计算的 token 保证。
def is_valid_conversation(conv):
    """Check conversation has valid structure for training."""
    if not conv or len(conv) < 2:
        return False
    has_assistant = any(m["role"] == "assistant" for m in conv)
    if not has_assistant:
        return False
    # Check minimum content length (need enough tokens for block_size=16)
    total_assistant_chars = sum(len(m["content"]) for m in conv if m["role"] == "assistant")
    if total_assistant_chars < 50:
        return False
    return True


# 固定随机种子后下载 train split，先抽样再规范化和过滤，逐行覆盖写出 UTF-8 JSON，并报告保留/跳过数量。
def main():
    parser = argparse.ArgumentParser(description="Prepare PerfectBlend for DFlash training")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument(
        "--sample-size", type=int, default=None, help="Number of samples (default: all)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-turns", type=int, default=2, help="Minimum conversation turns")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading mlabonne/open-perfectblend from HuggingFace...")
    ds = load_dataset("mlabonne/open-perfectblend")["train"]
    print(f"Loaded {len(ds)} samples")

    # 先抽样后过滤：请求 50000 条候选，若一部分对话不合法，最终输出会少于 50000。
    if args.sample_size and args.sample_size < len(ds):
        indices = random.sample(range(len(ds)), args.sample_size)
        ds = ds.select(indices)
        print(f"Subsampled to {len(ds)} samples")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    valid = 0
    skipped = 0
    # 写模式会替换同路径原文件；这不是增量追加或带断点的下载任务。
    with open(args.output, "w") as f:
        for idx, row in enumerate(tqdm(ds, desc="Processing")):
            conv = row.get("conversations", row.get("conversation", []))
            normalized = normalize_conversation(conv)
            if normalized is None or not is_valid_conversation(normalized):
                skipped += 1
                continue
            # 基础有效性与用户 min-turns 两层过滤；turn 在此按消息条数计，不是 user/assistant 成对轮数。
            if len(normalized) < args.min_turns:
                skipped += 1
                continue
            sample = {
                "id": row.get("id", f"perfectblend_{idx}"),
                "conversations": normalized,
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            valid += 1

    print(f"\nDone! Wrote {valid} samples to {args.output}")
    print(f"Skipped {skipped} invalid samples")
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
