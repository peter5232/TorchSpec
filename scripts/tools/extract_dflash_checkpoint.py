#!/usr/bin/env python3
# 把 DFlash 训练检查点提取成简单 .pt 权重字典，便于检查和独立加载。
# 与完整 HF 导出不同，这里不生成 config、不做服务端键映射，也不保留优化器恢复协议。

"""Extract DFlash draft model weights from FSDP checkpoint to a simple state_dict .pt file.

Uses the same loading approach as tools/convert_to_hf.py — no distributed setup needed.
"""

import argparse
import os

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.filesystem import FileSystemReader
from typing_extensions import override


# 允许用户传迭代目录或 model 子目录，统一交给 PyTorch DCP 文件读取器。
class _WrappedStorageReader(FileSystemReader):
    """FileSystemReader that auto-locates the 'model/' subdirectory."""

    # 发现 model/ 子目录时自动向下定位，再初始化父类 storage reader。
    def __init__(self, path: str):
        model_dir = os.path.join(path, "model")
        if os.path.isdir(model_dir):
            path = model_dir
        super().__init__(path)

    # 直接复用父类元数据读取，当前包装不修改其中内容。
    def read_metadata(self):
        metadata = super().read_metadata()
        return metadata


# 为未知完整权重键空间分配 CPU 张量，方便不实例化模型就提取 DCP。
class _EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    # 忽略 optimizer 项，按存储元数据分配张量并调用默认加载规划。
    @override
    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k:
                continue
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


# 先尝试普通 checkpoint 文件筛选，再尝试 no_dist DCP 提取；剥离 draft 前缀并排除词表映射，最后 torch.save 写出。
# 普通文件分支保留原键名，DCP 分支会去前缀；无 draft 键时退回保存所有项，因此输出格式需看日志核对。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir", required=True, help="Path to FSDP checkpoint directory (iter_XXXXXXX)"
    )
    parser.add_argument("--output", default="dflash_draft.pt", help="Output .pt file path")
    args = parser.parse_args()

    # Try loading as a simple checkpoint first
    # 普通文件先读到 CPU 并筛名字；此分支成功就直接保存，不执行后面的 DCP 前缀剥离。
    if os.path.isfile(args.checkpoint_dir):
        state = torch.load(args.checkpoint_dir, map_location="cpu")
        draft_keys = {
            k: v
            for k, v in state.items()
            if "draft_model" in k or "context_proj" in k or "context_norm" in k
        }
        if draft_keys:
            torch.save(draft_keys, args.output)
            print(f"Saved {len(draft_keys)} keys to {args.output}")
            return

    # Load from FSDP distributed checkpoint using no_dist mode
    ckpt_dir = args.checkpoint_dir
    print(f"Loading FSDP checkpoint from {ckpt_dir}...")

    state_dict = {}
    # DCP 由元数据恢复完整张量，no_dist 省掉多进程启动，但完整权重仍必须装进本机内存。
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=_WrappedStorageReader(ckpt_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )

    print(f"Loaded {len(state_dict)} keys from checkpoint")

    # Extract draft model keys
    draft_state = {}
    skipped = []
    for k, v in state_dict.items():
        if "draft_model." not in k:
            skipped.append(k)
            continue
        # Extract the part after "draft_model."
        new_key = k.split("draft_model.")[-1]
        if new_key in ("t2d", "d2t"):  # Vocab pruning tables
            continue
        draft_state[new_key] = v

    # 没有匹配时保留全部项作为诊断产物；因此生成了 .pt 文件并不等同正确提取了纯 draft。
    if not draft_state:
        print("Warning: no draft_model keys found. All keys:")
        for k in sorted(state_dict.keys()):
            print(f"  {k}")
        print("\nSaving all keys as-is...")
        draft_state = state_dict

    # 覆盖指定输出路径，不保存模型配置或服务端映射；后续加载者需要自行提供正确架构。
    torch.save(draft_state, args.output)
    print(f"\nSaved {len(draft_state)} draft model keys to {args.output}")
    for k in sorted(draft_state.keys())[:15]:
        shape = draft_state[k].shape if hasattr(draft_state[k], "shape") else "n/a"
        print(f"  {k}: {shape}")
    if len(draft_state) > 15:
        print(f"  ... ({len(draft_state) - 15} more)")

    if skipped:
        print(f"\nSkipped {len(skipped)} non-draft keys")


if __name__ == "__main__":
    main()
