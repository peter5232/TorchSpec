# 中文导读：离线回放格式由 dataset.json（语义）、manifest.jsonl（索引）和 samples/*.pt（张量）组成。
# 先昂贵地采集一次 teacher 特征，之后训练反复回放；代价是磁盘容量和读写吞吐。
# 最后层是否已过 norm 随数据保存，不能根据当前选中的 offline 后端重新猜测。

"""Small on-disk dataset used by offline replay."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

OFFLINE_SCHEMA_VERSION = 1
_TENSOR_NAMES = ("input_ids", "hidden_states", "target", "last_hidden_states")


# 维护 train/eval 两个清单及按 data_id 查询的索引；载入清单时不把全部张量读进内存。
class OfflineDataset:
    """Read and append replay records.

    The format intentionally has only three parts: ``dataset.json``, one
    ``manifest.jsonl``, and self-describing ``samples/*.pt`` files.
    """

    # 可创建或打开目录，校验格式版本、norm 约定、split、ID 唯一性和文件路径。
    # overwrite=True 会删除已有目录；普通读取不覆盖。索引路径须位于数据根目录内。
    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        create: bool = False,
        last_hidden_states_prenorm: bool | None = None,
        overwrite: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if overwrite and self.root.exists():
            shutil.rmtree(self.root)

        metadata_path = self.root / "dataset.json"
        if create and not metadata_path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(
                    {
                        "version": OFFLINE_SCHEMA_VERSION,
                        "last_hidden_states_prenorm": last_hidden_states_prenorm,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Offline dataset not found: {metadata_path}")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("version") != OFFLINE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported offline dataset version {self.metadata.get('version')!r}"
            )
        if (
            create
            and last_hidden_states_prenorm is not None
            and self.metadata.get("last_hidden_states_prenorm") != last_hidden_states_prenorm
        ):
            raise ValueError("Offline dataset uses a different hidden-state representation")

        self._rows: dict[str, list[dict[str, str]]] = {"train": [], "eval": []}
        self._by_id: dict[str, dict[str, str]] = {}
        manifest = self.root / "manifest.jsonl"
        if manifest.exists():
            with manifest.open(encoding="utf-8") as stream:
                for lineno, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    split = row.get("split")
                    data_id = str(row.get("data_id"))
                    if split not in self._rows:
                        raise ValueError(f"Invalid split in {manifest}:{lineno}")
                    if data_id in self._by_id:
                        raise ValueError(f"Duplicate data_id {data_id!r} in {manifest}")
                    path = (self.root / row["file"]).resolve()
                    if self.root not in path.parents or not path.is_file():
                        raise FileNotFoundError(f"Offline sample not found: {path}")
                    item = {
                        "split": split,
                        "data_id": data_id,
                        "file": row["file"],
                        "seq_len": row["seq_len"],
                    }
                    self._rows[split].append(item)
                    self._by_id[data_id] = item

    # 返回指定 split 的列表副本，调用者重排列表不改变内部顺序；行字典仍是浅共享。
    def rows(self, split: str) -> list[dict[str, str]]:
        return list(self._rows[split])

    # 生成 split 内 ID 集合，供去重或恢复处理进度。
    def ids(self, split: str) -> set[str]:
        return {row["data_id"] for row in self._rows[split]}

    # 返回清单条数，不扫描张量文件，也不重新校验磁盘内容。
    def count(self, split: str) -> int:
        return len(self._rows[split])

    # 按 ID/清单项使用 CPU mmap 读取张量字典，验证 ID 及必要的输入/辅助特征。
    # mmap 减少立即读取全部文件的压力，实际访问页面时仍可能发生磁盘 I/O。
    def load(self, row_or_id: dict[str, str] | str) -> dict[str, Any]:
        row = self._by_id[str(row_or_id)] if isinstance(row_or_id, str) else row_or_id
        path = self.root / row["file"]
        record = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(record, dict):
            raise ValueError(f"Offline sample must contain a dict: {path}")
        if str(record.get("data_id")) != row["data_id"]:
            raise ValueError(f"Offline sample data_id does not match manifest: {path}")
        if not all(
            isinstance(record.get(name), torch.Tensor) for name in ("input_ids", "hidden_states")
        ):
            raise ValueError(f"Offline sample has missing tensors: {path}")
        return record

    # 已存在同 split 的 ID 返回 False；新样本 detach 后存 CPU 连续张量，再提交清单记录。
    # 先写临时文件并原子替换，随后 fsync 清单，避免清单指向尚未写好的半文件。
    # 文件名用 ID 哈希；它是命名策略，不是张量内容校验。这里按单写者使用，没有多写者锁。
    def append(
        self,
        split: str,
        *,
        data_id: str,
        tensors: dict[str, torch.Tensor | None],
        packed_loss_mask: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        if split not in self._rows:
            raise ValueError(f"Unknown offline split: {split!r}")
        data_id = str(data_id)
        if data_id in self._by_id:
            if self._by_id[data_id]["split"] != split:
                raise ValueError(f"data_id {data_id!r} already exists in another split")
            return False

        saved = {
            name: value.detach().cpu().contiguous()
            for name, value in tensors.items()
            if name in _TENSOR_NAMES and isinstance(value, torch.Tensor)
        }
        if not {"input_ids", "hidden_states"}.issubset(saved):
            raise ValueError(f"Offline sample {data_id!r} has missing tensors")

        relative = Path("samples") / f"{hashlib.sha1(data_id.encode()).hexdigest()}.pt"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            **saved,
            "data_id": data_id,
            "packed_loss_mask": packed_loss_mask,
            "metadata": dict(metadata or {}),
        }
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        os.close(fd)
        try:
            torch.save(record, temporary)
            # 发布最终文件名之前内容已完成写入；消费者通过清单发现文件时应看到完整记录。
            # 但样本文件和清单不是一个跨文件事务，异常可能留下未被清单引用的孤立文件。
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

        row = {
            "split": split,
            "data_id": data_id,
            "file": relative.as_posix(),
            "seq_len": int(saved["input_ids"].numel()),
        }
        with (self.root / "manifest.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._rows[split].append(row)
        self._by_id[data_id] = row
        return True


# 用数据记录的 pre/post norm 语义覆盖运行参数；缺少该字段立即失败，避免 teacher 概率算错。
def configure_offline_args(dataset: OfflineDataset, args) -> None:
    """Use the representation detail recorded with the tensors."""
    value = dataset.metadata.get("last_hidden_states_prenorm")
    if value is None:
        raise ValueError("Offline dataset does not declare last_hidden_states_prenorm")
    args.last_hidden_states_prenorm = bool(value)
