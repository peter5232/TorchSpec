# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# 中文导读：数据形状、监督 mask 和数据源读取的公共工具。
# 先读 pack/unpack/resolve_loss_mask，再读 collator：前者确定训练语义，后者只把变长样本拼成矩形张量。
# 注意输入列表中的单样本常带 [1,T] 或 [1,T,H] 维度，cat 后才得到 [B,T] / [B,T,H]。

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import IterableDataset, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

from torchspec.models.ops.loss_mask import compute_assistant_loss_mask

_LOCAL_DATA_EXTS = frozenset({".json", ".jsonl", ".parquet", ".arrow", ".csv", ".tsv", ".txt"})


# 用路径前缀、常见扩展名或磁盘存在性区分本地路径与 Hub ID；可相对指定基目录探测。
def is_local_data_path(path: str, base_dir: str | None = None) -> bool:
    """True if *path* looks like a local file/directory rather than a HF Hub dataset ID.

    When *base_dir* is given, relative paths are probed against it instead of
    the process CWD.
    """
    if path.startswith((".", "/", "~")):
        return True
    if os.path.splitext(path)[1].lower() in _LOCAL_DATA_EXTS:
        return True
    probe = os.path.join(base_dir, path) if base_dir is not None else path
    return os.path.exists(probe)


# 对齐变长 token/特征/mask 后沿 batch 维拼接；不是 tokenizer，也不重新推断 assistant 区间。
class DataCollatorWithPadding:
    # 保存是否使用 USP；普通路径另做长度分桶，USP 则保留调用方给的 attention mask。
    def __init__(self, usp_enabled: bool = False):
        self.sp_degree = 1
        self.usp_enabled = usp_enabled

    # 将 [B,n,H] 沿序列维补零或截为 [B,N,H]；保持 dtype/device，不动 hidden 维。
    def paddingtensor(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        B, n, S = intensors.shape
        # Truncate if longer than target (can happen when loss_mask/hidden_states
        # length differs from input_ids after unpacking).
        if n > N:
            return intensors[:, :N, :]
        if n == N:
            return intensors
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype, device=intensors.device)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    # 将 [B,n] ID/mask 沿序列维对齐到 N；mask 补零使新增 padding 不参与监督/有效上下文。
    def paddingtensor2D(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        B, n = intensors.shape
        # Truncate if longer than target (prevents negative padding dimension
        # when loss_mask length differs from input_ids after collation).
        if n > N:
            return intensors[:, :N]
        if n == N:
            return intensors
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype, device=intensors.device)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    # 要求上游已经附上张量 mask；缺失则报错，而不是猜一个全一 mask 混入训练。
    def _get_loss_mask(self, item: Dict[str, Any]) -> torch.Tensor:
        """Read the materialized loss_mask tensor from the item.

        Callers (e.g. MooncakeDataset) are responsible for computing and
        attaching loss_mask before items reach the collator.
        """
        if "loss_mask" in item and isinstance(item["loss_mask"], torch.Tensor):
            return item["loss_mask"]
        raise KeyError(f"loss_mask not found in item: {item}")

    # 找到批内最大长度，普通路径再向上对齐到 128 的倍数，以减少编译器遇到的新形状。
    # 例如长度 130 和 200 都补到 256；token/hidden 可以补零，但 attention/loss mask 必须同时屏蔽这些位置。
    # USP 保留已有位置与注意力信息；可选 target/last_hidden_states 仅在全批均有时拼接。
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        max_length = ((max_length + self.sp_degree - 1) // self.sp_degree) * self.sp_degree

        if self.usp_enabled:
            attention_masks = [item["attention_mask"].long() for item in features]
        else:
            # Round up to nearest bucket to reduce unique shapes for torch.compile.
            # Without this, every batch gets a different padded length, causing
            # FlexAttention recompilation (~1s overhead per new shape).
            # 编译器常按形状缓存图/内核；长度 129、130、131 若都产生新形状，就可能反复编译。
            # 分桶让它们都走 256 长度的计算，用一些 padding 算力换取更少的编译和稳定运行。
            _BUCKET = 128
            max_length = ((max_length + _BUCKET - 1) // _BUCKET) * _BUCKET
            attention_masks = [torch.ones_like(item["input_ids"]).long() for item in features]

        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(mask, max_length) for mask in attention_masks]
        )
        # 输入 padding token 的数值不够表达“无效”，必须在 mask 中明确排除。
        # 例如原序列长度 3 补到 128，新增 125 个位置不能被当作 125 个真实监督样本。
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(self._get_loss_mask(item), max_length) for item in features]
        )
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_states": None,
            "target": None,
            "last_hidden_states": None,
        }
        if self.usp_enabled:
            max_position_length = max(item["position_ids"].shape[1] for item in features)
            batch["position_ids"] = torch.cat(
                [
                    self.paddingtensor2D(item["position_ids"], max_position_length)
                    for item in features
                ]
            )
        if all("hidden_states" in item for item in features):
            batch["hidden_states"] = torch.cat(
                [self.paddingtensor(item["hidden_states"], max_length) for item in features]
            )
            has_target = all(item.get("target") is not None for item in features)
            has_last_hs = all(item.get("last_hidden_states") is not None for item in features)
            if not has_target and not has_last_hs:
                pass
            if has_target:
                batch["target"] = torch.cat(
                    [self.paddingtensor(item["target"], max_length) for item in features]
                )
            if has_last_hs:
                batch["last_hidden_states"] = torch.cat(
                    [
                        self.paddingtensor(item["last_hidden_states"], max_length)
                        for item in features
                    ]
                )
        return batch


# 把二值 mask 编码成从零段开始的交替段长，压缩连续的用户/助手区域。
# 例如 [1,1,0,1] → [0,2,1,1]，开头的 0 不可省，否则解码时零/一奇偶性会颠倒。
def pack_loss_mask(loss_mask: torch.Tensor) -> List[int]:
    """
    Pack a loss_mask tensor into interleaved segment lengths.

    The returned list alternates between prompt and response lengths,
    always starting with prompt (even if length 0).

    Args:
        loss_mask: 1D tensor of 0s and 1s indicating which tokens contribute to loss.

    Returns:
        List of segment lengths: [prompt_len, response_len, prompt_len, response_len, ...]

    Example:
        loss_mask = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
        returns: [2, 3, 2, 2, 1]  # 2 prompt, 3 response, 2 prompt, 2 response, 1 prompt
    """
    if loss_mask.dim() > 1:
        loss_mask = loss_mask.squeeze()

    if len(loss_mask) == 0:
        return []

    lengths = []
    mask_list = loss_mask.tolist()
    current_val = 0
    current_len = 0

    for val in mask_list:
        if val == current_val:
            current_len += 1
        else:
            lengths.append(current_len)
            current_val = val
            current_len = 1

    lengths.append(current_len)
    return lengths


# 按段索引奇偶恢复一维 long mask；奇数段填一，偶数段保持零，空输入返回空张量。
def unpack_loss_mask(packed: Union[List[int], str]) -> torch.Tensor:
    """
    Unpack segment lengths back into a loss_mask tensor.

    Args:
        packed: List of segment lengths [prompt_len, response_len, prompt_len, ...],
                or a serialized packed_loss_mask string (e.g. "2,3,2,2,1").

    Returns:
        1D tensor of 0s and 1s.

    Example:
        packed = [2, 3, 2, 2, 1]
        returns: tensor([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    """
    if isinstance(packed, str):
        packed = deserialize_packed_loss_mask(packed)
    if not packed:
        return torch.tensor([], dtype=torch.long)

    total = sum(packed)
    loss_mask = torch.zeros(total, dtype=torch.long)
    pos = 0

    for i, length in enumerate(packed):
        if i % 2 == 1:
            loss_mask[pos : pos + length] = 1
        pos += length

    return loss_mask


# 优先用存储的 packed mask 并对齐到实际长度；否则按真实 token 的角色边界动态匹配。
# 已算出但全零的 mask 返回 None 表示无可训练位置；两种策略都不适用时返回长度 1 的全一占位值。
# 函数还会将有效 mask 写回 data；多模态必须事先删除旧 packed mask，否则优先级会阻止重算。
def resolve_loss_mask(
    data: Dict[str, Any],
    *,
    dynamic_loss_mask: bool = False,
    assistant_header_ids: Optional[List[int]] = None,
    end_token_ids: Optional[List[int]] = None,
    last_turn_loss_only: bool = False,
    skip_after_header: int = 0,
) -> torch.Tensor | None:
    """
    Two strategies, tried in order:
    1. ``packed_loss_mask`` key present → unpack it.
    2. ``dynamic_loss_mask`` enabled with valid header/end ids → compute from
       ``input_ids`` via :func:`compute_assistant_loss_mask`.
    """
    packed = data.get("packed_loss_mask")
    if packed is not None:
        mask = unpack_loss_mask(packed)
        input_ids = data.get("input_ids")
        if input_ids is not None:
            if input_ids.dim() == 2:
                input_ids = input_ids.squeeze(0)
            expected_len = input_ids.shape[-1]
            if mask.shape[0] > expected_len:
                mask = mask[:expected_len]
            elif mask.shape[0] < expected_len:
                mask = torch.nn.functional.pad(mask, (0, expected_len - mask.shape[0]))
        if not mask.any():
            return None
        data["loss_mask"] = mask
        return mask

    if dynamic_loss_mask and assistant_header_ids and end_token_ids:
        input_ids = data.get("input_ids")
        if input_ids is None:
            return None
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        per_sample = data.get("last_turn_loss_only")
        last_turn_only = per_sample if per_sample is not None else last_turn_loss_only
        if not isinstance(last_turn_only, bool):
            last_turn_only = False
        mask = compute_assistant_loss_mask(
            input_ids,
            assistant_header_ids,
            end_token_ids,
            last_turn_only=last_turn_only,
            skip_after_header=skip_after_header,
        )
        if not mask.any():
            return None
        data["loss_mask"] = mask
        return mask

    return torch.ones(1)


# 段长列表转逗号字符串，便于经过引擎/RPC 元数据通道，不需要传一个长 mask 张量。
def serialize_packed_loss_mask(packed: List[int]) -> str:
    """
    Serialize packed loss_mask to a comma-separated string.

    Args:
        packed: List of segment lengths from pack_loss_mask().

    Returns:
        Comma-separated string of integers.

    Example:
        packed = [2, 3, 2, 2, 1]
        returns: "2,3,2,2,1"
    """
    return ",".join(str(x) for x in packed)


# 将逗号字符串恢复为整数段长；这里仅解析，不做总长或非负校验，上层契约负责验证。
def deserialize_packed_loss_mask(s: str) -> List[int]:
    """
    Deserialize a comma-separated string back to packed loss_mask.

    Args:
        s: Comma-separated string of integers.

    Returns:
        List of segment lengths.

    Example:
        s = "2,3,2,2,1"
        returns: [2, 3, 2, 2, 1]
    """
    if not s:
        return []
    return [int(x) for x in s.split(",")]


# 只提取图片/视频引用，不下载媒体；缺失地址仍保留 None 位置，避免破坏占位符对应次序。
def extract_media_urls(messages: list) -> dict | None:
    """Extract image/video URLs from structured messages without loading them."""
    images = []
    videos = []

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image":
                if "image" in item:
                    images.append(item["image"])
                else:
                    images.append(None)
            elif item_type == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, dict) and "url" in image_url:
                    images.append(image_url["url"])
                else:
                    images.append(None)
            elif item_type == "video":
                if "video" in item:
                    videos.append(item["video"])
                else:
                    videos.append(None)

    if not images and not videos:
        return None
    return {"images": images or None, "videos": videos or None}


# 就地将结构化 content 改为文本和媒体占位符，必须先提取 URL，否则结构信息会丢失。
def flatten_multimodal_content(messages, image_placeholder="<image>"):
    """Convert list-type content parts to plain text strings.

    Transforms the standard HF multimodal format:
      [{"type":"image"}, {"type":"text","text":"Describe"}]
    into a single string:
      "<image>\\nDescribe"

    Messages with string content are left unchanged.
    Must be called AFTER extract_media_urls so structured info is captured first.
    """
    for msg in messages:
        content = msg.get("content")
        if content is None:
            msg["content"] = ""
            continue
        if not isinstance(content, (str, list)):
            raise ValueError(
                f"Message content must be a str or list, got {type(content).__name__}: {repr(content)[:100]}"
            )
        if not isinstance(content, list):
            continue
        text_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type", "")
            if part_type == "text":
                text_parts.append(part.get("text", ""))
            elif part_type in ("image", "image_url"):
                text_parts.append(image_placeholder)
            elif part_type == "video":
                text_parts.append("<video>")
        msg["content"] = "\n".join(text_parts)
    return messages


# 优先取离线 seq_len，再取 token 数，最后用格式化文本字符数作粗略成本估计。
# 字符数不是严格单调的 token 数，跨语言时尤其只是启发式；它不参与模型位置计算。
def sample_length_hint(sample: Dict[str, Any]) -> int:
    """Sequence length of a stored dataset entry, for length grouping.

    Exactly one of these three is present: ``seq_len`` on offline replay rows,
    the tokenized ``input_ids``, or — under ``defer_tokenization`` — only the
    formatted text, whose character count is monotone in token count and free.
    """
    if "seq_len" in sample:
        return sample["seq_len"]
    if "input_ids" in sample:
        return sample["input_ids"].numel()
    return len(sample["formatted_prompt"])


# 只在固定大小的小组内按长度降序重排，保留组间随机次序。
# 这样减少同批 padding，同时避免全局从短到长/长到短排序把数据变成长度课程。
def length_grouped_order(samples: List[Any], group_size: int) -> List[Any]:
    """Sort *samples* longest-first within fixed chunks of *group_size*.

    Chunks are positional, so only the order inside each one changes. That keeps
    a dispatch's sequences similar in length without turning length into a
    curriculum the way a global sort would.
    """
    if group_size <= 1:
        return list(samples)

    ordered = []
    for start in range(0, len(samples), group_size):
        chunk = samples[start : start + group_size]
        chunk.sort(key=sample_length_hint, reverse=True)
        ordered.extend(chunk)
    return ordered


# 为进度条估计本地 JSONL 行数或 Parquet 元数据行数，不支持/读取失败返回 None。
def estimate_row_count(data_path):
    if not os.path.isfile(data_path):
        return None
    if data_path.endswith(".jsonl"):
        with open(data_path, "rb") as f:
            return sum(1 for _ in f)
    if data_path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq

            return pq.ParquetFile(data_path).metadata.num_rows
        except Exception:
            return None
    if data_path.endswith(".json"):
        return None
    return None


# 根据首字符区分 JSON 数组与逐行 JSON，逐条 yield；JSONL 空行跳过。
def load_local_json(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            yield from json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


# 筛选指定后缀的仓库文件，若 data/ 下有匹配则优先使用那一组并稳定排序。
def _list_hub_data_files(data_path: str, suffixes: tuple[str, ...]) -> list[str]:
    """List repo files matching ``suffixes``, preferring those under ``data/``."""
    files = list_repo_files(data_path, repo_type="dataset")
    matching = sorted(f for f in files if f.endswith(suffixes))
    preferred = [f for f in matching if f.startswith("data/")]
    return preferred or matching


# 直接下载 JSON 文件并用标准库解析，绕过混合类型列导致的 Arrow schema 推断问题。
def _load_hub_json_files(data_path):
    """Download JSON/JSONL files from a HF Hub dataset and yield rows.

    Uses raw json module instead of load_dataset to avoid PyArrow schema
    inference failures on datasets with mixed-type columns.
    """
    data_files = _list_hub_data_files(data_path, (".jsonl", ".json"))
    if not data_files:
        raise ValueError(f"No JSON/JSONL files found in HF Hub dataset '{data_path}'.")

    for filename in data_files:
        local_path = hf_hub_download(repo_id=data_path, filename=filename, repo_type="dataset")
        yield from load_local_json(local_path)


# 绕过数据集卡片配置，直接用 Parquet 文件 URL 建流式数据集。
def _load_hub_parquet_dataset(data_path: str):
    """Stream parquet files from a HF Hub dataset.

    Used when ``load_dataset(repo_id, ...)`` fails due to schema/card parsing
    issues but the repo's parquet shards themselves are valid.
    """
    parquet_files = _list_hub_data_files(data_path, (".parquet",))
    if not parquet_files:
        raise ValueError(f"No parquet files found in HF Hub dataset '{data_path}'.")
    urls = [f"https://huggingface.co/datasets/{data_path}/resolve/main/{f}" for f in parquet_files]
    return load_dataset("parquet", data_files=urls, split="train", streaming=True)


# 本地文件/目录按格式读取，Hub 优先原生 streaming，schema 错误时尝试原始分片。
# 预分词语料保留全部列以免丢掉 ID/mask 和溯源信息；普通 Hub 对话仅保留约定的输入列。
def load_hf_dataset(data_path: str):
    """Load dataset as a streaming IterableDataset.

    Local paths are loaded directly; everything else goes to HF Hub.
    """
    data_path = os.path.expanduser(data_path)

    if is_local_data_path(data_path):
        if os.path.isfile(data_path):
            if data_path.endswith((".json", ".jsonl")):
                return IterableDataset.from_generator(
                    load_local_json, gen_kwargs={"data_path": data_path}
                )
            ext = os.path.splitext(data_path)[1].lower()
            fmt = {".parquet": "parquet", ".arrow": "arrow"}.get(ext, "json")
            return load_dataset(fmt, data_files=data_path, split="train", streaming=True)

        if os.path.isdir(data_path):
            patterns = {
                "json": ["*.json", "*.jsonl"],
                "parquet": ["*.parquet"],
                "arrow": ["*.arrow"],
            }
            for fmt, globs in patterns.items():
                files = []
                for g in globs:
                    files.extend(str(p) for p in Path(data_path).rglob(g))
                if files:
                    return load_dataset(
                        fmt, data_files=sorted(files), split="train", streaming=True
                    )
            raise ValueError(f"No supported dataset files found in local directory: {data_path}")

        raise FileNotFoundError(f"Local dataset path not found: {data_path}")

    # hub path — try native load_dataset first (handles Arrow, Parquet, etc.),
    # fall back to manual JSON download for repos with mixed-type columns
    _KEEP_COLUMNS = frozenset(
        {
            "id",
            "data_id",
            "conversations",
            "text",
            "messages",
            "tools",
            "generation_config",
        }
    )
    try:
        ds = load_dataset(data_path, split="train", streaming=True)
        columns = ds.column_names or []
        # A pretokenized repo's provenance columns cannot be enumerated ahead of
        # time, and dropping input_ids would silently demote it to raw text.
        if "input_ids" in columns:
            return ds
        drop_cols = [c for c in columns if c not in _KEEP_COLUMNS]
        if drop_cols:
            ds = ds.remove_columns(drop_cols)
        return ds
    except (ValueError, TypeError, ArithmeticError, KeyError) as e:
        # Schema inference / card-parsing failures (e.g., mixed-type columns in
        # Arrow/Parquet, malformed dataset_info YAML). Fall back to direct file
        # streaming — parquet first, then JSON/JSONL.
        import logging

        log = logging.getLogger(__name__)
        log.info(f"load_dataset failed for '{data_path}' ({e}), trying direct file streaming")
        try:
            return _load_hub_parquet_dataset(data_path)
        except ValueError as parquet_err:
            log.info(f"parquet streaming failed ({parquet_err}), falling back to JSON download")
            return IterableDataset.from_generator(
                _load_hub_json_files, gen_kwargs={"data_path": data_path}
            )
