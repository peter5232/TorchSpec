# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/EleutherAI/gpt-neox (Apache License 2.0)
#   - https://github.com/huggingface/transformers (Apache License 2.0)
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 中文导读：对话规范化、监督区间构建和草稿词表选择。
# 这里也保留旧版 .ckpt 离线读取器，它与 offline/dataset.py 的 manifest 回放格式不是同一个协议。
# 词表裁剪只缩小草稿输出空间；目标模型与 tokenizer 的 token ID 语义仍需保持一致。

import os
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numba
import numpy as np
import torch
from datasets import Dataset as HFDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

try:
    from qwen_vl_utils import process_vision_info

    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    process_vision_info = None

from torchspec.data.parse import create_parser
from torchspec.data.template import TEMPLATE_REGISTRY, ChatTemplate
from torchspec.data.utils import (
    pack_loss_mask,
    serialize_packed_loss_mask,
    unpack_loss_mask,
)
from torchspec.utils.logging import logger
from torchspec.utils.tensor import padding

# define a type called conversation
Conversation = List[Dict[str, Any]]

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
}

_REASONING_FIELDS = ("thinking", "thinking_content", "reasoning_content", "reasoning")


# 把 ShareGPT 的 from/value 改成 role/content，并保留工具字段，将多种思考字段归一。
# 已有标准格式直接返回；不要假设这个函数总会深拷贝消息，调用方后续就地修改可能影响原列表。
def _normalize_conversation(conversation: Conversation) -> Conversation:
    """
    Normalize conversation format to use role/content keys.
    Handles ShareGPT format (from/value) and converts to standard format (role/content).
    Extended fields such as tool_calls, name, and tool_call_id are preserved.
    """
    if not conversation:
        return conversation

    first_msg = conversation[0]
    if "role" in first_msg and "content" in first_msg:
        return conversation

    if "from" in first_msg and "value" in first_msg:
        normalized = []
        for msg in conversation:
            role = ROLE_MAPPING.get(msg["from"], msg["from"])
            entry = {
                key: value
                for key, value in msg.items()
                if key not in {"from", "value", *_REASONING_FIELDS}
            }
            entry.update(role=role, content=msg["value"])
            for field in _REASONING_FIELDS:
                if msg.get(field):
                    entry["reasoning_content"] = msg[field]
                    break
            normalized.append(entry)
        return normalized

    return conversation


# Copied from https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py
# 逐对话先格式化再编码成 ID 和监督 mask，过滤没有足够监督位置的样本。
# 每个保留样本增加一个 batch 维 [1,T]；attention_mask 标上下文，loss_mask 只标训练目标。
def preprocess_conversations(
    tokenizer: PreTrainedTokenizer,
    conversations: Union[List[Conversation], List[str]],
    chat_template: ChatTemplate,
    max_length: int = 2048,
    is_preformatted: bool = False,
    include_attention_mask: bool = True,
    use_packed_loss_mask: bool = True,
    add_generation_prompt: bool = False,
    return_formatted_text: bool = False,
    last_turn_loss_only: bool = False,
    min_loss_tokens: int = 0,  # DFlash: skip sequences with too few supervised tokens for anchor sampling
    **kwargs,
) -> Dict[str, List]:
    """
    Preprocess a batch of ShareGPT style conversations or pre-formatted text.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        conversations: A list of conversations (if is_preformatted=False) or
                      a list of pre-formatted text strings (if is_preformatted=True).
        chat_template: The chat template to use for formatting/identifying spans.
        max_length: The maximum length of the tokenized input.
        is_preformatted: Whether the input is already formatted text strings.
        include_attention_mask: Whether to include attention_mask in the output.
                               If False, attention_mask can be generated as ones on the inference side.
        use_packed_loss_mask: If True, store loss_mask as packed string (compact).
                             If False, store as tensor (legacy behavior).
        add_generation_prompt: Whether to append assistant header for generation.
                               Only used when is_preformatted=False.
        return_formatted_text: If True, include 'formatted_text' in the output.

    Returns:
        A dictionary containing:
            - input_ids: List of tokenized input IDs.
            - packed_loss_mask: List of packed loss mask strings (if use_packed_loss_mask=True).
            - loss_mask: List of loss mask tensors (if use_packed_loss_mask=False).
            - attention_mask: List of attention masks (only if include_attention_mask=True).
            - formatted_text: List of formatted text strings (only if return_formatted_text=True).
    """

    if use_packed_loss_mask:
        results = {"input_ids": [], "packed_loss_mask": []}
    else:
        results = {"input_ids": [], "loss_mask": []}
    if include_attention_mask:
        results["attention_mask"] = []
    if return_formatted_text:
        results["formatted_text"] = []

    parser = create_parser(tokenizer, chat_template)

    kwargs_list = [{} for _ in range(len(conversations))]
    for key, value_list in kwargs.items():
        for i, value in enumerate(value_list):
            kwargs_list[i][key] = value
    for source, kwargs_item in zip(conversations, kwargs_list):
        if not source:
            continue
        source = _normalize_conversation(source)

        if is_preformatted:
            formatted = source
        else:
            formatted = parser.format(
                source, add_generation_prompt=add_generation_prompt, **kwargs_item
            )
        input_ids, loss_mask = parser.parse(
            formatted,
            max_length,
            preformatted=True,
            last_turn_only=last_turn_loss_only,
        )

        # Skip sequences with insufficient supervised tokens. For DFlash, anchor
        # sampling needs enough valid positions; sequences with near-zero loss_mask
        # produce zero gradients and waste training compute.
        if loss_mask.sum() < max(1, min_loss_tokens):
            continue

        results["input_ids"].append(input_ids[None, :])
        if use_packed_loss_mask:
            packed = pack_loss_mask(loss_mask)
            results["packed_loss_mask"].append(serialize_packed_loss_mask(packed))
        else:
            results["loss_mask"].append(loss_mask[None, :])
        if include_attention_mask:
            results["attention_mask"].append(torch.ones_like(loss_mask)[None, :])
        if return_formatted_text:
            results["formatted_text"].append(formatted)
    return results


# 旧式 HF Dataset.map 批处理入口，检查模板、可选打乱、映射并设置 torch 输出格式。
# 它与 controller 使用的 load_conversation_dataset 返回类型不同：这里返回 HF Dataset，而非条目列表。
def build_eagle3_dataset(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    chat_template: Optional[str] = None,
    max_length: Optional[int] = 2048,
    shuffle_seed: Optional[int] = 42,
    num_proc: Optional[int] = 8,
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
    is_preformatted: Optional[bool] = False,
    include_attention_mask: bool = True,
) -> HFDataset:
    """
    build eagle3 dataset

    Args:
        dataset: HF dataset to process.
        tokenizer: The tokenizer to use for tokenization.
        chat_template: The chat template to use for formatting conversations.
                        This includes the system prompt and user/assistant tokens
                        required to delineate different parts of the conversation
                        for loss mask generation.
        max_length: The maximum length of the tokenized input.
        shuffle_seed: The seed for shuffling the dataset.
        num_proc: The number of processes to use for multiprocessing.
        cache_dir: The directory to use for caching the processed dataset.
        cache_key: The key to use for caching the processed dataset.
        is_preformatted: Whether the dataset contains preformatted text of the conversation
                        (e.g. includes system prompt, user and assistant start and end tokens)
                        and doesn't need to have the chat template applied.
                        Note that the chat_template still needs to be specified to determine
                        the assistant spans for loss mask generation.
                        If True, expects "text" column with ready-to-train text.
                        If False, expects "conversations" column with ShareGPT format.
        include_attention_mask: Whether to include attention_mask in the output.
                               If False, attention_mask can be generated as ones on the inference side.

    Returns:
        The processed HF dataset.
    """
    if chat_template is None:
        raise ValueError("chat_template must be provided for all dataset types")

    assert chat_template in TEMPLATE_REGISTRY.get_all_template_names(), (
        f"Chat template {chat_template} not found in TEMPLATE_REGISTRY, you may need to register it first"
    )

    template: ChatTemplate = TEMPLATE_REGISTRY.get(chat_template)

    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    original_cols = dataset.column_names

    # 按是否已经格式化选择 text 或 conversations 列；额外列转成逐样本 kwargs 传给预处理。
    def preprocess_function(examples):
        if is_preformatted:
            if "text" not in examples:
                raise ValueError(
                    f"Expected 'text' column for is_preformatted=True, but found columns: {list(examples.keys())}"
                )
            processed = preprocess_conversations(
                tokenizer,
                examples["text"],
                template,
                max_length,
                is_preformatted=True,
                include_attention_mask=include_attention_mask,
            )
        else:
            if "conversations" not in examples:
                raise ValueError(
                    f"Expected 'conversations' column for is_preformatted=False, but found columns: {list(examples.keys())}"
                )
            conversations = examples.pop("conversations")
            if "id" in examples:
                examples.pop("id")
            processed = preprocess_conversations(
                tokenizer,
                conversations,
                template,
                max_length,
                is_preformatted=False,
                include_attention_mask=include_attention_mask,
                **examples,
            )

        return processed

    # Process dataset only once
    if cache_dir and cache_key:
        load_from_cache_file = True
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_name = os.path.join(cache_dir, f"{cache_key}.pkl")
        print(f"dataset is cached at {cache_file_name}")
    elif cache_dir is None and cache_key is None:
        load_from_cache_file = False
        cache_file_name = None
        print("dataset is not cached")
    else:
        warnings.warn("cache_dir and cache_key must be provided together to make caching work")

    batch_size = 1000
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size,
        remove_columns=original_cols,
        load_from_cache_file=load_from_cache_file,
        cache_file_name=cache_file_name,
    )

    dataset.set_format(type="torch")
    return dataset


# ==============================
# Offline Eagle3 Dataset
# ==============================
# modified from https://github.com/NickL77/BaldEagle/blob/master/train/modules/data/data.py
# 递归收集给定后缀文件，默认 .ckpt；当前不排序，调用者不能依赖文件系统遍历顺序复现实验。
def list_local_files(path, suffixes=None):
    if suffixes is None:
        suffixes = [".ckpt"]
    datapaths = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapaths.append(file_path)
    datapaths = [f for f in datapaths if any(f.endswith(s) for s in suffixes)]
    return datapaths


# 旧版 Eagle3 checkpoint 特征数据集，提供 PyTorch map-style Dataset 接口。
class OfflineEagle3Dataset(torch.utils.data.Dataset):
    # 保存文件列表、可选样本变换和最大序列长度；构造时不一次加载所有特征。
    def __init__(self, datapath, transform=None, max_len=2048):
        self.datapaths = datapath
        self.transform = transform
        self._epoch = 0
        self.max_len = max_len

    # 读取旧字段 aux_hidden_states/hidden_states，截长并左移 input_ids 与 target，实现下一 token 对齐。
    # 末位 loss_mask 置零，因为移位后的末尾是补零项而没有真实未来目标；该逻辑属于旧格式约定。
    @staticmethod
    def process_data(data, max_len, transform=None):
        new_data = {}
        # Squeeze due to our data generation script adding a batch dimension
        hidden_states = data["aux_hidden_states"].squeeze(0)[:max_len][None, :]
        target = data["hidden_states"].squeeze(0)[:max_len][None, :]

        input_ids = data["input_ids"][:max_len][None, :]
        loss_mask = data["loss_mask"][:max_len][None, :]
        loss_mask[0, -1] = 0

        new_data["attention_mask"] = torch.ones_like(loss_mask, dtype=torch.long)
        new_data["loss_mask"] = loss_mask
        new_data["target"] = padding(target, left=False)
        new_data["hidden_states"] = hidden_states
        new_data["input_ids"] = padding(input_ids, left=False)
        if transform:
            new_data = transform(new_data)
        return new_data

    # 返回文件列表长度，每个文件视作一个样本。
    def __len__(self):
        return len(self.datapaths)

    # 按索引 torch.load 旧特征文件；该旧接口使用 weights_only=False。
    def _open_file(self, index):
        return torch.load(self.datapaths[index], weights_only=False)

    # 读样本失败时记录错误并退回第一个文件，再执行形状转换；因此读错不会自动减少数据集长度。
    def __getitem__(self, index):
        try:
            data = self._open_file(index)
        except Exception as e:
            logger.error("Failed to load %s: %s", self.datapaths[index], e)
            data = self._open_file(0)
        return self.process_data(data, self.max_len, self.transform)

    # 只记录 epoch 值；此类本身不会据此自动打乱文件列表。
    def set_epoch(self, epoch):
        self._epoch = epoch


# 扫描旧 .ckpt 文件并构造旧数据集，不读取新版 manifest 或 dataset.json。
def build_offline_eagle3_dataset(
    hidden_states_path: str,
    max_len: int = 2048,
) -> torch.utils.data.Dataset:
    return OfflineEagle3Dataset(
        list_local_files(hidden_states_path),
        max_len=max_len,
    )


# 合并所有 ID 和监督 mask，仅统计 mask=1 的目标 token 频率。
# 选择输出词表应反映助手回答而非全部提示词；没有 packed mask 的条目按全监督统计。
def _count_token_frequencies(prompts: list) -> Counter:
    """Count token frequencies from tokenized prompts using packed_loss_mask."""

    # Numba 编译的紧凑计数循环：只在监督位置增加对应 ID 的桶，避免逐 token Python 对象操作。
    @numba.njit(cache=True)
    def _histogram(ids, mask, counts):
        for i in range(len(ids)):
            if mask[i] == 1:
                counts[ids[i]] += 1

    all_ids = []
    all_masks = []
    for item in tqdm(prompts, desc="Preparing token data"):
        input_ids = item["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            ids_np = input_ids.squeeze().numpy().astype(np.int64)
        elif isinstance(input_ids, np.ndarray):
            ids_np = input_ids.astype(np.int64)
        else:
            ids_np = np.array(input_ids, dtype=np.int64)

        packed = item.get("packed_loss_mask")
        if packed is not None:
            mask_np = unpack_loss_mask(packed).numpy().astype(np.int64)
        else:
            mask_np = np.ones_like(ids_np)

        all_ids.append(ids_np)
        all_masks.append(mask_np)

    flat_ids = np.concatenate(all_ids)
    flat_masks = np.concatenate(all_masks)

    counts = np.zeros(int(flat_ids.max()) + 1, dtype=np.int64)
    _histogram(flat_ids, flat_masks, counts)

    nonzero = np.nonzero(counts)[0]
    return Counter(dict(zip(nonzero.tolist(), counts[nonzero].tolist())))


# 从已 tokenized 的条目统计频率，再选草稿词表并返回两个编码缓冲。
def generate_vocab_mapping(
    prompts: list,
    target_vocab_size: int,
    draft_vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate vocab mapping (d2t, t2d) from tokenized prompts.

    Args:
        prompts: List of dicts with input_ids and packed_loss_mask
                 (as returned by load_conversation_dataset).
        target_vocab_size: The target vocabulary size.
        draft_vocab_size: The draft vocabulary size.

    Returns:
        (d2t, t2d) tensor tuple for vocab pruning.
    """
    token_dict = _count_token_frequencies(prompts)
    print(f"Found {len(token_dict)} unique tokens from {sum(token_dict.values())} total tokens")

    return process_token_dict_to_mappings(
        token_dict,
        draft_vocab_size,
        target_vocab_size,
    )


# 选最高频 token 后按目标 ID 排序；不足草稿词表大小时用零频 token 补齐。
# 注意实际编码与旧 docstring 的字面描述不同：t2d 是全词表布尔选择掩码，d2t 是 ID 差值。
# 例如选中目标 ID [2,5,9]：d2t=[2,4,7]，草稿列 j 对应的目标 ID 为 j+d2t[j]；
# t2d 在 2、5、9 位置为 True，不能当作 target ID→draft ID 的整数查表。
def process_token_dict_to_mappings(
    token_dict: Counter,
    draft_vocab_size: int,
    target_vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process token_dict to create d2t and t2d mappings, with optional caching.

    Args:
        token_dict: A Counter object mapping token ids to their frequencies.
        draft_vocab_size: The size of the draft vocabulary.
        target_vocab_size: The size of the target vocabulary.

    Returns:
        A tuple containing:
            - d2t: A tensor mapping draft token ids to target token ids.
            - t2d: A tensor mapping target token ids to draft token ids.
    """
    if len(token_dict) < draft_vocab_size:
        existing_tokens = set(token_dict.keys())
        missing_tokens = set(range(draft_vocab_size)) - existing_tokens
        for token in missing_tokens:
            token_dict[token] = 0
            if len(token_dict) >= draft_vocab_size:
                break
        print(
            f"Added {draft_vocab_size - len(existing_tokens)} missing tokens to reach draft vocab size: {draft_vocab_size}"
        )
    else:
        warnings.warn(
            f"Unique tokens ({len(token_dict)}) exceed draft vocab size ({draft_vocab_size}). "
            f"{len(token_dict) - draft_vocab_size} tokens will be dropped from the vocab mapping."
        )
    total_frequency = sum(token_dict.values())
    top_N = token_dict.most_common(draft_vocab_size)
    top_N_frequency_sum = sum(freq for key, freq in top_N)

    if total_frequency == 0:
        print("Warning: Total token frequency is zero. All tokens will have zero ratio.")
        top_N_ratio = 0.0
    else:
        top_N_ratio = top_N_frequency_sum / total_frequency

    print(f"top {draft_vocab_size} token frequency ratio: {top_N_ratio:.2%}")
    used_tokens = sorted(key for key, freq in top_N if key < target_vocab_size)
    used_arr = torch.tensor(used_tokens, dtype=torch.long)

    # 这里保存差值是为了与草稿输出列索引相加后恢复目标 ID，不是直接保存目标 ID。
    # 如果拿 d2t[j] 直接当 tokenizer ID，除第零列外都会产生错映射；配套使用方必须遵守编码。
    d2t = used_arr - torch.arange(len(used_arr))
    t2d = torch.zeros(target_vocab_size, dtype=torch.bool)
    t2d[used_arr] = True

    return d2t, t2d
