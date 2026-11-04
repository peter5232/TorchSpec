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

# 中文导读：在线数据入口，将对话变成 InferenceInput 所需字段；此时尚未运行目标模型。
# 三种输入契约：原始消息先格式化+分词；延迟分词只传格式化文本；预分词数据直接校验 ID/mask。
# loss_mask 表示哪些 token 参与监督，不等于 attention_mask（哪些 token 是有效上下文）。
# 多模态占位符经引擎展开后长度会改变，因此旧 mask 必须丢弃并以实际 token 流重算。

import contextlib
import hashlib
import logging as _logging
import multiprocessing as mp
import os

import numpy as np
import torch
from tqdm import tqdm

from torchspec.data.parse import (
    create_parser,
    has_thinking_content,
    has_unbalanced_thinking_tags,
)
from torchspec.data.preprocessing import _normalize_conversation, preprocess_conversations
from torchspec.data.renderers import RENDERER_REGISTRY
from torchspec.data.template import TEMPLATE_REGISTRY
from torchspec.data.utils import (
    deserialize_packed_loss_mask,
    estimate_row_count,
    extract_media_urls,
    flatten_multimodal_content,
    load_hf_dataset,
    pack_loss_mask,
    serialize_packed_loss_mask,
    unpack_loss_mask,
)
from torchspec.utils.logging import logger
from torchspec.utils.processing import load_tokenizer

_logging.getLogger("transformers_modules").setLevel(_logging.ERROR)

_worker_state = {}

# Columns the pretokenized loader consumes or can re-derive itself. Everything
# else in the row is treated as caller-defined provenance metadata.
_PRETOKENIZED_CONTRACT_COLUMNS = frozenset(
    {
        "data_id",
        "id",
        "input_ids",
        "loss_mask",
        "packed_loss_mask",
        "seq_len",
        "loss_tokens",
    }
)


# 依据列 schema 同时识别 input_ids 和 mask；只出现其中之一就报错，避免把不完整预分词数据当原始对话。
def _is_pretokenized_dataset(dataset) -> bool:
    """Return whether an HF dataset exposes the offline token/mask contract."""
    columns = set(getattr(dataset, "column_names", None) or ())
    has_ids = "input_ids" in columns
    has_mask = bool(columns & {"packed_loss_mask", "loss_mask"})
    if has_ids != has_mask:
        raise ValueError(
            "Pretokenized datasets must provide input_ids together with "
            "packed_loss_mask or loss_mask"
        )
    return has_ids


# 保留调用方自定义的标量溯源字段，过滤张量/列表等大对象，防止它们长期留在控制器内存。
def _pretokenized_metadata(sample) -> dict:
    """Carry through the row's own provenance columns untouched.

    Offline tokenization pipelines record their own bookkeeping (source corpus,
    supervision policy, row fingerprints). The loader does not interpret any of
    it, so pass every non-contract column through instead of enumerating names.
    Non-scalar values are skipped so a stray tensor column cannot be pinned in
    memory for the lifetime of the dataset.
    """
    return {
        key: value
        for key, value in sample.items()
        if key not in _PRETOKENIZED_CONTRACT_COLUMNS
        and value is not None
        and isinstance(value, (str, int, float, bool))
    }


# 逐条检查 ID 整数列表、长度、监督段总长、非空监督和 ID 唯一性，不再套聊天模板。
# 例如 5 个 ID 的 packed mask 必须恰好覆盖 5 个位置；若同时带完整 mask，两种表示必须逐项一致。
# 超长直接拒绝而非截断，保持离线生产方定义的监督范围不被悄悄改变。
def _load_pretokenized_dataset(dataset, *, max_length: int, total=None):
    """Load already-rendered rows without applying a chat template again."""
    prompts = []
    seen_ids = set()
    for idx, sample in enumerate(tqdm(dataset, desc="Loading pretokenized dataset", total=total)):
        input_ids = sample.get("input_ids")
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if not isinstance(input_ids, list) or any(not isinstance(v, int) for v in input_ids):
            raise ValueError(f"Pretokenized sample {idx} has invalid input_ids")
        if not input_ids:
            raise ValueError(f"Pretokenized sample {idx} has empty input_ids")
        # Truncating here would drop supervised tokens the producer counted on,
        # so an over-length row is a corpus/config mismatch, not something to fix
        # silently.
        if len(input_ids) > max_length:
            raise ValueError(
                f"Pretokenized sample {idx} has seq_len={len(input_ids)} above "
                f"max_seq_length={max_length}; retokenize the corpus at this limit "
                f"or raise max_seq_length instead of truncating"
            )
        stored_seq_len = sample.get("seq_len")
        if stored_seq_len is not None and int(stored_seq_len) != len(input_ids):
            raise ValueError(f"Pretokenized sample {idx} seq_len metadata does not match input_ids")

        packed_loss_mask = sample.get("packed_loss_mask")
        explicit_loss_mask = sample.get("loss_mask")
        if hasattr(explicit_loss_mask, "tolist"):
            explicit_loss_mask = explicit_loss_mask.tolist()
        if packed_loss_mask is None:
            if not isinstance(explicit_loss_mask, list):
                raise ValueError(f"Pretokenized sample {idx} is missing its loss mask")
            packed_loss_mask = serialize_packed_loss_mask(
                pack_loss_mask(torch.tensor(explicit_loss_mask, dtype=torch.long))
            )
        # 段长的总和必须等于 ID 长度。比如 6 个 ID 配 "2,3,1"：前 2 个不监督、接着 3 个监督、末 1 个不监督。
        # 校验计数及两种 mask 编码的一致性，避免文件看似可加载但 token 目标已经错位。
        segments = deserialize_packed_loss_mask(packed_loss_mask)
        if any(length < 0 for length in segments) or sum(segments) != len(input_ids):
            raise ValueError(
                f"Pretokenized sample {idx} packed_loss_mask length does not match input_ids"
            )
        packed_loss_tokens = sum(segments[1::2])
        if packed_loss_tokens <= 0:
            raise ValueError(f"Pretokenized sample {idx} has no supervised tokens")
        # Both mask encodings may be present; disagreement means the producer
        # wrote them from different states and neither can be trusted.
        if explicit_loss_mask is not None:
            if (
                not isinstance(explicit_loss_mask, list)
                or explicit_loss_mask != unpack_loss_mask(packed_loss_mask).tolist()
            ):
                raise ValueError(
                    f"Pretokenized sample {idx} explicit loss_mask disagrees with packed mask"
                )
        stored_loss_tokens = sample.get("loss_tokens")
        if stored_loss_tokens is not None and int(stored_loss_tokens) != packed_loss_tokens:
            raise ValueError(
                f"Pretokenized sample {idx} loss_tokens metadata disagrees with packed mask"
            )

        data_id = str(sample.get("data_id") or sample.get("id") or f"sample_{idx}")
        if data_id in seen_ids:
            raise ValueError(f"Duplicate pretokenized data_id: {data_id}")
        seen_ids.add(data_id)
        prompts.append(
            {
                "data_id": data_id,
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "packed_loss_mask": packed_loss_mask,
                "formatted_prompt": None,
                "multimodal_inputs": None,
                "metadata": _pretokenized_metadata(sample),
            }
        )
    return prompts


# 每个 CPU worker 只加载一次 tokenizer 和 parser/renderer，避免每条样本重复初始化。
def _init_tokenize_worker(
    tokenizer_path,
    trust_remote_code,
    chat_template_name,
    last_turn_loss_only=False,
    min_loss_tokens=0,
    renderer_name=None,
):
    """Initializer for each worker process — loads tokenizer once."""
    _logging.getLogger("transformers_modules").setLevel(_logging.ERROR)
    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
    _worker_state["tokenizer"] = tokenizer
    _worker_state["renderer"] = (
        RENDERER_REGISTRY.create(renderer_name, tokenizer) if renderer_name else None
    )
    _worker_state["template"] = (
        TEMPLATE_REGISTRY.get(chat_template_name) if not renderer_name else None
    )
    _worker_state["preprocess"] = preprocess_conversations if not renderer_name else None
    _worker_state["last_turn_loss_only"] = last_turn_loss_only
    _worker_state["min_loss_tokens"] = min_loss_tokens


# auto 模式在原始消息中检测真实思考内容，再决定是否只监督最后一轮；显式布尔值直接遵从。
def _resolve_last_turn_loss_only(messages):
    ltlo = _worker_state.get("last_turn_loss_only", False)
    if ltlo == "auto":
        return has_thinking_content(messages)
    return bool(ltlo)


# 把 auto 的决定编码为 has_thinking 元数据；后续媒体展开导致 mask 重建时仍能复用同一监督策略。
def _auto_decision_metadata(last_turn_only: bool) -> dict:
    """Record a resolved "auto" decision so training can reapply it.

    A multimodal row's mask is discarded before training (it was rendered
    against unexpanded media placeholders), so the decision made here must
    travel as metadata or the training-time matcher has nothing to consult.
    """
    if _worker_state.get("last_turn_loss_only", False) != "auto":
        return {}
    return {"has_thinking": last_turn_only}


# 处理一条消息列表，产出 ID、紧凑 mask 和可选格式化文本；监督 token 不足则过滤。
# 跨 CPU 进程返回 NumPy 数组而不是每 token 一个 Python int 或 torch 共享内存句柄，降低 IPC 对象开销。
def _tokenize_single(args):
    """Worker function — tokenize one sample."""
    messages, tools, generation_config, max_length, train_with_decode = args
    last_turn_only = _resolve_last_turn_loss_only(messages)
    auto_metadata = _auto_decision_metadata(last_turn_only)
    renderer = _worker_state.get("renderer")
    if renderer is not None:
        if train_with_decode:
            raise ValueError("Registered dataset renderers do not support train_with_decode=True")
        input_ids, loss_mask = renderer.render(
            messages,
            tools,
            max_seq_length=max_length,
            last_turn_only=last_turn_only,
            generation_config=generation_config,
        )
        # A renderer owns its own supervision, so a sample with no supervised
        # token is unusable rather than merely short.
        min_loss_tokens = max(1, _worker_state.get("min_loss_tokens", 0))
        if not input_ids or sum(loss_mask) < min_loss_tokens:
            return None
        packed_loss_mask = serialize_packed_loss_mask(
            pack_loss_mask(torch.tensor(loss_mask, dtype=torch.long))
        )
        return {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "packed_loss_mask": packed_loss_mask,
            "formatted_prompt": None,
            **auto_metadata,
        }

    processed = _worker_state["preprocess"](
        _worker_state["tokenizer"],
        [messages],
        _worker_state["template"],
        max_length=max_length,
        is_preformatted=False,
        include_attention_mask=False,
        use_packed_loss_mask=True,
        add_generation_prompt=train_with_decode,
        return_formatted_text=True,
        last_turn_loss_only=last_turn_only,
        min_loss_tokens=_worker_state.get("min_loss_tokens", 0),
    )
    if not processed["input_ids"]:
        return None
    # Tensors pickle by /dev/shm handle (one fd per sample, exhausted at 1M) and
    # lists rebuild every token as a PyLong; numpy crosses as bytes, zero-copy back.
    return {
        "input_ids": processed["input_ids"][0].numpy(),
        "packed_loss_mask": processed["packed_loss_mask"][0],
        "formatted_prompt": processed["formatted_text"][0],
        **auto_metadata,
    }


# 延迟分词仍需加载 tokenizer 供聊天模板渲染，只是不在这里对整条文本编码；原函数总说明中的 no tokenizer 并非当前实际行为。
def _init_format_worker(
    tokenizer_path, trust_remote_code, chat_template_name, last_turn_loss_only=False
):
    _logging.getLogger("transformers_modules").setLevel(_logging.ERROR)
    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
    _worker_state["template"] = TEMPLATE_REGISTRY.get(chat_template_name)
    _worker_state["parser"] = create_parser(tokenizer, _worker_state["template"])
    _worker_state["last_turn_loss_only"] = last_turn_loss_only


# 只规范化消息和渲染文本，不扩展媒体占位符；引擎之后决定实际 ID 与视觉 token 数量。
def _format_single(args):
    """
    Worker function — format only, skip tokenization.
    """
    messages, _, _, _, train_with_decode = args
    messages = _normalize_conversation(messages)

    result = _auto_decision_metadata(_resolve_last_turn_loss_only(messages))

    parser = _worker_state["parser"]
    formatted = parser.format(
        messages, add_generation_prompt=train_with_decode, expand_media_tokens=False
    )
    if not formatted:
        return None
    result["formatted_prompt"] = formatted
    return result


# 动态 mask 模式下移除多模态的旧 mask；若旧缓存缺失 auto 决策元数据则报错，避免猜测监督范围。
def _drop_stale_multimodal_masks(
    prompts, dynamic_loss_mask: bool, last_turn_loss_only=False
) -> None:
    """Remove masks rendered before the engine expands media placeholders."""
    if not dynamic_loss_mask:
        return
    for prompt in prompts:
        if prompt.get("multimodal_inputs") is None:
            continue
        dropped = prompt.pop("packed_loss_mask", None)
        if (
            dropped is not None
            and last_turn_loss_only == "auto"
            and "has_thinking" not in (prompt.get("metadata") or {})
        ):
            # Only a cache written before auto decisions were recorded can get
            # here; the source messages are gone, so the decision that the
            # training-time matcher needs cannot be recovered.
            raise ValueError(
                f"Multimodal sample {prompt.get('data_id')} has no recorded "
                "last_turn_loss_only='auto' decision, so its dropped mask cannot be "
                "recomputed at training time. Delete the tokenization cache and re-run, "
                "or set dataset.last_turn_loss_only to an explicit true/false."
            )


# 加载数据、优先检测预分词契约，再选择 renderer 或模板并构建缓存 key。
# 原始对话分两遍处理：先提取消息/媒体，再并行格式化或分词，最后还原成带 data_id 的条目。
# 多进程池在收集大语料前创建，减少 fork 后写时复制的内存压力；缓存键包含渲染版本和监督选项。
def load_conversation_dataset(args):
    """Load conversation dataset and optionally tokenize for training.

    When defer_tokenization=True, only applies the chat template to produce
    formatted text — no tokenizer is loaded and no input_ids/loss_mask are
    generated. The inference engine handles tokenization and media token
    expansion; loss mask is computed at training time from the engine's
    actual input_ids.

    When defer_tokenization=False (default), fully tokenizes and produces
    input_ids + packed_loss_mask for the input_ids engine path. Tokenization
    goes through either the ``chat_template`` registry or, if ``renderer`` is
    set, a registered :class:`ConversationRenderer` that owns both prompt
    construction and loss-mask derivation.

    A dataset that already carries ``input_ids`` plus a loss mask is loaded
    directly and is never rendered or truncated again. Detection needs a typed
    schema, so this covers Parquet/Arrow paths and Hub repos; a JSONL file is
    always treated as raw conversations.

    Returns list of dicts. Fields depend on mode:
        defer_tokenization=True:  data_id, formatted_prompt, multimodal_inputs, metadata
        defer_tokenization=False: data_id, input_ids, packed_loss_mask, formatted_prompt, multimodal_inputs, metadata
    """
    prompt_key = getattr(args, "prompt_key", "text")
    chat_template_name = getattr(args, "chat_template", None)
    renderer_name = getattr(args, "renderer", None)
    max_length = args.max_seq_length
    defer_tokenization = getattr(args, "defer_tokenization", False)

    logger.info(f"Max sequence length allowed for training: {max_length}")

    if renderer_name and defer_tokenization:
        raise ValueError("Registered dataset renderers require defer_tokenization=False")
    if renderer_name and renderer_name not in RENDERER_REGISTRY.get_all_renderer_names():
        available = ", ".join(RENDERER_REGISTRY.get_all_renderer_names()) or "<none>"
        raise ValueError(f"Unknown dataset renderer {renderer_name!r}; available: {available}")

    hf_dataset = load_hf_dataset(args.train_data_path)

    # Detection comes before the renderer/template requirement below: a
    # pretokenized corpus is never rendered, so a config that sets neither is
    # complete rather than under-specified. The checks above are decidable from
    # the config alone and stay there.
    # schema 提前识别出 ID+mask 时直接走校验并 return，不要求 renderer/chat_template。
    # 否则对已格式化过的 token 再套一次模板，会重复角色头并改变所有监督位置。
    if _is_pretokenized_dataset(hf_dataset):
        if defer_tokenization:
            raise ValueError("Pretokenized datasets require defer_tokenization=False")
        logger.info("Detected offline pretokenized input_ids/loss-mask dataset")
        prompts = _load_pretokenized_dataset(
            hf_dataset,
            max_length=max_length,
            total=estimate_row_count(args.train_data_path),
        )
        logger.info(f"Loaded {len(prompts)} pretokenized samples without re-rendering")
        return prompts

    if not renderer_name and not chat_template_name:
        raise ValueError("Either renderer or chat_template must be set for dataset tokenization")

    custom_template = TEMPLATE_REGISTRY.get(chat_template_name) if not renderer_name else None

    dataset_name = os.path.basename(args.train_data_path)
    file_stat = ""
    if os.path.isfile(args.train_data_path):
        st = os.stat(args.train_data_path)
        file_stat = f"-{st.st_size}-{st.st_mtime}"
    last_turn_loss_only_flag = getattr(args, "last_turn_loss_only", False)
    train_with_decode = getattr(args, "train_with_decode", False)
    min_loss_tokens_val = getattr(args, "min_loss_tokens", 0)
    renderer_version = RENDERER_REGISTRY.cache_version(renderer_name) if renderer_name else None
    # 缓存 key 同时包含语料位置、模型、最大长度、模板/renderer 版本和监督策略。
    # 这些因素改变时通常应重新生成 ID/mask；本地单文件还纳入大小/修改时间，但这不是完整内容哈希。
    cache_params = (
        f"{dataset_name}-{args.train_data_path}{file_stat}-{args.target_model_path}"
        f"-{max_length}-template={chat_template_name}-renderer={renderer_name}"
        f"-renderer-version={renderer_version}-ltlo={last_turn_loss_only_flag}"
        f"-defer={defer_tokenization}-decode={train_with_decode}"
        f"-mlt={min_loss_tokens_val}"
    )
    cache_key = hashlib.md5(cache_params.encode()).hexdigest()
    cache_dir = os.path.join(getattr(args, "cache_dir", "./cache"), "tokenized_dataset")
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

    dynamic_loss_mask = getattr(args, "dynamic_loss_mask", False)

    # 命中缓存节省的是格式化/分词成本；它不是目标 hidden states 缓存。
    # 仍需处理媒体展开后的旧 mask 问题，不能因为缓存存在就跳过语义检查。
    if os.path.exists(cache_path):
        logger.info(f"Loading dataset from cache: {cache_path}")
        prompts = torch.load(cache_path, weights_only=False)
        _drop_stale_multimodal_masks(
            prompts,
            dynamic_loss_mask=dynamic_loss_mask,
            last_turn_loss_only=last_turn_loss_only_flag,
        )
        logger.info(f"Loaded {len(prompts)} cached samples")
        return prompts

    mode_label = "Formatting" if defer_tokenization else "Tokenizing"
    logger.info(f"{mode_label} dataset (cache will be saved to {cache_path})")

    total_estimate = estimate_row_count(args.train_data_path)
    num_proc = getattr(args, "num_proc", 64)

    last_turn_loss_only = getattr(args, "last_turn_loss_only", False)
    if defer_tokenization:
        # 延迟模式仍加载用于聊天模板的 tokenizer；只是把整段 tokenization 留给引擎。
        # 因此这里省掉的是本地编码与提前定义 token 位置，不是所有 tokenizer 初始化成本。
        worker_init = _init_format_worker
        worker_initargs = (args.target_model_path, True, chat_template_name, last_turn_loss_only)
        worker_fn = _format_single
        desc = "Formatting dataset"
    else:
        if last_turn_loss_only:
            logger.info(
                f"last_turn_loss_only={last_turn_loss_only}: "
                "loss mask will only cover the last assistant turn"
            )
        min_loss_tokens = getattr(args, "min_loss_tokens", 0)
        worker_init = _init_tokenize_worker
        worker_initargs = (
            args.target_model_path,
            True,
            chat_template_name,
            last_turn_loss_only,
            min_loss_tokens,
            renderer_name,
        )
        worker_fn = _tokenize_single
        desc = "Tokenizing dataset"

    # Fork before pass 1 fills raw_samples: CPython refcounts dirty every page a
    # worker reads, so inheriting the loaded corpus costs 4.5x to copy-on-write.
    if num_proc <= 1:
        worker_init(*worker_initargs)
        pool_ctx = contextlib.nullcontext()
    else:
        # 在收集 raw_samples 前创建池，使支持 fork 的环境不把已装入的大语料一起继承给 worker。
        # Python 引用计数更新会弄脏共享页；“fork 是写时复制”并不保证读遍大对象时没有内存膨胀。
        pool_ctx = mp.Pool(num_proc, initializer=worker_init, initargs=worker_initargs)

    with pool_ctx as pool:
        # Pass 1: collect and normalize raw samples (fast I/O, no tokenization)
        raw_samples = []
        for idx, sample in enumerate(
            tqdm(hf_dataset, desc="Loading samples", total=total_estimate)
        ):
            raw_prompt = sample.get(prompt_key, "")

            if not isinstance(raw_prompt, list):
                raise ValueError(
                    f"Expected conversation format (list of messages) for sample {idx}, got {type(raw_prompt)}"
                )

            messages = _normalize_conversation(raw_prompt)
            # 必须先提取结构化媒体引用，再将 content 展平为占位符文本；顺序颠倒会丢失图片地址。
            # 保留媒体与 tokenized 文本两种表示，让推理引擎后续执行自己的视觉预处理。
            multimodal_inputs = extract_media_urls(messages)
            if custom_template is not None:
                flatten_multimodal_content(messages, custom_template.image_placeholder)
            data_id = sample.get("id") or sample.get("data_id") or f"sample_{idx}"
            raw_samples.append(
                (
                    data_id,
                    messages,
                    multimodal_inputs,
                    sample.get("tools"),
                    sample.get("generation_config"),
                )
            )

        logger.info(
            f"Loaded {len(raw_samples)} samples, {mode_label.lower()} with {num_proc} workers..."
        )

        # Pass 2: process in parallel
        # 提交 CPU 预处理所需的消息、工具、样本生成配置和长度；媒体引用保留在主进程条目中。
        # 这里是 CPU 数据并行，不是 GPU DP/SP，也不会触发模型梯度通信。
        work_items = [
            (messages, tools, generation_config, max_length, train_with_decode)
            for _, messages, _, tools, generation_config in raw_samples
        ]

        if pool is None:
            results = [worker_fn(item) for item in tqdm(work_items, desc=desc)]
        else:
            results = list(
                tqdm(
                    # imap 保持输入次序，使下方 zip(raw_samples, results) 仍能按位置配回原 data_id。
                    # 若改成无序返回，就必须显式带回 ID，否则结果形状可能正常但对应了另一条样本。
                    pool.imap(worker_fn, work_items, chunksize=64),
                    total=len(work_items),
                    desc=desc,
                )
            )

    # Collect results
    prompts = []
    skipped = 0
    unbalanced_think = 0
    for (data_id, _, multimodal_inputs, _, _), result in zip(raw_samples, results):
        # 被 worker 判为空/监督不足的原始样本在训练开始前就过滤。
        # 这与训练 rank 已经同步取到样本后再丢弃不同：后者可能破坏 collective 次数，需要 neutralize。
        if result is None:
            skipped += 1
            continue
        formatted_prompt = result.get("formatted_prompt")
        if (
            not defer_tokenization
            and isinstance(formatted_prompt, str)
            and has_unbalanced_thinking_tags(formatted_prompt)
        ):
            unbalanced_think += 1
        metadata = {}
        if "has_thinking" in result:
            metadata["has_thinking"] = result["has_thinking"]

        entry = {
            "data_id": data_id,
            "metadata": metadata,
            "multimodal_inputs": multimodal_inputs,
            "formatted_prompt": formatted_prompt,
        }

        if not defer_tokenization:
            # worker 回传的 NumPy 字节缓冲可直接包装成 tensor，减少逐 token Python 对象重建。
            # 纯文本可沿用原 mask；图片展开数量由引擎决定，不能沿用预展开时的序列位置。
            entry["input_ids"] = torch.from_numpy(result["input_ids"])
            if multimodal_inputs is None:
                entry["packed_loss_mask"] = result["packed_loss_mask"]
            elif not dynamic_loss_mask:
                raise ValueError(
                    "Multimodal samples require dynamic_loss_mask=True because "
                    "the inference engine expands media placeholders after rendering"
                )

        prompts.append(entry)

    _drop_stale_multimodal_masks(
        prompts,
        dynamic_loss_mask=dynamic_loss_mask,
        last_turn_loss_only=last_turn_loss_only_flag,
    )

    if skipped:
        logger.warning(f"Skipped {skipped} samples (empty source or zero loss mask)")

    if unbalanced_think:
        logger.warning(
            f"{unbalanced_think}/{len(prompts)} samples have unbalanced <think>/</think> tags "
            "after chat-template formatting. This usually means the data was generated by a "
            "thinking model that emits the opening <think> in the generation prompt, so the saved "
            "assistant content lacks it and re-tokenization produces malformed turns. Restore the "
            "opening <think> in the data (or verify the chat template) before training."
        )

    os.makedirs(cache_dir, exist_ok=True)
    # 缓存只保存预处理条目，下一次训练仍需推理生产 teacher 特征（除非选择 offline 回放）。
    # 注意该写入不是临时文件+原子替换；不要据此假定并发作业共享缓存写入具备事务保证。
    torch.save(prompts, cache_path)
    logger.info(f"Saved {len(prompts)} samples to cache: {cache_path}")

    return prompts
