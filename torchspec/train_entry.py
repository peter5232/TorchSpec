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

# 中文阅读入口：这里是训练作业的装配层，不是模型 forward。
# 阅读顺序：parse_config → _resolve_batch_size → train_async_no_generation → controller/loop.py。
# 对熟悉单机 PyTorch 的读者：原来的 dataset/model/optimizer 被放进不同进程；
# 这个进程只建立资源、启动远程工作并等待依赖。Ray 的 .remote() 返回未来结果，
# ray.get 才等待其完成；因此代码顺序不等于 GPU 计算的实际完成顺序。

"""Training entry point for Eagle3 speculative decoding."""

import argparse
import os
import sys
import time

# Fix PyTorch 2.9+ TorchInductor GEMM backend regression: without this,
# FlexAttention backward pass hits NoValidChoicesError and training is 3x slower.
# See Phase E in docs/inference/dflash/training_results.md.
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,TRITON")
from collections import namedtuple
from contextlib import contextmanager
from typing import Any, Generator

import ray
import torch
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from torchspec import AutoDraftModelConfig
from torchspec.config.train_config import config_to_flat_args, load_config
from torchspec.config.utils import generate_draft_model_config
from torchspec.controller import (
    AsyncTrainingController,
    auto_calculate_training_steps,
    build_mooncake_config,
    run_training_loop,
    setup_async_training_with_engines,
)
from torchspec.inference.factory import prepare_inference_engines
from torchspec.ray.placement_group import (
    allocate_train_group,
    create_placement_groups,
)
from torchspec.training.trainer_actor import TrainerActor
from torchspec.transfer.mooncake.utils import launch_mooncake_master
from torchspec.utils.env import get_torchspec_env_vars
from torchspec.utils.logging import init_tracking, logger

_Phase = namedtuple("_Phase", ["name", "duration", "is_async", "blocked"])


# 初始化阶段计时器：区分异步任务从提交到完成的总时长与主进程真正等待的时长。
# 异步任务可以重叠，所以各阶段总时长之和不应当被当成整个初始化的耗时。
class _InitTimer:
    """Lightweight segmented timer for initialization phases."""

    # 记录总起点；已完成阶段和仍在运行的异步阶段分开保存，避免把尚未完成的工作计入报告。
    def __init__(self) -> None:
        self._t0 = time.time()
        self._phases: list[_Phase] = []
        self._pending: dict[str, float] = {}

    # 给同步代码块包一层计时上下文；yield 期间运行调用者的代码，之后记录阶段耗时。
    @contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        """Time a synchronous phase."""
        start = time.time()
        yield
        self._phases.append(_Phase(name, time.time() - start, is_async=False, blocked=0.0))

    # 在提交远程任务附近记录时间，不等待结果；同名阶段的起点会被更新。
    def begin_async(self, name: str) -> None:
        """Mark the start of an async operation (e.g., ray.remote dispatch)."""
        self._pending[name] = time.time()

    # 等待 refs 的实际结果，并返回 ray.get 原样返回的值。
    # 例如任务共运行 10 秒，主进程在第 8 秒才来等待：total≈10 秒，blocked≈2 秒。
    def wait(self, name: str, refs) -> Any:
        """Wrap ray.get for async phases. Returns the result."""
        if name not in self._pending:
            raise ValueError(f"No async phase '{name}' was started via begin_async()")
        t_before = time.time()
        result = ray.get(refs)
        t_after = time.time()
        dispatch_time = self._pending.pop(name)
        total = t_after - dispatch_time
        blocked = t_after - t_before
        self._phases.append(_Phase(name, total, is_async=True, blocked=blocked))
        return result

    # 打印各阶段的总耗时和等待耗时，帮助区分模型加载本身慢与主进程被依赖阻塞。
    def log_summary(self) -> None:
        total = time.time() - self._t0
        lines = ["Initialization timing:"]
        for p in self._phases:
            suffix = f"  (blocked {p.blocked:.2f}s)" if p.is_async else ""
            lines.append(f"  {p.name:<48s} {p.duration:>8.2f}s{suffix}")
        lines.append(f"  {'─' * 57}")
        lines.append(f"  {'Total':<48s} {total:>8.2f}s")
        logger.info("\n".join(lines))


# 将 YAML 和 CLI 覆盖项合成结构化配置，再转为历史调用接口使用的扁平 args。
# 这里还派生并行规模和 batch 大小；offline 模式会补有限样本池，避免从磁盘过量灌入内存。
# --print-config-only 在创建训练进程前退出，可先用它核对参数最终值。
def parse_config():
    """Parse YAML config and convert to flat args.

    Supports configs with sections matching the Config dataclass:
    model, dataset, training, debug, inference, logging, mooncake, decode.

    The config is flattened via config_to_flat_args(), with prefixed sections:
    mooncake_*, sglang_*, offline_*, decode_*.
    """

    parser = argparse.ArgumentParser(description="Eagle3 speculative decoding training")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--print-config-only", action="store_true", help="Print resolved config and exit"
    )

    args, unknown = parser.parse_known_args()

    config = load_config(
        config_path=args.config, cli_args=unknown if unknown else None, save_snapshot=True
    )

    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(config))

    if args.print_config_only:
        sys.exit(0)

    flat_args = config_to_flat_args(config)

    flat_args.rank = 0
    flat_args.world_size = flat_args.training_num_nodes * flat_args.training_num_gpus_per_node

    defaults = {
        "colocate": False,
        "dp_size": None,
        "save_debug_train_data": None,
    }
    for key, value in defaults.items():
        if not hasattr(flat_args, key) or getattr(flat_args, key) is None:
            setattr(flat_args, key, value)

    _resolve_batch_size(flat_args)
    _validate_usp_args(flat_args)

    if (
        getattr(flat_args, "inference_engine_type", None) == "offline"
        and getattr(flat_args, "max_sample_pool_size", 0) <= 0
    ):
        flat_args.max_sample_pool_size = max(
            flat_args.global_batch_size,
            getattr(flat_args, "inference_batch_size", 1)
            * getattr(flat_args, "offline_num_engines", 1)
            * 2,
        )
        logger.info(
            "Offline replay set max_sample_pool_size=%d for bounded Mooncake staging",
            flat_args.max_sample_pool_size,
        )

    return flat_args


# decode 训练若启用投机算法却没有草稿权重路径，先从训练组导出初始草稿供引擎加载。
# 该操作可能涉及所有训练 rank 的参数聚合，不能理解为任意时刻单进程保存一个 state_dict。
def _maybe_create_scratch_draft(args, train_group):
    """Auto-create scratch draft checkpoint for inference engine if not provided."""
    if (
        getattr(args, "train_with_decode", False)
        and getattr(args, "decode_speculative_algorithm", None)
        and getattr(args, "decode_speculative_draft_model_path", None) is None
    ):
        scratch_dir = os.path.join(getattr(args, "output_dir", "./outputs"), "scratch_draft_model")
        os.makedirs(scratch_dir, exist_ok=True)
        logger.info(f"Auto-creating scratch draft checkpoint at {scratch_dir}")
        train_group.save_draft_model_for_serving(scratch_dir)
        args.decode_speculative_draft_model_path = scratch_dir
        logger.info(f"Set decode_speculative_draft_model_path = {scratch_dir}")


# 把物理进程数转换为独立数据副本数，再计算一次 optimizer.step 实际消费的样本数。
# 普通路径：global_batch = micro_batch × DP × accumulation；例如 2×4×3=24。
# USP 路径中一个序列由 SP 个进程共同处理，所以 DP=world_size/SP，
# per_dp_rank_batch_size 固定为 1，不能把同一序列在 SP rank 上的副本重复计为新样本。
def _resolve_batch_size(args):
    """Derive dp_size, per_dp_rank_batch_size, dispatch_batch_size, and global_batch_size."""
    # 这一处是 batch 推导函数内部的总进程数；例如两节点各 4 卡得到 world_size=8。
    # DP=8 表示八份独立样本流；若 SP=4，则只有 DP=2 份独立样本流。
    world_size = args.training_num_nodes * args.training_num_gpus_per_node
    if getattr(args, "attention_backend", None) == "usp":
        sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
        if sp_size <= 0:
            raise ValueError(f"USP requires positive sp_size, got {sp_size}")
        if world_size % sp_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by USP sp_size ({sp_size})"
            )
        dp_size = getattr(args, "dp_size", None) or (world_size // sp_size)
        if dp_size * sp_size != world_size:
            raise ValueError(
                f"dp_size ({dp_size}) * sp_size ({sp_size}) must equal world_size ({world_size})"
            )
        args.dp_size = dp_size
        args.sp_size = sp_size
        # 四个 SP worker 各持一条序列的不同片段，整体仍算一条样本。
        # 例如 world=8、SP=4、累积=3，此处 global_batch 为 1×2×3=6，不能乘成 24。
        args.per_dp_rank_batch_size = 1
    else:
        dp_size = getattr(args, "dp_size", None) or world_size
        args.dp_size = dp_size
        sp_size = getattr(args, "sp_size", None)
        if sp_size is not None and sp_size != 1:
            raise NotImplementedError(
                f"Sequence parallel is not yet supported (got sp_size={sp_size})"
            )
        sp_size = sp_size or 1
        args.per_dp_rank_batch_size = args.micro_batch_size * sp_size

    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    args.global_batch_size = args.per_dp_rank_batch_size * dp_size * accumulation_steps


# 在分配 GPU 前拒绝尚未支持的组合：当前 USP 仅允许 sgl、REPLICATE、micro_batch=1。
# 这些是当前实现边界，不是序列并行在理论上必须具备的限制。
def _validate_usp_args(args) -> None:
    if getattr(args, "attention_backend", None) != "usp":
        return

    sp_size = getattr(args, "sp_size", None)
    if sp_size is None:
        sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
    if sp_size <= 1:
        raise NotImplementedError(f"USP requires sp_size > 1, got {sp_size}")

    inference_engine_type = getattr(args, "inference_engine_type", "sgl")
    if inference_engine_type != "sgl":
        raise ValueError(
            f"USP currently only supports inference_engine_type=sgl, got {inference_engine_type}"
        )

    fsdp_strategy = getattr(args, "fsdp_strategy", "REPLICATE").upper()
    if fsdp_strategy != "REPLICATE":
        raise NotImplementedError(
            f"USP currently only supports fsdp_strategy=REPLICATE, got {fsdp_strategy}"
        )

    micro_batch_size = getattr(args, "micro_batch_size", 1)
    if micro_batch_size != 1:
        raise NotImplementedError(
            f"USP currently only supports micro_batch_size=1, got {micro_batch_size}"
        )


# 优先读取明确提供的草稿配置，否则从目标模型结构派生配置；返回配置对象而非模型权重。
def _get_draft_model_config(args):
    """Resolve draft model config from args or auto-generate from target model."""

    draft_config_path = getattr(args, "draft_model_config", None)
    if draft_config_path is not None:
        return AutoDraftModelConfig.from_file(draft_config_path)

    config_dict = generate_draft_model_config(
        target_model_path=args.target_model_path,
        cache_dir=getattr(args, "model_download_dir", None),
    )
    return AutoDraftModelConfig.from_dict(config_dict)


# 检查 DFlash 系列所需的后端、block 大小、监督 token 数和目标层选择。
# 至少 2×block_size 个监督 token 是该入口的校验条件；DFlash2 还要求两处配置严格一致，
# 否则采集的特征层数或预测块几何与草稿网络期待的形状不同。
def _validate_and_configure_dflash(args, draft_model_config) -> None:
    """Validate DFlash-specific config and auto-set aux layer IDs.

    Called before dataset loading to fail fast on misconfigurations.
    """
    from torchspec.models.draft.dflash import DFlashConfig
    from torchspec.models.draft.dflash2 import DFlash2Config
    from torchspec.models.draft.dspark import DSparkConfig

    if not isinstance(draft_model_config, DFlashConfig):
        return

    is_dflash2 = isinstance(draft_model_config, DFlash2Config)
    is_dspark = isinstance(draft_model_config, DSparkConfig)
    algo = "DFlash2" if is_dflash2 else "DSpark" if is_dspark else "DFlash"

    if is_dflash2 and getattr(args, "attention_backend", None) == "usp":
        raise ValueError("DFlash2 does not support training.attention_backend=usp.")

    engine_type = getattr(args, "inference_engine_type", "hf")
    if engine_type not in ("vllm", "sgl", "trtllm", "offline"):
        raise NotImplementedError(
            f"{algo} supports inference_engine_type in "
            f"('vllm', 'sgl', 'trtllm', 'offline'), got '{engine_type}'."
        )
    if getattr(args, "defer_tokenization", False):
        raise NotImplementedError(f"{algo} does not support defer_tokenization=True.")
    block_size = getattr(args, "dflash_block_size", 16)
    if is_dflash2 and block_size != draft_model_config.block_size:
        raise ValueError(
            "training.dflash_block_size must match dflash_config.block_size "
            f"({block_size} != {draft_model_config.block_size})."
        )
    num_target_layers = getattr(args, "dflash_num_target_layers", 5)
    if is_dflash2 and num_target_layers != draft_model_config.num_target_layers:
        raise ValueError(
            "training.dflash_num_target_layers must match the number of "
            f"dflash_config.target_layer_ids ({num_target_layers} != "
            f"{draft_model_config.num_target_layers})."
        )
    min_loss = getattr(args, "min_loss_tokens", 0)
    if min_loss < 2 * block_size:
        raise ValueError(
            f"{algo} requires dataset.min_loss_tokens >= 2 * training.dflash_block_size "
            f"({min_loss} < {2 * block_size}). Set dataset.min_loss_tokens={2 * block_size}."
        )

    target_layer_ids = getattr(draft_model_config, "target_layer_ids", None)
    if not getattr(args, "aux_hidden_states_layers", None):
        from torchspec.models.draft.dflash import build_target_layer_ids

        if target_layer_ids is None:
            num_target = getattr(draft_model_config, "num_target_layers", 5)
            target_num_hidden = getattr(draft_model_config, "target_num_hidden_layers", 36)
            target_layer_ids = build_target_layer_ids(num_target, target_num_hidden)
        args.aux_hidden_states_layers = target_layer_ids
        logger.info(f"{algo}: set aux_hidden_states_layers = {target_layer_ids}")
    elif is_dflash2 and list(args.aux_hidden_states_layers) != list(target_layer_ids):
        raise ValueError(
            "inference.aux_hidden_states_layers must match "
            f"dflash_config.target_layer_ids ({list(args.aux_hidden_states_layers)} != "
            f"{list(target_layer_ids)})."
        )


# 装配完整训练流水线：控制器载入数据，同时预留资源和启动存储，然后创建训练/推理 actor。
# 数据集大小决定训练步数，词表映射需要 tokenized 数据，因此这些依赖必须在模型开训前完成。
# 名字保留了历史用法，实际也支持 train_with_decode；offline 则回放已采集的 teacher 特征。
# 控制器持有数据集，主进程只拿大小和句柄，避免把整个数据集反复经 Ray 对象存储传输。
def train_async_no_generation(args):
    """Entry point for Eagle3 asynchronous training.

    Supports prefill-only mode (default) and decode mode (train_with_decode=True)
    with speculative decoding. Uses distributed Ray actors with placement groups.
    Engines store tensors in mooncake and return keys to AsyncInferenceManager.
    """
    if (
        getattr(args, "train_with_decode", False)
        and getattr(args, "inference_engine_type", "sgl") != "sgl"
    ):
        raise ValueError("train_with_decode=True requires inference_engine_type=sgl")

    if getattr(args, "inference_engine_type", None) == "offline":
        from torchspec.offline.dataset import (
            OfflineDataset,
            configure_offline_args,
        )

        # offline 记录已经确定 token 和 norm 边界，启动时先读取这份语义。
        # 这里不从“当前引擎叫 offline”推导 post-norm，否则用 vLLM 采集的数据会被错误解释。
        offline_dataset = OfflineDataset(args.offline_data_path)
        configure_offline_args(offline_dataset, args)

    init_tracking(args)
    timer = _InitTimer()

    # [1] Create controller early (lightweight: only needs args + dp_size)
    with timer.phase("Create controller"):
        driver_node_id = ray.get_runtime_context().get_node_id()
        controller = AsyncTrainingController.options(
            runtime_env={"env_vars": get_torchspec_env_vars()},
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=driver_node_id, soft=False),
        ).remote(args, args.dp_size)

    # [1.5] Parse draft config + DFlash validation (before any async work)
    with timer.phase("Parse draft model config"):
        # 不同草稿算法需要不同采集层和 block 约定；先解析并校验再异步加载大数据集。
        # 例如 DFlash2 的显式层列表必须和推理采集列表一致，不能等收到张量后才碰 shape 错误。
        draft_model_config = _get_draft_model_config(args)
        args.draft_model_config_obj = draft_model_config

        _validate_and_configure_dflash(args, draft_model_config)

    # [2] Kick off dataset loading on controller (async — runs on actor while driver continues)
    # 数据加载不依赖 GPU 模型，先发 RPC 让控制器并行执行。
    # 下一段启动资源/存储时，CPU 可继续分词；直到需要数据长度时才等待这两个 refs。
    timer.begin_async("Dataset loading")
    dataset_size_ref = controller.load_dataset.remote(args)
    eval_dataset_size_ref = controller.load_eval_dataset.remote(args)

    # [3] Do initialization that doesn't depend on dataset in parallel
    with timer.phase("Driver-side init"):
        roles = (
            {"training"}
            if getattr(args, "inference_engine_type", None) == "offline"
            else {"training", "inference"}
        )
        pgs = create_placement_groups(args, roles=roles)
        launch_mooncake_master(args)
        mooncake_config = build_mooncake_config(args)

    # [4] Wait for dataset sizes (small ints, unlike the old ray.put of the full dataset)
    dataset_size, eval_dataset_size = timer.wait(
        "Dataset loading", [dataset_size_ref, eval_dataset_size_ref]
    )
    logger.info(f"Dataset loaded on controller: {dataset_size} train, {eval_dataset_size} eval")

    # [5] Auto-calculate training steps (needs dataset_size)
    with timer.phase("Auto-calculate training steps"):
        auto_calculate_training_steps(args, dataset_size)

    # [6] Generate vocab mapping on controller if vocab pruning is enabled
    # 若输出词表裁剪，所有训练 rank 必须使用完全相同的列顺序。
    # 在线可从本轮训练数据统计；offline 必须复用物化时的映射；已有权重映射也可明确保留。
    vocab_mapping = None
    draft_vocab_size = getattr(draft_model_config, "draft_vocab_size", None)
    vocab_size = draft_model_config.vocab_size
    if draft_vocab_size is not None and draft_vocab_size != vocab_size:
        with timer.phase("Vocab mapping"):
            if getattr(args, "keep_initial_vocab_mapping", False):
                logger.info(
                    "keep_initial_vocab_mapping=True: training against the mapping stored in the "
                    "loaded draft weights, so no mapping is computed for this dataset"
                )
            elif getattr(args, "inference_engine_type", None) == "offline":
                # offline 回放没有重新统计任意语料的自由：输出列与 token 的关系已经在物化时确定。
                # 缺少映射就失败，不能为凑维度临时生成一份看似合理的新映射。
                mapping_path = os.path.join(args.offline_data_path, "vocab_mapping.pt")
                if not os.path.isfile(mapping_path):
                    raise FileNotFoundError(
                        "Offline replay requires the vocabulary mapping produced during "
                        f"materialization: {mapping_path}"
                    )
                saved_mapping = torch.load(mapping_path, map_location="cpu", weights_only=True)
                vocab_mapping = (saved_mapping["d2t"], saved_mapping["t2d"])
            else:
                logger.info(
                    f"Computing vocab mapping on controller "
                    f"(target={vocab_size}, draft={draft_vocab_size})..."
                )
                vocab_mapping = ray.get(
                    controller.compute_vocab_mapping.remote(vocab_size, draft_vocab_size)
                )
            if vocab_mapping is not None:
                logger.info(
                    f"Generated vocab mapping: "
                    f"d2t={vocab_mapping[0].shape}, t2d={vocab_mapping[1].shape}"
                )

    # [7] Create training actors + inference engines (args now has num_train_steps)
    timer.begin_async("Actor initialization")
    with timer.phase("Allocate actors + dispatch init"):
        train_group = allocate_train_group(
            args=args,
            num_nodes=args.training_num_nodes,
            num_gpus_per_node=args.training_num_gpus_per_node,
            pg=pgs["training"],
            training_class=TrainerActor,
        )
        train_init_refs = train_group.async_init(
            args, role="training", mooncake_config=mooncake_config, with_ref=False
        )

        # Decode mode: create scratch draft checkpoint before inference engines
        # are prepared, since they need decode_speculative_draft_model_path on args.
        # This blocks on train actor init (FSDP gather), so inference engines are
        # dispatched after to maximize parallelism with the wait below.
        # 普通 prefill 可直接让训练与推理初始化并行；decode 投机引擎却可能先需要草稿权重文件。
        # 这时导出 scratch 依赖训练 actor 已完成初始化及 FSDP 参数聚合，所以这个调用可能等待训练端。
        # 必须在 prepare_inference_engines 前满足该依赖，否则引擎启动时拿不到草稿。
        _maybe_create_scratch_draft(args, train_group)

        inference_engines, engine_init_refs = prepare_inference_engines(
            args, pgs["inference"], mooncake_config
        )

    # [8] Wait for all actor init to complete concurrently
    # 训练与推理初始化 futures 合并等待，减少两边串行加载模型的总时间。
    # 只对训练返回值检查一致性；两类 actor 的初始化结果协议不同，不能混成一个断言。
    n_train = len(train_init_refs)
    logger.info(
        f"Waiting for {n_train} training actors and {len(engine_init_refs)} "
        f"inference engines to initialize in parallel..."
    )
    all_results = timer.wait("Actor initialization", train_init_refs + engine_init_refs)

    train_results = all_results[:n_train]
    assert len(set(train_results)) == 1
    logger.info(
        f"All {n_train} training actors and {len(engine_init_refs)} inference engines initialized"
    )

    if vocab_mapping is not None:
        # 等所有训练模型已构造好再装入统一映射；映射缓冲属于草稿输出语义，不能在不同 rank 自行随机选择。
        train_group.set_vocab_buffers(*vocab_mapping)
        logger.info("Loaded vocab mapping into training actors")

    # [9] Setup async training with pre-created controller
    with timer.phase("Setup async training"):
        # 最后连接各 actor 的队列和独立存储客户端，至此数据协议与进程资源都已就绪。
        # run_training_loop 将启动长期生产循环；前面只是装配，不会自动持续产生训练样本。
        controller, inference_manager = setup_async_training_with_engines(
            args, train_group, mooncake_config, inference_engines, controller=controller
        )

    timer.log_summary()

    # [10] Run training loop (no ray.put needed — dataset lives on controller)
    run_training_loop(
        args,
        controller,
        inference_manager,
        train_group,
        inference_engines=inference_engines,
        dataset_size=dataset_size,
        eval_dataset_size=eval_dataset_size,
    )


if __name__ == "__main__":
    args = parse_config()
    train_async_no_generation(args)
