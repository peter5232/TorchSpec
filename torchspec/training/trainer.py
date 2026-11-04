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

# 训练公共骨架：队列读取、设备 mesh、梯度累积、检查点和评估缓存在此衔接。
# 模型特有的损失留给子类；完整累积窗口通常对应一次 optimizer.step，而非一次取样。
# 队列提前结束时 global_step 仍递增，不能仅凭计数确认优化器已经更新。

import abc
import concurrent.futures
import dataclasses
import itertools
import json
import logging
import os
import time
from argparse import Namespace
from contextlib import nullcontext
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh

from torchspec.config.mooncake_config import MooncakeConfig
from torchspec.data.utils import DataCollatorWithPadding
from torchspec.training import checkpoint
from torchspec.training.data_fetcher import MooncakeDataFetcher, PrefetchedDataFetcher
from torchspec.training.fsdp import init_empty_weights
from torchspec.training.optimizer import BF16Optimizer
from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore
from torchspec.utils.distributed import get_gloo_group, get_usp_device_mesh, get_usp_grad_sync_mesh
from torchspec.utils.logging import logger
from torchspec.utils.processing import get_assistant_token_ids
from torchspec.utils.profiling import TrainProfiler
from torchspec.utils.train_dump import extract_gradients, extract_model_weights


@dataclasses.dataclass
class DeferredTrainMetrics:
    """One-step-delayed GPU metric transfer.

    The current step enqueues a non-blocking copy into pinned host memory. The
    next step is submitted before these values are materialized, allowing CPU
    launch work to overlap the previous step's device-to-host dependency.
    """

    cpu_values: torch.Tensor
    source_values: torch.Tensor
    ready_event: torch.cuda.Event
    metric_step: int
    finalize_values: Callable[[list[float]], dict]
    data_time: float = 0.0
    compute_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = dataclasses.field(
        default_factory=list
    )
    optimizer_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = dataclasses.field(
        default_factory=list
    )
    step_events: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None

    # 等待计时结束事件（或 D2H 完成事件），才能把 pinned CPU 数值转换为 Python 指标。
    # 保留 metric_step 防止延迟一轮的指标被记到当前步；读取结束后释放 CUDA 源张量引用。
    def finalize(self) -> dict:
        if self.step_events is not None:
            self.step_events[1].synchronize()
        else:
            self.ready_event.synchronize()

        metrics = self.finalize_values(self.cpu_values.tolist())
        metrics["_metrics_step"] = self.metric_step

        if self.step_events is not None:
            metrics["perf/step_time"] = (
                self.step_events[0].elapsed_time(self.step_events[1]) / 1000.0
            )
            metrics["perf/data_time"] = self.data_time
            metrics["perf/compute_time"] = (
                sum(start.elapsed_time(end) for start, end in self.compute_events) / 1000.0
            )
            metrics["perf/optimizer_time"] = (
                sum(start.elapsed_time(end) for start, end in self.optimizer_events) / 1000.0
            )

        # The copy and timing events are complete; release the CUDA source now.
        self.source_values = torch.empty(0)
        return metrics


class Trainer(abc.ABC):
    """Base trainer for async training pipeline.

    Provides shared infrastructure: device mesh, data fetching, training loop
    skeleton, checkpointing, and profiling. Subclasses implement model-specific
    logic via ``init_model``, ``_train_step``, and ``_aggregate_metrics``.
    """

    # 建立设备/并行上下文，并初始化模型、优化器、Mooncake、评估缓存及性能记录状态。
    # assistant 标记供动态 loss mask 使用；单线程 I/O executor 负责异步写评估缓存。
    def __init__(self, args: Namespace):
        self.args = args

        self._setup_device_mesh()
        torch.manual_seed(getattr(args, "seed", 42))

        self.fsdp_cpu_offload = getattr(args, "fsdp_cpu_offload", False)

        self.global_step = 0
        self.model = None
        self.draft_model = None
        self.optimizer: Optional[BF16Optimizer] = None
        self.lr_scheduler = None
        self.data_fetcher: Optional[MooncakeDataFetcher] = None
        self.train_queue = None
        self.mooncake_store: Optional[EagleMooncakeStore] = None
        self._eval_cache: list[dict] = []

        self.prof = TrainProfiler(args)

        self.dynamic_loss_mask = getattr(args, "dynamic_loss_mask", False)
        self.last_turn_loss_only = getattr(args, "last_turn_loss_only", False)
        self.assistant_header_ids, self.end_token_ids, self.skip_after_header = (
            get_assistant_token_ids(self.args)
        )

        self.save_debug_train_data = getattr(args, "save_debug_train_data", None)
        self.max_dump_steps = getattr(args, "max_dump_steps", 5)

        self._enable_perf_metrics = getattr(args, "enable_perf_metrics", True)
        self._pending_train_metrics: Optional[DeferredTrainMetrics] = None

        self._io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._eval_cache_save_future: Optional[concurrent.futures.Future] = None

    # ------------------------------------------------------------------
    # Device mesh
    # ------------------------------------------------------------------

    # 普通路径把全部 rank 作为 DP；USP 的 draft_dp 只连接不同样本副本。
    # USP 参数梯度则用覆盖所有 rank 的 grad_sync_mesh 同步，因为不同 SP rank 持有同一样本的不同位置。
    def _setup_device_mesh(self) -> None:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        self.cache_rank = rank

        usp_mesh = None
        if getattr(self.args, "attention_backend", None) == "usp":
            usp_mesh = get_usp_device_mesh()

        if usp_mesh is not None:
            self.mesh = usp_mesh
            self.dp_size = getattr(self.args, "dp_size", world_size)
            self.dp_mesh = usp_mesh["draft_dp"]
            self.grad_sync_mesh = get_usp_grad_sync_mesh()
            if self.grad_sync_mesh is None:
                raise RuntimeError("USP grad sync mesh has not been initialized")
            self.dp_group = usp_mesh.get_group("draft_dp")
            self.dp_rank = dist.get_rank(self.dp_group)
            logger.info(
                f"[Rank {rank}] Device mesh (USP): world_size={world_size}, dp_size={self.dp_size}, "
                f"dp_rank={self.dp_rank}, grad_sync_size={world_size}"
            )
            return

        self.dp_size = world_size
        self.dp_rank = rank

        self.mesh = init_device_mesh("cuda", mesh_shape=(self.dp_size,), mesh_dim_names=("dp",))
        self.dp_group = self.mesh.get_group("dp")
        self.dp_mesh = self.mesh
        self.grad_sync_mesh = self.dp_mesh

        logger.info(
            f"[Rank {rank}] Device mesh (1D): world_size={world_size}, dp_size={self.dp_size}"
        )

    # 返回构造权重的上下文工厂：rank 0 在 CPU 真正分配，其他 rank 在 meta 上只建立形状。
    # 后续 fsdp2_load_full_state_dict 广播实际值，避免每个进程重复读完整权重。
    def _get_init_weight_context_manager(self):
        """Meta-device context for non-rank-0 processes to save memory."""

        # 为 rank 0 返回 CPU 设备上下文，使模型构造阶段不提前占满 GPU。
        def cpu_init_weights():
            return torch.device("cpu")

        if dist.get_rank() != 0:
            return init_empty_weights
        return cpu_init_weights

    # ------------------------------------------------------------------
    # Mooncake store
    # ------------------------------------------------------------------

    # 从参数或给定配置创建训练侧读取客户端；replace 保留原配置不被原地改写。
    # 把写入池和全局段设为零，随后在当前 GPU 上 setup 并持有 store 供 fetcher 复用。
    def init_mooncake_store(
        self,
        mooncake_config: Optional[MooncakeConfig] = None,
    ) -> EagleMooncakeStore:
        if mooncake_config is None:
            mooncake_config = MooncakeConfig.from_flat_args(self.args)

        mooncake_config = dataclasses.replace(
            mooncake_config,
            global_segment_size=0,
            async_put_pool_size=0,
        )

        store = EagleMooncakeStore(mooncake_config)
        store.setup(device=torch.cuda.current_device())
        self.mooncake_store = store
        logger.info(f"[Rank {self.dp_rank}] EagleMooncakeStore initialized")
        return store

    # ------------------------------------------------------------------
    # Data queue
    # ------------------------------------------------------------------

    # 把控制器队列接到 DataLoader/collator；USP 明确要求每个 DP rank 的 batch 为 1。
    # 开启预取时后台读取到 CPU，主线程再搬 GPU，避免 TCP 读取与 GPU 计算直接争用。
    def set_train_queue(
        self,
        queue,
        mooncake_config: Optional[MooncakeConfig] = None,
        per_dp_rank_batch_size: int = 1,
    ) -> None:
        self.train_queue = queue
        self.per_dp_rank_batch_size = per_dp_rank_batch_size
        usp_enabled = getattr(self.args, "attention_backend", None) == "usp"
        if usp_enabled and per_dp_rank_batch_size != 1:
            raise ValueError("USP requires per_dp_rank_batch_size=1")
        if mooncake_config is not None and self.mooncake_store is None:
            self.init_mooncake_store(mooncake_config)

        # 先决定 batch 的组织方式，再建传输读取器。普通 collator 会把长度不同的样本补齐；
        # USP 的样本已经按序列分片，不能把它当成普通数据副本继续随意拼 batch。
        # 因此前面的 per_dp_rank_batch_size=1 校验是布局约束，不只是性能建议。
        collator = DataCollatorWithPadding(usp_enabled=usp_enabled)

        prefetch_depth = getattr(self.args, "prefetch_depth", 0)
        gpu_device = torch.cuda.current_device()

        # When prefetching, stage data on CPU to avoid GPU contention between
        # background Mooncake TCP transfers and forward/backward compute.
        fetch_device = "cpu" if prefetch_depth > 0 else gpu_device

        # 这里把“读取到哪个设备”与“最后在哪计算”分开。无预取时直接读当前 GPU；
        # 有预取时 CPU staging 的等待可放在后台线程，GPU 搬运则在主消费线程发起。
        # non_blocking 搬运返回不保证复制已完成，后续使用依赖 CUDA stream 的顺序约束。
        # mask 的匹配规则、最小监督数、TTT 尾部和序列上限必须同生产端/模型契约一致。
        inner_fetcher = MooncakeDataFetcher(
            queue=self.train_queue,
            mooncake_store=self.mooncake_store,
            collator=collator,
            device=fetch_device,
            batch_size=per_dp_rank_batch_size,
            assistant_header_ids=self.assistant_header_ids,
            end_token_ids=self.end_token_ids,
            dynamic_loss_mask=self.dynamic_loss_mask,
            last_turn_loss_only=self.last_turn_loss_only,
            skip_after_header=self.skip_after_header,
            min_loss_tokens=getattr(self.args, "min_loss_tokens", 0),
            usp_enabled=usp_enabled,
            ttt_length=getattr(self.args, "ttt_length", 1),
            max_seq_length=getattr(self.args, "max_seq_length", None),
        )

        # 队列深度只限制已经读好等待训练的 batch 数，不改变每次 optimizer.step 的样本数。
        # 若 depth=2、训练线程暂时较慢，第三次 put 会阻塞；这是控制 CPU 驻留数据的背压。
        if prefetch_depth > 0:
            self.data_fetcher = PrefetchedDataFetcher(
                inner_fetcher,
                prefetch_depth=prefetch_depth,
                target_device=gpu_device,
            )
            logger.info(
                f"[Rank {self.dp_rank}] Prefetched data fetcher initialized "
                f"(batch_size={per_dp_rank_batch_size}, prefetch_depth={prefetch_depth}, "
                f"staging=CPU, target={gpu_device})"
            )
        else:
            self.data_fetcher = inner_fetcher
            logger.info(
                f"[Rank {self.dp_rank}] Data fetcher initialized with batch_size={per_dp_rank_batch_size}"
            )

    # ------------------------------------------------------------------
    # Eval queue & CPU cache
    # ------------------------------------------------------------------

    # 独立建立评估读取器和 collator，并清空内存评估缓存。
    # 评估数据仍来自 Mooncake，但缓存完成后可重复评估，不必反复运行 teacher。
    def set_eval_queue(
        self,
        queue,
        mooncake_config: Optional[MooncakeConfig] = None,
        per_dp_rank_batch_size: int = 1,
    ) -> None:
        usp_enabled = getattr(self.args, "attention_backend", None) == "usp"
        if mooncake_config is not None and self.mooncake_store is None:
            self.init_mooncake_store(mooncake_config)

        collator = DataCollatorWithPadding(usp_enabled=usp_enabled)

        self._eval_data_fetcher = MooncakeDataFetcher(
            queue=queue,
            mooncake_store=self.mooncake_store,
            collator=collator,
            device=torch.cuda.current_device(),
            batch_size=per_dp_rank_batch_size,
            assistant_header_ids=self.assistant_header_ids,
            end_token_ids=self.end_token_ids,
            dynamic_loss_mask=self.dynamic_loss_mask,
            last_turn_loss_only=self.last_turn_loss_only,
            skip_after_header=self.skip_after_header,
            min_loss_tokens=getattr(self.args, "min_loss_tokens", 0),
            usp_enabled=usp_enabled,
            ttt_length=getattr(self.args, "ttt_length", 1),
            max_seq_length=getattr(self.args, "max_seq_length", None),
        )
        self._eval_collator = collator
        self._eval_cache: list[dict] = []
        logger.info(
            f"[Rank {self.dp_rank}] Eval data fetcher initialized "
            f"with batch_size={per_dp_rank_batch_size}"
        )

    # 从评估 fetcher 最多消费 count 个元素，把张量移到 CPU 后追加缓存，返回累计缓存数。
    def cache_eval_samples(self, count: int) -> int:
        for sample in itertools.islice(self._eval_data_fetcher, count):
            cpu_sample = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in sample.items()
            }
            self._eval_cache.append(cpu_sample)
        return len(self._eval_cache)

    # 等待上次写入完成，再对缓存列表取快照提交后台线程；各全局 rank 写自己的文件。
    # 列表快照并非 tensor 深拷贝，调用者不应在写盘期间原地修改缓存内容。
    def save_eval_cache(self, cache_dir: str) -> None:
        if not getattr(self, "_eval_cache", None):
            return
        self._wait_for_eval_cache_save()

        cache_snapshot = list(self._eval_cache)
        rank = self.cache_rank

        # 先 torch.save 到 .tmp，再 os.replace 发布文件，避免读者观察半写入的评估缓存。
        def _save() -> None:
            os.makedirs(cache_dir, exist_ok=True)
            path = os.path.join(cache_dir, f"eval_rank_{rank}.pt")
            tmp_path = path + ".tmp"
            torch.save(cache_snapshot, tmp_path)
            os.replace(tmp_path, path)
            logger.info(f"[Rank {rank}] Saved {len(cache_snapshot)} eval batches to {path}")

        self._eval_cache_save_future = self._io_executor.submit(_save)

    # 等待 Future 并传播写盘异常；完成后清空引用，供下一次保存或关闭使用。
    def _wait_for_eval_cache_save(self) -> None:
        fut = self._eval_cache_save_future
        if fut is not None:
            fut.result()
            self._eval_cache_save_future = None

    # 先等异步保存，按 cache_rank 读取对应缓存并使用 mmap 延迟映射。
    # 文件缺失或读取损坏返回 0；成功返回可用于后续评估的缓存元素数。
    def load_eval_cache(self, cache_dir: str) -> int:
        # Safe guard to wait for eval cache save to complete.
        self._wait_for_eval_cache_save()
        path = os.path.join(cache_dir, f"eval_rank_{self.cache_rank}.pt")
        if not os.path.exists(path):
            return 0
        try:
            self._eval_cache = torch.load(path, weights_only=False, mmap=True)
        except Exception as e:
            logger.warning(f"[Rank {self.dp_rank}] Corrupt eval cache at {path}, ignoring: {e}")
            return 0
        logger.info(
            f"[Rank {self.dp_rank}] Loaded {len(self._eval_cache)} eval batches from {path}"
        )
        return len(self._eval_cache)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    # 执行一个累积窗口并推进 profiler；Eagle 返回延迟对象时先提交当前步，再兑现上一步指标。
    # 首步可能返回空字典，最后一份指标必须通过 flush_pending_metrics 收尾。
    def train_from_queue(self, step: int, num_batches: int) -> dict:
        if self.data_fetcher is None:
            raise RuntimeError("Data fetcher not initialized. Call set_train_queue first.")
        perf = self._enable_perf_metrics
        if perf:
            t0 = time.time()
            step_start = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)
            step_start.record()
        metrics = self._train_core_from_queue(step=step, num_batches=num_batches)
        if isinstance(metrics, DeferredTrainMetrics):
            if perf:
                step_end.record()
                metrics.step_events = (step_start, step_end)
            # 先完成本步 CUDA 提交再读取上步 CPU buffer；等待上步事件可与本步设备计算重叠。
            previous_metrics = self._pending_train_metrics
            self._pending_train_metrics = metrics
            metrics = previous_metrics.finalize() if previous_metrics is not None else {}
        elif perf:
            # Immediate metric aggregation has already synchronized the step.
            metrics["perf/step_time"] = time.time() - t0
        self.prof.step(step=step)
        return metrics

    # 取出并清空最后一个待兑现指标对象；无待处理值时返回空字典，避免重复记录。
    def flush_pending_metrics(self) -> dict:
        """Materialize the final deferred training metrics, if any."""
        pending = self._pending_train_metrics
        self._pending_train_metrics = None
        return pending.finalize() if pending is not None else {}

    # 消费 num_batches 个 microbatch；只有最后一个触发梯度同步、dump 和优化器更新。
    # FSDP2 用 set_requires_gradient_sync，replicate 用 no_sync；子类负责 loss 缩放和 backward。
    # 计时用 CUDA Event 测设备执行，data_time 用 CPU 时钟测取数等待，二者不能简单当作互斥时间相加。
    def _train_core_from_queue(self, step: int, num_batches: int) -> dict:
        """Training loop skeleton.

        Calls ``_train_step`` for each micro-batch, wrapped with debug logging
        and optional dump extraction.  One optimizer step is performed after
        all micro-batches.  ``global_step`` counts optimizer steps.
        """
        self.model.train()
        accumulation_steps = num_batches

        all_step_metrics: list[dict] = []
        grad_norm = None

        perf = self._enable_perf_metrics
        if perf:
            data_time = 0.0
            compute_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            t_data_start = time.time()

        # Gradient sync control for micro-batch accumulation.
        # FSDP2 fully_shard: use set_requires_gradient_sync(bool)
        # replicate (DDP): use no_sync() context manager
        # compile wrapper 不是实际拥有 no_sync/set_requires_gradient_sync 的分布式对象，先取原模块。
        # 这里支持两种同步控制接口，但优化器更新仍位于统一循环的最后一个 microbatch。
        # no_sync 必须包住前向与反向，因为分布式模块可能在 forward 时设置后续梯度 hook 状态。
        _model = getattr(self.model, "_orig_mod", self.model)
        _set_grad_sync = getattr(_model, "set_requires_gradient_sync", None)
        _no_sync = getattr(_model, "no_sync", None) if _set_grad_sync is None else None

        batches = self.prof.iterate_train_actor(self._iter_batches_from_queue(num_batches))
        for batch_idx, batch in enumerate(batches):
            # 例如 num_batches=4，前 3 次只累积梯度，索引 3 才同步并更新。
            # 这个条件依赖数据源确实产出要求的数量；队列提前结束并不会在这里补造 batch。
            # 阅读调用链时需要同时检查控制器的投递数量和 sentinel 的发送时机。
            is_last = batch_idx == num_batches - 1

            if perf:
                data_time += time.time() - t_data_start
                evt_start = torch.cuda.Event(enable_timing=True)
                evt_end = torch.cuda.Event(enable_timing=True)
                evt_start.record()

            if logger.isEnabledFor(logging.DEBUG):
                self._log_batch_debug(batch, step, batch_idx, num_batches)

            # 前 K-1 次只在本 rank 累积；最后一次同步必须包含此前累积梯度，随后只更新一次参数。
            if _set_grad_sync is not None:
                _set_grad_sync(is_last)
                ctx = nullcontext()
            else:
                ctx = _no_sync() if (_no_sync is not None and not is_last) else nullcontext()

            with ctx:
                step_metrics = self._train_step(
                    batch=batch,
                    accumulation_steps=accumulation_steps,
                    step=step,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                )

            if is_last:
                # dump 捕获的是本窗口全部 microbatch 已累积后的梯度，以及更新之前的模型参数。
                # 随后 optimizer 会清掉梯度并回写新权重，因此挪到 step 之后就失去这组对照关系。
                self._maybe_dump(batch, step_metrics, step, batch_idx)
                _evt_opt_s = torch.cuda.Event(enable_timing=True)
                _evt_opt_e = torch.cuda.Event(enable_timing=True)
                _evt_opt_s.record()
                grad_norm = self.optimizer.step()
                _evt_opt_e.record()
                step_metrics["_opt_events"] = (_evt_opt_s, _evt_opt_e)

            if perf:
                evt_end.record()
                compute_events.append((evt_start, evt_end))

            all_step_metrics.append(step_metrics)

            if perf:
                t_data_start = time.time()

        # 当前实现无条件推进计数：若队列提前结束，is_last 未成立，就没有 optimizer.step，
        # 也不会在这里清除未完成窗口的梯度。排查步数时要同时核对实际消费的 batch 数。
        self.global_step += 1

        # 模型 loss 的分母与日志指标的分母由子类决定；基类不尝试再次平均 loss。
        # 返回值也可能是延迟读取对象：不要在公共循环中对每个 CUDA 标量 .item()，
        # 否则会破坏 Eagle 使用 pinned 双缓冲隐藏指标同步的设计。
        metrics = self._aggregate_metrics(all_step_metrics, step, grad_norm=grad_norm)
        if perf:
            optimizer_events = []
            for m in all_step_metrics:
                if "_opt_events" in m:
                    optimizer_events.append(m["_opt_events"])

            if isinstance(metrics, DeferredTrainMetrics):
                metrics.data_time = data_time
                metrics.compute_events = compute_events
                metrics.optimizer_events = optimizer_events
            else:
                # Immediate aggregation synchronized CUDA, so event queries are ready.
                metrics["perf/data_time"] = data_time
                # compute event 的结束点在 optimizer 之后记录，因此最后一段 compute 含 optimizer 时间。
                # optimizer_time 是该段的细分，不应再和 compute_time 相加解释为总 GPU 时间。
                metrics["perf/compute_time"] = (
                    sum(start.elapsed_time(end) for start, end in compute_events) / 1000.0
                )
                metrics["perf/optimizer_time"] = (
                    sum(start.elapsed_time(end) for start, end in optimizer_events) / 1000.0
                )

        return metrics

    # 用 islice 限制本轮消费数量，队列中的剩余样本留给后续优化器步。
    def _iter_batches_from_queue(self, num_batches: int):
        yield from itertools.islice(self.data_fetcher, num_batches)

    # ------------------------------------------------------------------
    # Checkpointing & persistence
    # ------------------------------------------------------------------

    # checkpoint_dir 未配置时直接返回；否则交给分布式 checkpoint.save 保存训练状态。
    # force_sync 目前只是接口兼容参数，此实现未据此选择异步保存。
    def save_model(self, step: int, force_sync: bool = False) -> None:
        if not self.args.checkpoint_dir:
            return
        checkpoint.save(self, step=step)

    # 把模型 config 和权重转换为 serving schema 后写入 config.json/pytorch_model.bin。
    # 有自定义转换方法时必须调用它，训练内部参数名不一定是推理引擎期望的名字。
    def _write_serving_artifacts(self, model, state_dict: dict, output_dir: str) -> None:
        if hasattr(model, "config"):
            config = (
                model.config_for_serving() if hasattr(model, "config_for_serving") else model.config
            )
            if isinstance(config, dict):
                with open(os.path.join(output_dir, "config.json"), "w") as config_file:
                    json.dump(config, config_file, indent=2)
            else:
                config.save_pretrained(output_dir)
        if hasattr(model, "state_dict_for_serving"):
            state_dict = model.state_dict_for_serving(state_dict)
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    # 把本 rank 异常转成字符串，通过 Gloo all_gather_object 收集后返回首个错误。
    # 这样写盘 rank 失败时其他 rank 也能走一致的失败/回退分支，避免卡在集合通信。
    def _synchronize_serving_error(self, error: Exception | None) -> str | None:
        if not dist.is_initialized():
            return None if error is None else f"{type(error).__name__}: {error}"

        group = get_gloo_group()
        errors = [None] * dist.get_world_size(group)
        local_error = None if error is None else f"{type(error).__name__}: {error}"
        dist.all_gather_object(errors, local_error, group=group)
        return next((message for message in errors if message is not None), None)

    # 先集体收集完整 CPU state_dict，再由 dp_rank 0 写推理可加载的权重。
    # 自定义 schema 下 FULL_SHARD 收集失败不可退回局部分片 state_dict；replicate 才尝试完整本地权重回退。
    def save_draft_model_for_serving(self, output_dir: str) -> None:
        """Save draft model in HuggingFace format for serving update."""
        os.makedirs(output_dir, exist_ok=True)

        model = self.draft_model
        if hasattr(model, "module"):
            model = model.module
        # 训练 wrapper、DDP wrapper 和实际 draft 的 state_dict 命名可能不同。
        # custom_schema 标记模型提供了专门面向 serving 的配置/键名转换；仅保存训练配置未必能被推理端读取。
        custom_schema = hasattr(model, "config_for_serving") or hasattr(
            model, "state_dict_for_serving"
        )

        save_error = None
        try:
            # 收集完整权重是所有参与 rank 的 collective 工作，不应放进 dp_rank==0 条件。
            # 只有最终文件写入由指定 rank 执行；CPU offload 避免导出时再保留一份完整 GPU 权重。
            state_dict = get_model_state_dict(
                self.draft_model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )

            if self.dp_rank == 0:
                if hasattr(model, "save_pretrained"):
                    if custom_schema:
                        self._write_serving_artifacts(model, state_dict, output_dir)
                    else:
                        if hasattr(model, "config"):
                            model.config.save_pretrained(output_dir)
                        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                    logger.info(f"[Rank {self.dp_rank}] Saved draft model to {output_dir}")
                else:
                    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                    logger.info(
                        f"[Rank {self.dp_rank}] Saved draft model state dict to {output_dir}"
                    )

        except Exception as error:
            save_error = error

        if custom_schema:
            synchronized_error = self._synchronize_serving_error(save_error)
            if synchronized_error is None:
                return

            strategy = getattr(getattr(self, "args", None), "fsdp_strategy", "REPLICATE")
            # FULL_SHARD 的本地 state_dict 可能只覆盖部分参数存储，失败后不可拿它充当完整权重。
            # 这里先把错误同步给所有 rank，再统一失败，防止仅写盘者抛错而其他人继续等待 barrier。
            if strategy.upper() == "FULL_SHARD":
                if save_error is not None:
                    raise save_error
                raise RuntimeError(
                    f"Failed to save a full serving state dict on another rank: "
                    f"{synchronized_error}"
                )

            logger.warning(
                f"[Rank {self.dp_rank}] Failed to save with FSDP2 state dict, trying fallback: "
                f"{synchronized_error}"
            )
            # REPLICATE 下各 rank 有完整参数，允许尝试本地完整 state_dict 的转换导出。
            # 回退仍可能因 I/O/schema 转换失败，因此第二次错误也需要跨 rank 同步，不能只记录 warning。
            fallback_error = None
            try:
                if self.dp_rank == 0:
                    self._write_serving_artifacts(model, model.state_dict(), output_dir)
                    logger.info(
                        f"[Rank {self.dp_rank}] Saved draft model using save_pretrained to {output_dir}"
                    )
            except Exception as error:
                fallback_error = error

            synchronized_error = self._synchronize_serving_error(fallback_error)
            if synchronized_error is not None:
                if fallback_error is not None:
                    raise fallback_error
                raise RuntimeError(
                    f"Failed to save serving artifacts on another rank: {synchronized_error}"
                )
            return

        if save_error is not None:
            logger.warning(
                f"[Rank {self.dp_rank}] Failed to save with FSDP2 state dict, trying fallback: "
                f"{save_error}"
            )
            if self.dp_rank == 0:
                if hasattr(model, "save_pretrained"):
                    model.save_pretrained(output_dir)
                    logger.info(
                        f"[Rank {self.dp_rank}] Saved draft model using save_pretrained to {output_dir}"
                    )
                else:
                    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    logger.info(
                        f"[Rank {self.dp_rank}] Saved draft model state dict to {output_dir}"
                    )

        if dist.is_initialized():
            dist.barrier()

    # 委托 checkpoint.load 返回恢复载荷；RNG、步数和 continual-training 收尾由 finalize_load 负责。
    def load_checkpoint(self) -> dict | None:
        return checkpoint.load(self)

    # 等待评估缓存写完并关闭线程池，再释放 Mooncake 客户端持有的传输资源。
    def close(self) -> None:
        self._wait_for_eval_cache_save()
        self._io_executor.shutdown(wait=True)
        if self.mooncake_store is not None and hasattr(self.mooncake_store, "close"):
            self.mooncake_store.close()
            logger.info(f"[Rank {self.dp_rank}] EagleMooncakeStore closed")

    # ------------------------------------------------------------------
    # Debug logging & dump helpers
    # ------------------------------------------------------------------

    # 读取 input_ids 的 [B,S] 与可选 hidden_states 形状，仅输出 batch 定位信息，不打印张量值。
    def _log_batch_debug(self, batch: dict, step: int, batch_idx: int, num_batches: int) -> None:
        batch_size = batch["input_ids"].shape[0]
        seq_len = batch["input_ids"].shape[1]
        hs_shape = (
            tuple(batch["hidden_states"].shape) if batch.get("hidden_states") is not None else None
        )
        logger.debug(
            f"step={step} batch={batch_idx}/{num_batches} | "
            f"batch_size={batch_size}, seq_len={seq_len}, "
            f"input_ids={tuple(batch['input_ids'].shape)}, hidden_states={hs_shape}"
        )

    # 仅在配置 dump 路径且下一次优化器步未超过 max_dump_steps 时启用调试快照。
    def _should_dump_step(self) -> bool:
        return bool(self.save_debug_train_data and (self.global_step + 1) <= self.max_dump_steps)

    # 在更新权重之前提取当前模型梯度和权重 CPU 副本，并委托模型专用 dump hook 写盘。
    def _maybe_dump(self, batch: dict, step_metrics: dict, step: int, batch_idx: int) -> None:
        if not self._should_dump_step():
            return

        self._save_dump_data(
            batch=batch,
            step_metrics=step_metrics,
            gradients=extract_gradients(self.model),
            model_weights=extract_model_weights(self.model),
            step=step,
            batch_idx=batch_idx,
        )

    # 基类不写文件；子类可把统一的 batch、梯度、权重和自身损失指标组织成可比较快照。
    def _save_dump_data(
        self,
        *,
        batch: dict,
        step_metrics: dict,
        gradients: dict,
        model_weights: dict,
        step: int,
        batch_idx: int,
    ) -> None:
        """Save debug dump data. Override in subclass for model-specific dumps."""

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    # 子类契约：建立模型及优化器并处理恢复；返回启动步信息，基类不规定具体模型结构。
    @abc.abstractmethod
    def init_model(self, *args, **kwargs) -> int:
        """Initialize model, optimizer, and load checkpoint.

        Returns:
            Start step.
        """
        ...

    # 子类契约：对一个 microbatch 前向及反向，返回用于窗口汇总的指标张量。
    # 不能在这里调用 optimizer.step，更新归公共循环统一管理。
    @abc.abstractmethod
    def _train_step(
        self,
        batch: dict,
        accumulation_steps: int,
        step: int,
        batch_idx: int,
        num_batches: int,
    ) -> dict:
        """Run forward + backward for a single micro-batch.

        The optimizer step is handled by the base class after the last
        micro-batch — do NOT call ``self.optimizer.step()`` here.

        Returns:
            Dict of step-level metrics for later aggregation.
        """
        ...

    # 子类契约：全部 microbatch 完成并更新优化器后汇总指标；grad_norm 是裁剪前的范数。
    @abc.abstractmethod
    def _aggregate_metrics(
        self, all_step_metrics: list[dict], step: int, *, grad_norm: torch.Tensor = None
    ) -> dict:
        """Aggregate per-step metrics into a single metrics dict.

        Called once per optimizer step after all micro-batches.
        """
        ...
