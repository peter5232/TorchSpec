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

# 中文导读：异步生产者调度器。它把控制器中的输入分批发给引擎，再把完成的 key 送回控制器。
# asyncio 在等待远程 GPU 计算时让出执行权；它不会把 Python 代码本身变成 GPU 并行计算。
# 水位线控制本地输入预取，样本池容量控制未消费输出，两者分别缓冲生产前和生产后的速度波动。
# 引擎选择是轮询，加一个全局信号量限制总并发，并不是按每台引擎的实时负载精确调度。

"""Watermark-buffered, backpressure-aware batch dispatcher for inference engines.

Sits between the training controller and Ray inference engine actors.
Refills its prompt buffer when it drops below a low-watermark, dispatches
fixed-size batches to engines in round-robin order, and forwards the
resulting mooncake keys back to the controller's sample pool.

NOTE: The inference engine stores tensors in mooncake.
This dispatcher only receives keys and forwards them to training.

Data flow:
  Controller.prompt_buffer → dispatcher._prompt_buffer → Engine.generate()
  → Engine stores tensors in mooncake → returns dict with mooncake_key
  → dispatcher → Controller.sample_pool

Backpressure:
  When sample_pool exceeds _max_pool_size, pauses dispatching until
  training catches up and consumes data.
"""

import asyncio
import time
from collections import deque
from typing import Any

import ray
from ray.exceptions import RayActorError

from torchspec.utils.logging import logger
from torchspec.utils.types import InferenceInput, InferenceOutput

MOONCAKE_BACKPRESSURE_POLL_INTERVAL = 0.5  # seconds
MOONCAKE_BACKPRESSURE_LOG_INTERVAL = 5.0  # seconds


# 封装引擎句柄和全局并发额度；信号量数量=每引擎配置额度×引擎数。
class EnginePool:
    """Round-robin engine selection with semaphore-based concurrency control."""

    # 拒绝空引擎列表并建立总信号量；注意并没有为每个引擎分别建立 semaphore。
    def __init__(self, engines: list, max_concurrent_per_engine: int = 1):
        if not engines:
            raise ValueError("engines must be non-empty")
        self._engines = engines
        self._next_idx = 0
        self._max_concurrent = max_concurrent_per_engine * len(engines)
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

    # 返回下一个引擎并环绕索引，不检查这个引擎是否空闲。
    # 因此 round-robin 保证提交顺序轮换，不保证各引擎完成速度相同。
    def pick(self):
        """Return the next engine in round-robin order."""
        engine = self._engines[self._next_idx]
        self._next_idx = (self._next_idx + 1) % len(self._engines)
        return engine

    # 提供共享并发闸门，调用方用 async with 在成功或异常时都归还额度。
    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore

    # 暴露允许同时挂起的总批次数，上层据此限制 asyncio task 数。
    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    # 返回引擎实例数，用于日志与容量估算，而不是 GPU 总数（一个引擎可能占多张卡）。
    def __len__(self) -> int:
        return len(self._engines)


# 暂存引擎回包中的可选性能指标；缺失字段不填零，避免把“不提供”误算成零延迟/零接受率。
class MetricsCollector:
    """Collects per-sample metrics from engine outputs and aggregates on flush."""

    # engine output key → buffer key
    _KEYS = {
        "e2e_latency": "e2e_latencies",
        "prompt_tokens": "prompt_tokens",
        "completion_tokens": "completion_tokens",
        "spec_accept_rate": "spec_accept_rates",
        "spec_accept_length": "spec_accept_lengths",
    }

    # buffer key → (wandb metric prefix, include min/max)
    _AGGREGATIONS = {
        "e2e_latencies": ("inference/e2e_latency", True),
        "prompt_tokens": ("inference/prompt_tokens", False),
        "completion_tokens": ("inference/completion_tokens", False),
        "spec_accept_rates": ("inference/spec_accept_rate", False),
        "spec_accept_lengths": ("inference/spec_accept_length", False),
    }

    # 为各类指标建立独立列表，记录的是每个回包的观测值。
    def __init__(self):
        self._buffer: dict[str, list] = {k: [] for k in self._AGGREGATIONS}

    # 仅收集本回包确实包含的字段；不同后端可以只提供部分指标。
    def record(self, output: dict) -> None:
        """Record metrics from a single engine output dict."""
        for key, buffer_key in self._KEYS.items():
            if key in output:
                self._buffer[buffer_key].append(output[key])

    # 逐字段求观测值的算术平均，延迟额外给 min/max，然后清空本轮列表。
    # 这是回包级平均，不是把不同 token 数的请求按 token 加权后的平均。
    def flush(self) -> dict:
        """Return aggregated metrics and reset."""
        metrics = {}
        for buffer_key, (prefix, with_minmax) in self._AGGREGATIONS.items():
            values = self._buffer[buffer_key]
            if not values:
                continue
            metrics[f"{prefix}_avg"] = sum(values) / len(values)
            if with_minmax:
                metrics[f"{prefix}_min"] = min(values)
                metrics[f"{prefix}_max"] = max(values)
        self._buffer = {k: [] for k in self._AGGREGATIONS}
        return metrics


# 持有 actor 句柄和本地任务集合；它管理请求生命周期，不承载 teacher 的计算图。
@ray.remote
class AsyncInferenceManager:
    """Dispatches inference batches to distributed Ray engine actors.

    Main loop::

        Controller ──get_prompts──→ [_prompt_buffer]
                                         │
                        ┌────────────────┘
                        ▼
                  _dispatch_ready_batches ──→ EnginePool ──→ engine.generate()
                        │                        ▲
                        │ (backpressure:          │ semaphore
                        │  pool full → wait)      │
                        ▼                         │
                  _collect_completed ◄─── _pending_tasks
                        │
                        ▼
                  _forward_results ──→ Controller.push_inference_results()

    Public API (called via Ray remote):
      run(), stop(), flush_metrics(), get_status(), get_pool_status()
    """

    # 初始化输入缓冲、并发池、背压阈值和运行开关；记录是否延迟分词及是否执行 decode。
    # return_hidden_states 来自 compute_logits_in_trainer：在训练端重建 logits 可降低传输大词表张量的开销。
    def __init__(
        self,
        args,
        controller,
        inference_engines: list | None = None,
        max_concurrent_batches: int = 1,
    ):
        self.args = args
        self.controller = controller

        self._engines = EnginePool(inference_engines or [], max_concurrent_batches)
        self._metrics = MetricsCollector()

        self._prompt_buffer: deque[InferenceInput] = deque()
        self._buffer_low_watermark = getattr(args, "inference_buffer_threshold", 32)
        self._refill_size = getattr(args, "inference_fetch_batch", 1)
        self._batch_size = getattr(args, "inference_batch_size", 8)
        self._max_pool_size = getattr(args, "max_sample_pool_size", 0)

        self._running = True
        self._pending_tasks: set[asyncio.Task] = set()
        self._source_dry = False

        self._defer_tokenization = getattr(args, "defer_tokenization", False)
        self._return_hidden_states = getattr(args, "compute_logits_in_trainer", True)
        self._train_with_decode = getattr(args, "train_with_decode", False)

        self._enable_perf_metrics = getattr(args, "enable_perf_metrics", True)
        self._perf_window_seconds: float = 10.0
        # Each entry: (num_samples, elapsed_seconds, finish_timestamp)
        self._batch_times: deque[tuple[int, float, float]] = deque(maxlen=50)

        self._last_pool_full_log_time = 0.0

        if self._max_pool_size > 0:
            logger.info(f"Flow control enabled: max_pool_size={self._max_pool_size} samples")
        else:
            logger.warning("Flow control disabled: max_pool_size=0 (unlimited generation)")

    # -- Public API ----------------------------------------------------------

    # 后台 task 补输入，前台持续提交和收集；停机时取消补货并等待已经发出的任务。
    # 没有 pending task 时 sleep 让出事件循环，避免空转阻止 stop 等 RPC 被处理。
    async def run(self) -> None:
        """Main loop: refill runs in background, dispatch → collect in foreground."""
        logger.info(
            f"AsyncInferenceManager starting ({len(self._engines)} engines, "
            f"max_concurrent={self._engines.max_concurrent})"
        )

        refill_task = asyncio.create_task(self._continuous_refill())

        while self._running:
            await self._dispatch_ready_batches()

            if not self._pending_tasks:
                await asyncio.sleep(0.01)
                continue

            await self._collect_completed()

        refill_task.cancel()
        try:
            await refill_task
        except asyncio.CancelledError:
            pass
        await self._drain()
        logger.info("AsyncInferenceManager stopped")

    # 不断尝试补充输入并短暂让出控制；实际是否需要补货由 _refill_buffer 的水位判断。
    async def _continuous_refill(self) -> None:
        """Background task: continuously refill prompt buffer."""
        while self._running:
            await self._refill_buffer()
            await asyncio.sleep(0.001)  # yield to event loop

    # 设置协作式退出标志，不取消已提交的引擎工作；run 最后会 drain 在途结果。
    def stop(self) -> None:
        """Signal the main loop to stop after current iteration."""
        self._running = False

    # 取走并重置回包统计，另外附加滑动窗口的容量估计。
    def flush_metrics(self) -> dict:
        """Return aggregated metrics and reset the buffer."""
        metrics = self._metrics.flush()
        if self._enable_perf_metrics:
            metrics.update(self._compute_perf_metrics())
        return metrics

    # 用近期各批耗时总和估计单并发槽吞吐，再乘总并发额度估计容量。
    # 多个批次时间可以重叠，因此这个值是容量估计，不应解读为全流水线真实墙钟吞吐。
    def _compute_perf_metrics(self) -> dict:
        """Compute throughput metrics over a sliding time window."""
        if not self._batch_times:
            return {}

        cutoff = time.time() - self._perf_window_seconds
        recent = [
            (samples, elapsed)
            for samples, elapsed, finished_at in self._batch_times
            if finished_at > cutoff
        ]
        if not recent:
            return {}

        total_samples = sum(samples for samples, _ in recent)
        total_time = sum(elapsed for _, elapsed in recent)
        if total_time <= 0.001:
            return {}

        per_slot_rate = total_samples / total_time
        return {
            "perf/infer_capacity": per_slot_rate * self._engines.max_concurrent,
            "perf/infer_batch_time": total_time / len(recent),
        }

    # 汇总本地缓冲、运行状态和远端样本池，便于区分输入缺货、引擎繁忙和输出积压。
    async def get_status(self) -> dict:
        """Get current status of inference manager."""
        status = {
            "mode": f"distributed_engines ({len(self._engines)})",
            "prompt_buffer_size": len(self._prompt_buffer),
            "running": self._running,
            "pending_tasks": len(self._pending_tasks),
            "engine_semaphore_available": getattr(self._engines.semaphore, "_value", None),
        }
        status.update(await self.get_pool_status())
        return status

    # 查询控制器样本池；查询异常用 -1 表示未知，避免报告为正常空池。
    async def get_pool_status(self) -> dict:
        """Get current sample pool status."""
        try:
            pool_size = await self.controller.get_pool_size.remote()
        except Exception:
            pool_size = -1

        return {
            "max_pool_size": self._max_pool_size,
            "current_pool_size": pool_size,
            "pool_usage_percent": (
                int(pool_size * 100 / self._max_pool_size)
                if self._max_pool_size > 0 and pool_size >= 0
                else 0
            ),
            "flow_control_enabled": self._max_pool_size > 0,
        }

    # -- Main loop internals -------------------------------------------------

    # 在输入够一批且任务额度未满时创建任务；数据源暂空时也允许不足一批的尾部发出。
    # 每次提交前检查输出池容量，使训练端慢下来时生产端也减速。source_dry 表示本轮取不到，非永久 EOF。
    async def _dispatch_ready_batches(self) -> None:
        """Dispatch batches from buffer, up to the engine concurrency limit."""
        while (
            len(self._prompt_buffer) >= self._batch_size
            and len(self._pending_tasks) < self._engines.max_concurrent
        ):
            await self._await_pool_capacity()
            entries = self._take_batch(self._batch_size)
            task = asyncio.create_task(self._dispatch_batch(entries))
            self._pending_tasks.add(task)

        if (
            self._source_dry
            and 0 < len(self._prompt_buffer) < self._batch_size
            and len(self._pending_tasks) < self._engines.max_concurrent
        ):
            await self._await_pool_capacity()
            entries = self._take_batch(self._batch_size)
            self._pending_tasks.add(asyncio.create_task(self._dispatch_batch(entries)))

    # 等待任意一个任务完成后立即转发其结果，不等待本轮所有引擎，从而减少快任务被慢任务拖住。
    async def _collect_completed(self) -> None:
        """Wait for at least one dispatch to finish and forward results."""
        # FIRST_COMPLETED 只要求至少一个完成，返回 done 可能包含多个已完成任务。
        # 后续将仍在运行的集合保留，下一次循环继续提交/收集，使快慢引擎都能持续工作。
        done, self._pending_tasks = await asyncio.wait(
            self._pending_tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            await self._forward_results(task.result())

    # 关闭阶段丢弃尚未提交的输入，等待已提交任务完成并转发结果；不再等待样本池空位。
    async def _drain(self) -> None:
        # Currently drain is only called on shutdown, so we don't need to await pool.
        if self._prompt_buffer:
            logger.info(
                f"Drain: discarding {len(self._prompt_buffer)} buffered prompts on shutdown"
            )
            self._prompt_buffer.clear()
        if self._pending_tasks:
            done, _ = await asyncio.wait(self._pending_tasks, return_when=asyncio.ALL_COMPLETED)
            for task in done:
                await self._forward_results(task.result())
            self._pending_tasks.clear()

    # -- Buffer management ---------------------------------------------------

    # 本地输入低于水位才远程补货；输出池已满时连输入预取也暂停。
    # 一次请求 max(缺口, refill_size) 条，减少逐条 RPC；未取到时设置 source_dry 以触发尾批提交。
    async def _refill_buffer(self) -> None:
        """Fetch prompts from controller if buffer is below watermark.

        Requests enough prompts to fill the buffer up to the watermark in one
        call, avoiding multiple round-trips to the controller.
        """
        if len(self._prompt_buffer) >= self._buffer_low_watermark:
            return

        if self._max_pool_size > 0:
            pool_size = await self.controller.get_pool_size.remote()
            if pool_size >= self._max_pool_size:
                now = time.monotonic()
                if now - self._last_pool_full_log_time >= 2.0:
                    self._last_pool_full_log_time = now
                    logger.debug(
                        f"Skipping prompt fetch: pool full ({pool_size}/{self._max_pool_size})"
                    )
                return

        need = self._buffer_low_watermark - len(self._prompt_buffer)
        fetch_size = max(need, self._refill_size)
        entries = await self.controller.get_prompts.remote(fetch_size)
        if entries:
            self._prompt_buffer.extend(entries)
            self._source_dry = False
            logger.debug(f"Fetched {len(entries)} prompts, buffer_size={len(self._prompt_buffer)}")
        else:
            self._source_dry = True

    # 从本地队头最多移走 n 条，不会在这里等待更多输入。
    def _take_batch(self, n: int) -> list[InferenceInput]:
        """Pop up to n entries from internal buffer."""
        n = min(n, len(self._prompt_buffer))
        return [self._prompt_buffer.popleft() for _ in range(n)]

    # -- Backpressure --------------------------------------------------------

    # 协作式轮询输出池，满时暂停新提交，停机标志可终止等待。
    # 阈值按样本条数而非精确字节检查；在途请求仍可能完成，所以它不是严格的存储字节上限。
    async def _await_pool_capacity(self) -> None:
        """Block until sample pool has capacity."""
        if self._max_pool_size <= 0:
            return

        pool_size = await self.controller.get_pool_size.remote()
        if pool_size < self._max_pool_size:
            return

        wait_start = time.time()
        last_log_time = wait_start

        logger.warning(
            f"Sample pool full, pausing generation: pool_size={pool_size}/{self._max_pool_size}"
        )

        while pool_size >= self._max_pool_size and self._running:
            await asyncio.sleep(MOONCAKE_BACKPRESSURE_POLL_INTERVAL)
            pool_size = await self.controller.get_pool_size.remote()

            now = time.time()
            if now - last_log_time >= MOONCAKE_BACKPRESSURE_LOG_INTERVAL:
                wait_duration = now - wait_start
                logger.info(
                    f"Still waiting for pool capacity (waited {wait_duration:.1f}s): "
                    f"pool_size={pool_size}/{self._max_pool_size}"
                )
                last_log_time = now

        wait_duration = time.time() - wait_start
        if not self._running:
            logger.info(f"Pool wait aborted (shutdown requested) after {wait_duration:.1f}s")
            return
        logger.info(
            f"Pool capacity available after {wait_duration:.1f}s, "
            f"pool_size={pool_size}/{self._max_pool_size}"
        )

    # -- Engine dispatch -----------------------------------------------------

    # 将条目整理为引擎统一参数。提前分词时把 ID 列表放入 Ray；延迟分词时传格式化文本。
    # 延迟模式不能沿用旧 token 的 mask，因为后端 tokenizer/多模态展开可能改变位置；mask 应以真实 ID 为准。
    def _prepare_engine_inputs(self, entries: list[InferenceInput]) -> dict:
        """Pack entries into engine.generate() kwargs."""
        packed_loss_mask_list = [e.packed_loss_mask for e in entries]
        data_ids = [e.data_id for e in entries]
        multimodal_inputs = [e.multimodal_inputs for e in entries]
        has_multimodal = any(m is not None for m in multimodal_inputs)

        if self._defer_tokenization:
            input_ids_ref = None
            packed_loss_mask_list = None
            assert all(e.formatted_prompt is not None for e in entries), (
                "formatted_prompt is required when defer_tokenization is True"
            )
            formatted_prompts = [e.formatted_prompt for e in entries]
        else:
            # 这次 Ray 对象传输的是较小的输入 token 列表；teacher 的大 hidden states 仍走 Mooncake。
            # 两种传输通道用途不同，因此“控制面只传元数据”不应被误读成任何地方都不使用 Ray 传 tensor。
            input_ids_ref = ray.put([e.input_ids for e in entries])
            formatted_prompts = None

        return {
            "data_ids": data_ids,
            "input_ids_ref": input_ids_ref,
            "packed_loss_mask_list": packed_loss_mask_list,
            "formatted_prompts": formatted_prompts,
            "return_last_hidden_states": self._return_hidden_states,
            "return_logits": not self._return_hidden_states,
            "multimodal_inputs": multimodal_inputs if has_multimodal else None,
        }

    # 在并发额度内选择引擎并等待 generate 或 generate_with_decode，按输入顺序配对返回结果。
    # 结果条数不一致时整批标为异常；actor 死亡通知控制器并停止管理器，普通异常则变为样本级失败。
    async def _dispatch_batch(
        self, entries: list[InferenceInput]
    ) -> list[tuple[InferenceInput, dict | Exception]]:
        """Send a batch to the next engine and return (entry, output) pairs."""
        async with self._engines.semaphore:
            kwargs = self._prepare_engine_inputs(entries)
            engine = self._engines.pick()

            try:
                if self._enable_perf_metrics:
                    t0 = time.time()
                generate_method = (
                    engine.generate_with_decode if self._train_with_decode else engine.generate
                )
                outputs = await generate_method.remote(
                    data_id=kwargs["data_ids"],
                    input_ids_ref=kwargs["input_ids_ref"],
                    packed_loss_mask_list=kwargs["packed_loss_mask_list"],
                    formatted_prompts=kwargs["formatted_prompts"],
                    return_last_hidden_states=kwargs["return_last_hidden_states"],
                    return_logits=kwargs["return_logits"],
                    multimodal_inputs=kwargs["multimodal_inputs"],
                )
                if self._enable_perf_metrics:
                    now = time.time()
                    self._batch_times.append((len(entries), now - t0, now))
                if len(outputs) != len(entries):
                    logger.error(
                        f"Engine returned {len(outputs)} results for "
                        f"{len(entries)} entries (expected equal)"
                    )
                    err = ValueError(f"output count mismatch: {len(outputs)} vs {len(entries)}")
                    return [(entry, err) for entry in entries]
                return list(zip(entries, outputs, strict=True))
            # 进程死亡与单批失败不能采用同一种处理：actor 死亡说明生产者能力已丢失，
            # 必须让控制器把错误传给训练主循环，否则训练 rank 可能永远等待下一份样本。
            except RayActorError as e:
                logger.critical(f"Engine actor died, terminating inference manager: {e}")
                self._running = False
                await self.controller.set_inference_error.remote(str(e))
                raise
            except Exception as e:
                import traceback

                logger.error(f"Engine dispatch failed: {e}\n{traceback.format_exc()}")
                return [(entry, e) for entry in entries]

    # -- Result processing ---------------------------------------------------

    # 过滤缺少 Mooncake key 的无效输出，合并输入和引擎元数据，并保留真实 shape/dtype/mask。
    # 同名 metadata 由引擎值覆盖；packed_loss_mask 若引擎提供则优先于预处理值。
    def _parse_engine_output(self, entry: InferenceInput, output: dict) -> InferenceOutput | None:
        """Convert engine output dict to InferenceOutput."""
        if not isinstance(output, dict) or "mooncake_key" not in output:
            logger.debug(
                f"Skipping invalid engine output for data_id={entry.data_id}: {type(output)}"
            )
            return None

        self._metrics.record(output)

        metadata = dict(entry.metadata or {})
        metadata.update(output.get("metadata", {}) or {})

        return InferenceOutput(
            data_id=entry.data_id,
            mooncake_key=output["mooncake_key"],
            tensor_shapes=output.get("tensor_shapes", {}),
            tensor_dtypes=output.get("tensor_dtypes", {}),
            packed_loss_mask=output.get("packed_loss_mask", entry.packed_loss_mask),
            metadata=metadata,
        )

    # 跳过异常和无效回包，将有效结果一次性送入控制器并返回成功条数；此处没有自动重试。
    async def _forward_results(self, results: list[tuple[InferenceInput, Any | Exception]]) -> int:
        """Parse results and forward to controller. Returns success count."""
        inference_results = []

        for entry, result in results:
            if isinstance(result, Exception):
                continue

            inference_result = self._parse_engine_output(entry, result)
            if inference_result is not None:
                inference_results.append(inference_result)

        if inference_results:
            await self.controller.push_inference_results.remote(inference_results)
            logger.debug(f"Forwarded {len(inference_results)} results to controller")

        return len(inference_results)
