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

# 中文导读：训练流水线的样本账本。负责“哪条数据该给谁”，不负责 teacher forward。
# prompt_buffer 保存待推理输入，sample_pool 保存已完成的 Mooncake key；train_queues 送往训练 rank。
# 这些容器中的大模型输出只是元数据，真正的 hidden states 留在 Mooncake。
# 本文件还保留整个已处理数据集以便下一 epoch 重排；这是与仅保存输出 key 不同的一层状态。
# 注意模块原有 batch 公式概述；当前 USP 的派生规则应以 train_entry._resolve_batch_size 为准。

"""
Async Training Controller for decoupled inference and training.

Data flow:
  load_dataset(args) → Stored Dataset → Prompt Buffer → Inference Manager
    → Sample Pool → Train Queues → Training Workers
  (Stored dataset is retained for epoch reloads and vocab mapping computation.)

Controller manages the tokenized dataset (for epoch reloads and vocab mapping),
prompt metadata, and mooncake keys. Actual inference tensor data is stored in
mooncake; the controller only tracks keys and byte sizes for backpressure.

Batch Size Design:
  micro_batch_size                   # Samples per GPU per dispatch (user config)
  per_dp_rank_batch_size             # = micro_batch_size * sp_size (derived)
  dispatch_batch_size            # = per_dp_rank_batch_size * dp_size (samples per dispatch)
  global_batch_size                  # = dispatch_batch_size * accumulation_steps (per optimizer step)

  Example with micro_batch_size=2, sp_size=1, dp_size=4, accumulation_steps=2:
    - per_dp_rank_batch_size = 2 * 1 = 2
    - dispatch_batch_size = 2 * 4 = 8
    - global_batch_size = 8 * 2 = 16

  Data flow per optimizer step (with accumulation_steps=2):
    1. Controller dispatches 8 samples per dispatch, 2 dispatches per optimizer step
    2. Each DP rank receives 2 samples per dispatch (4 total per optimizer step)
    3. Each train actor calls train_from_queue(num_batches=accumulation_steps)
    4. Forward/backward for each micro-batch, optimizer step after last one
"""

import copy
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import ray
from ray.util.queue import Queue

from torchspec.data.utils import length_grouped_order
from torchspec.training.data_fetcher import TrainSample
from torchspec.utils.logging import logger
from torchspec.utils.memory import estimate_tensor_bytes
from torchspec.utils.types import InferenceInput, InferenceOutput

_estimate_bytes = estimate_tensor_bytes


# 滑动窗口统计样本吞吐；总计数与短窗口速率分开，既能看瞬时趋势也能计算全程平均。
@dataclass
class SpeedMonitor:
    """Tracks throughput over a sliding time window."""

    window_seconds: float = 10.0
    _events: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _total_count: int = 0

    # 在锁内添加 (时间, 数量)，累计总数并裁掉过期事件，避免并发更新破坏队列。
    def record(self, count: int = 1) -> None:
        """Record count entries at current time."""
        now = time.time()
        with self._lock:
            self._events.append((now, count))
            self._total_count += count
            self._prune_old_events(now)

    # 移除窗口左侧的旧事件；只调整用于测速的历史，不减少全程总计数。
    def _prune_old_events(self, now: float) -> None:
        """Remove events outside the window."""
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    # 用窗口内样本数除以从最老事件到当前的时间；空窗口或不足 1ms 返回零，避免无穷大。
    def get_speed(self) -> float:
        """Get current speed in entries/sec over the window."""
        now = time.time()
        with self._lock:
            self._prune_old_events(now)
            if not self._events:
                return 0.0

            window_count = sum(count for _, count in self._events)
            oldest_time = self._events[0][0]
            elapsed = now - oldest_time
            if elapsed < 0.001:
                return 0.0
            return window_count / elapsed

    # 返回启动以来累计处理的条数；不是当前窗口中的样本数。
    def get_total_count(self) -> int:
        """Get total count since start."""
        return self._total_count


# Ray actor 是远程常驻对象；保存数据、分发元数据和统计容量，使多个生产者共享一个调度视图。
@ray.remote
class AsyncTrainingController:
    """Central controller for async training pipeline.

    Responsibilities:
      - Loads and stores tokenized datasets for training and eval
      - Computes vocab mappings for draft model pruning
      - Manages prompt buffer (samples waiting for inference)
      - Manages sample pool (completed inferences waiting for training)
      - Dispatches batches to per-DP training queues when pool is full
      - Monitors inference and training throughput
    """

    # 分别初始化待推理输入、训练结果池、评估结果池及队列。
    # queue_count=DP×SP：同一 DP 组内的 SP rank 收到同一个样本基础 key 的描述，
    # 但当前 USP 由生产侧预先切片，每个消费者实际读取自己的 base_usp{rank} 分片 key。
    # _pool_bytes 只跟踪尚在训练 sample_pool 中的估算字节数，不等于整个 Mooncake 的实时物理占用。
    def __init__(self, args, dp_size: int):
        self.args = args
        self.dp_size = dp_size
        self.sp_size = (
            getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
            if getattr(args, "attention_backend", None) == "usp"
            else 1
        )
        self.queue_count = dp_size * self.sp_size

        self.prompt_buffer: deque[InferenceInput] = deque()
        self._prompt_lock = threading.Lock()

        self.sample_pool: deque[InferenceOutput] = deque()
        self._pool_lock = threading.Lock()
        # 字节账本对应“尚未 dispatch 的训练结果”。张量从池里移到训练队列后仍可能在存储中。
        # 所以调度背压必须结合在途量和客户端容量理解，不能把这个字段当作显存 allocator 的计量。
        self._pool_bytes = 0
        self._sample_bytes: dict[str, int] = {}

        # 此处没有为 Ray 队列指定 maxsize；生产节奏主要由外层 loop 的 queued_batches/prefetch 控制。
        # 输出池阈值与队列条数不是同一约束，样本离开 pool 后仍可能在队列里等待很久。
        self.train_queues = [Queue() for _ in range(self.queue_count)]

        # Eval: separate pool and queues so eval data never mixes with training
        self.eval_pool: deque[InferenceOutput] = deque()
        self._eval_pool_lock = threading.Lock()
        self._eval_data_ids: set[str] = set()
        self._eval_expected_count: int = 0
        self._eval_dispatched_samples: int = 0
        self.eval_queues = [Queue() for _ in range(self.queue_count)]

        self.batch_id = 0
        # DP=2、每 DP 两条时，一次 dispatch 消耗 4 条独立样本；SP=2 虽有 4 个队列，
        # 不再把 dispatch 乘 2。若梯度累积为 3，外层需 3 次 dispatch 才凑齐一次 optimizer step。
        self.dispatch_batch_size = args.per_dp_rank_batch_size * dp_size
        self.eval_dispatch_batch_size = dp_size
        self._data_id_counter = 0

        self._stored_dataset: list | None = None
        self._stored_eval_dataset: list | None = None
        self._dataset_epoch: int = 0
        self._dataset_seed: int = getattr(args, "seed", 42)
        self._shuffle_dataset: bool = getattr(args, "shuffle_dataset", True)
        self._length_group_size: int = getattr(args, "length_group_size", 32)

        self._start_time = time.time()
        self._inference_monitor = SpeedMonitor(window_seconds=10.0)
        self._training_monitor = SpeedMonitor(window_seconds=10.0)
        self._last_dispatch_log_time = 0.0
        self._inference_error: str | None = None

    # 生成此控制器生命周期内递增的样本标识，用于关联异步请求与结果；它不是数据内容的哈希。
    def _generate_data_id(self) -> str:
        self._data_id_counter += 1
        return f"data_{self._data_id_counter}"

    # ─────────────────────────────────────────────────────────────
    # Dataset Loading
    # ─────────────────────────────────────────────────────────────

    # 把原始条目统一成 InferenceInput 并追加到待推理队列，返回加入数量。
    # 已有 input_ids 却无 loss mask 时要求开启动态 mask，否则无法知道哪些 token 应监督。
    # 不在这里生成 teacher 特征；这里只准备后续推理的输入。
    def add_dataset(self, dataset: list) -> int:
        with self._prompt_lock:
            for sample in dataset:
                if isinstance(sample, dict):
                    data_id = sample.get("data_id") or self._generate_data_id()
                    input_ids = sample.get("input_ids")
                    packed_loss_mask = sample.get("packed_loss_mask")
                    if (
                        input_ids is not None
                        and packed_loss_mask is None
                        and not getattr(self.args, "dynamic_loss_mask", False)
                    ):
                        raise ValueError(
                            f"packed_loss_mask is required when input_ids is provided "
                            f"(data_id={data_id}). Enable dynamic_loss_mask or use "
                            f"defer_tokenization=True to match engine-produced token IDs."
                        )
                    entry = InferenceInput(
                        data_id=data_id,
                        prompt=sample.get("prompt", sample),
                        input_ids=input_ids,
                        packed_loss_mask=packed_loss_mask,
                        formatted_prompt=sample.get("formatted_prompt"),
                        metadata=sample.get("metadata", {}),
                        multimodal_inputs=sample.get("multimodal_inputs"),
                    )
                else:
                    entry = InferenceInput(
                        data_id=self._generate_data_id(),
                        prompt=sample,
                    )
                self.prompt_buffer.append(entry)
            return len(dataset)

    # 按 train/eval 选择数据来源；offline 只读取清单 ID 和长度，在线模式执行对话预处理。
    # 评估用浅拷贝 args 改路径，避免意外把后续训练的数据路径改成验证集。
    def _load_dataset_split(self, args, split: str) -> list:
        """Load one split from either replay records or conversation data."""
        if split not in ("train", "eval"):
            raise ValueError(f"Unknown dataset split: {split!r}")

        if getattr(args, "inference_engine_type", None) == "offline":
            from torchspec.offline.dataset import OfflineDataset

            dataset = OfflineDataset(args.offline_data_path)
            return [
                {
                    "data_id": str(row["data_id"]),
                    "metadata": {"offline_replay": True},
                    "seq_len": row["seq_len"],
                }
                for row in dataset.rows(split)
            ]

        data_path = (
            args.train_data_path if split == "train" else getattr(args, "eval_data_path", None)
        )
        if not data_path:
            return []

        from torchspec.data.dataset import load_conversation_dataset

        dataset_args = args
        if split == "eval":
            dataset_args = copy.copy(args)
            dataset_args.train_data_path = data_path
            if getattr(args, "eval_prompt_key", None):
                dataset_args.prompt_key = args.eval_prompt_key
        return load_conversation_dataset(dataset_args)

    # 载入并保留训练数据；空数据立即失败，防止后续消费者一直等待不可能产生的样本。
    def load_dataset(self, args) -> int:
        """Load and store the training dataset for later epochs."""
        self._stored_dataset = self._load_dataset_split(args, "train")
        if not self._stored_dataset:
            if getattr(args, "inference_engine_type", None) == "offline":
                raise ValueError("Offline dataset has no train samples")
            raise ValueError(
                f"Training dataset is empty after processing. "
                f"Check train_data_path='{args.train_data_path}', "
                f"max_seq_length={getattr(args, 'max_seq_length', None)}, "
                f"and chat_template settings."
            )
        logger.info(f"Controller loaded dataset: {len(self._stored_dataset)} samples")
        return len(self._stored_dataset)

    # 复制保存的数据，再以 seed+epoch 打乱、按长度局部分组并跳过已消费前缀。
    # 随机种子可复现顺序，但异步在途结果没有完整保存，因此 resume 的数据进度仅尽力恢复。
    def _prepare_dataset(self, skip: int = 0) -> list:
        """Return dataset for the current epoch, optionally shuffled and length-grouped.

        When shuffle is enabled the ordering is deterministic from
        (seed + epoch), so resume can reconstruct the same epoch ordering
        and approximately skip samples consumed by completed optimizer
        steps.  This is best-effort only because async prompt/result
        buffers may still contain in-flight samples.
        """
        data = list(self._stored_dataset)
        if self._shuffle_dataset:
            import random

            rng = random.Random(self._dataset_seed + self._dataset_epoch)
            rng.shuffle(data)

        # 先打乱再局部长度分组：保留随机组顺序，同时让相邻 batch 的序列长度更接近。
        # 如果全局排序，训练前后见到的内容长度分布会系统变化；这里避免这种额外课程效应。
        data = length_grouped_order(data, self._length_group_size)

        if skip > 0:
            skip = min(skip, len(data))
            data = data[skip:]

        shuffle_tag = (
            f"seed {self._dataset_seed}+{self._dataset_epoch}"
            if self._shuffle_dataset
            else "shuffle disabled"
        )
        logger.info(
            f"Prepared dataset ({len(data)} samples, {shuffle_tag}, "
            f"length group {self._length_group_size}" + (f", skipped {skip})" if skip > 0 else ")")
        )
        return data

    # 设定当前 epoch 后提交重排后的数据；skip 用于恢复到已完成 optimizer 步数对应的近似位置。
    def submit_training_dataset(self, epoch: int = 0, skip: int = 0) -> int:
        """Submit the stored training dataset to the prompt buffer for inference.

        Args:
            epoch: Current epoch number (for deterministic shuffle seed).
            skip: Number of samples to skip from the start (for resume mid-epoch).
        """
        assert self._stored_dataset is not None, "No stored dataset to submit"
        self._dataset_epoch = epoch
        return self.add_dataset(self._prepare_dataset(skip=skip))

    # 递增 epoch 并重新投递保留的数据集，不重新从磁盘 tokenize。
    def reload_dataset(self) -> int:
        """Re-add the stored dataset to the prompt buffer (epoch reload)."""
        assert self._stored_dataset is not None, "No stored dataset to reload"
        self._dataset_epoch += 1
        # 追加新 epoch，不清空此前的 prompt、结果池或训练队列；异步晚到结果可能跨 epoch 边界。
        # 因而 _dataset_epoch 描述投递顺序，不能充当每个已完成回包的精确训练 epoch 标签。
        return self.add_dataset(self._prepare_dataset())

    # 将验证集截到 DP 数量的整数倍，确保每次评估分发时各数据副本都有样本。
    # 例如 10 条、DP=4，只保留 8 条；这是本实现的组批取舍，并非验证集本身只有 8 条。
    def load_eval_dataset(self, args) -> int:
        """Load eval dataset on the controller and store it. Returns size (0 if none)."""
        raw_dataset = self._load_dataset_split(args, "eval")
        raw_count = len(raw_dataset)
        # Truncate to a multiple of dp_size so every dispatch is a full batch
        usable = (raw_count // self.dp_size) * self.dp_size
        if usable < raw_count:
            logger.info(
                f"Eval dataset truncated from {raw_count} to {usable} samples "
                f"(dp_size={self.dp_size})"
            )
        self._stored_eval_dataset = raw_dataset[:usable]
        count = len(self._stored_eval_dataset)
        logger.info(f"Controller loaded eval dataset: {count} samples")
        return count

    # 返回已载入训练集长度；尚未载入时显式报错，不能用零冒充正常空集。
    def get_dataset_size(self) -> int:
        if self._stored_dataset is None:
            raise RuntimeError(
                "get_dataset_size() called but no dataset has been loaded. "
                "Call load_dataset() first."
            )
        return len(self._stored_dataset)

    # 返回截断后的验证集大小，未配置验证数据时返回零供上层关闭评估。
    def get_eval_dataset_size(self) -> int:
        return len(self._stored_eval_dataset) if self._stored_eval_dataset is not None else 0

    # 从已 tokenized 的训练数据统计草稿词表映射；纯文本延迟分词模式没有 input_ids，不能直接统计。
    def compute_vocab_mapping(self, target_vocab_size: int, draft_vocab_size: int) -> tuple:
        """Generate vocab mapping on the controller using the stored dataset.

        Requires the dataset to have been loaded with defer_tokenization=False,
        since vocab mapping needs input_ids.
        """
        from torchspec.data.preprocessing import generate_vocab_mapping

        assert self._stored_dataset is not None, "No stored dataset for vocab mapping"
        assert "input_ids" in self._stored_dataset[0], (
            "compute_vocab_mapping requires input_ids in dataset. "
            "Set defer_tokenization=False to enable tokenization."
        )
        return generate_vocab_mapping(
            prompts=self._stored_dataset,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
        )

    # ─────────────────────────────────────────────────────────────
    # Interface for Inference Manager
    # ─────────────────────────────────────────────────────────────

    # 从待推理队列头部取走最多 num_prompts 条；取走后由推理管理器负责它们的后续处理。
    def get_prompts(self, num_prompts: int) -> list[InferenceInput]:
        """Inference manager gets prompts with data_ids.

        Args:
            num_prompts: Maximum number of prompts to fetch.

        Returns:
            List of InferenceInput objects.
        """
        with self._prompt_lock:
            entries = []
            for _ in range(min(num_prompts, len(self.prompt_buffer))):
                # pop 是所有权交接而非借阅；后续引擎失败没有自动把这个 entry 放回控制器。
                entries.append(self.prompt_buffer.popleft())
            return entries

    # 按 data_id 将结果分流到训练或评估池；训练项按 shape/dtype 估算存储字节。
    # 这里只保存 key 和元数据，不下载张量。返回本次添加训练结果后的池字节，只有评估结果时为零。
    def push_inference_results(self, results: list[InferenceOutput]) -> int:
        """Inference sends back (data_id, mooncake_key) pairs.

        Controller stores the keys and tracks exact bytes for backpressure.
        Eval results (identified by data_id) are routed to the eval pool.

        Args:
            results: List of InferenceOutput containing data_id, mooncake_key,
                    tensor_shapes, and tensor_dtypes.

        Returns:
            Current pool bytes after adding results. This allows inference manager
            to implement Mooncake backpressure.
        """
        eval_results = []
        train_results = []
        # 分流依据是提交 eval 时登记的 ID 集合，而非 key 名称或张量 shape；
        # 若训练与评估复用同一 data_id（离线回放保留原 ID），调用方须保证命名空间不歧义。
        for result in results:
            if result.data_id in self._eval_data_ids:
                eval_results.append(result)
            else:
                train_results.append(result)

        if eval_results:
            with self._eval_pool_lock:
                self.eval_pool.extend(eval_results)

        pool_bytes = 0
        if train_results:
            with self._pool_lock:
                for result in train_results:
                    sample_bytes = estimate_tensor_bytes(
                        result.tensor_shapes or {},
                        result.tensor_dtypes or {},
                    )
                    # 估算使用逻辑 shape/dtype，不逐个枚举 PP 层键、重复 ids、USP 重叠片段或复制副本。
                    # 此表也不去重：协议要求每次发布 key 唯一，重复 key 会覆盖条目但仍再次增加总账。
                    self._sample_bytes[result.mooncake_key] = sample_bytes
                    self._pool_bytes += sample_bytes
                self.sample_pool.extend(train_results)
                pool_bytes = self._pool_bytes

        self._inference_monitor.record(len(results))
        return pool_bytes

    # 报告仍在控制器中的输入数；不包含推理管理器本地缓冲和引擎中正在执行的请求。
    def get_prompt_buffer_size(self) -> int:
        """Get current size of prompt buffer."""
        return len(self.prompt_buffer)

    # ─────────────────────────────────────────────────────────────
    # Interface for Training
    # ─────────────────────────────────────────────────────────────

    # 将队列句柄交给训练组；SP 开启时队列实际按 DP×SP 展开。
    def get_train_queues(self) -> list[Queue]:
        """Get the per-DP training queues."""
        return self.train_queues

    # 把等待分发的训练和评估条目一起计数，供推理侧限流。
    # 已发到训练队列但尚未删除的张量不在此计数里，因此它不是所有存储对象的总数。
    def get_pool_size(self) -> int:
        """Total mooncake-resident samples (training + eval) for backpressure.

        Always includes eval pool so that backpressure accounts for mooncake
        segment capacity used by eval data.  Without this, eval data occupies
        mooncake outside backpressure awareness and the segment overflows.
        """
        train_size = len(self.sample_pool)
        with self._eval_pool_lock:
            return train_size + len(self.eval_pool)

    # 在锁内返回训练结果池的字节账本；评估池和在途/已分发对象不包含在这个值中。
    def get_pool_bytes(self) -> int:
        """Get current bytes in sample pool (for Mooncake backpressure)."""
        with self._pool_lock:
            return self._pool_bytes

    # ─────────────────────────────────────────────────────────────
    # Dispatch Logic
    # ─────────────────────────────────────────────────────────────

    # 记录推理 actor 的致命错误，让下一次训练分发抛出异常，而非无限等待新样本。
    def set_inference_error(self, msg: str) -> None:
        """Called by the inference manager when a fatal error occurs."""
        self._inference_error = msg

    # 只有结果池足够组成一个完整 dispatch 才取出样本并发到各训练队列。
    # 取出时扣减池字节仅表示控制权交给消费者，并不代表 Mooncake 内存此时已释放。
    # 返回 False 表示暂时缺数据；致命推理错误则抛异常，二者必须区分。
    def try_dispatch_batch(self) -> bool:
        """Try to dispatch one batch to training queues.

        Only dispatches when sample pool has enough samples (>= dispatch_batch_size).
        Dispatches TrainSample objects that MooncakeDataFetcher can consume.
        Subtracts bytes from pool tracking when dispatching.

        Returns:
            True if a batch was dispatched, False if not enough samples.

        Raises:
            RuntimeError: If the inference manager has reported a fatal error.
        """
        # 先检查致命错误，再查看是否凑齐；即便池中还有完整旧批次，也不会继续分发掩盖引擎死亡。
        if self._inference_error is not None:
            raise RuntimeError(f"Inference engine failed: {self._inference_error}")

        with self._pool_lock:
            pool_size = len(self.sample_pool)
            now = time.time()
            should_log = (now - self._last_dispatch_log_time) >= 2.0
            if pool_size < self.dispatch_batch_size:
                if should_log:
                    self._last_dispatch_log_time = now
                    logger.debug(
                        f"try_dispatch_batch: pool_size={pool_size} < dispatch_batch_size={self.dispatch_batch_size}, not dispatching"
                    )
                return False

            if should_log:
                self._last_dispatch_log_time = now
                logger.debug(
                    f"try_dispatch_batch: pool_size={pool_size} >= dispatch_batch_size={self.dispatch_batch_size}, dispatching batch {self.batch_id}"
                )

            # 从 pool 按到达顺序取一整批，之后才在批内做 DP 负载平衡；
            # 因此全局训练顺序受异步完成顺序影响，并不等于最初数据集的排列顺序。
            batch_results = []
            for _ in range(self.dispatch_batch_size):
                result = self.sample_pool.popleft()
                sample_bytes = self._sample_bytes.pop(result.mooncake_key, 0)
                # 这一减法只是结果池出账。真正 remove 要等消费者安全拿到所需数据后执行，
                # 当前 USP 消费者读取并清理各自分片 key；这里的基础 key 出账不代表所有分片均已消费。
                self._pool_bytes -= sample_bytes
                batch_results.append(result)

        # 出池和入各队列不是事务：这里失败时可能只有部分 rank 收到样本，池字节已扣减。
        # 锁外发送缩短池临界区，但没有自动回滚；True 仅在所有 put 正常返回后才成立。
        self._dispatch_to_queues(batch_results, self.train_queues)

        self._training_monitor.record(self.dispatch_batch_size)
        logger.debug(
            f"Dispatched batch {self.batch_id} with {self.dispatch_batch_size} samples "
            f"to {self.dp_size} queues at t={time.time():.3f}"
        )
        self.batch_id += 1
        return True

    # 从 input_ids 的最后一维读取序列长度；缺少形状时以零参与负载排序。
    @staticmethod
    def _seq_len(result: InferenceOutput) -> int:
        shapes = result.tensor_shapes or {}
        ids_shape = shapes.get("input_ids")
        return ids_shape[-1] if ids_shape else 0

    # 按序列长度从长到短，把样本放到当前总长度最小且未满的 DP rank。
    # 例如长度 [8,7,2,1]、DP=2、每 rank 两条，分为 [8,1] 与 [7,2]，都承担 9 个 token。
    # 同时限制条数，避免某 rank 提前耗尽 batch；不规则数量或每 rank 至多一条时回退轮询。
    def _partition_results(self, results: list[InferenceOutput]) -> list[list[InferenceOutput]]:
        """Partition InferenceOutputs across DP ranks.

        When each rank receives more than one sample per dispatch, uses
        longest-first greedy bin-packing with a per-rank capacity cap so
        that ranks see similar total sequence load. Falls back to
        round-robin when there is at most one sample per rank (e.g. eval
        dispatch, or training with per_dp_rank_batch_size=1) or when
        len(results) is not divisible by dp_size — preserving the old
        round-robin behavior for irregular batch sizes.
        """
        partitions: list[list[InferenceOutput]] = [[] for _ in range(self.dp_size)]
        if self.dp_size <= 1 or len(results) <= self.dp_size or len(results) % self.dp_size != 0:
            for i, result in enumerate(results):
                partitions[i % self.dp_size].append(result)
            return partitions

        # 贪心平衡同时满足“每 rank 条数相等”和“累计 token 尽量接近”。
        # 前者保证每个消费者取到预期数量，后者减少同步训练中等待最慢 rank 的时间。
        # 贪心负载用 token 数代理算量，不估计 attention 的二次项、padding、图像展开等真实成本。
        # 同负载时 min 先选 rank 小者；同长度样本保留 sorted 的稳定输入顺序。
        capacity = len(results) // self.dp_size
        loads = [0] * self.dp_size
        for result in sorted(results, key=self._seq_len, reverse=True):
            min_rank = min(
                (r for r in range(self.dp_size) if len(partitions[r]) < capacity),
                key=lambda r: loads[r],
            )
            partitions[min_rank].append(result)
            loads[min_rank] += self._seq_len(result)
        return partitions

    # 将推理结果包装成 TrainSample；不同 DP rank 收不同分组，同一 DP 的 SP rank 收同一描述。
    # 复制的是基础 key/shape/mask 等元数据，不是在 Python 中复制大张量。
    # 当前 USP 生产侧已切片，消费者按 rank 派生分片 key，并重建对应的 mask/position 信息。
    def _dispatch_to_queues(
        self,
        batch_results: list[InferenceOutput],
        queues: list[Queue],
    ) -> None:
        """Partition results across DP ranks and push TrainSamples into queues."""
        partitioned = self._partition_results(batch_results)
        for dp_rank, results in enumerate(partitioned):
            for result in results:
                metadata = getattr(result, "metadata", {}) or {}
                last_turn_loss_only = metadata.get("has_thinking")
                sample = TrainSample(
                    mooncake_key=result.mooncake_key,
                    tensor_shapes=result.tensor_shapes,
                    tensor_dtypes=result.tensor_dtypes,
                    packed_loss_mask=result.packed_loss_mask,
                    last_turn_loss_only=last_turn_loss_only,
                    metadata=metadata,
                    data_id=result.data_id,
                )
                if self.sp_size > 1 and len(queues) == self.queue_count:
                    # 把同一个 TrainSample 放到同一 DP 组各独立队列，相当于广播元数据。
                    # 若只让多个 rank 消费同一条共享队列，一次 get 会取走条目，其他 rank 并不会自动得到副本。
                    # DP=2、SP=2：DP0 的 A/B 放进队列 0 和 1；DP1 的 C/D 放进 2 和 3。
                    # 四条逻辑样本产生八次元数据 put，但同组 SP rank 协作处理相同样本，不是八条独立数据。
                    start = dp_rank * self.sp_size
                    for rank in range(start, start + self.sp_size):
                        queues[rank].put(sample)
                else:
                    queues[dp_rank].put(sample)

    # 把单条服务端采集结果复用批量入池路径，保持统计与 train/eval 分流规则一致。
    def push_inference_sample(self, sample: InferenceOutput) -> int:
        """Add a single inference sample to the training pool.

        This method is used by OnlineServingController to add samples
        captured from the inference engine's /generate API.

        Args:
            sample: InferenceOutput containing mooncake_key and tensor metadata.

        Returns:
            Current pool bytes after adding the sample.
        """
        return self.push_inference_results([sample])

    # ─────────────────────────────────────────────────────────────
    # Eval Pipeline
    # ─────────────────────────────────────────────────────────────

    # 构建评估输入并记录 ID 集合以识别异步回包。
    # 在线 ID 加 eval_ 前缀；offline 必须保留清单中的原始 ID 才能找到磁盘样本。
    def _build_eval_entries(self, dataset: list) -> list[InferenceInput]:
        eval_entries: list[InferenceInput] = []
        for sample in dataset:
            if isinstance(sample, dict):
                raw_id = sample.get("data_id") or self._generate_data_id()
                data_id = (
                    str(raw_id)
                    if getattr(self.args, "inference_engine_type", None) == "offline"
                    else f"eval_{raw_id}"
                )
                self._eval_data_ids.add(data_id)
                entry = InferenceInput(
                    data_id=data_id,
                    prompt=sample.get("prompt", sample),
                    input_ids=sample.get("input_ids"),
                    packed_loss_mask=sample.get("packed_loss_mask"),
                    formatted_prompt=sample.get("formatted_prompt"),
                    metadata=sample.get("metadata", {}),
                    multimodal_inputs=sample.get("multimodal_inputs"),
                )
            else:
                data_id = f"eval_{self._generate_data_id()}"
                self._eval_data_ids.add(data_id)
                entry = InferenceInput(data_id=data_id, prompt=sample)
            eval_entries.append(entry)
        return eval_entries

    # 将验证集切片插入待推理队列前端；先 reversed 再 extendleft，保持切片内部原有顺序。
    # 小块提交使训练端有机会及时取走结果，避免整个验证集同时占满张量存储。
    def submit_eval_chunk(self, start: int, end: int) -> int:
        """Submit a slice of the stored eval dataset for inference."""
        assert self._stored_eval_dataset is not None, "No stored eval dataset"
        chunk = self._stored_eval_dataset[start:end]
        if not chunk:
            return 0

        if start == 0:
            self._eval_expected_count = len(self._stored_eval_dataset)
            self._eval_dispatched_samples = 0

        eval_entries = self._build_eval_entries(chunk)

        # 优先级只作用于控制器尚未被取走的输入：manager 本地预取和引擎在途的训练批次不会被抢占。
        with self._prompt_lock:
            self.prompt_buffer.extendleft(reversed(eval_entries))
        logger.info(
            f"Eval: submitted chunk [{start}:{end}] "
            f"({len(chunk)} samples, total_expected={self._eval_expected_count})"
        )
        return len(chunk)

    # 在评估锁内读取等待分发的结果数量，用于判断缓存构建进度。
    def get_eval_pool_size(self) -> int:
        with self._eval_pool_lock:
            return len(self.eval_pool)

    # 返回独立的评估队列句柄，避免验证样本被训练迭代器消费。
    def get_eval_queues(self) -> list[Queue]:
        return self.eval_queues

    # 凑齐一个评估 dispatch 后出池，按同一分区规则发送并累计已分发数；不足则返回 False。
    def try_dispatch_eval_batch(self) -> bool:
        """Dispatch one eval batch from the pool if enough samples are available."""
        # eval 一次每 DP 一条，再广播给该 DP 的 SP 队列；不同于训练的 per_dp_rank_batch_size。
        # 本方法没有训练分发中的 _inference_error 检查，调用方 eval.generate_eval_cache 另有空闲超时。
        bs = self.eval_dispatch_batch_size
        with self._eval_pool_lock:
            if len(self.eval_pool) < bs:
                return False
            batch_results = [self.eval_pool.popleft() for _ in range(bs)]

        self._dispatch_to_queues(batch_results, self.eval_queues)
        self._eval_dispatched_samples += bs
        logger.debug(
            f"Eval: dispatched batch ({self._eval_dispatched_samples}/"
            f"{self._eval_expected_count} samples)"
        )
        return True

    # 确认预期样本已到齐且没有完整批次滞留，再清空评估 ID 和计数状态。
    # 这里只清控制器跟踪状态，张量的下载、缓存和释放由消费者/存储层处理。
    def finalize_eval_dispatch(self) -> None:
        """Assert all eval batches were dispatched, then clean up tracking state.

        Raises AssertionError if not all expected samples have arrived or
        undispatched full batches remain in the pool.
        """
        with self._eval_pool_lock:
            arrived = self._eval_dispatched_samples + len(self.eval_pool)
            pool_remaining = len(self.eval_pool)

        assert self._eval_expected_count > 0 and arrived >= self._eval_expected_count, (
            f"finalize_eval_dispatch called before all samples arrived "
            f"(arrived={arrived}, expected={self._eval_expected_count})"
        )
        assert pool_remaining < self.eval_dispatch_batch_size, (
            f"finalize_eval_dispatch called with undispatched full batches "
            f"(pool={pool_remaining}, batch_size={self.eval_dispatch_batch_size})"
        )

        with self._eval_pool_lock:
            if pool_remaining > 0:
                logger.info(
                    f"Eval: dropping {pool_remaining} leftover samples that didn't fill a batch"
                )
                # 这里只丢元数据，不调用 Mooncake remove；若真有尾部对象，其远端回收不能由此推断。
                self.eval_pool.clear()

        self._eval_data_ids.clear()
        self._eval_expected_count = 0
        self._eval_dispatched_samples = 0
        logger.info("Eval: dispatch finalized, tracking state cleared")

    # ─────────────────────────────────────────────────────────────
    # Status and Monitoring
    # ─────────────────────────────────────────────────────────────

    # 报告控制器本地排队状态，适合判断数据是否不足；不代表引擎内部的实时请求数。
    def get_status(self) -> dict:
        """Get current status of controller."""
        return {
            "prompt_buffer_size": len(self.prompt_buffer),
            "sample_pool_size": len(self.sample_pool),
            "batches_dispatched": self.batch_id,
            "dispatch_batch_size": self.dispatch_batch_size,
        }

    # 同时输出短窗口速率、全程样本数和平均速率；training_speed 统计的是分发速度，不是 GPU 实测速度。
    def get_speeds(self) -> dict:
        """Get current throughput speeds in entries/sec."""
        elapsed = time.time() - self._start_time
        return {
            "inference_speed": round(self._inference_monitor.get_speed(), 2),
            "training_speed": round(self._training_monitor.get_speed(), 2),
            "inference_total": self._inference_monitor.get_total_count(),
            "training_total": self._training_monitor.get_total_count(),
            "elapsed_seconds": round(elapsed, 1),
            "avg_inference_speed": round(
                self._inference_monitor.get_total_count() / max(elapsed, 0.001), 2
            ),
            "avg_training_speed": round(
                self._training_monitor.get_total_count() / max(elapsed, 0.001), 2
            ),
        }

    # 将队列状态和吞吐统计合并为一次远程返回，减少主循环多次 RPC。
    def get_full_status(self) -> dict:
        """Get complete status including speeds."""
        status = self.get_status()
        status.update(self.get_speeds())
        return status

    # 给每个训练队列放入 None 结束哨兵，通知阻塞读取的消费者停止；不强行杀掉 actor。
    def shutdown(self) -> None:
        """Signal training workers to stop by sending None to queues."""
        # FIFO 中已有样本仍排在 None 前面，消费者取到哨兵才退出；这不是清空队列或取消远程 GET。
        # 此方法只遍历训练队列，不向 eval_queues 发送停止信号。
        for q in self.train_queues:
            q.put(None)
        logger.info("Controller shutdown: sent stop signals to training queues")
