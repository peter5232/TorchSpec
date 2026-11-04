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

# 提供遵守对象租约 TTL 的后台延迟删除器，使用有界新任务队列和失败重试列表。
# 当前 EagleMooncakeStore 消费清理采用 force=True 直接删除；不要把本模块误认为每次 get 的必经步骤。

"""Deferred deletion manager for Mooncake objects with lease TTL."""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from torchspec.utils.logging import logger


# 一组待删除键的重试状态：入队时间用于租约等待，最近尝试时间用于重试间隔，attempts 限制总尝试次数。
@dataclass
class DeleteTask:
    """A deferred deletion task."""

    keys: List[str]  # List of Mooncake keys to delete
    base_key: str  # Original base key for logging
    enqueue_time: float  # When this was queued
    attempts: int = 0  # Number of delete attempts
    last_attempt_time: float = 0.0  # Last attempt timestamp
    max_attempts: int = 3  # Maximum retry attempts


# 把可能需要等待 TTL 的删除移出数据消费线程，并累计任务级别的成功、重试和放弃计数。
class DeferredDeleteManager:
    """Manages deferred deletions with TTL awareness and retry logic.

    Mooncake objects have a lease TTL during which they cannot be deleted.
    This manager queues deletions and processes them after the TTL expires.
    """

    # 创建有界队列、统计锁和停止事件，然后启动守护线程；构造对象已经产生后台执行副作用。
    def __init__(
        self,
        store: Any,
        ttl_seconds: float = 5.0,
        ttl_buffer_seconds: float = 0.5,
        check_interval: float = 1.0,
        max_queue_size: int = 10000,
        retry_interval: float = 2.0,
    ):
        """Initialize the deferred delete manager.

        Args:
            store: Mooncake store instance with remove() method
            ttl_seconds: Mooncake lease TTL duration
            ttl_buffer_seconds: Extra buffer to add to TTL before deletion
            check_interval: How often to check for deletions (seconds)
            max_queue_size: Maximum pending deletions
            retry_interval: Wait time between retries (seconds)
        """
        self.store = store
        self.ttl_seconds = ttl_seconds
        self.ttl_buffer_seconds = ttl_buffer_seconds
        self.check_interval = check_interval
        self.retry_interval = retry_interval

        self.delete_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.retry_queue: List[DeleteTask] = []  # Tasks to retry

        # Statistics
        self.stats = {
            "enqueued": 0,
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "retried": 0,
            "abandoned": 0,  # Exceeded max attempts
        }
        self.stats_lock = threading.Lock()

        # Background thread
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        logger.info(
            "Manager started: ttl=%.1fs, buffer=%.1fs, check_interval=%.1fs",
            ttl_seconds,
            ttl_buffer_seconds,
            check_interval,
        )

    # 以当前时间创建任务并非阻塞入队；队列满返回 False，调用方据此知道回收请求未被接受。
    def enqueue_delete(
        self,
        keys: List[str],
        base_key: str,
        max_attempts: int = 3,
    ) -> bool:
        """Queue a deletion task.

        Args:
            keys: List of Mooncake keys to delete
            base_key: Original base key for logging
            max_attempts: Maximum retry attempts

        Returns:
            True if enqueued successfully, False if queue is full
        """
        task = DeleteTask(
            keys=keys,
            base_key=base_key,
            enqueue_time=time.time(),
            max_attempts=max_attempts,
        )

        try:
            self.delete_queue.put_nowait(task)
            with self.stats_lock:
                self.stats["enqueued"] += 1
            logger.debug("Enqueued deletion for %s (%d keys)", base_key, len(keys))
            return True
        except queue.Full:
            logger.error("Queue full! Cannot enqueue deletion for %s", base_key)
            return False

    # 每轮先看重试，再从新队列取任务；空队列正常轮询，意外异常记录后短暂退避。
    def _worker_loop(self):
        """Background worker that processes deletions."""
        logger.info("Worker thread started")

        while not self._stop_event.is_set():
            try:
                # Process retry queue first (these are already past TTL)
                self._process_retry_queue()

                # Check for new tasks from main queue
                try:
                    task = self.delete_queue.get(timeout=self.check_interval)
                    self._process_task(task)
                except queue.Empty:
                    pass  # No tasks, continue loop

            except Exception as e:
                logger.error("Worker error: %s", e, exc_info=True)
                time.sleep(1.0)  # Brief pause on error

        logger.info("Worker thread stopped")

    # 从入队时刻算起等待 TTL 加缓冲时间，再执行删除；这不是查询对象创建时间，而是保守的额外等待。
    def _process_task(self, task: DeleteTask):
        """Process a single deletion task."""
        current_time = time.time()
        elapsed = current_time - task.enqueue_time
        wait_time = self.ttl_seconds + self.ttl_buffer_seconds

        # Check if TTL has expired
        if elapsed < wait_time:
            # Not ready yet, wait
            remaining = wait_time - elapsed
            logger.debug(
                "Waiting %.1fs for TTL on %s",
                remaining,
                task.base_key,
            )
            time.sleep(remaining)

        # Attempt deletion
        self._attempt_delete(task)

    # 检查每项距离上次尝试的时间，只重试达到间隔者，未到期的留待后续轮询。
    def _process_retry_queue(self):
        """Process tasks in the retry queue."""
        if not self.retry_queue:
            return

        current_time = time.time()
        still_waiting = []

        for task in self.retry_queue:
            # Check if enough time has passed since last attempt
            if current_time - task.last_attempt_time >= self.retry_interval:
                self._attempt_delete(task)
            else:
                still_waiting.append(task)

        self.retry_queue = still_waiting

    # 逐键删除并把已不存在当成功；只把失败键保留到下次任务，达到最大尝试数后记录放弃。
    def _attempt_delete(self, task: DeleteTask):
        """Attempt to delete keys for a task."""
        task.attempts += 1
        task.last_attempt_time = time.time()

        # 尝试次数按任务计数，而不是按每个键计数；第二次及以后同时增加 retried。
        with self.stats_lock:
            self.stats["attempted"] += 1
            if task.attempts > 1:
                self.stats["retried"] += 1

        logger.debug(
            "Attempting delete for %s (attempt %d/%d)",
            task.base_key,
            task.attempts,
            task.max_attempts,
        )

        removed_count = 0
        failed_keys = []

        # 一组字段可能部分已删除，逐键统计并把失败原因保留下来；-704 说明目标已经不存在，无需再重试。
        for key in task.keys:
            try:
                result = self.store.remove(key)

                # remove() returns 0 on success, -704 (OBJECT_NOT_FOUND) if
                # the key is already gone — both count as success.
                if result is None or result == 0 or result == -704:
                    removed_count += 1
                    logger.debug("Successfully removed %s", key)
                else:
                    logger.warning(
                        "Remove returned error code %s for %s",
                        result,
                        key,
                    )
                    failed_keys.append((key, f"error_code_{result}"))

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.warning("Exception removing %s: %s", key, error_msg)
                failed_keys.append((key, error_msg))

        # Check results
        # 只有本次剩余键全部完成才算任务成功；部分失败会缩小下一次需要删除的键集合。
        if removed_count == len(task.keys):
            # Complete success
            with self.stats_lock:
                self.stats["succeeded"] += 1
            logger.debug(
                "Successfully deleted %s (all %d keys removed, attempt %d)",
                task.base_key,
                len(task.keys),
                task.attempts,
            )
        elif failed_keys:
            # Some or all keys failed
            if task.attempts < task.max_attempts:
                # Retry
                logger.warning(
                    "Partial failure for %s: %d/%d keys failed, will retry (attempt %d/%d)",
                    task.base_key,
                    len(failed_keys),
                    len(task.keys),
                    task.attempts,
                    task.max_attempts,
                )
                # Update task to only retry failed keys
                # 已经删除的字段不重复入队，避免大样本某一个键失败时反复操作其他字段。
                task.keys = [key for key, _ in failed_keys]
                self.retry_queue.append(task)
            else:
                # Exceeded max attempts
                with self.stats_lock:
                    self.stats["failed"] += 1
                    # 达到上限就放弃当前任务，不无限占用后台线程；遗留对象的清理只能依赖后续管理措施。
                    self.stats["abandoned"] += 1
                logger.error(
                    "Abandoned deletion for %s after %d attempts: %d keys still failed",
                    task.base_key,
                    task.attempts,
                    len(failed_keys),
                )
                # Log failed keys
                for key, error in failed_keys[:5]:  # Show first 5
                    logger.error("Failed key: %s (%s)", key, error)

    # 持锁复制统计字典，调用者不会拿到可直接修改内部计数的对象。
    def get_stats(self) -> Dict[str, int]:
        """Get deletion statistics."""
        with self.stats_lock:
            return self.stats.copy()

    # 将新任务队列和重试列表长度相加，提供近似待回收积压量，不是尚未删除的对象字节数。
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.delete_queue.qsize() + len(self.retry_queue)

    # 设置退出信号并最多等待十秒线程结束，记录最终统计；这不是保证清空所有剩余删除任务的 drain。
    def stop(self):
        """Stop the worker thread."""
        logger.info("Stopping worker thread...")
        self._stop_event.set()
        self._worker_thread.join(timeout=10.0)

        # Log final statistics
        stats = self.get_stats()
        logger.info(
            "Final stats: enqueued=%d, succeeded=%d, failed=%d, abandoned=%d, queue_size=%d",
            stats["enqueued"],
            stats["succeeded"],
            stats["failed"],
            stats["abandoned"],
            self.get_queue_size(),
        )

    # 析构时仅在后台线程存在且仍运行时请求停止，处理可能的部分初始化状态。
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "_worker_thread") and self._worker_thread.is_alive():
            self.stop()
