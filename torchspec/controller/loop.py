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

# 中文导读：主循环把异步样本生产与同步 optimizer step 连接起来。
# 熟悉 PyTorch 的读者可把 train_from_queue 看成远程执行的“取若干 micro-batch、backward、step”。
# 一个 dispatch 只是各 DP rank 的一次供料，accumulation_steps 次供料才形成一次 optimizer step。
# 评估、保存、权重同步都在这个边界安排，防止把尚未完成的梯度累积误认为一个已完成训练步。

"""Pipeline training loop: main training loop with sync training and async inference."""

import re
import shutil
import tempfile
import time
from pathlib import Path

import ray
import wandb
from tqdm import tqdm

from torchspec.controller.eval import (
    generate_eval_cache,
    run_eval,
    setup_eval,
    update_checkpoint_eval_meta,
)
from torchspec.utils.logging import get_tb_writer, logger


# 输出训练日志并发送到已启用的跟踪后端；推理与训练指标可以使用不同横轴步数。
def _write_training_metrics(metrics: dict, train_step: int, inference_step: int) -> None:
    loss = metrics.get("train/avg_loss")
    if isinstance(loss, (int, float)):
        logger.info(f"TRAIN_STEP step={train_step} loss={loss:.6f}")

    if getattr(wandb, "run", None) is not None:
        wandb.log(metrics)

    tb_writer = get_tb_writer()
    if tb_writer is not None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                scalar_step = inference_step if key.startswith("inference/") else train_step
                tb_writer.add_scalar(key, value, scalar_step)


# 仅在 decode 训练且到同步间隔时导出草稿并通知各引擎从磁盘加载。
# 训练进程与推理进程不共享 nn.Module，训练端更新参数不会自动更新服务端副本。
# 当前实现使用本地临时目录，多节点可见性受文件系统部署约束；finally 保证清理临时目录。
def _maybe_sync_draft_weights(args, completed_steps, train_group, inference_engines):
    """Sync draft model weights to inference engines (decode mode only)."""
    weight_sync_enabled = getattr(args, "decode_weight_sync_enabled", False)
    weight_sync_interval = getattr(args, "decode_weight_sync_interval", 500)
    if not (
        getattr(args, "train_with_decode", False)
        and weight_sync_enabled
        and inference_engines
        and weight_sync_interval > 0
        and completed_steps > 0
        and completed_steps % weight_sync_interval == 0
    ):
        return

    # NOTE: uses local tmp dir; for multi-node, ensure output_dir is on shared filesystem.
    tmp_dir = tempfile.mkdtemp(prefix="draft_weight_sync_")
    try:
        logger.info(f"Step {completed_steps}: saving draft model to {tmp_dir}")
        train_group.save_draft_model_for_serving(tmp_dir)

        logger.info(f"Step {completed_steps}: updating {len(inference_engines)} engine(s)")
        update_results = ray.get(
            [engine.update_weights_from_disk.remote(tmp_dir) for engine in inference_engines]
        )
        for i, res in enumerate(update_results):
            log_fn = logger.info if res.get("success") else logger.warning
            log_fn(f"Engine {i}: success={res.get('success')}, message={res.get('message')}")
    except Exception:
        logger.exception(f"Step {completed_steps}: weight sync failed, skipping")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temp dir {tmp_dir}")


# interval>0 才启用周期保存，再检查整除；零或负数明确关闭这项调度。
def _is_save_interval_step(step: int, interval: int) -> bool:
    return interval > 0 and step % interval == 0


# 只按 iter_数字 目录的步数排序删除较早项，保留非该命名的元数据文件。
# 保留 best_* 指示文件不等于保证它指向的旧 iter 目录也保留，读者需区分指针和实际权重。
def _cleanup_old_checkpoints(checkpoint_dir: str | None, max_checkpoints: int) -> None:
    """Delete old checkpoints, keeping only the most recent `max_checkpoints`.

    Checkpoint directories are named ``iter_NNNNNNN`` where N is the step number.
    The ``latest_checkpointed_iteration.txt`` and ``best_*`` files are preserved.
    """
    if not checkpoint_dir or max_checkpoints <= 0:
        return

    base_dir = Path(checkpoint_dir).expanduser()
    if not base_dir.exists():
        return

    # Find all iter_* directories, sorted by step number
    iter_dirs = sorted(
        (d for d in base_dir.iterdir() if d.is_dir() and re.match(r"iter_\d+", d.name)),
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )

    if len(iter_dirs) <= max_checkpoints:
        return

    # Delete oldest checkpoints, keep the newest max_checkpoints
    to_delete = iter_dirs[: len(iter_dirs) - max_checkpoints]
    for old_dir in to_delete:
        logger.info(f"Removing old checkpoint: {old_dir}")
        try:
            shutil.rmtree(old_dir)
        except OSError as e:
            logger.warning(f"Failed to remove old checkpoint {old_dir}: {e}")


# 先停推理管理器并等待它退出，再关闭引擎和 Mooncake master；各阶段异常仅记日志。
# 这样清理失败不会遮住原来的训练异常，且尽量避免存储先退出而引擎仍在写入。
def _safe_training_cleanup(
    args, inference_manager, inference_future, inference_engines=None
) -> None:
    """Best-effort teardown for inference manager and mooncake master actor."""
    if inference_manager is not None:
        try:
            ray.get(inference_manager.stop.remote())
        except Exception as exc:
            logger.warning(f"Failed to stop inference manager: {exc}")
        if inference_future is not None:
            try:
                ray.get(inference_future)
            except Exception as exc:
                logger.warning(
                    f"Inference manager run loop exited with error during cleanup: {exc}"
                )

    if inference_engines:
        logger.info(f"Shutting down {len(inference_engines)} inference engine(s)...")
        shutdown_refs = []
        for engine in inference_engines:
            try:
                shutdown_refs.append(engine.shutdown.remote())
            except Exception as exc:
                logger.warning(f"Failed to initiate engine shutdown: {exc}")
        for ref in shutdown_refs:
            try:
                ray.get(ref, timeout=30)
            except Exception as exc:
                logger.warning(f"Engine shutdown timed out or failed: {exc}")

    mooncake_master_actor = getattr(args, "_mooncake_master_actor", None)
    if mooncake_master_actor is not None:
        try:
            ray.get(mooncake_master_actor.shutdown.remote(), timeout=10)
        except Exception as exc:
            logger.warning(f"Failed to shutdown mooncake master actor: {exc}")


# 先构建/加载固定评估特征，再提交训练数据，随后按 optimizer 步数运行。
# 外层计步，内层尝试凑齐并预取 dispatch；当前一步已备齐时即开算，不必等所有预取槽填满。
# 每次同时向全部训练 actor 发请求后 ray.get 等待：DP rank 必须共同参与梯度通信。
# 恢复位置按已完成步数推算，异步队列尾部不能保证逐样本精确恢复。
def training_loop(
    args,
    controller,
    inference_manager,
    train_group,
    inference_engines=None,
    dataset_size=None,
    eval_dataset_size=None,
):
    """Run the training loop with sync training and async inference.

    Training is synchronous - waits for each step to complete.
    Inference runs in background, continuously producing data.

    Each optimizer step (with draft_accumulation_steps dispatches):
      1. Controller dispatches per_dp_rank_batch_size * dp_size samples, accumulation_steps times
      2. Each DP rank receives per_dp_rank_batch_size * accumulation_steps samples total
      3. train_from_queue(num_batches=accumulation_steps) processes all micro-batches
      4. Optimizer steps after the last micro-batch

    completed_steps counts optimizer steps (consistent with lr_total_steps).

    Args:
        args: Configuration arguments.
        controller: AsyncTrainingController ray actor handle (dataset already loaded).
        inference_manager: AsyncInferenceManager ray actor handle.
        train_group: Training group with set_train_queues method.
        dataset_size: Number of training samples. If None, queried from controller.
        eval_dataset_size: Number of eval samples. If None, queried from controller. 0 means no eval.
    """
    if dataset_size is None:
        dataset_size = ray.get(controller.get_dataset_size.remote())
    if dataset_size <= 0:
        raise ValueError(
            f"Training dataset size is {dataset_size}. "
            f"Ensure controller.load_dataset() was called before run_training_loop()."
        )
    if eval_dataset_size is None:
        eval_dataset_size = ray.get(controller.get_eval_dataset_size.remote())

    # ── Eval setup ──────────────────────────────────────────────
    eval_state = setup_eval(controller, train_group, args, eval_dataset_size)

    # Inference engine is alive. Run eval hs generation first.
    if eval_state.eval_enabled and not eval_state.eval_cache_loaded:
        generate_eval_cache(controller, train_group, eval_state)

    eval_interval = eval_state.eval_interval
    eval_enabled = eval_state.eval_enabled
    best_eval_score = eval_state.best_eval_score

    dp_size = (
        getattr(args, "dp_size", None) or args.training_num_nodes * args.training_num_gpus_per_node
    )
    num_steps = args.num_train_steps
    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    # steps_per_epoch in optimizer steps, pre-computed in auto_calculate_training_steps
    steps_per_epoch = getattr(args, "steps_per_epoch", dataset_size // args.global_batch_size)
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    num_epochs = getattr(args, "num_epochs", 1)

    # Submit training data AFTER eval hs generation so that training prompts don't
    # leak into the inference pipeline during eval.
    # Resume is best-effort: completed optimizer steps determine epoch/skip, but
    # async prompt/result buffers can still lose or replay a small tail.
    start_step = ray.get(train_group._actor_handlers[0].get_global_step.remote())
    resume_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
    resume_skip = (start_step % steps_per_epoch) * args.global_batch_size if start_step > 0 else 0
    ray.get(controller.submit_training_dataset.remote(epoch=resume_epoch, skip=resume_skip))

    logger.info(
        f"Starting: num_steps={num_steps}, num_epochs={num_epochs}, "
        f"steps_per_epoch={steps_per_epoch}, global_batch_size={args.global_batch_size}, "
        f"accumulation_steps={accumulation_steps}, "
        f"dp_size={dp_size}, per_dp_rank_batch_size={args.per_dp_rank_batch_size}"
    )

    enable_perf = getattr(args, "enable_perf_metrics", True)
    # 即便配置不额外预取，也至少凑齐一次 optimizer step 所需的累积批次。
    # 例如累积=3、prefetch_depth=1，实际至少备 3 个 dispatch，才能让所有 rank 完成本次更新。
    prefetch_batches = max(accumulation_steps, getattr(args, "prefetch_depth", 0))

    completed_steps = start_step
    current_epoch = completed_steps // steps_per_epoch + 1
    steps_in_current_epoch = completed_steps % steps_per_epoch
    if start_step > 0:
        logger.info(f"Resuming from step {start_step} (epoch {current_epoch})")
    dispatch_attempts = 0
    consecutive_failures = 0
    queued_batches = 0
    last_saved_step: int | None = None
    previous_dispatch_wait: float | None = None
    progress = tqdm(total=num_steps, desc="Training", unit="step", initial=start_step)
    while completed_steps < num_steps:
        # 限制预取不跨本轮 epoch/总训练步数边界，避免结束后多余样本长期留在队列。
        # 计数单位在这里仍是 optimizer step，下一行乘 accumulation_steps 才得到 dispatch 数。
        remaining_steps = min(
            num_steps - completed_steps,
            steps_per_epoch - steps_in_current_epoch,
        )
        target_queued_batches = min(
            prefetch_batches,
            remaining_steps * accumulation_steps,
        )
        if enable_perf:
            t_dispatch = time.time()
        status = None
        while queued_batches < target_queued_batches:
            dispatch_attempts += 1

            dispatched = ray.get(controller.try_dispatch_batch.remote())
            if dispatched:
                queued_batches += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                # 已经有完整一步的数据时，停止追求额外预取并立即开算。
                # 否则为了填满预取窗口而继续等待，会让本来能训练的 GPU 空闲，削弱流水线重叠收益。
                if queued_batches >= accumulation_steps:
                    # The current step can run while inference refills the pool.
                    target_queued_batches = queued_batches
                    consecutive_failures = 0
                    continue

                # Only fetch status when needed for logging or reload decision
                if dispatch_attempts % 100 == 0 or consecutive_failures >= 500:
                    status = ray.get(controller.get_full_status.remote())

                if dispatch_attempts % 100 == 0 and status is not None:
                    logger.debug(
                        f"Epoch {current_epoch}, Step {completed_steps}: "
                        f"dispatch failed {dispatch_attempts} times "
                        f"(consecutive={consecutive_failures}), "
                        f"pool_size={status['sample_pool_size']}, "
                        f"need={status['dispatch_batch_size']}"
                    )

                should_reload = False
                if steps_in_current_epoch >= steps_per_epoch:
                    should_reload = True
                elif (
                    consecutive_failures >= 500
                    and (completed_steps > 0 or queued_batches > 0)
                    and status is not None
                    and status["sample_pool_size"] < status["dispatch_batch_size"]
                    and status.get("prompt_buffer_size", 0) == 0
                ):
                    logger.warning(
                        f"Pool insufficient for dispatch "
                        f"(pool_size={status['sample_pool_size']}, "
                        f"need={status['dispatch_batch_size']}, "
                        f"{queued_batches}/{accumulation_steps} batches queued, "
                        f"{steps_in_current_epoch}/{steps_per_epoch} steps in epoch). "
                        f"Reloading dataset."
                    )
                    should_reload = True

                if should_reload:
                    if completed_steps < num_steps:
                        current_epoch += 1
                        steps_in_current_epoch = 0
                        consecutive_failures = 0
                        logger.info(f"Dataset exhausted, reloading (epoch {current_epoch})...")
                        ray.get(controller.reload_dataset.remote())
                    else:
                        logger.info("Max steps reached, stopping")
                        break

                time.sleep(0.01)
        else:
            # The current optimizer step is fully queued.
            if enable_perf:
                dispatch_wait = time.time() - t_dispatch
            else:
                dispatch_wait = 0.0

            train_futures = [
                actor.train_from_queue.remote(
                    step=completed_steps,
                    num_batches=accumulation_steps,
                )
                for actor in train_group._actor_handlers
            ]

            # 每个 actor 内部做前向/反向和同步通信；driver 在这里等待整组完成。
            # 先向所有 rank 提交再统一等待，不能在创建列表时逐个等待，否则首个 rank 可能卡在 collective。
            train_results = ray.get(train_futures)
            queued_batches -= accumulation_steps
            completed_steps += 1

            # Log metrics from training (use rank 0's metrics - they're already all-reduced)
            metrics = train_results[0] if train_results and train_results[0] else {}
            # 训练侧为减少 CUDA 同步可能延后一轮才取回指标，因此日志所属 step 可能不是刚完成的 step。
            # 调度等待时间也要对齐这份指标，避免把前一步 compute 和后一步 wait 拼成虚假的性能分解。
            metric_step = int(metrics.pop("_metrics_step", completed_steps))
            metric_dispatch_wait = (
                previous_dispatch_wait
                if metric_step != completed_steps and previous_dispatch_wait is not None
                else dispatch_wait
            )
            previous_dispatch_wait = dispatch_wait
            if metrics:
                # Add step counters for wandb x-axis (required in shared mode)
                metrics["train/step"] = metric_step
                metrics["inference/step"] = completed_steps

                # Add inference metrics (e2e_latency, spec metrics, etc.)
                inference_metrics = ray.get(inference_manager.flush_metrics.remote())
                metrics.update(inference_metrics)

                if enable_perf:
                    metrics["perf/dispatch_wait"] = metric_dispatch_wait
                    step_time = metrics.get("perf/step_time", 0)
                    if step_time > 0:
                        metrics["perf/train_capacity"] = args.global_batch_size / step_time

                    if metric_step % 5 == 0 or metric_step <= 5:
                        data_t = metrics.get("perf/data_time", 0)
                        compute_t = metrics.get("perf/compute_time", 0)
                        fwd_t = metrics.get("perf/forward_time", 0)
                        bwd_t = metrics.get("perf/backward_time", 0)
                        opt_t = metrics.get("perf/optimizer_time", 0)
                        logger.info(
                            f"TIMING step={metric_step}: "
                            f"step={step_time:.3f}s "
                            f"data={data_t:.3f}s "
                            f"compute={compute_t:.3f}s "
                            f"[fwd={fwd_t:.3f}s bwd={bwd_t:.3f}s opt={opt_t:.3f}s] "
                            f"dispatch={metric_dispatch_wait:.3f}s"
                        )

                _write_training_metrics(metrics, metric_step, completed_steps)

            # ── Eval at explicit interval (if configured) ─────────
            # Skip if a checkpoint save is about to run (it will eval anyway)
            save_due = _is_save_interval_step(completed_steps, args.save_interval)
            if eval_interval > 0 and completed_steps % eval_interval == 0 and not save_due:
                run_eval(completed_steps, train_group, eval_enabled)

            steps_in_current_epoch += 1
            dispatch_attempts = 0

            status = ray.get(controller.get_full_status.remote())
            postfix = {
                "loss": f"{metrics.get('train/avg_loss', 0):.3f}",
                "acc": f"{metrics.get('train/avg_acc', 0):.3f}",
                "acc_len": f"{metrics.get('train/simulated_acc_len', 0):.2f}",
                "thru": f"{status['inference_speed']:.1f}",
            }
            if enable_perf:
                postfix["I"] = f"{metrics.get('perf/infer_capacity', 0):.1f}"
                postfix["T"] = f"{metrics.get('perf/train_capacity', 0):.1f}"
                postfix["wait"] = f"{dispatch_wait:.1f}s"
                postfix["pool"] = status["sample_pool_size"]
            postfix["epoch"] = f"{current_epoch}/{num_epochs}"
            # Set postfix before update so tqdm emits only one line
            progress.set_postfix(postfix, refresh=False)
            progress.update(1)

            if _is_save_interval_step(completed_steps, args.save_interval):
                eval_metrics = run_eval(completed_steps, train_group, eval_enabled)
                logger.info(f"Saving checkpoint at step {completed_steps}...")
                train_group.save_model(completed_steps)
                last_saved_step = completed_steps
                best_eval_score = update_checkpoint_eval_meta(
                    args.checkpoint_dir, completed_steps, eval_metrics, best_eval_score
                )
                max_ckpts = getattr(args, "max_checkpoints", 0)
                if max_ckpts > 0:
                    _cleanup_old_checkpoints(args.checkpoint_dir, max_ckpts)

            _maybe_sync_draft_weights(args, completed_steps, train_group, inference_engines)

            # Check if epoch completed
            if steps_in_current_epoch >= steps_per_epoch:
                logger.info(
                    f"Epoch {current_epoch} completed ({steps_in_current_epoch} steps). "
                    f"Total steps: {completed_steps}/{num_steps}"
                )

                if (
                    args.save_per_epoch
                    and args.checkpoint_dir
                    and last_saved_step != completed_steps
                ):
                    eval_metrics = run_eval(completed_steps, train_group, eval_enabled)
                    logger.info(
                        f"Saving checkpoint at end of epoch {current_epoch} "
                        f"(step {completed_steps})..."
                    )
                    train_group.save_model(completed_steps)
                    last_saved_step = completed_steps
                    best_eval_score = update_checkpoint_eval_meta(
                        args.checkpoint_dir, completed_steps, eval_metrics, best_eval_score
                    )
                    max_ckpts = getattr(args, "max_checkpoints", 0)
                    if max_ckpts > 0:
                        _cleanup_old_checkpoints(args.checkpoint_dir, max_ckpts)

                if completed_steps < num_steps:
                    current_epoch += 1
                    steps_in_current_epoch = 0
                    logger.info(f"Dataset exhausted, reloading (epoch {current_epoch})...")
                    ray.get(controller.reload_dataset.remote())
                else:
                    logger.info("Max steps reached")
                    break
            continue

        # Inner while broke (max steps reached during reload), break outer loop
        break

    final_train_results = train_group.flush_pending_train_metrics()
    if not isinstance(final_train_results, (list, tuple)):
        final_train_results = []
    final_metrics = final_train_results[0] if final_train_results and final_train_results[0] else {}
    if final_metrics:
        final_metric_step = int(final_metrics.pop("_metrics_step", completed_steps))
        final_metrics["train/step"] = final_metric_step
        final_metrics["inference/step"] = completed_steps
        if enable_perf and previous_dispatch_wait is not None:
            final_metrics["perf/dispatch_wait"] = previous_dispatch_wait
            step_time = final_metrics.get("perf/step_time", 0)
            if step_time > 0:
                final_metrics["perf/train_capacity"] = args.global_batch_size / step_time
        _write_training_metrics(final_metrics, final_metric_step, completed_steps)

    progress.close()

    # Always save a final checkpoint unless saved.
    if args.checkpoint_dir and last_saved_step != completed_steps:
        eval_metrics = run_eval(completed_steps, train_group, eval_enabled)
        logger.info(f"Saving final checkpoint at step {completed_steps}...")
        train_group.save_model(completed_steps, force_sync=True)
        best_eval_score = update_checkpoint_eval_meta(
            args.checkpoint_dir, completed_steps, eval_metrics, best_eval_score
        )
        max_ckpts = getattr(args, "max_checkpoints", 0)
        if max_ckpts > 0:
            _cleanup_old_checkpoints(args.checkpoint_dir, max_ckpts)

    final_status = ray.get(controller.get_full_status.remote())
    logger.info(
        f"Training completed: {completed_steps} steps in {final_status['elapsed_seconds']:.1f}s | "
        f"avg inference={final_status['avg_inference_speed']:.1f} entries/s | "
        f"avg training={final_status['avg_training_speed']:.1f} entries/s"
    )


# 启动后台推理后进入同步训练循环；try/finally 确保正常退出和异常退出都执行资源清理。
def run_training_loop(
    args,
    controller,
    inference_manager,
    train_group,
    inference_engines=None,
    dataset_size=None,
    eval_dataset_size=None,
):
    inference_future = inference_manager.run.remote()
    try:
        return training_loop(
            args,
            controller,
            inference_manager,
            train_group,
            inference_engines=inference_engines,
            dataset_size=dataset_size,
            eval_dataset_size=eval_dataset_size,
        )
    finally:
        _safe_training_cleanup(
            args=args,
            inference_manager=inference_manager,
            inference_future=inference_future,
            inference_engines=inference_engines,
        )
