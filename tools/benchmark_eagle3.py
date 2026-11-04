#!/usr/bin/env python3
# 用合成输入测 EAGLE3 完整训练步的显存和时间，包含前向、反向、BF16 优化器更新。
# 它不运行 teacher 或远程传输，因此结果不能直接等同生产流水线吞吐；需要 CUDA 和所选 attention 后端。

"""Benchmark Eagle3 end-to-end training (forward + backward + optimizer step).

Measures:
  1. Memory efficiency: max sequence length via binary search
  2. Speed: full training step time at various sequence lengths

Gradient checkpointing is enabled by default. Use --no-grad-ckpt to disable.

Usage:
    python tools/benchmark_eagle3.py [--hidden-size 4096] [--vocab-size 151936]
"""

import argparse
import gc
import time
from typing import Tuple

import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from torchspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3
from torchspec.models.eagle3 import (
    Eagle3Model,
    compute_lazy_target_padded,
    compute_target_p_padded,
)
from torchspec.training.optimizer import BF16Optimizer


# ---------------------------------------------------------------------------
# Helpers (aligned with max_seq_search.py)
# ---------------------------------------------------------------------------
# 先同步 GPU，再收集 Python 垃圾和释放空闲缓存；仍被模型或优化器引用的显存不会被清除。
def clear_memory():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


# 根据 H/V 构造随机 Llama draft 的维度配置，draft_vocab_size 等于完整词表，不加载预训练权重。
def make_config(H, V, num_heads=32, num_kv_heads=8):
    config = LlamaConfig(
        hidden_size=H,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=H * 4,
        max_position_embeddings=131072,
        vocab_size=V,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_scaling=None,
        pretraining_tp=1,
        pad_token_id=0,
    )
    config.draft_vocab_size = V
    return config


# 实例化 BF16 draft、冻结 embedding，包装多步 EAGLE3 并启用训练模式；checkpointing 与 attention 后端由参数控制。
def make_eagle3_model(
    config,
    length=7,
    gradient_checkpointing=True,
    attention_backend="flex_attention",
    device="cuda",
):
    draft_model = LlamaForCausalLMEagle3(config, attention_backend=attention_backend)
    draft_model = draft_model.to(device=device, dtype=torch.bfloat16)
    draft_model.freeze_embedding()
    model = Eagle3Model(
        draft_model,
        length=length,
        attention_backend=attention_backend,
        gradient_checkpointing=gradient_checkpointing,
    )
    model.train()
    return model


# 为 draft 建 BF16Optimizer，测量包括 FP32 主权重及 AdamW 状态带来的成本。
def make_optimizer(model):
    return BF16Optimizer(model.draft_model, lr=1e-4)


# 清空底层 Adam 动量/方差，减少上一候选长度残留影响；不重新随机化模型或重建优化器对象。
def reset_optimizer_state(optimizer: BF16Optimizer):
    """Reset optimizer state to free memory from momentum/variance buffers."""
    optimizer.optimizer.state.clear()


# 生成 [B,T] token/mask、[B,T,3H] 辅助表示及 [B,T,H] 目标表示；它们是随机数，不能用 loss 判断真实收敛。
def create_synthetic_batch(B, T, H, V, device="cuda"):
    """Create synthetic training batch matching Eagle3 expected format."""
    input_ids = torch.randint(0, V, (B, T), device=device)
    attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
    loss_mask = (torch.rand(B, T, device=device) > 0.5).float()
    hidden_states = torch.randn(B, T, H * 3, device=device, dtype=torch.bfloat16)
    target_hidden_states = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "hidden_states": hidden_states,
        "target_hidden_states": target_hidden_states,
    }


# 逐键删除 batch 对张量的引用；若计算图仍引用这些张量，实际分配不会立即释放。
def cleanup_batch(batch: dict):
    """Delete batch tensors and free memory."""
    for k in list(batch.keys()):
        del batch[k]


# 按是否裁剪词表选择 eager 目标概率或 lazy 目标表示，再运行多步 draft 并返回每步 loss。
def run_eagle3_forward(model, batch):
    eagle3 = model.module if hasattr(model, "module") else model
    draft_model = eagle3.draft_model
    _, lm_head_weight, _ = draft_model.get_lm_head_params()
    t2d = draft_model.t2d if eagle3.vocab_pruning else None

    # 裁剪词表需要把 teacher 输出投到有效词表集合；完整词表路径用 lazy 目标避免一次物化全部概率。
    if t2d is not None:
        target = compute_target_p_padded(
            target_hidden_states=batch["target_hidden_states"],
            target_lm_head_weight=lm_head_weight,
            t2d=t2d,
            loss_mask=batch["loss_mask"],
            length=eagle3.length,
        )
    else:
        target = compute_lazy_target_padded(
            batch["target_hidden_states"],
            lm_head_weight,
            eagle3.length,
        )

    plosses, _, acces, _, _ = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        target=target,
        loss_mask=batch["loss_mask"],
        hidden_states=batch["hidden_states"],
    )
    return plosses


# 以 0.8 的幂衰减组合各步 loss，反向并更新优化器；不同展开步不是独立 batch。
def run_eagle3_backward_and_update(plosses, optimizer):
    """Run backward pass and optimizer step (matches eagle3_trainer.py)."""
    ploss_weight = [0.8**i for i in range(len(plosses))]
    ploss = sum(ploss_weight[i] * plosses[i] for i in range(len(plosses)))
    ploss.backward()
    optimizer.step()
    return ploss


# 每步重新构造合成数据、执行前向反向和更新，再释放 batch 引用；计时会包含随机输入生成成本。
def run_train_step(model, optimizer, B, T, H, V):
    """Full training step: forward + backward + optimizer step."""
    batch = create_synthetic_batch(B, T, H, V)
    plosses = run_eagle3_forward(model, batch)
    loss = run_eagle3_backward_and_update(plosses, optimizer)
    cleanup_batch(batch)
    return loss


# ---------------------------------------------------------------------------
# 1. Memory benchmark: binary search for max sequence length
# ---------------------------------------------------------------------------
# 重置优化器状态后执行预热与测试步，同步后读取峰值显存；仅把明确 OOM 当不可行，其他 RuntimeError 继续抛出。
def try_seq_len(model, optimizer, T, B, H, V, warmup_iters=2, test_iters=3) -> Tuple[bool, float]:
    reset_optimizer_state(optimizer)
    clear_memory()
    torch.cuda.reset_peak_memory_stats()

    try:
        # 每候选执行多步以触发 Adam 状态分配等首次开销；只成功前向不足以证明训练能放下。
        for i in range(warmup_iters + test_iters):
            run_train_step(model, optimizer, B, T, H, V)
            if i == warmup_iters - 1:
                clear_memory()

        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        clear_memory()
        return True, peak_mem
    except torch.cuda.OutOfMemoryError:
        reset_optimizer_state(optimizer)
        clear_memory()
        return False, 0.0
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            reset_optimizer_state(optimizer)
            clear_memory()
            return False, 0.0
        raise


# 按 step 网格二分可容纳的 T，成功增大下界、失败缩小上界；依赖可行性近似单调这一假设。
def binary_search_max_seq(
    model, optimizer, B, H, V, lo=256, hi=131072, step=256, verbose=True
) -> Tuple[int, float]:
    lo = lo // step * step
    hi = hi // step * step
    best = lo
    best_mem = 0.0

    if verbose:
        print(f"  {'T':>8s}  {'status':>6s}  {'peak_mem':>10s}")

    # 例如 step=256，候选始终落在该网格上；报告的是测试网格中的最大成功点，而不是逐 token 精确极限。
    while lo <= hi:
        mid = (lo + hi) // 2
        mid = mid // step * step
        if mid == 0:
            mid = step

        ok, peak_mem = try_seq_len(model, optimizer, mid, B, H, V)
        if verbose:
            status = "OK" if ok else "OOM"
            mem_str = f"{peak_mem:.2f} GB" if peak_mem > 0 else "N/A"
            print(f"  {mid:>8d}  {status:>6s}  {mem_str:>10s}")
        if ok:
            best = mid
            best_mem = peak_mem
            lo = mid + step
        else:
            hi = mid - step

    # best 初始为下界，若所有测试都失败仍可能返回该值且 best_mem=0；应结合成功日志解释结果。
    return best, best_mem


# 创建一组模型/优化器做容量搜索，结束后删除并清理缓存，返回最大候选及对应峰值。
def benchmark_memory(
    B,
    H,
    V,
    config,
    length=7,
    step=256,
    gradient_checkpointing=True,
    attention_backend="flex_attention",
):
    ckpt_str = "ON" if gradient_checkpointing else "OFF"
    print("=" * 70)
    print("MEMORY BENCHMARK: max sequence length (binary search)")
    print(f"  B={B}, H={H}, V={V}, length={length}, step={step}")
    print(f"  gradient_checkpointing={ckpt_str}")
    print("=" * 70)

    print()
    clear_memory()
    model = make_eagle3_model(
        config,
        length=length,
        gradient_checkpointing=gradient_checkpointing,
        attention_backend=attention_backend,
    )
    optimizer = make_optimizer(model)
    max_seq, peak_mem = binary_search_max_seq(model, optimizer, B, H, V, step=step)
    del model, optimizer
    clear_memory()

    print()
    print(f"  Result: max_seq = {max_seq},  peak = {peak_mem:.2f} GB")
    print()
    return max_seq, peak_mem


# ---------------------------------------------------------------------------
# 2. Speed benchmark: full training step time
# ---------------------------------------------------------------------------
# 预热后重置峰值，用前后 CUDA 同步包住每个完整训练步的墙钟计时，返回平均毫秒和峰值 GiB。
def time_eagle3(model, optimizer, B, T, H, V, warmup=3, repeats=10) -> Tuple[float, float]:
    reset_optimizer_state(optimizer)
    clear_memory()

    for _ in range(warmup):
        run_train_step(model, optimizer, B, T, H, V)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    # 预热包含编译和优化器状态首次分配，正式计时把它们排除；峰值统计仍含当前常驻权重与状态。
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        # CUDA 异步提交很快，若不在前后 synchronize，CPU 墙钟只会量到发射开销。
        # 这里计时还包含随机 batch 构造与 Python 调度，不能与纯 kernel event 时间直接比较。
        t0 = time.perf_counter()
        run_train_step(model, optimizer, B, T, H, V)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    return sum(times) / len(times), peak_mem


# 把 OOM 及所有 RuntimeError 都转换成无穷耗时；因此表中 OOM 标记也可能代表其他运行错误。
def _try_time(model, optimizer, B, T, H, V, **kwargs):
    try:
        return time_eagle3(model, optimizer, B, T, H, V, **kwargs)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        reset_optimizer_state(optimizer)
        clear_memory()
        return float("inf"), 0.0


# 每个 T 独立创建模型和优化器，减少跨长度状态干扰；收集步耗时和显存并格式化输出。
def benchmark_speed(
    B,
    H,
    V,
    config,
    seq_lens,
    length=7,
    gradient_checkpointing=True,
    attention_backend="flex_attention",
):
    ckpt_str = "ON" if gradient_checkpointing else "OFF"
    print("=" * 70)
    print("SPEED BENCHMARK: full training step (fwd + bwd + optimizer) (ms)")
    print(f"  B={B}, H={H}, V={V}, length={length}")
    print(f"  gradient_checkpointing={ckpt_str}")
    print("=" * 70)

    results = []

    for T in seq_lens:
        clear_memory()
        model = make_eagle3_model(
            config,
            length=length,
            gradient_checkpointing=gradient_checkpointing,
            attention_backend=attention_backend,
        )
        optimizer = make_optimizer(model)
        ms, mem = _try_time(model, optimizer, B, T, H, V)
        del model, optimizer
        clear_memory()

        results.append({"T": T, "ms": ms, "mem": mem})

        # 把无穷时间显示为 OOM，零显存显示 N/A；只是报告约定，并没有进一步判别异常原因。
        def _fmt(ms, mem):
            ms_s = f"{ms:.1f}" if ms != float("inf") else "OOM"
            mem_s = f"{mem:.1f}" if mem > 0 else "N/A"
            return f"{ms_s:>8s} ms {mem_s:>5s} GB"

        print(f"  T={T:>6d}: {_fmt(ms, mem)}")

    print()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# 读取模型/后端维度，先搜索容量，再测能放下的候选长度并打印硬件与配置，便于解释结果边界。
def main():
    parser = argparse.ArgumentParser(description="Benchmark Eagle3 end-to-end training")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=7, help="Number of draft steps")
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument(
        "--mem-step", type=int, default=1024, help="Step size for memory binary search"
    )
    parser.add_argument(
        "--no-grad-ckpt", action="store_true", help="Disable gradient checkpointing"
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["sdpa", "fa", "flex_attention"],
    )
    args = parser.parse_args()

    B = args.batch_size
    H = args.hidden_size
    V = args.vocab_size
    grad_ckpt = not args.no_grad_ckpt
    attention_backend = args.attention_backend

    config = make_config(H, V, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads)

    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {total / 1024**3:.1f} GB total, {free / 1024**3:.1f} GB free")
    print(f"Config: B={B}, H={H}, V={V}, length={args.length}")
    print(f"        num_heads={args.num_heads}, num_kv_heads={args.num_kv_heads}")
    print(f"        gradient_checkpointing={'ON' if grad_ckpt else 'OFF'}")
    print(f"        attention_backend={attention_backend}")
    print()

    # 1. Memory benchmark
    # 先建立容量上界，再限制速度测试候选，减少明显无法容纳的运行。
    max_seq, peak_mem = benchmark_memory(
        B,
        H,
        V,
        config,
        length=args.length,
        step=args.mem_step,
        gradient_checkpointing=grad_ckpt,
        attention_backend=attention_backend,
    )

    # 2. Speed benchmark
    # 速度比较应保留相同 H/V、batch、展开长度、checkpointing 和后端；只比较一列毫秒会混入不同工作量。
    speed_lens = [s for s in [512, 1024, 2048, 4096, 8192, 16384, 32768] if s <= max_seq]
    if not speed_lens:
        speed_lens = [256, 512]
    speed_results = benchmark_speed(
        B,
        H,
        V,
        config,
        speed_lens,
        length=args.length,
        gradient_checkpointing=grad_ckpt,
        attention_backend=attention_backend,
    )

    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Config: B={B}, H={H}, V={V}, length={args.length}")
    print(f"  Gradient checkpointing: {'ON' if grad_ckpt else 'OFF'}")
    print(f"  Max sequence length: {max_seq}")
    print(f"  Peak memory at max seq: {peak_mem:.2f} GB")
    print()
    print(f"  {'T':>8s}  {'step_ms':>10s}  {'peak_mem':>10s}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}")
    for r in speed_results:
        ms_s = f"{r['ms']:.1f}" if r["ms"] != float("inf") else "OOM"
        mem_s = f"{r['mem']:.1f} GB" if r["mem"] > 0 else "N/A"
        print(f"  {r['T']:>8d}  {ms_s:>8s} ms  {mem_s:>10s}")
    print()


if __name__ == "__main__":
    main()
