#!/usr/bin/env python3
# 用真实 teacher embedding/LM head 和随机隐藏状态搜索可训练的最大序列长度。
# 会加载模型文件、占用 CUDA 并执行真实反向和优化器步；不启动分布式训练或推理服务。

"""Find maximum sequence length before OOM for Eagle3 training.

This tool performs binary search to find the maximum sequence length that
fits in GPU memory for a given model configuration and batch size.

Matches the actual training logic from Eagle3Trainer:
- Uses BF16Optimizer (fp32 master weights + AdamW)
- Uses TargetLMHead to compute target logits
- Runs forward/backward/optimizer step

Usage:
    python tools/max_seq_search.py \
        --target-model /path/to/target/model \
        --batch-size 1 \
        --min-seq-len 128 \
        --max-seq-len 32768
"""

import argparse
import sys
from typing import Optional

import torch
import torch.nn as nn

from torchspec import AutoDraftModelConfig, AutoEagle3DraftModel, Eagle3Model
from torchspec.config.utils import generate_draft_model_config
from torchspec.models.eagle3 import compute_lazy_target_padded, compute_target_p_padded
from torchspec.models.target.target_utils import TargetLMHead
from torchspec.training.optimizer import BF16Optimizer
from torchspec.utils.memory import available_memory, clear_memory


# 构造 [B,S,3H] 辅助状态和 [B,S,H] 目标状态；loss 可覆盖随机约八成位置，或仅尾部指定比例。
def create_synthetic_batch(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    draft_vocab_size: int,
    device: str = "cuda",
    last_turn_loss: bool = False,
    loss_ratio: float = 0.01,
) -> dict:
    """Create synthetic training batch matching Eagle3 expected format."""
    input_ids = torch.randint(0, draft_vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    if last_turn_loss:
        loss_len = max(1, int(seq_len * loss_ratio))
        loss_mask = torch.zeros(batch_size, seq_len, device=device)
        loss_mask[:, -loss_len:] = 1.0
    else:
        loss_mask = (torch.rand(batch_size, seq_len, device=device) > 0.2).float()

    hidden_states = torch.randn(
        batch_size, seq_len, hidden_dim * 3, dtype=torch.bfloat16, device=device
    )
    target_hidden_states = torch.randn(
        batch_size, seq_len, hidden_dim, dtype=torch.bfloat16, device=device
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "hidden_states": hidden_states,
        "target_hidden_states": target_hidden_states,
    }


# 显式配置优先，否则从 target 推导 draft 配置；加载并冻结 teacher embedding，再构建 EAGLE3 训练包装。
def create_eagle3_model(
    target_model_path: str,
    draft_config_path: Optional[str] = None,
    draft_vocab_size: Optional[int] = None,
    ttt_length: int = 7,
    device: str = "cuda",
    attention_backend: str = "flex_attention",
    embedding_key: str = "model.embed_tokens.weight",
    trust_remote_code: bool = False,
    gradient_checkpointing: bool = False,
    verbose: bool = True,
) -> Eagle3Model:
    """Create and initialize Eagle3 model."""
    if draft_config_path is not None:
        draft_model_config = AutoDraftModelConfig.from_file(draft_config_path)
        if verbose:
            print(f"Loaded draft config from: {draft_config_path}")
    else:
        if verbose:
            print("Auto-generating draft config from target model...")
        config_dict = generate_draft_model_config(
            target_model_path=target_model_path,
            template_config_path=None,
            cache_dir=None,
        )
        config_dict["tie_word_embeddings"] = False
        if draft_vocab_size is not None:
            config_dict["draft_vocab_size"] = draft_vocab_size
        elif "draft_vocab_size" not in config_dict or config_dict["draft_vocab_size"] is None:
            config_dict["draft_vocab_size"] = config_dict.get("vocab_size")

        from transformers import LlamaConfig

        draft_model_config = LlamaConfig.from_dict(config_dict)
        if verbose:
            print(
                f"  hidden_size={config_dict.get('hidden_size')}, "
                f"vocab_size={config_dict.get('vocab_size')}, "
                f"draft_vocab_size={config_dict.get('draft_vocab_size')}"
            )

    draft_model = AutoEagle3DraftModel.from_config(
        draft_model_config,
        attention_backend=attention_backend,
        torch_dtype=torch.bfloat16,
    )

    # 仅加载 teacher embedding，teacher 的完整层不在这个模型中；后面的目标特征仍是合成随机值。
    draft_model.load_embedding(target_model_path, embedding_key=embedding_key)
    draft_model.freeze_embedding()

    # 多步展开和 checkpointing 改变激活保存/重算量，会明显影响最大长度，不能当作无关开关。
    eagle3_model = Eagle3Model(
        draft_model=draft_model,
        length=ttt_length,
        attention_backend=attention_backend,
        gradient_checkpointing=gradient_checkpointing,
    )

    eagle3_model = eagle3_model.to(device)
    return eagle3_model


# 使用指定 teacher LM head，否则退到 draft 的头；按词表裁剪状态构造目标并收集多步 loss。
def run_eagle3_forward(
    model: nn.Module,
    batch: dict,
    target_lm_head_weight: Optional[torch.Tensor] = None,
) -> list[torch.Tensor]:
    eagle3 = model.module if hasattr(model, "module") else model
    draft_model = eagle3.draft_model

    if target_lm_head_weight is None:
        _, target_lm_head_weight, _ = draft_model.get_lm_head_params()

    t2d = draft_model.t2d if eagle3.vocab_pruning else None

    if t2d is not None:
        target = compute_target_p_padded(
            target_hidden_states=batch["target_hidden_states"],
            target_lm_head_weight=target_lm_head_weight,
            t2d=t2d,
            loss_mask=batch["loss_mask"],
            length=eagle3.length,
        )
    else:
        target = compute_lazy_target_padded(
            batch["target_hidden_states"],
            target_lm_head_weight,
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


# 衰减加权各步 loss 并除 accumulation_steps 后执行反向和 step；这里每次调用仍会更新，不实现多次调用后才 step 的累积调度。
def run_eagle3_backward_and_update(
    plosses: list[torch.Tensor],
    optimizer: BF16Optimizer,
    accumulation_steps: int = 1,
) -> torch.Tensor:
    """Run backward pass and optimizer step (matches eagle3_trainer.py)."""
    ploss_weight = [0.8**i for i in range(len(plosses))]
    ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))]) / accumulation_steps
    ploss.backward()
    optimizer.step()
    return ploss


# 删除合成 batch 的字段引用，为候选长度之间回收中间张量创造条件。
def cleanup_batch(batch: dict):
    """Delete batch tensors and free memory."""
    for k in list(batch.keys()):
        del batch[k]


# 清空优化器动量/方差缓存，下一次实际 step 会重建；FP32 主参数等其他状态仍存在。
def reset_optimizer_state(optimizer: BF16Optimizer):
    """Reset optimizer state to free memory from momentum/variance buffers."""
    optimizer.optimizer.state.clear()


# 在指定长度运行预热和测试步，预热结束后重置峰值统计；OOM 返回失败及简短原因，非 OOM 不当成容量不足。
def test_seq_len(
    model: nn.Module,
    optimizer: BF16Optimizer,
    seq_len: int,
    batch_size: int,
    hidden_dim: int,
    draft_vocab_size: int,
    target_lm_head_weight: Optional[torch.Tensor] = None,
    warmup_iters: int = 2,
    test_iters: int = 3,
    last_turn_loss: bool = False,
    loss_ratio: float = 0.01,
) -> tuple[bool, float, Optional[str]]:
    """Test if a sequence length fits in memory.

    Returns (success, peak_memory_gb, error_message).
    """
    reset_optimizer_state(optimizer)
    clear_memory()
    torch.cuda.reset_peak_memory_stats()

    try:
        # 每轮包括目标分布构造、draft 前向、反向和优化器更新，覆盖比推理前向更多的显存对象。
        for i in range(warmup_iters + test_iters):
            batch = create_synthetic_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                draft_vocab_size=draft_vocab_size,
                last_turn_loss=last_turn_loss,
                loss_ratio=loss_ratio,
            )

            plosses = run_eagle3_forward(model, batch, target_lm_head_weight=target_lm_head_weight)
            run_eagle3_backward_and_update(plosses, optimizer)
            cleanup_batch(batch)

            # 预热结束后清缓存并重置峰值，后续统计偏向稳定状态；优化器常驻状态不会因此消失。
            if i == warmup_iters - 1:
                clear_memory()
                torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        clear_memory()
        return True, peak_mem, None

    except torch.cuda.OutOfMemoryError as e:
        reset_optimizer_state(optimizer)
        clear_memory()
        return False, 0.0, f"OOM: {str(e)[:100]}"
    # 只把错误文本确为 out of memory 的运行异常折成失败，形状/后端错误应立即暴露。
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            reset_optimizer_state(optimizer)
            clear_memory()
            return False, 0.0, f"OOM: {str(e)[:100]}"
        raise


# 从低到高按固定网格逐一尝试，首次失败即结束；避免二分先尝试超大张量导致分配器状态干扰后续测试。
def linear_search_max_seq_len(
    model: nn.Module,
    optimizer: BF16Optimizer,
    batch_size: int,
    hidden_dim: int,
    draft_vocab_size: int,
    min_seq_len: int,
    max_seq_len: int,
    step_size: int = 1024,
    target_lm_head_weight: Optional[torch.Tensor] = None,
    warmup_iters: int = 2,
    test_iters: int = 3,
    verbose: bool = True,
    last_turn_loss: bool = False,
    loss_ratio: float = 0.01,
) -> tuple[int, float]:
    """Linear search from low to high to find max sequence length.

    More reliable than binary search as it avoids memory fragmentation
    from failing on large sequences first.

    Returns (max_seq_len, peak_memory_gb).
    """
    # 下界向下、上界向上对齐，实际尝试可超出用户原始数值范围。
    # 若 min_seq_len 小于 step_size，lo 可变成零；应读实际 Testing 日志，不把参数原值当作已测试最小值。
    lo = (min_seq_len // step_size) * step_size
    hi = ((max_seq_len + step_size - 1) // step_size) * step_size
    best = lo
    best_mem = 0.0

    if verbose:
        mem = available_memory()
        print(f"GPU Memory: {mem['total_GB']:.2f} GB total, {mem['free_GB']:.2f} GB free")
        print(f"Search range: [{lo}, {hi}] with step size {step_size}")
        print("-" * 60)

    current = lo
    while current <= hi:
        if verbose:
            print(f"Testing seq_len={current}...", end=" ", flush=True)

        success, peak_mem, error = test_seq_len(
            model=model,
            optimizer=optimizer,
            seq_len=current,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            draft_vocab_size=draft_vocab_size,
            target_lm_head_weight=target_lm_head_weight,
            warmup_iters=warmup_iters,
            test_iters=test_iters,
            last_turn_loss=last_turn_loss,
            loss_ratio=loss_ratio,
        )

        # 线性法遇首个失败即停，隐含后续更长也不可行的假设；分配器和编译分桶可能使边界不完全平滑。
        if success:
            if verbose:
                print(f"OK (peak {peak_mem:.2f} GB)")
            best = current
            best_mem = peak_mem
            current += step_size
        else:
            if verbose:
                print(f"FAIL {error}")
            break

    return best, best_mem


# 把边界按 step 对齐后做二分，成功点作为 best；网格精度和显存碎片都会影响报告的容量边界。
def binary_search_max_seq_len(
    model: nn.Module,
    optimizer: BF16Optimizer,
    batch_size: int,
    hidden_dim: int,
    draft_vocab_size: int,
    min_seq_len: int,
    max_seq_len: int,
    step_size: int = 1024,
    target_lm_head_weight: Optional[torch.Tensor] = None,
    warmup_iters: int = 2,
    test_iters: int = 3,
    verbose: bool = True,
    last_turn_loss: bool = False,
    loss_ratio: float = 0.01,
) -> tuple[int, float]:
    """Binary search to find maximum sequence length before OOM.

    Note: May be less accurate than linear search due to memory fragmentation
    when large allocations fail.

    Returns (max_seq_len, peak_memory_gb).
    """
    lo = min_seq_len
    hi = max_seq_len
    best = min_seq_len
    best_mem = 0.0

    lo = (lo // step_size) * step_size
    hi = ((hi + step_size - 1) // step_size) * step_size

    if verbose:
        mem = available_memory()
        print(f"GPU Memory: {mem['total_GB']:.2f} GB total, {mem['free_GB']:.2f} GB free")
        print(f"Search range: [{lo}, {hi}] with step size {step_size}")
        print("-" * 60)

    while lo <= hi:
        # 在 step 网格上选中点，不是每个整数长度都测试；例如 1024 粒度只给出约千 token 精度的边界。
        mid = ((lo + hi) // 2 // step_size) * step_size

        if verbose:
            print(f"Testing seq_len={mid}...", end=" ", flush=True)

        success, peak_mem, error = test_seq_len(
            model=model,
            optimizer=optimizer,
            seq_len=mid,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            draft_vocab_size=draft_vocab_size,
            target_lm_head_weight=target_lm_head_weight,
            warmup_iters=warmup_iters,
            test_iters=test_iters,
            last_turn_loss=last_turn_loss,
            loss_ratio=loss_ratio,
        )

        if success:
            if verbose:
                print(f"OK (peak {peak_mem:.2f} GB)")
            best = mid
            best_mem = peak_mem
            lo = mid + step_size
        else:
            if verbose:
                print(f"FAIL {error}")
            hi = mid - step_size

    # 如果没有成功点，best 仍是初始化的 min_seq_len；应结合 best_mem 和试验记录判断，而不能只看返回正数。
    return best, best_mem


# 加载 draft、BF16Optimizer 和 target LM head，再选择线性/二分搜索；最后报告的是当前合成输入和训练配置下的局部容量。
def main():
    parser = argparse.ArgumentParser(
        description="Find maximum sequence length before OOM for Eagle3 training"
    )
    parser.add_argument(
        "--target-model",
        required=True,
        help="Path or name of target model",
    )
    parser.add_argument(
        "--draft-config",
        default=None,
        help="Path to draft model config JSON file (auto-generated if not provided)",
    )
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="Key for embedding weights in target model (default: model.embed_tokens.weight)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for testing (default: 1)",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=128,
        help="Minimum sequence length to test (default: 128)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=65536,
        help="Maximum sequence length to test (default: 65536)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=1024,
        help="Step size for binary search (default: 1024)",
    )
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="TTT (test-time training) length (default: 7)",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["sdpa", "flash_attn", "fa3", "flex_attention"],
        help="Attention backend (default: flex_attention)",
    )
    parser.add_argument(
        "--lm-head-key",
        type=str,
        default="lm_head.weight",
        help="Key for lm_head weights in target model (default: lm_head.weight)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer (default: 1e-4)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Max gradient norm for clipping (default: 0.5)",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=100000,
        help="Total steps for LR scheduler (default: 100000)",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=2,
        help="Number of warmup iterations per test (default: 2)",
    )
    parser.add_argument(
        "--test-iters",
        type=int,
        default=3,
        help="Number of test iterations after warmup (default: 3)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print final result",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use linear search (low to high) instead of binary search. More reliable but slower.",
    )
    parser.add_argument(
        "--no-grad-ckpt",
        action="store_true",
        help="Disable gradient checkpointing (enabled by default)",
    )
    parser.add_argument(
        "--draft-vocab-size",
        type=int,
        default=None,
        help="Draft vocab size (overrides auto-generated config). Enables vocab pruning when < target vocab.",
    )
    parser.add_argument(
        "--vocab-mapping",
        type=str,
        default=None,
        help="Path to vocab mapping .pt file (contains t2d/d2t tensors for vocab pruning)",
    )
    parser.add_argument(
        "--last-turn-loss",
        action="store_true",
        help="Only compute loss on the last turn (tail of sequence). Use --loss-ratio to control the fraction.",
    )
    parser.add_argument(
        "--loss-ratio",
        type=float,
        default=0.01,
        help="Fraction of sequence that has loss when --last-turn-loss is set (default: 0.01 = 1%%)",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print("Eagle3 Maximum Sequence Length Search")
        print("=" * 60)
        print(f"Target model: {args.target_model}")
        print(f"Draft config: {args.draft_config or '(auto-generated)'}")
        print(f"LM head key: {args.lm_head_key}")
        print(f"Embedding key: {args.embedding_key}")
        print(f"Batch size: {args.batch_size}")
        print(f"TTT length: {args.ttt_length}")
        print(f"Attention backend: {args.attention_backend}")
        print(f"Search range: [{args.min_seq_len}, {args.max_seq_len}]")
        print(f"Step size: {args.step_size}")
        grad_ckpt = not args.no_grad_ckpt
        print(f"Gradient checkpointing: {'ON' if grad_ckpt else 'OFF'}")
        print(f"Vocab mapping: {args.vocab_mapping or '(none)'}")
        if args.last_turn_loss:
            print(f"Last-turn loss: ON (ratio={args.loss_ratio:.2%})")
        print("=" * 60)

    if verbose:
        print("\n[1/4] Loading draft model...")

    # 按真实 target 的维度/embedding 建 draft，其他训练特征随机生成；这不是加载完整 teacher 执行真实数据。
    model = create_eagle3_model(
        target_model_path=args.target_model,
        draft_config_path=args.draft_config,
        draft_vocab_size=args.draft_vocab_size,
        ttt_length=args.ttt_length,
        attention_backend=args.attention_backend,
        embedding_key=args.embedding_key,
        trust_remote_code=args.trust_remote_code,
        gradient_checkpointing=not args.no_grad_ckpt,
        verbose=verbose,
    )
    model.train()

    hidden_dim = model.draft_model.config.hidden_size
    vocab_size = model.draft_model.config.vocab_size
    draft_vocab_size = model.draft_model.config.draft_vocab_size

    if verbose:
        print(
            f"Draft model ready: hidden_dim={hidden_dim}, vocab_size={vocab_size}, draft_vocab_size={draft_vocab_size}"
        )
        print(f"Vocab pruning: {model.vocab_pruning}")

    if verbose:
        print("\n[2/4] Creating BF16Optimizer (fp32 master weights + AdamW)...")

    optimizer = BF16Optimizer(
        model.draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        total_steps=args.total_steps,
    )

    if verbose:
        trainable_params = sum(p.numel() for p in model.draft_model.parameters() if p.requires_grad)
        print(
            f"Optimizer created: {trainable_params / 1e6:.2f}M trainable params (fp32 copies allocated)"
        )

    if verbose:
        print("\n[3/4] Loading target lm_head...")

    # 额外加载 teacher 的完整输出头，容量搜索因此包含目标词表投影权重，而不是只估算 draft 本身。
    target_lm_head = TargetLMHead.from_pretrained(
        model_path=args.target_model,
        lm_head_key=args.lm_head_key,
        trust_remote_code=args.trust_remote_code,
    )

    target_lm_head_weight = target_lm_head.lm_head.weight

    if verbose:
        lm_head_params = sum(p.numel() for p in target_lm_head.parameters())
        print(f"Target lm_head loaded: {lm_head_params / 1e6:.2f}M parameters")
        print()

    search_method = "linear" if args.linear else "binary"
    if verbose:
        print(f"[4/4] {search_method.capitalize()} search for max sequence length...")

    # 当前 --vocab-mapping 只在参数和日志中出现，没有在此加载并设置 t2d/d2t；不要据 CLI 名称假定裁剪映射已生效。
    search_fn = linear_search_max_seq_len if args.linear else binary_search_max_seq_len
    max_seq, peak_mem = search_fn(
        model=model,
        optimizer=optimizer,
        batch_size=args.batch_size,
        hidden_dim=hidden_dim,
        draft_vocab_size=draft_vocab_size,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        step_size=args.step_size,
        target_lm_head_weight=target_lm_head_weight,
        warmup_iters=args.warmup_iters,
        test_iters=args.test_iters,
        verbose=verbose,
        last_turn_loss=args.last_turn_loss,
        loss_ratio=args.loss_ratio,
    )

    print()
    print("=" * 60)
    print(f"Maximum sequence length: {max_seq}")
    print(f"  Peak memory at max seq: {peak_mem:.2f} GB")
    print(f"  batch_size={args.batch_size}, ttt_length={args.ttt_length}")
    grad_ckpt = not args.no_grad_ckpt
    print(f"  gradient_checkpointing={'ON' if grad_ckpt else 'OFF'}")
    if args.last_turn_loss:
        loss_tokens = max(1, int(max_seq * args.loss_ratio))
        print(f"  last_turn_loss=ON (ratio={args.loss_ratio:.2%}, ~{loss_tokens} loss tokens)")
    print("=" * 60)

    return max_seq


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
