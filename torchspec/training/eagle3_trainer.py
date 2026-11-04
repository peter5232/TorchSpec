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

# Eagle 训练适配层：把 teacher 特征变成多步训练目标，连接 draft、FSDP 和公共训练循环。
# 这里的展开深度是训练监督位置；simulated_acc_len 是统计近似，不执行线上接受/拒绝采样。

from argparse import Namespace
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from torchspec import AutoDraftModelConfig, AutoEagle3DraftModel, Eagle3Model
from torchspec.models.anchored_eagle3 import AnchoredEagle3Model
from torchspec.models.draft.eagle4_moe import Eagle4Router
from torchspec.models.eagle3 import compute_lazy_target_padded, compute_target_p_padded
from torchspec.training import checkpoint
from torchspec.training.fsdp import apply_fsdp2, fsdp2_load_full_state_dict
from torchspec.training.optimizer import BF16Optimizer
from torchspec.training.trainer import DeferredTrainMetrics, Trainer
from torchspec.utils.distributed import get_gloo_group
from torchspec.utils.logging import logger
from torchspec.utils.tensor import padding
from torchspec.utils.train_dump import dump_eagle3_batch


# 返回长度为 n 的深度权重；显式列表必须精确匹配，否则默认第 i 步为 0.8**i。
def _position_decay_weights(n: int, weights: Optional[list[float]] = None) -> list[float]:
    """Per-position TTT loss weights.

    If ``weights`` is supplied, it is used as-is (after coercion to ``list[float]``).
    Otherwise defaults to ``[0.8**i for i in range(n)]``.
    """
    if weights is not None:
        if len(weights) != n:
            raise ValueError(f"ploss_weights length {len(weights)} does not match ttt_length {n}")
        return [float(w) for w in weights]
    return [0.8**i for i in range(n)]


class Eagle3Trainer(Trainer):
    """Eagle3-specific trainer.

    Extends ``Trainer`` with Eagle3 model initialisation, forward/backward,
    and metric aggregation.
    """

    # 在公共训练状态上缓存各深度损失权重及其总和，并预留双缓冲 CPU 指标存储。
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.target_lm_head: Optional[torch.nn.Module] = None
        self._ploss_weights = tuple(
            _position_decay_weights(args.ttt_length, getattr(args, "ploss_weights", None))
        )
        self._ploss_weight_sum = sum(self._ploss_weights)
        self._metric_buffers: list[torch.Tensor] = []

    # rank 0 加载初始 draft/teacher embedding，其他 rank 用 meta 构造；冻结 embedding 后统一广播。
    # 将训练 wrapper 按 grad_sync_mesh 包装，建立 FP32 master 优化器并恢复检查点，最后载入冻结 teacher head。
    # warm start 的全词表 head 可先用 teacher 补齐旧裁剪词表缺失行；embedding 最后加载以保证 teacher 权重优先。
    def init_model(
        self,
        draft_model_config: AutoDraftModelConfig,
        target_model_path: str,
        mooncake_config=None,
    ) -> int:
        if mooncake_config is not None:
            from torchspec.transfer.mooncake.utils import (
                check_mooncake_master_available,
            )

            check_mooncake_master_available(
                mooncake_config.master_server_address, mooncake_config.metadata_server
            )

        init_context = self._get_init_weight_context_manager()

        # 初始化分两层：先建 draft 网络，再建负责展开和 loss 的 Eagle3Model。
        # 非主 rank 的 meta 参数只有 shape/dtype；不能在真实权重广播前进行有效前向或导出。
        with init_context():
            draft_model = AutoEagle3DraftModel.from_config(
                draft_model_config,
                attention_backend=self.args.attention_backend,
                torch_dtype=torch.bfloat16,
            )

        initial_draft_model_path = getattr(self.args, "initial_draft_model_path", None)
        freeze_lm_head = getattr(self.args, "freeze_lm_head", False)
        # A fresh frozen head is seeded from the target, and so is a full-vocab dense head before
        # a warm start, so the rows a pruned EAGLE3 checkpoint lacks start from the target; the
        # checkpoint then overwrites what it has. The resume path is resolved rather than merely
        # present: load() skips a load_path that resolves to nothing.
        # 判定的是实际可解析的续训目录；仅填写了不存在的 load_path 不应阻止 teacher 初始化。
        seed_lm_head_from_target = not checkpoint.resolve_resume_model_dir(self.args) and (
            (freeze_lm_head and not initial_draft_model_path)
            or bool(initial_draft_model_path)
            and not hasattr(draft_model, "t2d")
            and not draft_model.is_lm_head_factorized
        )
        if dist.get_rank() == 0:
            if seed_lm_head_from_target:
                draft_model.load_lm_head(
                    target_model_path,
                    lm_head_key=getattr(self.args, "lm_head_key", "lm_head.weight"),
                )
                logger.info(f"[Rank 0] Seeded draft lm_head from {target_model_path}")

            if initial_draft_model_path:
                loaded_from = checkpoint.load_initial_draft_weights(
                    draft_model, initial_draft_model_path
                )
                logger.info(f"[Rank 0] Loaded initial draft weights from {loaded_from}")

            # Must stay after the initial checkpoint, so the target's embedding wins.
            draft_model.load_embedding(
                target_model_path,
                embedding_key=getattr(self.args, "embedding_key", "model.embed_tokens"),
            )

        # teacher embedding 是输入语义空间的固定坐标；把它冻结后不占优化器 master/moment。
        # 可选冻结 lm_head 与冻结 embedding 是两个不同决策；前者会改变 checkpoint 中可训练参数集合。
        draft_model.freeze_embedding()
        frozen_parts = ["embedding"]
        if freeze_lm_head:
            draft_model.freeze_lm_head()
            frozen_parts.append("lm_head")

        dist.barrier(group=get_gloo_group())

        frozen_count = sum(p.numel() for p in draft_model.parameters() if not p.requires_grad)
        trainable_count = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        logger.info(
            f"[Rank {self.dp_rank}] Draft model: {trainable_count:,} trainable, "
            f"{frozen_count:,} frozen ({', '.join(frozen_parts)}) parameters"
        )

        num_anchors = getattr(self.args, "eagle3_num_anchors", 0)
        anchor_max_gap = getattr(self.args, "eagle3_anchor_max_gap", None)
        model_kwargs = dict(
            draft_model=draft_model,
            length=self.args.ttt_length,
            attention_backend=self.args.attention_backend,
            gradient_checkpointing=getattr(self.args, "gradient_checkpointing", True),
            loss_type=getattr(self.args, "loss_type", "forward_kl"),
            lk_eta=getattr(self.args, "lk_eta", 3.0),
        )
        # 训练展开可选 anchor 采样 wrapper，但仍把同一个 draft_model 放进统一的训练接口。
        # anchor 控制选哪些位置展开，并不是在线生成时创建多少条独立请求。
        if num_anchors:
            eagle3_model = AnchoredEagle3Model(
                num_anchors=num_anchors, anchor_max_gap=anchor_max_gap, **model_kwargs
            )
            logger.info(
                f"[Rank {self.dp_rank}] Anchored Eagle3: {num_anchors} anchors per sample "
                f"(max gap {anchor_max_gap})"
            )
        else:
            eagle3_model = Eagle3Model(**model_kwargs)

        # 先记录 rank 0 构造的完整状态，再对模型应用分布式包装。
        # 随后 set_model_state_dict 负责把完整值放进包装后的布局；顺序反过来可能拿到分片而非原始完整权重。
        full_state = eagle3_model.state_dict() if dist.get_rank() == 0 else {}

        midlayer_modules = [
            m
            for name, m in eagle3_model.named_modules()
            if isinstance(m, torch.nn.Linear) and "midlayer" in name
        ]
        eagle3_model = apply_fsdp2(
            eagle3_model,
            mesh=self.grad_sync_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
            modules_to_shard=midlayer_modules,
        )

        eagle3_model = fsdp2_load_full_state_dict(
            eagle3_model,
            full_state,
            self.grad_sync_mesh,
            cpu_offload=True if self.fsdp_cpu_offload else None,
        )

        self.model = eagle3_model
        self.eagle3 = self.model.module if hasattr(self.model, "module") else self.model
        self.draft_model = self.eagle3.draft_model
        self.t2d = self.draft_model.t2d if self.eagle3.vocab_pruning else None

        decay_style = getattr(self.args, "lr_decay_style", "cosine")
        wsd_decay_steps = None
        wsd_decay_style = None
        if decay_style == "WSD":
            wsd_ratio = getattr(self.args, "lr_wsd_decay_ratio", 0.2)
            wsd_decay_steps = int(wsd_ratio * self.args.lr_total_steps)
            wsd_decay_style = getattr(self.args, "lr_wsd_decay_style", "cosine")
        # optimizer 必须在分布式模型布局确定之后创建，master/grad buffer 才与实际参数布局一致。
        # 调度器以 optimizer step 为单位；TTT 深度更多并不会自动多推进几次学习率。
        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 0.0),
            max_grad_norm=self.args.max_grad_norm,
            warmup_ratio=getattr(self.args, "warmup_ratio", 0.1),
            total_steps=self.args.lr_total_steps,
            decay_style=decay_style,
            min_lr=getattr(self.args, "min_lr", 0.0),
            wsd_decay_steps=wsd_decay_steps,
            wsd_decay_style=wsd_decay_style,
        )
        self.lr_scheduler = self.optimizer.lr_scheduler

        checkpoint_payload = checkpoint.load(self)
        # load 读模型与优化器，finalize_load 才处理 RNG、global_step 和 continual-training master。
        # 两步共同完成恢复；只执行 load 并不等价于整个训练状态已经准备好。
        checkpoint.finalize_load(self, checkpoint_payload)

        self._last_hs_prenorm = getattr(self.args, "last_hidden_states_prenorm", False)

        if getattr(self.args, "compute_logits_in_trainer", True):
            self._init_target_lm_head(target_model_path)

        if self.target_lm_head is None:
            raise ValueError(
                "target_lm_head is required but was None. "
                "Set compute_logits_in_trainer=True or provide a TargetLMHead."
            )
        self.target_lm_head_weight = self.target_lm_head.lm_head.weight
        self.verifier_norm = self.target_lm_head.norm

        # 可选注意力内核预热在正式 profiler 启动之前执行。
        # 这把首次编译/特化成本与稳态训练分开；它不是又执行一次带梯度的训练样本。
        if getattr(self.args, "attention_backend", None) == "fa4":
            from torchspec.models.draft.llama3_eagle import (
                _has_cute_dsl,
                warmup_flash_attention_masked,
            )

            if _has_cute_dsl:
                cfg = self.draft_model.config
                head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
                warmup_flash_attention_masked(
                    q_len=self.args.max_seq_length,
                    num_heads=cfg.num_attention_heads,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=head_dim,
                    dtype=torch.bfloat16,
                    device=torch.cuda.current_device(),
                )

        self.prof.on_init_end()

        logger.info(f"[Rank {self.dp_rank}] Eagle3 model initialized with FSDP2")

        return 0

    # ------------------------------------------------------------------
    # Eagle3-specific helpers
    # ------------------------------------------------------------------

    # 从 teacher 同步加载冻结 lm_head；只有输入最后层特征为 pre-norm 时才额外加载 norm。
    def _init_target_lm_head(self, target_model_path: str) -> None:
        """Initialize TargetLMHead for computing logits from last_hidden_states.

        Only rank 0 loads the weights, then broadcasts to other ranks.
        The lm_head is kept frozen and not wrapped with FSDP.
        """
        from torchspec.models.target.target_utils import load_synced_target_lm_head

        self.target_lm_head = load_synced_target_lm_head(
            target_model_path,
            load_norm=self._last_hs_prenorm,
            lm_head_key=getattr(self.args, "lm_head_key", "lm_head.weight"),
            norm_key=getattr(self.args, "norm_key", "model.norm.weight"),
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
        )

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    # 把 [B,S] token 和 [B,S,H] teacher 末层特征向左移一位并尾部补零，与下一 token 监督对齐。
    # pre-norm 特征先经冻结 norm；裁剪词表路径计算压缩概率，全词表路径传递惰性目标避免常驻 [B,S,V]。
    # 辅助层特征不做同样的左移；位置、attention mask、loss mask 一起传给训练 wrapper。
    def _forward(
        self, batch: dict
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        input_ids = padding(batch["input_ids"], left=False).cuda()
        target_hidden_states = padding(batch["last_hidden_states"], left=False).cuda()

        if self.verifier_norm is not None:
            with torch.no_grad():
                target_hidden_states = self.verifier_norm(target_hidden_states)

        # loss_mask 保留原位置布局，具体各展开深度的可监督位置由 Eagle wrapper 处理。
        # 这里仅移除兼容输入的末尾单例维，不能像 input_ids 一样无条件再左移一次。
        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)
        loss_mask = loss_mask.cuda()

        # t2d 标识目标词表中保留的 token；裁剪后的监督概率与 draft head 的输出维必须一致。
        if self.t2d is not None:
            target = compute_target_p_padded(
                target_hidden_states=target_hidden_states,
                target_lm_head_weight=self.target_lm_head_weight,
                t2d=self.t2d,
                loss_mask=loss_mask,
                length=self.eagle3.length,
            )
        else:
            target = compute_lazy_target_padded(
                target_hidden_states,
                self.target_lm_head_weight,
                self.eagle3.length,
            )
        # 显式释放本地引用，避免不再需要的中间量延长驻留。
        # 但惰性目标若仍引用其底层 tensor，实际存储仍会存活到目标计算完成；del 不是强制释放 CUDA 内存。
        del target_hidden_states

        # 模型输入同时包括 student 的辅助特征和 teacher 监督目标，两条用途不同。
        # 返回 plosses 用于 backward，vlosses 用于日志；统计 count 必须保留，才能跨不同 mask 密度正确池化。
        plosses, vlosses, acces, acc_counts, alphas = self.model(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"].cuda(),
            target=target,
            loss_mask=loss_mask,
            hidden_states=batch["hidden_states"].cuda(),
            position_ids=batch.get("position_ids").cuda()
            if batch.get("position_ids") is not None
            else None,
        )
        return plosses, vlosses, acces, acc_counts, alphas

    # 将各展开深度 loss 乘权重后求和，再除 microbatch 累积次数并 backward。
    # 注意这里不除权重总和；日志 avg_loss 除权重和，因而日志尺度与实际反向标量不同。
    def _backward(self, plosses: List[torch.Tensor], accumulation_steps: int = 1) -> torch.Tensor:
        ploss = (
            sum(self._ploss_weights[i] * plosses[i] for i in range(len(plosses)))
            / accumulation_steps
        )
        ploss.backward()
        return ploss

    # ------------------------------------------------------------------
    # Eval (no-grad forward on CPU-cached data)
    # ------------------------------------------------------------------

    # no_grad 前向返回逐深度 loss、accuracy、有效计数和 alpha 的 detached 向量，不更新优化器。
    def eval_forward(self, batch: dict) -> dict:
        """Single forward pass without backward — returns per-position metrics."""
        with torch.no_grad():
            _plosses, vlosses, acces, acc_counts, alphas = self._forward(batch)
        return {
            "vlosses": torch.stack(vlosses).detach(),
            "acces": torch.stack(acces).detach(),
            "acc_counts": torch.stack(acc_counts).detach(),
            "alphas": torch.stack(alphas).detach(),
        }

    # 把 CPU 缓存按评估 batch 大小重新 collate，转 GPU 后收集指标；结束后恢复 train 模式。
    def eval_from_cache(self) -> dict:
        """Run forward-only eval over all CPU-cached eval samples.

        Samples are stored individually (no padding). We re-collate them into
        batches of ``eval_micro_batch_size`` (or ``micro_batch_size``) so the
        eval forward batch size is independent of cache generation throughput.
        """
        if not getattr(self, "_eval_cache", None):
            return {}

        eval_mbs = getattr(self.args, "eval_micro_batch_size", None) or self.args.micro_batch_size

        self.model.eval()
        all_metrics: list[dict] = []
        for i in range(0, len(self._eval_cache), eval_mbs):
            chunk = self._eval_cache[i : i + eval_mbs]
            batch = self._eval_collator(chunk)
            gpu_batch = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            all_metrics.append(self.eval_forward(gpu_batch))

        # Switch back to training mode
        self.model.train()

        return self._aggregate_eval_metrics(all_metrics)

    # accuracy 和 alpha 先乘有效 token 数、跨 chunk/rank 累加，再统一相除。
    # 例如 1 个 token 全对与 9 个 token 全错应得 0.1，不能把两个 batch accuracy 平均成 0.5。
    # vloss 则保持 chunk 均值再跨 rank AVG；连续准确率乘积之和只用于估计接受长度。
    def _aggregate_eval_metrics(self, all_step_metrics: list[dict]) -> dict:
        if not all_step_metrics:
            return {}

        avg_vlosses = torch.stack([m["vlosses"] for m in all_step_metrics]).mean(dim=0)
        # Weight by acc_counts (valid-token count per eval-cache chunk) rather than
        # averaging per-chunk accuracies directly — chunks can have very different
        # numbers of loss positions, so a naive mean-of-means over- or under-weights
        # small chunks relative to a true token-weighted mean.
        acc_counts = torch.stack([m["acc_counts"] for m in all_step_metrics]).sum(dim=0)
        acc_correct = torch.stack([m["acces"] * m["acc_counts"] for m in all_step_metrics]).sum(
            dim=0
        )
        alpha_weighted = torch.stack([m["alphas"] * m["acc_counts"] for m in all_step_metrics]).sum(
            dim=0
        )

        # 所有 rank 都参与归约，不能只让 rank 0 执行此函数。
        # 损失按当前实现先求本地 chunk 均值，再 AVG；准确率/alpha 是总分子/总有效数。
        # 两种统计口径并不相同，特别是最后一个评估 batch 较小时。
        dist.all_reduce(avg_vlosses, op=dist.ReduceOp.AVG)
        dist.all_reduce(acc_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(alpha_weighted, op=dist.ReduceOp.SUM)

        denom = acc_counts.clamp_min(1.0)
        avg_acces = acc_correct / denom
        avg_alphas = alpha_weighted / denom

        avg_acc_scalar = avg_acces.mean().item()
        avg_alpha_scalar = avg_alphas.mean().item()

        # 此指标假设用各深度 accuracy 的乘积近似连续正确的概率。
        # 例如 a=[0.8,0.5,0.25]，结果为 0.8+0.4+0.1=1.3；
        # 真实接受率还受路径相关性、采样分布等影响，不能把 1.3 当成现场解码测量。
        cumulative = 1.0
        simulated_acc_len = 0.0
        for i in range(avg_acces.shape[0]):
            cumulative *= avg_acces[i].item()
            simulated_acc_len += cumulative

        ploss_weights = torch.tensor(
            _position_decay_weights(
                avg_vlosses.shape[0], getattr(self.args, "ploss_weights", None)
            ),
            device=avg_vlosses.device,
        )
        weighted_avg_loss = (avg_vlosses * ploss_weights).sum().item() / ploss_weights.sum().item()

        metrics: dict = {
            "eval/avg_loss": weighted_avg_loss,
            "eval/avg_acc": avg_acc_scalar,
            "eval/avg_alpha": avg_alpha_scalar,
            "eval/simulated_acc_len": simulated_acc_len,
        }
        for i in range(avg_vlosses.shape[0]):
            metrics[f"eval/ploss_{i}"] = avg_vlosses[i].item()
            metrics[f"eval/acc_{i}"] = avg_acces[i].item()
            metrics[f"eval/alpha_{i}"] = avg_alphas[i].item()

        if dist.get_rank() == 0:
            logger.info(
                f"eval: loss={weighted_avg_loss:.4f}, acc={avg_acc_scalar:.4f}, "
                f"sim_acc_len={simulated_acc_len:.2f}"
            )

        return metrics

    # ------------------------------------------------------------------
    # Subclass contract implementations
    # ------------------------------------------------------------------

    # 调用前向和按累积次数缩放的反向；返回 detached 指标避免汇总列表持有整张计算图。
    def _train_step(
        self,
        batch: dict,
        accumulation_steps: int,
        step: int,
        batch_idx: int,
        num_batches: int,
    ) -> dict:
        plosses, vlosses, acces, acc_counts, alphas = self._forward(batch)
        total_loss = self._backward(plosses, accumulation_steps=accumulation_steps)

        return {
            "plosses": torch.stack(plosses).detach(),
            "vlosses": torch.stack(vlosses).detach(),
            "acces": torch.stack(acces).detach(),
            "acc_counts": torch.stack(acc_counts).detach(),
            "alphas": torch.stack(alphas).detach(),
            "plosses_raw": [p.detach() for p in plosses],
            "acces_raw": [a.detach() for a in acces],
            "total_loss": total_loss.detach(),
        }

    # 把 Eagle 每深度 loss/accuracy 与公共循环抓取的梯度、权重写入统一 debug dump。
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
        dump_eagle3_batch(
            self.args,
            step=step,
            batch_idx=batch_idx,
            batch=batch,
            plosses=step_metrics.get("plosses_raw"),
            acces=step_metrics.get("acces_raw"),
            gradients=gradients,
            total_loss=step_metrics.get("total_loss"),
            model_weights=model_weights,
        )

    # 每优化器步更新 MoE router bias；对准确率/alpha 累加分子分母，对 vloss 取均值。
    # 把所有指标打包成一个 float32 张量，用两个 pinned CPU buffer 轮换非阻塞复制，交给下一步兑现。
    def _aggregate_metrics(
        self, all_step_metrics: list[dict], step: int, *, grad_norm: torch.Tensor = None
    ) -> dict | DeferredTrainMetrics:
        if not all_step_metrics:
            return {}

        # Runs once per optimizer step, on every rank.
        for router in (m for m in self.draft_model.modules() if isinstance(m, Eagle4Router)):
            imbalance = router.update_bias()
            if step % 50 == 0:
                logger.info(f"step {step}: MoE max/mean expert load {imbalance:.2f}")

        avg_vlosses = torch.stack([m["vlosses"] for m in all_step_metrics]).mean(dim=0)
        # Weight by acc_counts instead of averaging per-accumulation-step accuracies
        # directly: microbatches can carry very different numbers of loss positions,
        # so a mean-of-means over/under-weights small microbatches relative to a true
        # token-weighted mean across both gradient-accumulation steps and DP ranks.
        acc_counts_sum = torch.stack([m["acc_counts"] for m in all_step_metrics]).sum(dim=0)
        acc_correct_sum = torch.stack([m["acces"] * m["acc_counts"] for m in all_step_metrics]).sum(
            dim=0
        )
        alpha_weighted_sum = torch.stack(
            [m["alphas"] * m["acc_counts"] for m in all_step_metrics]
        ).sum(dim=0)

        num_depths = avg_vlosses.shape[0]
        # 把三个长度为 depth 的向量拼成一次归约，减少小标量 collective 的次数。
        # 打包顺序也是 _build_train_metrics 的解包协议；修改两端任一端都要保持一致。
        weighted_counts = torch.cat((acc_correct_sum, acc_counts_sum, alpha_weighted_sum))
        dist.all_reduce(avg_vlosses, op=dist.ReduceOp.AVG)
        dist.all_reduce(weighted_counts, op=dist.ReduceOp.SUM)

        grad_norm_value = (
            grad_norm.to(device=avg_vlosses.device, dtype=avg_vlosses.dtype)
            if grad_norm is not None
            else avg_vlosses.new_zeros(())
        )
        packed_metrics = (
            torch.cat(
                (
                    avg_vlosses,
                    weighted_counts,
                    grad_norm_value.reshape(1),
                )
            )
            .detach()
            .float()
        )
        metric_step = self.global_step
        learning_rate = self.optimizer.get_learning_rate()

        # 闭包固定本次 global_step、学习率和深度数，延迟读取时仍生成本步而非下步的指标。
        def finalize_values(metric_values: list[float]) -> dict:
            return self._build_train_metrics(
                metric_values,
                num_depths=num_depths,
                metric_step=metric_step,
                learning_rate=learning_rate,
            )

        if packed_metrics.device.type != "cuda":
            return finalize_values(packed_metrics.tolist())

        # 每个 CPU buffer 大小为 4*depth+1：loss、correct、count、alpha_sum 和范数。
        # 首次分配后反复使用，假定本次训练的深度固定；pin_memory 是非阻塞 D2H 的关键前提。
        if not self._metric_buffers:
            self._metric_buffers = [
                torch.empty(
                    packed_metrics.numel(),
                    dtype=torch.float32,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(2)
            ]
        # 两个 buffer 对应交替步号：本步复制不覆盖仍待兑现的上一步 CPU 数值。
        cpu_values = self._metric_buffers[metric_step % len(self._metric_buffers)]
        cpu_values.copy_(packed_metrics, non_blocking=True)
        ready_event = torch.cuda.Event()
        # 事件记录在 copy_ 之后，所以等待事件意味着 CPU 数组可读取。
        # DeferredTrainMetrics 同时持有 packed_metrics 的强引用，防止源存储在异步复制完成前被回收复用。
        ready_event.record()
        return DeferredTrainMetrics(
            cpu_values=cpu_values,
            source_values=packed_metrics,
            ready_event=ready_event,
            metric_step=metric_step,
            finalize_values=finalize_values,
        )

    # 解包 [loss, correct, count, alpha_sum, grad_norm]，用 max(count,1) 避免空监督除零。
    # 连续接受长度近似为 a0+a0*a1+…；如准确率 [0.8,0.5] 得 1.2，并非实测解码吞吐。
    def _build_train_metrics(
        self,
        metric_values: list[float],
        *,
        num_depths: int,
        metric_step: int,
        learning_rate: float,
    ) -> dict:
        ploss_values = metric_values[:num_depths]
        acc_correct_values = metric_values[num_depths : 2 * num_depths]
        acc_counts_values = metric_values[2 * num_depths : 3 * num_depths]
        alpha_weighted_values = metric_values[3 * num_depths : 4 * num_depths]
        grad_norm_scalar = metric_values[-1]

        acc_values = [
            correct / max(count, 1.0)
            for correct, count in zip(acc_correct_values, acc_counts_values)
        ]
        alpha_values = [
            alpha / max(count, 1.0)
            for alpha, count in zip(alpha_weighted_values, acc_counts_values)
        ]

        # 现在已经是 Python float 列表，求均值/构造字典不会再触发每个 CUDA 标量一次同步。
        # 此处 loss 除权重和用于稳定展示尺度，但 _backward 的加权和未除权重和。
        weighted_avg_loss_value = (
            sum(loss * weight for loss, weight in zip(ploss_values, self._ploss_weights))
            / self._ploss_weight_sum
        )
        avg_acc_scalar = sum(acc_values) / num_depths
        avg_alpha_scalar = sum(alpha_values) / num_depths
        cumulative = 1.0
        simulated_acc_len_value = 0.0
        for acc in acc_values:
            cumulative *= acc
            simulated_acc_len_value += cumulative

        # metric_step 和 learning_rate 来自创建延迟对象时的快照。
        # 若此时重新读取 optimizer 当前 lr/global_step，日志会被错误地标到已经开始的下一步。
        metrics = {
            "train/avg_loss": weighted_avg_loss_value,
            "train/avg_acc": avg_acc_scalar,
            "train/avg_alpha": avg_alpha_scalar,
            "train/simulated_acc_len": simulated_acc_len_value,
            "train/grad_norm": grad_norm_scalar,
            "train/global_step": metric_step,
            "train/lr": learning_rate,
            "train/step": metric_step,
        }

        for i in range(num_depths):
            metrics[f"train/ploss_{i}"] = ploss_values[i]
            metrics[f"train/acc_{i}"] = acc_values[i]
            metrics[f"train/alpha_{i}"] = alpha_values[i]

        if dist.get_rank() == 0:
            logger.debug(f"step {metric_step}: {metrics}")

        return metrics
