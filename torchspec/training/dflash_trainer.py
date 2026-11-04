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

# DFlash 系列共用训练器：构造 block/anchor 目标，支持 DFlash2 与 DSpark 通过 hook 更换网络。
# 其梯度按整个累积窗口的有效监督分母归一化，不能直接套用逐 batch 平均 loss 的直觉。

"""DFlash trainer — extends Trainer with DFlash-specific model init, forward, and metrics."""

from argparse import Namespace
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from torchspec.models.dflash import DFlashModel
from torchspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from torchspec.training import checkpoint
from torchspec.training.fsdp import apply_fsdp2, fsdp2_load_full_state_dict
from torchspec.training.optimizer import BF16Optimizer
from torchspec.training.trainer import Trainer
from torchspec.utils.distributed import get_gloo_group
from torchspec.utils.logging import logger


class DFlashTrainer(Trainer):
    """DFlash-specific trainer.

    Extends ``Trainer`` with DFlash model initialisation (dual-source KV draft model),
    forward/backward with anchor sampling + block-causal mask, and metric aggregation.

    DSparkTrainer is a subclass that overrides the build hooks:
    - `_draft_config_class`
    - `_build_draft_model`
    - `_build_training_wrapper`
    """

    _draft_config_class = DFlashConfig
    _anchor_slot_offset = 1
    _extra_loss_component_keys: list[str] = []

    # 按配置构造基础 DFlash draft；子类覆盖此 hook 时仍复用公共初始化及优化器流程。
    def _build_draft_model(self, config):
        """Instantiate the draft network. Overridden by subclasses."""
        return DFlashDraftModel(config)

    # 把 draft 与 anchor 数、block 大小、目标类型和损失权重封装成 DFlashModel。
    def _build_training_wrapper(self, draft_model):
        """Wrap the draft network with the training-objective module."""
        return DFlashModel(
            draft_model=draft_model,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            loss_objective=self.loss_objective,
            dpace_alpha=self.dpace_alpha,
            loss_decay_gamma=self.loss_decay_gamma,
            ce_loss_alpha=self.ce_loss_alpha,
            l1_loss_alpha=self.l1_loss_alpha,
        )

    # 记录目标层数、anchor/block 参数和损失配置，并初始化冻结 teacher head/norm 的可选状态。
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.target_lm_head: Optional[torch.nn.Module] = None
        self.num_target_layers = getattr(args, "dflash_num_target_layers", 5)
        self.block_size = getattr(args, "dflash_block_size", 16)
        self.num_anchors = getattr(args, "dflash_num_anchors", 512)
        self.loss_objective = getattr(args, "dflash_loss_objective", "decay")
        self.dpace_alpha = getattr(args, "dflash_dpace_alpha", 0.5)
        self.loss_decay_gamma = getattr(args, "dflash_loss_decay_gamma", 7.0)
        self.ce_loss_alpha = getattr(args, "dflash_ce_loss_alpha", 1.0)
        self.l1_loss_alpha = getattr(args, "dflash_l1_loss_alpha", 0.0)
        self._last_hs_prenorm = getattr(args, "last_hidden_states_prenorm", False)
        self.verifier_norm: Optional[torch.nn.Module] = None

    # 接受路径、字典或配置对象，补齐 teacher 层数等信息后构造 draft。
    # rank 0 加载初始权重及 teacher embedding，冻结 embedding，再按 dp_mesh 包装和广播。
    # 建立优化器/调度器并恢复状态，最后加载目标 head；可选 torch.compile 后取原始 wrapper 访问模型字段。
    def init_model(
        self,
        draft_model_config,
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

        with init_context():
            cfg_cls = self._draft_config_class

            # 路径、dict 和已解析配置对象最终统一为同一 config，后续构建 hook 不需重复处理三种输入。
            # config 的目标层数、hidden size 与实际 teacher 输出形状必须一致，否则最后维拆分会错位。
            if isinstance(draft_model_config, str):
                config = cfg_cls.from_pretrained(draft_model_config)
            elif isinstance(draft_model_config, dict):
                config = cfg_cls(**draft_model_config)
            elif isinstance(draft_model_config, DFlashConfig):
                config = draft_model_config
            else:
                raise TypeError(
                    f"Unsupported draft_model_config type: {type(draft_model_config).__name__}. "
                    f"Expected str, dict, or {cfg_cls.__name__}."
                )

            if not hasattr(config, "num_target_layers") or config.num_target_layers is None:
                config.num_target_layers = self.num_target_layers
            if not hasattr(config, "target_hidden_size") or config.target_hidden_size is None:
                config.target_hidden_size = config.hidden_size
            if (
                not hasattr(config, "target_num_hidden_layers")
                or config.target_num_hidden_layers is None
            ):
                from transformers import AutoConfig

                target_config = AutoConfig.from_pretrained(
                    target_model_path,
                    trust_remote_code=getattr(self.args, "trust_remote_code", True),
                )
                config.target_num_hidden_layers = target_config.num_hidden_layers

            draft_model = self._build_draft_model(config)

        if dist.get_rank() == 0:
            initial_path = getattr(self.args, "initial_draft_model_path", None)
            if initial_path:
                checkpoint.load_initial_dflash_weights(draft_model, initial_path)
            draft_model.load_embedding(
                target_model_path,
                embedding_key=getattr(self.args, "embedding_key", "model.embed_tokens.weight"),
            )

        # embedding 由 teacher 提供并冻结；draft 主体转 BF16 以降低前向与激活开销。
        # 优化器后面另外建立 FP32 master，模型 dtype 并不说明 Adam 的更新也以 BF16 累计。
        draft_model.freeze_embedding()
        draft_model = draft_model.to(torch.bfloat16)

        dist.barrier(group=get_gloo_group())

        frozen_count = sum(p.numel() for p in draft_model.parameters() if not p.requires_grad)
        trainable_count = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        logger.info(
            f"[Rank {self.dp_rank}] DFlash draft model: {trainable_count:,} trainable, "
            f"{frozen_count:,} frozen (embedding) parameters"
        )

        dflash_model = self._build_training_wrapper(draft_model)

        # 非零 rank 在 meta 上建模，只知道形状；barrier 仅对齐进度，不会传播权重值。
        # 真正把 rank 0 的 teacher embedding/初始 draft 同步给其他 rank 的是下面的
        # fsdp2_load_full_state_dict；因此不能用 barrier 代替该状态加载步骤。
        full_state = dflash_model.state_dict() if dist.get_rank() == 0 else {}

        dflash_model = apply_fsdp2(
            dflash_model,
            mesh=self.dp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
            modules_to_shard=list(draft_model.layers),
        )

        dflash_model = fsdp2_load_full_state_dict(
            dflash_model,
            full_state,
            self.dp_mesh,
            cpu_offload=True if self.fsdp_cpu_offload else None,
        )

        # compile 在 FSDP 包装和权重广播之后，模型属性访问则解开 _orig_mod/module。
        # 训练调用仍走 compiled wrapper，访问 self.dflash 只是取得目标配置及 draft 引用。
        if getattr(self.args, "compile_model", False):
            logger.info("Compiling DFlash model with torch.compile (inductor backend)")
            dflash_model = torch.compile(dflash_model)

        self.model = dflash_model
        # Unwrap torch.compile and/or DDP module wrappers to access underlying DFlashModel
        _unwrapped = getattr(self.model, "_orig_mod", self.model)  # torch.compile
        self.dflash = getattr(_unwrapped, "module", _unwrapped)  # DDP/replicate
        self.draft_model = self.dflash.draft_model

        total_steps = self.args.lr_total_steps
        decay_style = getattr(self.args, "lr_decay_style", "cosine")
        warmup_ratio = getattr(self.args, "warmup_ratio", 0.1)

        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 0.0),
            max_grad_norm=self.args.max_grad_norm,
            warmup_ratio=warmup_ratio,
            total_steps=total_steps,
            decay_style=decay_style if decay_style != "WSD" else "cosine",
            min_lr=getattr(self.args, "min_lr", 0.0),
        )

        # 普通优化器构造先使用兼容曲线，WSD 分支再替换 scheduler。
        # self.lr_scheduler 在此分支后读取，确保 checkpoint 保存的是实际在用的调度器。
        if decay_style == "WSD" and total_steps:
            from torchspec.training.lr_scheduler import LRSchedulerWithWarmup

            wsd_ratio = getattr(self.args, "wsd_decay_ratio", 0.2)
            self.optimizer.scheduler = LRSchedulerWithWarmup(
                self.optimizer.optimizer,
                max_lr=self.args.learning_rate,
                total_steps=total_steps,
                warmup_steps=int(warmup_ratio * total_steps),
                decay_style="WSD",
                min_lr=getattr(self.args, "min_lr", 0.0),
                wsd_decay_steps=int(wsd_ratio * total_steps),
                wsd_decay_style=getattr(self.args, "wsd_decay_style", "cosine"),
            )

        self.lr_scheduler = self.optimizer.lr_scheduler

        checkpoint_payload = checkpoint.load(self)
        checkpoint.finalize_load(self, checkpoint_payload)

        # teacher head 独立于可训练 draft，用于把 teacher/预测特征映射到监督空间。
        # 它的权重被冻结，不应出现在 draft optimizer 的参数列表中。
        self._init_target_lm_head(target_model_path)

        if self.target_lm_head is None:
            raise ValueError(
                "target_lm_head is required but was None. Ensure _init_target_lm_head succeeded."
            )
        # 冻结 W 不等于禁用 F.linear 的 autograd：draft logits = h @ W.T，
        # 仍需通过固定 W 把 dL/dlogits 传回 h，即 dL/dh = dL/dlogits @ W。
        # 因此这里只冻结 head 权重，不能把整个 draft→logits→loss 放进 no_grad。
        self.target_lm_head_weight = self.target_lm_head.lm_head.weight

        self.prof.on_init_end()

        logger.info(f"[Rank {self.dp_rank}] DFlash model initialized with FSDP2")

        return 0

    # ------------------------------------------------------------------
    # Target LM head (same as Eagle3Trainer)
    # ------------------------------------------------------------------

    # teacher head 总是加载；norm 仅在输入是 pre-norm 且当前目标实际使用末层特征时加载。
    # 这与 _forward 的条件一致，防止既没加载 norm 却使用未归一化特征。
    def _init_target_lm_head(self, target_model_path: str) -> None:
        from torchspec.models.target.target_utils import load_synced_target_lm_head

        # The norm exists to make the target's pre-norm hidden states usable, so a
        # run whose objective never reads them must not fail on loading it.
        load_norm = self._last_hs_prenorm and self.dflash.uses_target_hidden_states
        if self._last_hs_prenorm and not load_norm:
            logger.info(
                "[Rank %s] last_hidden_states_prenorm is set, but the objective does not read "
                "the target's last hidden states; skipping the verifier norm",
                self.dp_rank,
            )

        self.target_lm_head = load_synced_target_lm_head(
            target_model_path,
            load_norm=load_norm,
            lm_head_key=getattr(self.args, "lm_head_key", "lm_head.weight"),
            norm_key=getattr(self.args, "norm_key", "model.norm.weight"),
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
        )
        self.verifier_norm = self.target_lm_head.norm

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    # 沿最后维拆 [B,S,L*H] 为 L 个 [B,S,H] 层特征；顺序必须与生产端拼接层顺序一致。
    def _split_hidden_states(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Split concatenated hidden states [B, seq_len, num_layers*D] into per-layer list.

        The target model concatenates hidden states from `num_target_layers` layers
        along the last dimension. We split them back into a list of [B, seq_len, D] tensors.
        """
        total_dim = hidden_states.shape[-1]
        # 这里没有校验 total_dim 能否整除层数，输入契约由配置和生产端共同保证。
        # 例如 12 维/3 层得到 3 片各 4 维；13 维/3 层却会 split 出 4,4,4,1，
        # 所以“函数返回 list”本身不能证明它恰好包含 num_target_layers 个合法层。
        per_layer_dim = total_dim // self.num_target_layers
        return list(hidden_states.split(per_layer_dim, dim=-1))

    # 将 token、mask、辅助特征搬到 CUDA，并按目标需要读取/归一化末层 teacher 特征。
    # 网络返回标量、逐位置统计、附加损失项及原始 (分子,分母)，后者用于全窗口梯度归一化。
    def _forward(
        self, batch: dict
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        device = torch.device("cuda")
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        hidden_states = batch["hidden_states"].to(device, non_blocking=True)

        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)
        loss_mask = loss_mask.to(device, non_blocking=True)

        # Gated on the same predicate as the norm load: feeding the loss an
        # unnormalised pre-norm tensor is silently wrong, so a run that skipped
        # the norm must not consume these at all.
        # 有些目标只需要硬 token/其他特征，不读取 teacher 末层 hidden state。
        # 初始化是否加载 verifier norm 与这里是否消费末层特征使用同一 predicate，避免 silent 数值偏移。
        last_hidden_states = None
        if self.dflash.uses_target_hidden_states and batch.get("last_hidden_states") is not None:
            last_hidden_states = batch["last_hidden_states"].to(device, non_blocking=True)
            if self.verifier_norm is not None:
                with torch.no_grad():
                    last_hidden_states = self.verifier_norm(last_hidden_states)

        # 例如 B=2,S=128,L=5,H=4096，输入末维为 20480，拆成 5 个 [2,128,4096]。
        # split 结果通常仍是共享存储的视图；删除原变量名不会释放这些视图还需要的存储。
        hidden_states_list = self._split_hidden_states(hidden_states)
        del hidden_states

        (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
            loss_terms,
        ) = self.model(
            input_ids=input_ids,
            hidden_states_list=hidden_states_list,
            loss_mask=loss_mask,
            lm_head_weight=self.target_lm_head_weight,
            last_hidden_states=last_hidden_states,
        )

        # loss/accuracy 用来展示，loss_terms 才是训练窗口归一化的原料。
        # 调用者不能只取 loss.backward() 替代原始分子累积，否则不同 mask 数量的 batch 权重会改变。
        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
            loss_terms,
        )

    # 把传入标量除 accumulation_steps 后 backward，返回未缩放标量；训练调用方传的是损失分子。
    def _backward(self, loss: torch.Tensor, accumulation_steps: int = 1) -> torch.Tensor:
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        return loss

    # 跨 dp_group 求有效分母总和 D；当前梯度已除累积次数 K 并经 FSDP 平均 R 次，故再乘 R*K/D。
    # 例如 R=2、K=4、D=80，当前分子梯度和/8 再乘 0.1，最终成为分子梯度和/80。
    # D=0 时用设备侧 where 将梯度置零，避免每步 .item() 同步和各 rank 分支分歧。
    def _normalize_accumulated_gradients(
        self,
        local_denominator: torch.Tensor,
        accumulation_steps: int,
    ) -> torch.Tensor:
        """Rescale accumulated gradients into a globally token-pooled mean.

        Backward ran on raw loss numerators, so gradients currently carry
        ``sum(numerator) / accumulation_steps``, and FSDP additionally *means*
        them over the shard group — hence the ``world_size * accumulation_steps``
        factor over the pooled denominator.

        The empty-window case goes through ``torch.where`` rather than a Python
        branch: a fully masked accumulation window is legal here (see the
        all-zero ``loss_mask`` fallback in ``DFlashModel``) and must not cost a
        device-to-host sync on every optimizer step.
        """
        # detach 阻断的是归一化分母的梯度路径；clone 另给 all_reduce 可原地修改的存储，
        # 防止把本地累积计数覆盖成全局计数。D-PACE 的权重也已在模型中按 no_grad 构造，
        # 本步骤优化的是“固定本次权重”的加权损失，不对权重选择策略本身求导。
        denominator = local_denominator.detach().clone()
        group = getattr(self, "dp_group", None)
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size(group=group)
            if world_size > 1:
                dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=group)

        # 归约后的梯度是 SUM(grad numerator)/(R*K)，以下系数把它变成 SUM(grad numerator)/D。
        scale = torch.where(
            denominator > 0,
            (world_size * accumulation_steps) / denominator.clamp(min=1e-6),
            torch.zeros_like(denominator),
        )
        with torch.no_grad():
            for parameter in self.optimizer.model_params:
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
        return denominator

    # ------------------------------------------------------------------
    # Eval (no-grad forward on CPU-cached data) — mirrors Eagle3Trainer
    # ------------------------------------------------------------------

    # 仅做 no_grad 前向，把逐位置损失/准确率/计数及附加分子分母交给评估汇总。
    def eval_forward(self, batch: dict) -> dict:
        """Single forward pass without backward — returns per-position tensors."""
        with torch.no_grad():
            (
                _,
                _,
                loss_per_position,
                acc_per_position,
                count_per_position,
                loss_components,
                _,
            ) = self._forward(batch)
        return {
            "loss_pp": loss_per_position.detach(),
            "acc_pp": acc_per_position.detach(),
            "count_pp": count_per_position.detach(),
            **loss_components,
        }

    # 先把逐位置均值乘 count 恢复分子，跨 microbatch 和默认进程组 SUM，再除总 count。
    # 空位置用 count.clamp(min=1) 得到零值，返回计数供后续跨位置加权。
    def _reduce_position_metrics(
        self,
        all_step_metrics: list[dict],
        *,
        loss_key: str,
        acc_key: str,
        count_key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 每份输入是 [block_size]，stack 后为 [microbatch数, block_size]。
        # 同一槽位若两批均值 2、5，计数 3、1，则池化为 (2*3+5*1)/4=2.75，
        # 不是 (2+5)/2=3.5；无监督位置 count=0，不应获得一个批次的投票权。
        loss_sum_pp = torch.stack([m[loss_key] * m[count_key] for m in all_step_metrics]).sum(dim=0)
        correct_sum_pp = torch.stack([m[acc_key] * m[count_key] for m in all_step_metrics]).sum(
            dim=0
        )
        count_pp = torch.stack([m[count_key] for m in all_step_metrics]).sum(dim=0)

        dist.all_reduce(loss_sum_pp, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_sum_pp, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_pp, op=dist.ReduceOp.SUM)

        safe_count_pp = count_pp.clamp(min=1.0)
        avg_loss_pp = loss_sum_pp / safe_count_pp
        avg_acc_pp = correct_sum_pp / safe_count_pp
        return avg_loss_pp, avg_acc_pp, count_pp

    # accuracy 按所有预测位置 token 数加权；loss 额外按 exp(-k/gamma) 衰减位置。
    # 这是日志标量的计算规则，不应据此推断所有目标函数都优化同一标量。
    def _compute_scalar_metrics(
        self,
        pred_loss_pp: torch.Tensor,
        pred_acc_pp: torch.Tensor,
        pred_count_pp: torch.Tensor,
    ) -> Tuple[float, float]:
        safe_total_count = pred_count_pp.sum().clamp(min=1.0)
        avg_acc = ((pred_acc_pp * pred_count_pp).sum() / safe_total_count).item()

        gamma = self.loss_decay_gamma
        if gamma is not None and gamma > 0:
            k = torch.arange(pred_loss_pp.shape[0], device=pred_loss_pp.device)
            weights = torch.exp(-k.float() / gamma)
        else:
            weights = torch.ones_like(pred_loss_pp)

        weighted_counts = pred_count_pp * weights
        safe_weighted_count = weighted_counts.sum().clamp(min=1.0)
        avg_loss = ((pred_loss_pp * weighted_counts).sum() / safe_weighted_count).item()
        return avg_loss, avg_acc

    # 对各附加损失的 (numerator,denominator) 一起打包归约，仅最后相除并同步到 CPU。
    # 若先平均各 microbatch 的均值，会把监督稀疏的小 batch 赋予过大权重。
    def _reduce_loss_components(self, all_step_metrics: list[dict], prefix: str) -> dict:
        """Reduce extra loss components into ``{prefix}{key}`` token-pooled means.

        Components arrive as ``(numerator, denominator)`` pairs and are pooled the
        same way the optimized objective is: sum both sides over the accumulation
        window and the data-parallel group, then divide once. Averaging pre-divided
        per-micro-batch means instead would weight a sparsely supervised
        micro-batch the same as a dense one, so the component curves could diverge
        from the loss actually being optimized.

        All keys share one all-reduce and one host sync.
        """
        keys: list[str] = []
        totals: list[torch.Tensor] = []
        # 附加项由具体模型类声明，例如 selector_loss 或 ce/l1/confidence_loss。
        # 每项的 denominator 可能对应不同监督区域，因此需要逐项保留自己的分母，不能套用主 loss 的 count。
        for key in self._extra_loss_component_keys:
            pairs = [m[key] for m in all_step_metrics if key in m]
            if not pairs:
                continue
            keys.append(key)
            totals.append(
                torch.stack(
                    [
                        torch.stack(
                            (
                                numerator.detach().float().reshape(()),
                                denominator.detach().float().reshape(()),
                            )
                        )
                        for numerator, denominator in pairs
                    ]
                ).sum(dim=0)
            )
        if not keys:
            return {}

        # 形状为 [组件数,2]；列 0 是分子，列 1 是分母。
        # 归约之后一次 tolist 同步，比为每个组件做两次 .item() 更少阻塞 CPU 发起后续任务。
        packed = torch.stack(totals)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        pooled = (packed[:, 0] / packed[:, 1].clamp(min=1e-6)).tolist()
        return {f"{prefix}{key}": value for key, value in zip(keys, pooled, strict=True)}

    # 按 eval_micro_batch_size 重组 CPU 样本，切 eval/no_grad 计算，再恢复 train 并汇总。
    def eval_from_cache(self) -> dict:
        """Run forward-only eval over all CPU-cached eval samples.

        Samples are stored individually (no padding). Re-collate into batches
        of ``eval_micro_batch_size`` (or ``micro_batch_size``) so the eval
        forward batch size is independent of cache generation throughput.
        """
        if not getattr(self, "_eval_cache", None):
            return {}

        eval_mbs = getattr(self.args, "eval_micro_batch_size", None) or self.args.micro_batch_size

        self.model.eval()
        all_metrics: list[dict] = []
        for i in range(0, len(self._eval_cache), eval_mbs):
            # 缓存由 batch_size=1 的 fetcher 建立，但其中可能已经含长度分桶 padding。
            # 此处只组合缓存项，不按 attention_mask 去 padding；collator 普通分支还会
            # 重建 attention_mask。解释“无 padding 样本”的 docstring 时应以这条实际数据流为准。
            chunk = self._eval_cache[i : i + eval_mbs]
            batch = self._eval_collator(chunk)
            gpu_batch = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            all_metrics.append(self.eval_forward(gpu_batch))

        self.model.train()

        return self._aggregate_eval_metrics(all_metrics)

    # 按 count 池化逐位置指标，去掉模型约定的 anchor 槽，再生成 loss、accuracy 与接受长度近似。
    def _aggregate_eval_metrics(self, all_step_metrics: list[dict]) -> dict:
        if not all_step_metrics:
            return {}

        avg_loss_pp, avg_acc_pp, count_pp = self._reduce_position_metrics(
            all_step_metrics,
            loss_key="loss_pp",
            acc_key="acc_pp",
            count_key="count_pp",
        )

        # Drop anchor slot (index 0) — see _aggregate_metrics for rationale.
        # 先跨 batch/rank 池化，再按模型约定去掉非预测槽。
        # 不能先把所有位置均值平均成一个标量，否则短序列尾部稀疏的位置会获得不合理权重。
        pred_loss_pp = avg_loss_pp[self._anchor_slot_offset :]
        pred_acc_pp = avg_acc_pp[self._anchor_slot_offset :]
        pred_count_pp = count_pp[self._anchor_slot_offset :]

        cumulative = 1.0
        simulated_acc_len = 0.0
        for i in range(pred_acc_pp.shape[0]):
            cumulative *= pred_acc_pp[i].item()
            simulated_acc_len += cumulative

        weighted_avg_loss, avg_acc = self._compute_scalar_metrics(
            pred_loss_pp, pred_acc_pp, pred_count_pp
        )

        metrics: dict = {
            "eval/avg_loss": weighted_avg_loss,
            "eval/avg_acc": avg_acc,
            "eval/simulated_acc_len": simulated_acc_len,
        }
        for i in range(pred_loss_pp.shape[0]):
            metrics[f"eval/ploss_{i}"] = pred_loss_pp[i].item()
            metrics[f"eval/acc_{i}"] = pred_acc_pp[i].item()

        # 额外损失用自身分子分母池化，和逐位置 accuracy 的 count 不混用。
        # 主指标主要描述预测质量，附加指标用于判断辅助 head/蒸馏项是否学习正常。
        metrics.update(self._reduce_loss_components(all_step_metrics, "eval/"))

        if dist.get_rank() == 0:
            logger.info(
                f"eval: loss={weighted_avg_loss:.4f}, acc={avg_acc:.4f}, "
                f"sim_acc_len={simulated_acc_len:.2f}"
            )

        return metrics

    # ------------------------------------------------------------------
    # Subclass contract implementations
    # ------------------------------------------------------------------

    # 为前反向记录 CUDA Event；累加每个 microbatch 的有效分母，对原始分子 backward。
    # 最后一个 microbatch 同步完成后才重标定全部梯度，公共基类随后进行裁剪和参数更新。
    def _train_step(
        self,
        batch: dict,
        accumulation_steps: int,
        step: int,
        batch_idx: int,
        num_batches: int,
    ) -> dict:
        evt_fwd_s = torch.cuda.Event(enable_timing=True)
        evt_fwd_e = torch.cuda.Event(enable_timing=True)
        evt_bwd_s = torch.cuda.Event(enable_timing=True)
        evt_bwd_e = torch.cuda.Event(enable_timing=True)

        evt_fwd_s.record()
        (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
            loss_terms,
        ) = self._forward(batch)
        evt_fwd_e.record()

        # 跨 batch 长度/mask 不同，分母不能假设相同；只反向分子，待窗口结束统一除总分母。
        # 对 base DFlash，分母是 objective_weights.sum()，未必是整数 token 数。
        # 例如两个有效槽的权重为 1 和 0.5，CE 为 2 和 4，则分子=4、分母=1.5；
        # count_per_position 仍各计 1，accuracy 也不因第二槽训练权重较小而减半计数。
        loss_numerator, loss_denominator = loss_terms
        # 累积窗口起点重置 denominator，后续 microbatch 原地相加。
        # 这里 detach 是因为分母只用于归一化计数，不应把它连接进训练计算图。
        if batch_idx == 0:
            self._accumulated_loss_denominator = loss_denominator.detach().clone()
        else:
            self._accumulated_loss_denominator.add_(loss_denominator.detach())

        evt_bwd_s.record()
        self._backward(loss_numerator, accumulation_steps=accumulation_steps)
        # 公共循环已在最后一个 microbatch 打开梯度同步；backward 返回后再统一乘归一化系数。
        # 优化器裁剪要在此之后，否则先裁未归一化的分子梯度会改变实际更新方向的尺度。
        if batch_idx == num_batches - 1:
            self._normalize_accumulated_gradients(
                self._accumulated_loss_denominator,
                accumulation_steps,
            )
        evt_bwd_e.record()

        # 前反向 Event 和 detached 指标留给窗口结束后汇总，原始图则可随当前 batch 释放。
        # 额外 loss_components 保留各自的 (numerator,denominator) 统计，不在此提前平均。
        return {
            "loss": loss.detach(),
            "accuracy": accuracy.detach(),
            "loss_per_position": loss_per_position.detach(),
            "acc_per_position": acc_per_position.detach(),
            "count_per_position": count_per_position.detach(),
            "total_loss": loss.detach(),
            "_fwd_events": (evt_fwd_s, evt_fwd_e),
            "_bwd_events": (evt_bwd_s, evt_bwd_e),
            **loss_components,
        }

    # 池化窗口指标后按 _anchor_slot_offset 选预测槽：DFlash 跳过第 0 槽，DSpark 从第 0 槽开始。
    # 输出每个深度和整体统计，并读取前反向 Event 时间；此路径的标量读取会同步设备。
    def _aggregate_metrics(
        self, all_step_metrics: list[dict], step: int, *, grad_norm: torch.Tensor = None
    ) -> dict:
        if not all_step_metrics:
            return {}

        avg_loss_pp, avg_acc_pp, count_pp = self._reduce_position_metrics(
            all_step_metrics,
            loss_key="loss_per_position",
            acc_key="acc_per_position",
            count_key="count_per_position",
        )

        # Skip index 0 (anchor slot, always zero); indices 1..B-1 are the
        # predicted tokens at 1..B-1 steps past the anchor. Re-index to 0..B-2
        # so the naming matches Eagle3 (acc_0 = first predicted token).
        pred_loss_pp = avg_loss_pp[self._anchor_slot_offset :]
        pred_acc_pp = avg_acc_pp[self._anchor_slot_offset :]
        pred_count_pp = count_pp[self._anchor_slot_offset :]

        # Simulated accepted length: acc_0 + acc_0*acc_1 + ... + prod(acc_0..acc_{B-2})
        # Models the expected number of consecutively accepted draft tokens.
        cumulative = 1.0
        simulated_acc_len = 0.0
        for i in range(pred_acc_pp.shape[0]):
            cumulative *= pred_acc_pp[i].item()
            simulated_acc_len += cumulative

        # 预测准确率按 count 池化；展示 loss 还带深度衰减权重。
        # 当训练目标不是简单 decay CE 时，这个展示标量不能替代实际被优化的目标函数定义。
        avg_loss, avg_acc = self._compute_scalar_metrics(pred_loss_pp, pred_acc_pp, pred_count_pp)

        metrics = {
            "train/avg_loss": avg_loss,
            "train/avg_acc": avg_acc,
            "train/simulated_acc_len": simulated_acc_len,
            "train/grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
            "train/global_step": self.global_step,
            "train/lr": self.optimizer.get_learning_rate(),
            "train/step": step,
        }

        for i in range(pred_loss_pp.shape[0]):
            metrics[f"train/ploss_{i}"] = pred_loss_pp[i].item()
            metrics[f"train/acc_{i}"] = pred_acc_pp[i].item()

        metrics.update(self._reduce_loss_components(all_step_metrics, "train/"))

        # Sub-timing breakdown (forward vs backward)
        # 前面多个 .item() 已使 CUDA 工作完成，才能安全读取事件间 elapsed_time。
        # 这些分项和基类 compute_time 的覆盖范围不同；optimizer 位于基类最后一段 compute 之内。
        fwd_ms = sum(
            m["_fwd_events"][0].elapsed_time(m["_fwd_events"][1])
            for m in all_step_metrics
            if "_fwd_events" in m
        )
        bwd_ms = sum(
            m["_bwd_events"][0].elapsed_time(m["_bwd_events"][1])
            for m in all_step_metrics
            if "_bwd_events" in m
        )
        metrics["perf/forward_time"] = fwd_ms / 1000.0
        metrics["perf/backward_time"] = bwd_ms / 1000.0

        if dist.get_rank() == 0 and (step % 5 == 0 or step <= 5):
            logger.info(
                f"COMPUTE_BREAKDOWN step={step}: forward={fwd_ms:.1f}ms backward={bwd_ms:.1f}ms"
            )

        if dist.get_rank() == 0:
            logger.debug(f"step {step}: {metrics}")

        return metrics
