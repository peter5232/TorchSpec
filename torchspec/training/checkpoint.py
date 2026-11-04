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

# 训练状态持久化与 warm start：DCP 保存分布式训练状态，safetensors 初始权重用于模型初始化。
# 恢复训练需匹配 Adam 状态与 FP32 master；continual_training 则保留新超参数并重置训练进度。

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from torchspec.models.draft.eagle4_moe import stack_latent_experts
from torchspec.models.draft.keymap import to_internal_keys
from torchspec.utils.logging import logger


class ModelState(Stateful):
    """Wrapper for model state only."""

    # 保存被 FSDP/replicate 包装的模型引用，作为 DCP Stateful 接口的适配器。
    def __init__(self, model):
        self.model = model

    # 通过分布式 get_state_dict 导出模型状态，而非假设每个 rank 都持有完整普通 tensor。
    def state_dict(self):
        model_state_dict, _ = get_state_dict(self.model, optimizers=[])
        return {"model": model_state_dict}

    # 将 model 字段交给 set_state_dict 恢复模型分布式布局，不触及优化器。
    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, optimizers=[], model_state_dict=state_dict["model"], optim_state_dict=None
        )


# 按 named_parameters 中可训练参数的顺序生成名字，与 BF16Optimizer 的 master 列表一一对应。
def _fp32_param_names(optimizer) -> list[str]:
    """Names for a ``BF16Optimizer``'s fp32 master copies, in ``fp32_params`` order."""
    return [name for name, param in optimizer.model.named_parameters() if param.requires_grad]


class OptimizerState(Stateful):
    """Wrapper for optimizer + fp32 master params."""

    # 保存模型/优化器引用，并判断是否使用带独立 FP32 master 的封装，以选择对应序列化方式。
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self._is_bf16_optimizer = hasattr(optimizer, "sync_fp32_params_from_model")

    # BF16 路径额外按参数名保存 master，避免冻结策略改变时仅靠列表位置误配。
    # 普通优化器则由分布式 get_state_dict 提取优化器状态。
    def state_dict(self):
        if self._is_bf16_optimizer:
            names = _fp32_param_names(self.optimizer)
            return {
                "optim": self.optimizer.state_dict(),
                # Keyed by name: which parameters are trainable can differ between the run
                # that saved and the run that loads.
                "fp32_params": {
                    name: param.data
                    for name, param in zip(names, self.optimizer.fp32_params, strict=True)
                },
            }
        _, optimizer_state_dict = get_state_dict(self.model, optimizers=[self.optimizer])
        return {"optim": optimizer_state_dict}

    # 先恢复优化器，再按名字把 master 拷回；若当前可训练参数在快照中缺失则报错。
    # 参数冻结集合不同的模型不能被当作完全相同的优化器续训，可改走 continual training。
    def load_state_dict(self, state_dict):
        if self._is_bf16_optimizer:
            self.optimizer.load_state_dict(state_dict["optim"])
            saved = state_dict.get("fp32_params")
            if saved:
                names = _fp32_param_names(self.optimizer)
                missing = [name for name in names if name not in saved]
                if missing:
                    raise ValueError(
                        "Optimizer checkpoint does not cover every trainable parameter: "
                        f"{missing[:5]}{'...' if len(missing) > 5 else ''}. The saved run trained "
                        "a different parameter set (for example it froze the lm_head and this one "
                        "does not). Use training.continual_training=true to start a fresh "
                        "optimizer from those weights instead of resuming this one."
                    )
                with torch.no_grad():
                    for name, master in zip(names, self.optimizer.fp32_params, strict=True):
                        master.data.copy_(saved[name])
                logger.info("Restored fp32 master params from checkpoint")
            return
        set_state_dict(
            self.model,
            optimizers=[self.optimizer],
            model_state_dict=None,
            optim_state_dict=state_dict["optim"],
        )


class LRSchedulerState(Stateful):
    """Wrapper for LR scheduler state only."""

    # 持有调度器引用，使 DCP 能独立保存学习率计数及配置状态。
    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    # 把 scheduler.state_dict 放在固定 lr_scheduler 字段中，与模型/优化器目录分离。
    def state_dict(self):
        return {"lr_scheduler": self.lr_scheduler.state_dict()}

    # 将保存的调度状态交还原 scheduler，恢复它的计数和曲线参数。
    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


# 读取 JSON 元数据；缺失或 JSON 损坏返回空字典，使兼容路径可用目录步号恢复。
def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse checkpoint metadata at {path}")
        return {}


# 先写相邻临时 JSON 文件，再原子替换目标，避免读者读到半份元数据。
def _write_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    tmp_path.replace(path)


# 初始权重路径必须指向现存 .safetensors 或含 model.safetensors 的目录；拼错即失败。
# 这比可选 resume 路径严格，防止用户以为 warm start 实际却从随机权重训练。
def resolve_initial_draft_checkpoint(path: str) -> Path:
    """Resolve ``model.initial_draft_model_path`` to the ``model.safetensors`` file it names.

    Either the file itself or the directory holding it is accepted; anything else is rejected
    rather than ignored, since a typo here would start a full training run from random weights.
    """
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.safetensors"
    if checkpoint_path.suffix != ".safetensors" or not checkpoint_path.is_file():
        raise ValueError(
            "initial_draft_model_path must be a serving-format model.safetensors file "
            f"(or the directory holding it), got {path!r}"
        )
    return checkpoint_path


# 转换专家键及裁剪 head；缺失的新 norm/MoE 部分以预定初值补齐。
# attention norm 变体用堆叠 QKV 的截断 SVD 拆成共享 eh_proj 和窄 Q/K/V，因此不是逐位等价转换。
@torch.no_grad()
def _upgrade_to_eagle4(tensors: dict, expected: dict) -> dict:
    """Initialize the EAGLE-4 parts a published EAGLE3 or EAGLE-4 draft does not carry."""
    # Latent MoE experts are served one key per expert; keys the model does not own keep their
    # serving names through the first to_internal_keys.
    tensors = to_internal_keys(stack_latent_experts(tensors), expected)

    # Pruned head: draft row i predicts target token i + d2t[i]; other rows keep the head the
    # trainer seeded from the target.
    if "d2t" in tensors and "d2t" not in expected:
        # 举例：旧 draft 行 0、1 的 d2t 偏移为 3、6，它们对应 teacher 行 3、7。
        # 新全词表 head 先保留 expected 中其他行，只覆盖这两行，避免缺失词表随机混入旧 head。
        d2t = tensors.pop("d2t")
        tensors.pop("t2d", None)
        head = expected["lm_head.weight"].clone()
        # d2t 保存的是偏移而非绝对 token id：draft 行 i 映射到 teacher 行 i+d2t[i]。
        head[torch.arange(len(d2t)) + d2t] = tensors["lm_head.weight"].to(head.dtype)
        tensors["lm_head.weight"] = head

    # Post-MLP norm starts as the final norm, as SGLang's EAGLE-4 loader does.
    if "midlayer.post_mlp_layernorm.weight" in expected:
        tensors.setdefault("midlayer.post_mlp_layernorm.weight", tensors["norm.weight"])

    # attn_norm: SVD-factor the stacked EAGLE3 q/k/v into eh_proj (2H -> H) and H-wide q/k/v.
    eh_proj = "midlayer.eh_proj.weight"
    if eh_proj in expected and eh_proj not in tensors:
        qkv = [f"midlayer.self_attn.{p}_proj.weight" for p in "qkv"]
        sizes = [tensors[k].shape[0] for k in qkv]
        # 将 Q/K/V 按输出行堆叠，分解为 UΣVh；Vh 前 rank 行作为共享输入投影，
        # UΣ 前 rank 列按原 Q/K/V 输出行数拆回三个投影。
        # 若保留 rank 小于矩阵真实秩，这就是低秩近似，warm start 后仍需训练适应。
        u, s, vh = torch.linalg.svd(torch.cat([tensors[k] for k in qkv]).float(), False)
        rank = expected[eh_proj].shape[0]
        tensors[eh_proj] = vh[:rank]
        tensors.update(zip(qkv, (u[:, :rank] * s[:rank]).split(sizes)))
        tensors["midlayer.pre_attention_layernorm.weight"] = torch.ones(rank)

    # MoE: routed experts keep their no-op init (zero down projections) and a shared expert as
    # wide as the dense MLP takes it over. A latent MoE cannot reuse it and starts fresh.
    dense = [f"layers.0.mlp.{p}_proj.weight" for p in ("gate", "up", "down")]
    # dense MLP 只在共享专家形状完全匹配时搬过去，路由专家保留构造时的初始化。
    # latent MoE 结构不一定能承接 dense 权重，因此不能把有同样 gate/up/down 名字理解为形状兼容。
    if "midlayer.mlp.gate.weight" in expected and dense[0] in tensors:
        mlp = {
            k.replace("layers.0.mlp.", "midlayer.mlp.shared_experts."): tensors.pop(k)
            for k in dense
        }
        tensors.update({k: v for k, v in expected.items() if k.startswith("midlayer.mlp.")})
        if all(k in expected and expected[k].shape == v.shape for k, v in mlp.items()):
            tensors.update(mlp)

    return tensors


# 先读取并映射键，校验新增模块要么完整出现要么全部缺失，再检查所有共享形状。
# 验证全部通过才 load_state_dict，允许缺失项用构造初值填充，避免错误权重部分写入模型。
def _load_initial_weights(
    draft_model: torch.nn.Module,
    path: str,
    *,
    upgrade_keys: set[str],
    allow_missing_embedding: bool,
    error_type: type[Exception] = RuntimeError,
    upgrade=None,
) -> Path:
    """Load shared weights strictly; new modules must be wholly absent or complete."""
    checkpoint_path = resolve_initial_draft_checkpoint(path)
    expected = draft_model.state_dict()
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        tensors = to_internal_keys({key: f.get_tensor(key) for key in f.keys()}, expected)
    if upgrade is not None:
        tensors = upgrade(tensors, expected)

    # 新增模块只允许“全部有”或“全部没有”，避免一半层来自 checkpoint、另一半随机初始化。
    # 例如 selector 有两个参数却只提供一个时必须失败，而不是宽松 strict=False 静默接受。
    present_upgrade = upgrade_keys.intersection(tensors)
    if present_upgrade and present_upgrade != upgrade_keys:
        raise error_type(
            f"Incomplete {type(draft_model).__name__} upgrade weights: "
            f"missing={sorted(upgrade_keys - present_upgrade)}"
        )
    allowed_missing = {"embed_tokens.weight"} if allow_missing_embedding else set()
    if not present_upgrade:
        allowed_missing |= upgrade_keys
    missing = set(expected) - set(tensors) - allowed_missing
    unexpected = set(tensors) - set(expected)
    # 参数名存在也不代表可加载：最后维可能因词表裁剪或层结构变动而改变。
    # 同时收集 missing/unexpected/shape_mismatch，能在修改模型前一次说明不兼容原因。
    mismatched = {
        key for key in tensors.keys() & expected.keys() if tensors[key].shape != expected[key].shape
    }
    if missing or unexpected or mismatched:
        raise error_type(
            f"Incompatible initial {type(draft_model).__name__} checkpoint: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
            f"shape_mismatch={sorted(mismatched)}"
        )
    # Validate before modifying the model. Retain only explicitly allowed initialization.
    # 显式合并 expected 与 tensors，再 strict=True 加载：允许的缺失项保留已构造的初值。
    # 这比对任意缺失键都容忍更可控，日志会列出哪些参数没有从初始权重得到。
    retained = set(expected) - set(tensors)
    draft_model.load_state_dict({**expected, **tensors}, strict=True)
    logger.info(
        "Loaded %d initial draft tensors from %s; retained initialization for %s",
        len(tensors),
        checkpoint_path,
        sorted(retained),
    )
    return checkpoint_path


# 只允许 DFlash2 conv/selector 或 DSpark Markov/confidence 新模块整组缺失。
# backbone 必须兼容；embedding 允许缺失因为 trainer 会从 teacher 单独加载。
def load_initial_dflash_weights(draft_model: torch.nn.Module, path: str) -> Path:
    """Warm-start DFlash, DFlash2 or DSpark from a compatible published backbone.

    DFlash2 convolutions/selector or DSpark Markov/confidence heads may be wholly
    absent, retaining their constructor initialization. Partial additions and missing
    backbone weights are errors. The embedding is loaded separately from the target.
    """
    from torchspec.models.draft.dflash2 import DFlash2DraftModel
    from torchspec.models.draft.dspark import DSparkDraftModel

    extra = set()
    if isinstance(draft_model, DFlash2DraftModel):
        extra = {
            key
            for key in draft_model.state_dict()
            if ".attention_conv." in key
            or ".mlp_conv." in key
            or key.startswith("candidate_selector.")
        }
    elif isinstance(draft_model, DSparkDraftModel):
        extra = {
            key
            for key in draft_model.state_dict()
            if key.startswith(("markov_head.", "confidence_head."))
        }
    return _load_initial_weights(
        draft_model, path, upgrade_keys=extra, allow_missing_embedding=True, error_type=ValueError
    )


# 为 Eagle 允许缺失 fc_norm 和独立加载的 embedding，Eagle4 架构额外经过升级转换。
# 这是 warm start，不恢复 Adam、scheduler 或步数，也不保证更换结构后输出保持相同。
def load_initial_draft_weights(draft_model: torch.nn.Module, path: str) -> Path:
    """Load published draft weights, including a compatible EAGLE3 -> 3.1 warm start.

    EAGLE's new per-input FC norms may be wholly absent; their unit-scale constructor
    initialization is retained. Post-norm feedback is selected by config.norm_output
    and adds no weights. This changes computation, so it is a training initialization,
    not an output-preserving conversion. Head and vocabulary mappings remain strict.
    The frozen EAGLE embedding may be absent because the trainer loads it from target.
    This path does not restore optimizer or scheduler state.
    """
    from torchspec.models.draft.base import Eagle3DraftModel

    is_eagle = isinstance(draft_model, Eagle3DraftModel)
    extra = {key for key in draft_model.state_dict() if is_eagle and key.startswith("fc_norm.")}
    is_eagle4 = "LlamaForCausalLMEagle4" in (draft_model.config.architectures or ())
    return _load_initial_weights(
        draft_model,
        path,
        upgrade_keys=extra,
        allow_missing_embedding=is_eagle,
        upgrade=_upgrade_to_eagle4 if is_eagle4 else None,
    )


# 按 ckpt_step 或 latest tracker 定位 iter_XXXXXXX/model；没有可恢复文件时返回 None。
# 不能仅根据 load_path 字符串非空就认定权重一定会加载。
def resolve_resume_model_dir(args: Any) -> Path | None:
    """Model directory ``training.load_path`` resolves to, or None if there is nothing to resume.

    Resolution is deliberately soft so a run can point at an output directory that has no
    checkpoint yet, but callers that need weights to actually arrive must check the result
    rather than the path string.
    """
    load_root = getattr(args, "load_path", None)
    if load_root is None:
        return None

    root_path = Path(load_root).expanduser()
    if not root_path.exists():
        logger.info(f"Checkpoint directory {root_path} not found; skipping load.")
        return None

    target_step = getattr(args, "ckpt_step", None)
    if target_step is None:
        tracker_file = root_path / "latest_checkpointed_iteration.txt"
        if not tracker_file.exists():
            logger.info(f"No tracker file at {tracker_file}; skipping load.")
            return None
        target_step = int(tracker_file.read_text().strip())

    model_dir = root_path / f"iter_{target_step:07d}" / "model"
    if not model_dir.exists():
        logger.info(f"Model checkpoint {model_dir} not found; skipping load.")
        return None
    return model_dir


# 先加载模型，再按续训模式加载优化器、可选 scheduler、RNG 和 metadata 并返回载荷。
# 模型已恢复而 optimizer 缺失/失败必须终止：旧 FP32 master 会在下一步覆盖已恢复模型。
def load(actor: Any) -> dict[str, Any] | None:
    """Load checkpoint from disk.

    Model weights and optimizer state live in separate directories. A resume requires both:
    restoring weights without the matching optimizer state would leave the fp32 masters stale.
    Use training.continual_training=true to start from the weights alone.
    """
    model_dir = resolve_resume_model_dir(actor.args)
    if model_dir is None:
        return None

    checkpoint_dir = model_dir.parent
    target_step = int(checkpoint_dir.name.removeprefix("iter_"))
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    # Load model weights (always)
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}

    try:
        dcp.load(state_dict=state_dict, checkpoint_id=str(model_dir))
        logger.info(f"Loaded model from {model_dir}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_dir}: {e}")
        return None

    # 模型权重两种模式都要读取，但 optimizer 和学习率计数的语义不同。
    # resume 延续同一条优化轨迹；continual training 以旧权重开始新轨迹，不能继续旧 Adam moments。
    continual_training = getattr(actor.args, "continual_training", False)

    # Keep optimizer/LR state out of continual training so it starts fresh.
    load_optimizer = not continual_training and hasattr(actor, "optimizer")
    if load_optimizer and optimizer_dir.exists():
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        try:
            dcp.load(state_dict=optim_state_dict, checkpoint_id=str(optimizer_dir))
            logger.info(f"Loaded optimizer from {optimizer_dir}")
        except Exception as e:
            # The model is already restored but the fp32 masters still hold their pre-load
            # values, and the first step would copy those back over the loaded weights.
            raise RuntimeError(
                f"Failed to load optimizer from {optimizer_dir}: {e}. The model weights were "
                "restored, so continuing would let stale fp32 master params overwrite them on "
                "the first optimizer step. Set training.continual_training=true to start a "
                "fresh optimizer from these weights."
            ) from e
    # 这是模型已成功恢复、但优化器目录不存在的情形。
    # 单纯 warning 然后继续会在第一步把构造时 master 的随机值覆盖到模型，因此必须显式失败。
    elif load_optimizer:
        raise RuntimeError(
            f"No optimizer checkpoint at {optimizer_dir}. Resuming without it would leave the "
            "fp32 master params holding their pre-load values, which the first optimizer step "
            "would copy over the restored weights. Set training.continual_training=true to "
            "start a fresh optimizer from these weights."
        )

    # Load LR scheduler state (optional)
    load_lr_scheduler = (
        not continual_training and hasattr(actor, "lr_scheduler") and lr_scheduler_dir.exists()
    )
    if load_lr_scheduler:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        try:
            dcp.load(state_dict=lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))
            logger.info(f"Loaded LR scheduler from {lr_scheduler_dir}")
        except Exception as e:
            logger.warning(f"Failed to load LR scheduler from {lr_scheduler_dir}: {e}")
    elif hasattr(actor, "lr_scheduler"):
        logger.info(
            f"LR scheduler checkpoint not found at {lr_scheduler_dir}, skipping LR scheduler load."
        )

    # RNG/metadata 先作为载荷返回，最后统一在 finalize_load 应用。
    # 这避免模型构造或权重加载过程中额外消耗随机数后，误以为 RNG 已精确恢复到训练边界。
    rng_state = None
    rng_path = checkpoint_dir / "rng.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")

    metadata = _read_checkpoint_metadata(checkpoint_dir / "meta.json")

    return {
        "rng": rng_state,
        "metadata": metadata,
        "iteration": target_step,
        "optimizer_dir": optimizer_dir,
    }


# continual training 优先从旧 optimizer 取精确 master，但恢复新 param_groups 并清空 Adam moments。
# 旧优化器不可用时从 BF16 模型同步 master 兜底，此路径有精度损失。
def _restore_fp32_master_params(actor: Any, optim_dir: Path) -> None:
    """Sync BF16Optimizer's fp32 master params after model-only checkpoint load.

    When optimizer state is skipped for continual training, the fp32 master copies
    still hold pre-checkpoint (random init) values.  The first optimizer step
    would copy these back to the model, overwriting the loaded weights.

    Strategy: temporarily load the optimizer checkpoint to recover the fp32 master
    params, then restore the freshly configured param_groups and clear Adam state
    so continual training still uses the new optimizer hyperparameters. Falls
    back to copying from the bf16 model weights if the optimizer checkpoint is
    unavailable.
    """
    opt = actor.optimizer
    if not hasattr(opt, "fp32_params"):
        return

    if optim_dir.exists() and (optim_dir / ".metadata").exists():
        try:
            # 暂存的是本次新配置的 group 超参数，排除 params 本身以保留正确 tensor 引用。
            # 读取旧 optimizer 会覆盖 group 配置，所以稍后必须恢复这些新值，而不是只清 moments。
            fresh_param_groups = [
                {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
                for group in opt.optimizer.param_groups
            ]
            optim_state = OptimizerState(actor.model, opt)
            optim_sd = {"optim_state": optim_state}
            dcp.load(state_dict=optim_sd, checkpoint_id=str(optim_dir))
            for group, fresh_group in zip(opt.optimizer.param_groups, fresh_param_groups):
                params = group["params"]
                group.clear()
                group.update(copy.deepcopy(fresh_group))
                group["params"] = params
            # continual training 只借旧快照的精确权重；动量/二阶矩必须清空，学习率使用本次新配置。
            opt.optimizer.state.clear()
            logger.info(f"Loaded fp32 master params from {optim_dir}")
            return
        except Exception as e:
            logger.warning(f"Failed to load fp32 params from optimizer checkpoint: {e}")

    # 例如旧 master 某值含 BF16 无法表达的低位，单从 BF16 权重升级 FP32 只会补零扩精度。
    # 因此此路径能保证下一步不被陈旧 master 覆盖，但无法还原原先 master 的全部数值精度。
    if hasattr(opt, "sync_fp32_params_from_model"):
        opt.sync_fp32_params_from_model()
        logger.info("Synced optimizer fp32 master params from bf16 model weights (lossy)")


# 普通续训恢复 RNG/global_step/start_step；continual training 保持新进度并单独同步 master。
# 最后 CUDA synchronize + barrier，保证所有 rank 恢复完成再进入训练。
def finalize_load(actor: Any, checkpoint_payload: dict[str, Any] | None) -> None:
    if checkpoint_payload is None:
        dist.barrier()
        return

    continual_training = getattr(actor.args, "continual_training", False)

    if checkpoint_payload.get("rng") is not None and not continual_training:
        rng_state = checkpoint_payload["rng"]
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    metadata = checkpoint_payload.get("metadata") or {}
    iteration = checkpoint_payload.get("iteration")
    if metadata and not continual_training:
        actor.global_step = int(metadata.get("global_step", actor.global_step))
        next_step = metadata.get("next_step") or metadata.get("next_inference_id")
        if next_step is not None:
            actor.args.start_step = next_step
    elif iteration is not None and not continual_training:
        if getattr(actor.args, "start_step", None) is None:
            actor.args.start_step = iteration

    if continual_training and hasattr(actor, "optimizer"):
        _restore_fp32_master_params(actor, checkpoint_payload["optimizer_dir"])

    torch.cuda.synchronize()
    dist.barrier()


# 所有 rank 集体保存模型、优化器和 scheduler 到独立 DCP 目录，rank 0 写 RNG/元数据/tracker。
# 目录步号是传入 step+1；tracker 最后写表示这次保存的发布点，首尾 barrier 协调参与者。
def save(actor: Any, step: int) -> None:
    """Save checkpoint to disk.

    Saves model weights and optimizer state to separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    torch.cuda.synchronize()

    base_dir = Path(actor.args.checkpoint_dir).expanduser()
    # 外部 step 可能是 0 起点，目录用下一边界号；metadata 同时保存 step 与 next_step。
    # 恢复时要看 next_step/global_step 的用途，不能仅把目录编号当作训练过程中所有计数器。
    step_id = step + 1
    checkpoint_dir = base_dir / f"iter_{step_id:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if dist.get_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        optimizer_dir.mkdir(parents=True, exist_ok=True)
        lr_scheduler_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # Save model weights
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}
    # DCP 是集体保存：各 rank 写自己的状态/参与元数据协调。
    # 创建目录可由 rank 0 完成，但真正 dcp.save 不能只让 rank 0 调用。
    dcp.save(state_dict, checkpoint_id=str(model_dir))

    if hasattr(actor, "optimizer") and actor.optimizer is not None:
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        dcp.save(optim_state_dict, checkpoint_id=str(optimizer_dir))

    if hasattr(actor, "lr_scheduler") and actor.lr_scheduler is not None:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        dcp.save(lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))

    if dist.get_rank() == 0:
        rng_state = {"torch": torch.get_rng_state()}
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_state, checkpoint_dir / "rng.pt")

        metadata = {
            "iteration": step_id,
            "step": step,
            "inference_id": step,  # compat: old checkpoints use this key
            "next_step": step + 1,
            "next_inference_id": step + 1,  # compat: old checkpoints use this key
            "global_step": actor.global_step,
            "world_size": dist.get_world_size(),
            "timestamp": time.time(),
        }
        _write_checkpoint_metadata(checkpoint_dir / "meta.json", metadata)

        # 模型、optimizer、scheduler、RNG 和 metadata 都写完后才更新 latest tracker。
        # tracker 给下次启动提供默认恢复入口；显式 ckpt_step 则可以绕过它选择历史目录。
        tracker_file = base_dir / "latest_checkpointed_iteration.txt"
        tracker_file.write_text(str(step_id))
        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    dist.barrier()
