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

# Llama/GQA EAGLE3/4 参数骨干及多种等价训练注意力实现。
# 训练缓存的 depth 维代表多步隐藏态展开：首轮完整因果前缀，后续轮只允许同槽对角连接。
# PyTorch 层、RoPE、编译 mask、Flash 自定义反传和 USP 通信集中在此，建议先读文件末尾模型类，再按后端深入。
import math
import os
from functools import partial
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.activations import ACT2FN
from transformers.models.llama.configuration_llama import LlamaConfig

from torchspec.config.utils import resolve_rope_theta
from torchspec.models.draft.base import Eagle3DraftModel, build_lm_head, draft_sliding_window
from torchspec.models.draft.eagle4_moe import Eagle4MoE
from torchspec.models.ops.flex_attention import (
    compile_friendly_flex_attention,
    eagle3_block_mask,
    generate_eagle3_mask,
)
from torchspec.utils.distributed import get_sp_ring_group, get_sp_ulysses_group
from torchspec.utils.logging import logger, print_with_rank

_flash_attn_import_error: ImportError | None = None
try:
    # cutlass-dsl >= 4.4.2 removed cutlass.utils.ampere_helpers, but FA4
    # still imports it for SMEM_CAPACITY. Inject a shim before any FA4 import.
    import importlib  # noqa: E401
    import importlib.util

    if importlib.util.find_spec("cutlass.utils") and not importlib.util.find_spec(
        "cutlass.utils.ampere_helpers"
    ):
        import types  # noqa: E401

        import cutlass.utils

        _shim = types.ModuleType("cutlass.utils.ampere_helpers")
        _shim.SMEM_CAPACITY = {
            "sm80": (164 - 1) * 1024,
            "sm86": (100 - 1) * 1024,
            "sm89": (100 - 1) * 1024,
        }
        import sys

        sys.modules["cutlass.utils.ampere_helpers"] = _shim

    from flash_attn.cute import flash_attn_func as _flash_attn_func
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
except ImportError as _e:
    _flash_attn_import_error = _e
    _flash_attn_func = None
    _flash_attn_fwd = None
    _flash_attn_bwd = None


# 集中尝试导入标准 FlashAttention 的定长/变长前后向和去 padding 工具。
# 依赖缺失时返回一组 None 和原异常，让不使用该后端的模型仍能导入。
def _import_standard_flash_attn():
    try:
        import flash_attn as mod
        from flash_attn import flash_attn_varlen_func as varlen_func
        from flash_attn.bert_padding import pad_input, unpad_input
        from flash_attn.flash_attn_interface import (
            _flash_attn_backward as backward,
        )
        from flash_attn.flash_attn_interface import (
            _flash_attn_forward as forward,
        )
        from flash_attn.flash_attn_interface import (
            _flash_attn_varlen_backward as varlen_backward,
        )
        from flash_attn.flash_attn_interface import (
            _flash_attn_varlen_forward as varlen_forward,
        )
    except ImportError as exc:
        return None, None, None, None, None, None, None, None, exc

    return (
        mod,
        varlen_func,
        pad_input,
        unpad_input,
        forward,
        backward,
        varlen_forward,
        varlen_backward,
        None,
    )


(
    _std_flash_attn_mod,
    _std_flash_attn_varlen_func,
    _std_flash_pad_input,
    _std_flash_unpad_input,
    _std_flash_attn_forward,
    _std_flash_attn_backward,
    _std_flash_attn_varlen_forward,
    _std_flash_attn_varlen_backward,
    _std_flash_attn_import_error,
) = _import_standard_flash_attn()


# 在用户实际选择标准 FA 时给出包含原导入异常的错误，而不是到 kernel 调用处才出现 NoneType 错误。
def _raise_standard_flash_attn_unavailable() -> None:
    raise RuntimeError(
        "LlamaFlashAttention requires the standard flash-attn interface "
        f"(import error: {_std_flash_attn_import_error!r})"
    )


try:
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils as fa_cute_utils
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    _has_cute_dsl = True

    # 为可用的 CUTLASS DSL 安装幂等补丁：允许落盘编译缓存，并按环境变量补编译优化选项。
    # 这是导入期对第三方类方法的修改；编译速度/运行速度的取舍由 opt-level 与 ptxas 设置控制。
    def _patch_cutlass_compilation() -> None:
        """Two patches to reduce flash_attn cute DSL compilation overhead.

        Patch 1 — Disk cache (BaseDSL.compile_and_cache):
            flash_attn hardcodes no_cache=True, bypassing CUTE_DSL_CACHE_DIR.
            Overriding to False lets compiled cubins be written on the first run
            and read back on every subsequent process start (seconds, not minutes).
            Set CUTE_DSL_CACHE_DIR to a persistent path to survive reboots:
                export CUTE_DSL_CACHE_DIR=/scratch/$USER/cutlass_cache

        Patch 2 — Compilation opt-level (Compiler._compile):
            flash_attn passes options="--enable-tvm-ffi" with no explicit
            --opt-level, so the parser default (3) is used. Lower levels reduce
            LLVM-IR and ptxas work at the cost of slightly slower kernels.

            Controlled by TORCHSPEC_FLASH_ATTN_OPT_LEVEL (default: 3):
              3  – current default; ~12 min; fastest kernel
              2  – ~6–8 min; <5% slower kernel  (recommended for training)
              1  – ~3–5 min; ~15% slower kernel
              0  – ~1–2 min; ~50% slower kernel  (debugging only)

            TORCHSPEC_FLASH_ATTN_PTXAS_OPT controls ptxas separately
            (default: 3). Use 1 for additional compile-time savings when
            combined with opt-level 1:
                export TORCHSPEC_FLASH_ATTN_OPT_LEVEL=1
                export TORCHSPEC_FLASH_ATTN_PTXAS_OPT=1
        """
        import os

        # ── Patch 1: disk cache ───────────────────────────────────────────────
        try:
            from cutlass.base_dsl.dsl import BaseDSL
        except ImportError:
            return
        # 补丁用类级标记保证幂等，避免多次导入/初始化把 wrapper 一层层套在原方法外。
        if not getattr(BaseDSL, "_disk_cache_enabled", False):
            _orig_compile_and_cache = BaseDSL.compile_and_cache

            # 保留原编译参数，只把 no_cache 改 False，允许读取/写入持久缓存以减少每进程重复编译。
            def _compile_and_cache_with_disk(
                self,
                module,
                module_hash,
                function_name,
                pipeline,
                args_spec,
                no_cache,
                *args,
                **kwargs,
            ):
                return _orig_compile_and_cache(
                    self,
                    module,
                    module_hash,
                    function_name,
                    pipeline,
                    args_spec,
                    False,  # override no_cache=True from Compiler._compile
                    *args,
                    **kwargs,
                )

            BaseDSL.compile_and_cache = _compile_and_cache_with_disk
            BaseDSL._disk_cache_enabled = True

            from cutlass.base_dsl.cache_helpers import default_generated_ir_path

            cache_dir = os.environ.get("CUTE_DSL_CACHE_DIR", default_generated_ir_path)
            logger.debug(
                "CUTLASS DSL disk cache enabled → %s  "
                "(set CUTE_DSL_CACHE_DIR for a persistent location)",
                cache_dir,
            )

        # ── Patch 2: opt-level injection ──────────────────────────────────────
        # cute.compile is a CompileCallable instance; _compile is the instance method.
        try:
            from cutlass.base_dsl.compiler import CompileCallable
        except ImportError:
            return
        if getattr(CompileCallable, "_opt_level_patched", False):
            return

        # 优化级别是编译成本与运行性能的取舍，不影响 attention 的目标数学定义；历史耗时不应当作当前硬件保证。
        _opt_level = int(os.environ.get("TORCHSPEC_FLASH_ATTN_OPT_LEVEL", "3"))
        _ptxas_opt = int(os.environ.get("TORCHSPEC_FLASH_ATTN_PTXAS_OPT", "3"))

        _orig_compile_callable = CompileCallable._compile

        # 只在 options 未显式指定相应选项时追加环境配置，再调用原编译函数，尊重调用方已有选择。
        def _compile_with_opt_level(self, func, *args, **kwargs):
            options: str = kwargs.get("options", "") or ""
            if "--opt-level" not in options:
                options = options.strip() + f" --opt-level {_opt_level}"
            if "--ptxas-options" not in options and _ptxas_opt != 3:
                options = options + f" --ptxas-options=-O{_ptxas_opt}"
            kwargs["options"] = options.strip()
            return _orig_compile_callable(self, func, *args, **kwargs)

        # 替换全局第三方方法，影响后续编译调用；这也是为何补丁集中在单一入口并保存原 callable。
        CompileCallable._compile = _compile_with_opt_level
        CompileCallable._opt_level_patched = True

        if _opt_level != 3 or _ptxas_opt != 3:
            logger.debug(
                f"flash_attn compilation: opt-level={_opt_level}, "
                f"ptxas-opt={_ptxas_opt}  "
                f"(TORCHSPEC_FLASH_ATTN_OPT_LEVEL / TORCHSPEC_FLASH_ATTN_PTXAS_OPT)"
            )

    _patch_cutlass_compilation()

except ImportError:
    _has_cute_dsl = False

# Detect SM major version once at import time.
# SM90 (H100/H200) requires block_sparse_tensors in _flash_attn_bwd with mask_mod.
_cuda_sm_major: int = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0


# 通过新增组维 expand 再 reshape，把 [B,Hkv,S,D] 变为 [B,Hq,S,D]；GQA 多个 Q 头共享一份 KV。
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# 将最后维两半旋转为 [-后半,前半]；这与 DeepSeek 相邻通道对旋转的权重布局不同。
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# 以真实 position_ids 查 cos/sin，广播到 head 维，再按 x*cos+rotate_half(x)*sin 旋转 Q/K。
@torch.compile(dynamic=True)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# 按 mrope_section 切通道并循环选择时间/高度/宽度三套 cos/sin；文本三坐标相同则退化成普通 RoPE。
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    # 建立逆频率与非持久 cos/sin 缓存；若配置 Llama3 缩放，按波长区间选择保持、缩小或平滑插值频率。
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=None,
        low_freq_factor=None,
        high_freq_factor=None,
        orig_max_position=None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # Llama3 style rotary embedding frequency scaling
        if all(
            v is not None
            for v in [
                scaling_factor,
                low_freq_factor,
                high_freq_factor,
                orig_max_position,
            ]
        ):
            logger.info(
                f"Using Llama3 style rotary embedding with scaling_factor={scaling_factor}, low_freq_factor={low_freq_factor}, high_freq_factor={high_freq_factor}, orig_max_position={orig_max_position}"
            )
            self.scaling_factor = scaling_factor
            self.low_freq_factor = low_freq_factor
            self.high_freq_factor = high_freq_factor
            self.orig_max_position = orig_max_position

            # 由原训练窗口与阈值换成波长边界：长波低频需要更多伸缩，短波高频保留局部位置分辨能力。
            low_freq_wavelen = orig_max_position / low_freq_factor
            high_freq_wavelen = orig_max_position / high_freq_factor
            wave_len = 2 * math.pi / inv_freq

            if low_freq_factor != high_freq_factor:
                smooth = (orig_max_position / wave_len - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
            else:
                smooth = 0

            # 分三段构造新逆频率，高频不变、低频除 factor，中间用 smooth 插值避免不连续。
            new_freqs = torch.where(
                wave_len < high_freq_wavelen,
                inv_freq,
                torch.where(
                    wave_len > low_freq_wavelen,
                    inv_freq / self.scaling_factor,
                    (1 - smooth) * inv_freq / self.scaling_factor + smooth * inv_freq,
                ),
            )
            inv_freq = new_freqs

        # 频率与 cos/sin 是可重建缓存，不随 checkpoint 保存；它们会跟随模块设备迁移。
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings + 20,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    # 位置和逆频率外积后复制成两半布局，保存 [1,1,S,D] cos/sin；缓存长度随需求增长。
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    # 超出缓存长度才重建，返回请求前缀并匹配 x.dtype；x 的值不参与角度计算。
    @torch.compile(dynamic=True)
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class NoRotaryEmbedding(LlamaRotaryEmbedding):
    """NoPE as an identity rotation (cos=1, sin=0), so every attention backend runs unchanged."""

    # 先沿用缓存结构，再令 cos=1/sin=0，使所有后端无需新增分支就实现无位置旋转。
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        super()._set_cos_sin_cache(seq_len, device, dtype)
        self.cos_cached.fill_(1)
        self.sin_cached.zero_()


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    # 先记录缩放因子再调用父构造，父类预建缓存时会动态调用本类重写的构建方法。
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    # 把位置 t 除以 scaling_factor 后乘逆频率，压缩相位增长来支持更长上下文。
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    # 保存 NTK 缩放因子，复用基础 RoPE 的参数和缓存分配入口。
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    # 只有超出原上下文长度才按当前 seq_len 重算 base/逆频率，再构造角度缓存；长度改变会改变频率。
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaMutiRotaryEmbedding(LlamaRotaryEmbedding):
    # 继承标准频率表并保存额外幅度因子，forward 改用三维 position_ids 而非单一 seq_len。
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__(dim, max_position_embeddings, base, device)
        self.scaling_factor = scaling_factor

    # 将逆频率扩为时间/高/宽三路，在禁用 autocast 的 float32 中与 [3,B,1,S] 坐标相乘。
    # 返回三路 cos/sin，后续 apply_multimodal 按通道区间选择相应路。
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = (
            x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.scaling_factor
            sin = emb.sin() * self.scaling_factor

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Inverse dim formula to find dim based on number of rotations
# 由原上下文内期望旋转次数反解频率维号，作为 YaRN 插值/外推切换区间的边界依据。
def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
# 对两个旋转次数边界分别 floor/ceil 并钳制到合法维度，避免 ramp 区间越界。
def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


# scale<=1 时幅度不变，否则用 1+0.1*mscale*log(scale) 校准长上下文注意力强度。
def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# 生成 [0,1] 截断的线性斜坡，在两频率方案间平滑过渡；相同边界加 0.001 避免除零。
def yarn_linear_ramp_mask(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001  # Prevent singularity
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class LlamaYarnRotaryEmbedding(LlamaRotaryEmbedding):
    # 保存原上下文、快慢频率阈值与幅度配置，再让父类调用 YaRN 专用缓存构造。
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    # 用 ramp 混合未缩放外推频率与除 factor 的插值频率，并将 cos/sin 乘幅度比值。
    # 频率选择与 MLA softmax_scale 的 mscale_all_dim 配合，不能独立更改其一。
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        # 分别构造外推（保留原频率）与插值（频率除 factor）方案，之后按通道频率分段混合。
        freq_extra = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        # 阈值由原上下文旋转次数反解；ramp 为每个频率通道提供平滑混合比例。
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        # 幅度校准不仅改变角度：cos/sin 同乘这个比值，MLA 的 score scale 会配合 mscale_all_dim。
        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached",
            (emb.cos() * _mscale)[None, None, :, :].to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (emb.sin() * _mscale)[None, None, :, :].to(dtype),
            persistent=False,
        )


# 兼容 rope 配置为 dict 或对象，缺失时返回调用方 default，使不同 Transformers 配置形态共享构造路径。
def rope_config_get(rope_scaling, key, default=None):
    """Read a key out of a rope_scaling config, which may be a dict or an object."""
    if rope_scaling is None:
        return default
    if isinstance(rope_scaling, dict):
        return rope_scaling.get(key, default)
    return getattr(rope_scaling, key, default)


# 集中解析 NoPE/default/linear/dynamic/llama3/mrope/yarn；维度由 GQA head_dim 或 MLA rope_dim 提供。
# 按实际各分支传参构造，未知类型报错；不要假定每个缩放分支都使用完全相同的参数集合。
def build_rotary_embedding(config, dim: int, max_position_embeddings: int):
    """Build the rotary embedding a config asks for.

    Every draft that uses these rotary classes goes through here, and `dim` is the
    only thing that differs between callers — the head dim for GQA blocks, the
    rope side dim for MLA ones. Each attention block used to carry its own copy of
    this branch ladder, and the copies drifted: the MLA one dropped `base` on the
    YaRN branch, so drafts trained at 10000 while their config declared otherwise.
    Adding a scaling type or an argument in one place now reaches every draft.
    """
    # NoPE 通过恒等旋转表实现，因此下游 attention 不需要再为有无 RoPE 分裂前向路径。
    if (
        getattr(config, "use_rope", True) is False
        or getattr(config, "position_embedding_type", None) == "none"
        or getattr(config, "partial_rotary_factor", None) == 0
    ):
        return NoRotaryEmbedding(dim, max_position_embeddings=max_position_embeddings)

    # 统一读取 type/rope_type 两种命名，resolve_rope_theta 处理模型配置中的频率基数。
    rope_scaling = config.rope_scaling
    rope_theta = resolve_rope_theta(config)
    rget = partial(rope_config_get, rope_scaling)
    scaling_type = rget("rope_type", rget("type"))

    if rope_scaling is None or scaling_type in (None, "default"):
        return LlamaRotaryEmbedding(
            dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    scaling_factor = rget("factor")

    # linear/dynamic 两条分支强制存在 factor；当前调用参数列表以源码为准，并非所有分支都显式传 rope_theta。
    if scaling_type == "linear":
        if scaling_factor is None:
            raise ValueError("Linear RoPE scaling requires 'factor' in rope_scaling config.")
        return LlamaLinearScalingRotaryEmbedding(
            dim,
            max_position_embeddings=max_position_embeddings,
            scaling_factor=scaling_factor,
        )
    if scaling_type == "dynamic":
        if scaling_factor is None:
            raise ValueError("Dynamic RoPE scaling requires 'factor' in rope_scaling config.")
        return LlamaDynamicNTKScalingRotaryEmbedding(
            dim,
            max_position_embeddings=max_position_embeddings,
            scaling_factor=scaling_factor,
        )
    if scaling_type == "llama3":
        return LlamaRotaryEmbedding(
            dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_factor=scaling_factor if scaling_factor is not None else 1.0,
            low_freq_factor=rget("low_freq_factor"),
            high_freq_factor=rget("high_freq_factor"),
            orig_max_position=rget("original_max_position_embeddings"),
        )
    if scaling_type == "mrope":
        return LlamaMutiRotaryEmbedding(dim, max_position_embeddings=max_position_embeddings)
    # YaRN 额外需要原上下文、beta 与 mscale，缺少这些信息不能仅凭 factor 重建相同旋转。
    if scaling_type == "yarn":
        return LlamaYarnRotaryEmbedding(
            dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            original_max_position_embeddings=rget("original_max_position_embeddings"),
            scaling_factor=scaling_factor,
            beta_fast=rget("beta_fast"),
            beta_slow=rget("beta_slow"),
            mscale=rget("mscale"),
            mscale_all_dim=rget("mscale_all_dim"),
        )
    raise ValueError(f"Unknown RoPE scaling type {scaling_type}")


_SNAP_Q = 128  # Alignment granularity for Q_LEN bucketing.


# 向上补齐到指定倍数，例如 129 按 128 对齐得 256，用固定 bucket 减少编译形状种类。
def _snap_to_multiple(n: int, multiple: int) -> int:
    """Round *n* up to the next multiple of *multiple*."""
    return ((n + multiple - 1) // multiple) * multiple


# 至少 min_val，再用 bit_length 向上取二次幂，保证后续取模能用按位与替代。
def _snap_to_power_of_2(n: int, min_val: int = 128) -> int:
    """Round *n* up to the next power of 2, at least *min_val*."""
    n = max(n, min_val)
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# EAGLE3 mask_mod mode selection
# ---------------------------------------------------------------------------
# Available modes (SM100+ only; SM90 always uses "closure"):
#   "seqlen_po2" - (default) read Q_LEN from seqlen_info, bitmask modulo.
#                  Requires power-of-2 Q_LEN.  Single compiled kernel.
#   "seqlen"     - read Q_LEN from seqlen_info, runtime integer modulo.
#                  Single compiled kernel but ~1.5-3x slower per-call.
#   "dynamic"    - read Q_LEN from aux_tensors.  Single compiled kernel
#                  but vec_size penalty (fwd: 2->1, bwd: 4->1).
#   "simple"     - closure-captured Q_LEN without valid_seq_len checks.
#                  One kernel per Q_LEN bucket (same as "closure" but fewer ops).
#   "closure"    - original closure-captured Q_LEN + valid_seq_len.
#                  One kernel per Q_LEN bucket.  Used on SM90.
_VALID_MASK_MODES = {"seqlen_po2", "seqlen", "dynamic", "simple", "closure"}
_EAGLE3_MASK_MODE: str = "seqlen_po2"


# 验证模式名后更新进程级 mask 策略；硬件有效模式仍由 _effective_mask_mode 决定。
def set_eagle3_mask_mode(mode: str) -> None:
    """Set the EAGLE3 mask_mod strategy for SM100+.

    Has no effect on SM90 (always uses "closure").
    Valid modes: "seqlen_po2", "seqlen", "dynamic", "simple", "closure".
    """
    global _EAGLE3_MASK_MODE
    if mode not in _VALID_MASK_MODES:
        raise ValueError(f"Unknown eagle3 mask mode {mode!r}, must be one of {_VALID_MASK_MODES}")
    _EAGLE3_MASK_MODE = mode


# 读取用户选择的全局策略，不应用 SM90 的强制 closure 覆盖。
def get_eagle3_mask_mode() -> str:
    """Return the current EAGLE3 mask_mod strategy."""
    return _EAGLE3_MASK_MODE


# SM90 固定 closure 以满足其反向块稀疏要求，其余设备使用已设置模式。
def _effective_mask_mode() -> str:
    """Return the effective mask mode, accounting for SM90 override."""
    if _cuda_sm_major == 9:
        return "closure"
    return _EAGLE3_MASK_MODE


# seqlen_po2 要求二次幂长度，其余模式仅按 128 对齐；padding 必须和 mask 的模数保持一致。
def _snap_q_len(q_len: int, mode: str) -> int:
    """Snap *q_len* to the bucket size required by *mode*.

    "seqlen_po2" requires power-of-2 Q_LEN; all other modes use _SNAP_Q alignment.
    """
    if mode == "seqlen_po2":
        return _snap_to_power_of_2(q_len)
    return _snap_to_multiple(q_len, _SNAP_Q)


# Caches for cute.jit mask_mod functions.
# _flash_mask_mod_cache: (Q_LEN, valid_seq_len) -> compiled mask_mod (SM90).
# _flash_mask_mod_simple_cache: Q_LEN -> compiled mask_mod (SM100+, no valid_seq_len).
# Keyed by integer constants so the same function object is returned for repeated calls,
# allowing _flash_attn_fwd/_flash_attn_bwd to reuse their compiled CUDA kernels.
_flash_mask_mod_cache: dict = {}
_flash_mask_mod_simple_cache: dict = {}


# 按 (Q_LEN,valid_seq_len) 缓存 CuTe mask 闭包，使同 bucket 复用函数和编译 kernel。
# 首轮含因果与有效边界检查，后续轮按同槽相等关系连接，不是整个 suffix 都可见。
def _make_eagle3_flash_mask_mod(Q_LEN: int, valid_seq_len: int):
    """Build a cute.jit mask_mod for the EAGLE3 attention pattern.

    KV layout: [k0 (Q_LEN tokens), k1, k2, ..., kN], total KV = Q_LEN * lck.

    Causal part  (kv_idx < Q_LEN):
        q_idx >= kv_idx AND kv_idx < valid_seq_len AND q_idx < valid_seq_len
    Suffix part  (kv_idx >= Q_LEN):
        (kv_idx % Q_LEN) < valid_seq_len AND (kv_idx - q_idx) % Q_LEN == 0

    Both Q_LEN and valid_seq_len are compile-time constants captured via closure,
    so no aux_tensors are needed and the SM90 backward kernel works correctly.

    Results are cached by (Q_LEN, valid_seq_len) to prevent recompilation.
    """
    # 函数对象与闭包常量会影响编译缓存，复用同一个 shape 的闭包有助于跨 batch 重用 kernel。
    cache_key = (Q_LEN, valid_seq_len)
    if cache_key in _flash_mask_mod_cache:
        return _flash_mask_mod_cache[cache_key]

    if not _has_cute_dsl:
        return None

    # 把闭包中的长度变成 SSA 常量，再组合首轮因果和 suffix 余数相等，屏蔽无效序列槽。
    @cute.jit
    def _eagle3_mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors: None,
    ) -> cute.TensorSSA:
        q_len_ssa = fa_cute_utils.scalar_to_ssa(Q_LEN, cutlass.Int32)
        seq_len_ssa = fa_cute_utils.scalar_to_ssa(valid_seq_len, cutlass.Int32)
        zero_ssa = fa_cute_utils.scalar_to_ssa(0, cutlass.Int32)

        # Causal part: kv token lives in k0
        in_k0 = kv_idx < q_len_ssa
        causal = q_idx >= kv_idx
        kv_valid = kv_idx < seq_len_ssa
        q_valid = q_idx < seq_len_ssa
        causal_part = in_k0 & causal & kv_valid & q_valid

        # Suffix part: kv token lives in k1..kN (diagonal EAGLE3 pattern)
        in_suffix = kv_idx >= q_len_ssa
        kv_mod = kv_idx % q_len_ssa
        suffix_padding = kv_mod < seq_len_ssa
        diff = kv_idx - q_idx
        diagonal = (diff % q_len_ssa) == zero_ssa
        suffix_part = in_suffix & suffix_padding & diagonal

        return causal_part | suffix_part

    # 可读名字用于区分不同 bucket 的 mask；真正的条件仍由闭包中的两种长度共同决定。
    _eagle3_mask_mod.__name__ = f"eagle3_flash_mask_Q{Q_LEN}_S{valid_seq_len}"
    _flash_mask_mod_cache[cache_key] = _eagle3_mask_mod
    return _eagle3_mask_mod


# valid_seq_len 等于 bucket 长度时去掉恒真边界检查，只按 Q_LEN 缓存简化闭包。
def _make_eagle3_flash_mask_mod_simple(Q_LEN: int):
    """Simplified mask_mod for SM100+ where valid_seq_len == Q_LEN.

    Mathematically equivalent to _make_eagle3_flash_mask_mod(Q_LEN, Q_LEN),
    but with the three always-True checks (kv_valid, q_valid, suffix_padding)
    removed at source rather than relying on the compiler to eliminate them.
    Fewer closure variables -> simpler hash -> fewer SSA ops in kernel.
    """
    if Q_LEN in _flash_mask_mod_simple_cache:
        return _flash_mask_mod_simple_cache[Q_LEN]

    if not _has_cute_dsl:
        return None

    # 首轮 q>=kv，后续 (kv-q)%Q_LEN==0；模数是闭包编译常量。
    @cute.jit
    def _eagle3_mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors: None,
    ) -> cute.TensorSSA:
        q_len_ssa = fa_cute_utils.scalar_to_ssa(Q_LEN, cutlass.Int32)
        zero_ssa = fa_cute_utils.scalar_to_ssa(0, cutlass.Int32)
        in_k0 = kv_idx < q_len_ssa
        causal_part = in_k0 & (q_idx >= kv_idx)
        in_suffix = kv_idx >= q_len_ssa
        diff = kv_idx - q_idx
        suffix_part = in_suffix & ((diff % q_len_ssa) == zero_ssa)
        return causal_part | suffix_part

    _eagle3_mask_mod.__name__ = f"eagle3_flash_mask_simple_Q{Q_LEN}"
    _flash_mask_mod_simple_cache[Q_LEN] = _eagle3_mask_mod
    return _eagle3_mask_mod


_eagle3_flash_mask_mod_dynamic = None


# 创建单个函数从 aux_tensors 读取运行时 Q_LEN，减少按长度重编译，但引入辅助张量读取和向量化代价。
def _make_eagle3_flash_mask_mod_dynamic():
    """Build a single cute.jit mask_mod that reads Q_LEN from aux_tensors at runtime.

    Unlike _make_eagle3_flash_mask_mod_simple which captures Q_LEN as a closure
    variable (compile-time constant, one kernel per Q_LEN bucket), this version
    reads Q_LEN from aux_tensors[0] at runtime, enabling a single compiled kernel
    for all Q_LEN values.

    Trade-off: avoids per-bucket recompilation, but has_aux_tensors=True reduces
    vec_size (fwd: 2->1, bwd: 4->1) and Q_LEN becomes a runtime value (global
    memory read + loss of constant folding / branch elimination).
    """
    global _eagle3_flash_mask_mod_dynamic
    if _eagle3_flash_mask_mod_dynamic is not None:
        return _eagle3_flash_mask_mod_dynamic

    if not _has_cute_dsl:
        return None

    # 从单元素 int32 辅助张量取模数，其他因果/同槽条件与静态闭包一致。
    @cute.jit
    def _eagle3_mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        # Read Q_LEN from aux_tensors[0] (a 1-element Int32 tensor)
        q_len_ssa = fa_cute_utils.scalar_to_ssa(aux_tensors[0][0], cutlass.Int32)
        zero_ssa = fa_cute_utils.scalar_to_ssa(0, cutlass.Int32)
        in_k0 = kv_idx < q_len_ssa
        causal_part = in_k0 & (q_idx >= kv_idx)
        in_suffix = kv_idx >= q_len_ssa
        diff = kv_idx - q_idx
        suffix_part = in_suffix & ((diff % q_len_ssa) == zero_ssa)
        return causal_part | suffix_part

    _eagle3_mask_mod.__name__ = "eagle3_flash_mask_dynamic"
    _eagle3_flash_mask_mod_dynamic = _eagle3_mask_mod
    return _eagle3_mask_mod


_eagle3_flash_mask_mod_seqlen = None


# 用 kernel 的 seqlen_info 取得运行时 Q_LEN，避免 aux_tensors 对向量化的影响，仍需整数取模。
def _make_eagle3_flash_mask_mod_seqlen():
    """Build a single cute.jit mask_mod that reads Q_LEN from seqlen_info.seqlen_q.

    The flash_attn compile cache key includes mask_mod_hash (from closure variables)
    but NOT tensor shapes (seqlen dimensions are dynamic). This means:
    - Closure approach: different Q_LEN -> different closure -> different hash -> recompile.
    - This approach: single function, no closure -> fixed hash -> single compiled kernel.

    seqlen_info.seqlen_q == Q_LEN because we call _flash_attn_fwd with q of shape
    [bsz, q_len, num_heads, head_dim] and no cu_seqlens_q/seqused_q.

    No aux_tensors needed -> no vec_size penalty (fwd: 2, bwd: 4 unchanged).
    """
    global _eagle3_flash_mask_mod_seqlen
    if _eagle3_flash_mask_mod_seqlen is not None:
        return _eagle3_flash_mask_mod_seqlen

    if not _has_cute_dsl:
        return None

    # 读取 seqlen_q 作为每轮块宽，首轮三角与后续对角共用一个谓词。
    @cute.jit
    def _eagle3_mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors: None,
    ) -> cute.TensorSSA:
        q_len_ssa = fa_cute_utils.scalar_to_ssa(seqlen_info.seqlen_q, cutlass.Int32)
        zero_ssa = fa_cute_utils.scalar_to_ssa(0, cutlass.Int32)
        in_k0 = kv_idx < q_len_ssa
        causal_part = in_k0 & (q_idx >= kv_idx)
        in_suffix = kv_idx >= q_len_ssa
        diff = kv_idx - q_idx
        suffix_part = in_suffix & ((diff % q_len_ssa) == zero_ssa)
        return causal_part | suffix_part

    _eagle3_mask_mod.__name__ = "eagle3_flash_mask_seqlen"
    _eagle3_flash_mask_mod_seqlen = _eagle3_mask_mod
    return _eagle3_mask_mod


_eagle3_flash_mask_mod_seqlen_po2 = None


# 运行时读取二次幂 Q_LEN，把取模替换成 diff&(Q_LEN-1)，减少整数除法成本。
# 例如 Q_LEN=256，diff=512 的低 8 位全零，代表同一槽跨了两轮。
def _make_eagle3_flash_mask_mod_seqlen_po2():
    """Mask_mod reading Q_LEN from seqlen_info.seqlen_q, using bitmask for modulo.

    Requires Q_LEN to be a power of 2 (use _snap_to_power_of_2 for bucketing).
    Replaces the expensive runtime `diff % Q_LEN == 0` with `diff & (Q_LEN-1) == 0`,
    which is a single AND instruction.

    Combined with reading Q_LEN from seqlen_info (no aux_tensors -> no vec_size penalty),
    this should match or approach closure-captured compile-time constant performance.
    """
    global _eagle3_flash_mask_mod_seqlen_po2
    if _eagle3_flash_mask_mod_seqlen_po2 is not None:
        return _eagle3_flash_mask_mod_seqlen_po2

    if not _has_cute_dsl:
        return None

    # 构建 Q_LEN-1 的位掩码，suffix 的位与为零才可见；非二次幂长度不满足这条等价关系。
    @cute.jit
    def _eagle3_mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors: None,
    ) -> cute.TensorSSA:
        q_len_ssa = fa_cute_utils.scalar_to_ssa(seqlen_info.seqlen_q, cutlass.Int32)
        one_ssa = fa_cute_utils.scalar_to_ssa(1, cutlass.Int32)
        zero_ssa = fa_cute_utils.scalar_to_ssa(0, cutlass.Int32)
        mask_ssa = q_len_ssa - one_ssa  # Q_LEN - 1 = bitmask for power-of-2
        in_k0 = kv_idx < q_len_ssa
        causal_part = in_k0 & (q_idx >= kv_idx)
        in_suffix = kv_idx >= q_len_ssa
        diff = kv_idx - q_idx
        suffix_part = in_suffix & ((diff & mask_ssa) == zero_ssa)
        return causal_part | suffix_part

    _eagle3_mask_mod.__name__ = "eagle3_flash_mask_seqlen_po2"
    _eagle3_flash_mask_mod_seqlen_po2 = _eagle3_mask_mod
    return _eagle3_mask_mod


# 按有效模式同时返回 CuTe 元素谓词与 Flex 块扫描谓词；dynamic 模式额外传长度张量。
# 两个谓词必须描述同一可见性，否则块跳过规则会漏掉合法 attention 项。
def _build_eagle3_mask_pair(
    q_len: int,
    kv_len: int,
    bsz: int,
    lck: int,
    device: torch.device,
) -> tuple:
    """Return (mask_mod_cute, mask_mod_flex, aux_tensors) for the current SM architecture.

    SM90: always uses "closure" mode (block_sparse backward required).
    SM100+: uses the mode selected by set_eagle3_mask_mode().
    """
    mode = _effective_mask_mode()
    aux_tensors = None

    if mode == "closure":
        mask_mod_cute = _make_eagle3_flash_mask_mod(q_len, q_len)
    elif mode == "simple":
        mask_mod_cute = _make_eagle3_flash_mask_mod_simple(q_len)
    elif mode == "dynamic":
        mask_mod_cute = _make_eagle3_flash_mask_mod_dynamic()
        aux_tensors = [torch.tensor([q_len], dtype=torch.int32, device=device)]
    elif mode == "seqlen":
        mask_mod_cute = _make_eagle3_flash_mask_mod_seqlen()
    elif mode == "seqlen_po2":
        mask_mod_cute = _make_eagle3_flash_mask_mod_seqlen_po2()
    else:
        raise ValueError(f"Unknown eagle3 mask mode {mode!r}")

    # Always build a flex-compatible mask_mod for block-sparse iteration.
    mask_mod_flex = generate_eagle3_mask(
        Q_LEN=q_len,
        KV_LEN=kv_len,
        lck=lck,
    )

    return mask_mod_cute, mask_mod_flex, aux_tensors


# Block-sparse tensor caches for FA4 forward and backward.
#
# BlockMask.as_tuple(flatten=True) layout:
#   [0] seq_lengths_q, [1] seq_lengths_kv,
#   [2] kv_num_blocks, [3] kv_indices, [4] full_kv_num_blocks, [5] full_kv_indices,
#   [6] q_num_blocks,  [7] q_indices,  [8] full_q_num_blocks,  [9] full_q_indices,
#   [10] KV_BLOCK_SIZE, [11] Q_BLOCK_SIZE, [12] mask_mod
#
# Forward needs KV-direction (indices 2-5): for each Q block, which KV blocks to visit.
# Backward needs Q-direction (indices 6-9): for each KV block, which Q blocks affect it.

_BM_KV_NUM_BLOCKS = 2
_BM_KV_INDICES = 3
_BM_FULL_KV_NUM_BLOCKS = 4
_BM_FULL_KV_INDICES = 5
_BM_Q_NUM_BLOCKS = 6
_BM_Q_INDICES = 7
_BM_FULL_Q_NUM_BLOCKS = 8
_BM_FULL_Q_INDICES = 9

# Level 1: B=1, H=1 raw tensors keyed by (q_len, kv_len, block_size, device_index).
_block_mask_raw_cache: dict = {}
# Level 2: expanded BlockSparseTensorsTorch keyed by (q_len, kv_len, bsz, num_heads, direction, block_size, device_index).
_block_sparse_cache: dict = {}


# SM100+ 使用 (Q=256,KV=128)，其余使用 128 方块，匹配 FA4 实际 CTA/subtile 布局。
def _fa4_block_size() -> tuple:
    """Return (Q_BLOCK_SIZE, KV_BLOCK_SIZE) for FA4 block-sparse iteration.

    SM100+ (Blackwell) requires Q_BLOCK_SIZE=256 for both forward and backward
    due to subtile_factor=2 (q_stage=2, two M-tiles per CTA).
    SM90 (Hopper) uses 128 x 128.
    """
    if _cuda_sm_major >= 10:
        return (256, 128)
    return (128, 128)


# 先缓存 B=H=1 的原始稀疏索引，再按 batch/head 和 forward/backward 方向扩展成连续元数据。
# forward 按 Q 查 KV 块，backward 按 KV 查 Q 块；两套索引不能互换。
def _get_block_sparse(
    mask_mod_flex,
    q_len,
    kv_len,
    bsz,
    num_heads,
    device,
    direction="kv",
):
    """Return BlockSparseTensorsTorch for FA4 forward or backward.

    direction="kv": KV-direction tensors for forward (skip empty KV blocks per Q block).
    direction="q":  Q-direction tensors for backward (skip empty Q blocks per KV block).

    The EAGLE3 mask is batch/head-independent, so we compute with B=1, H=1
    and expand to (bsz, num_heads).
    """
    block_size = _fa4_block_size()
    device_idx = device.index or 0

    # 第一层缓存不含 B/H，因为 EAGLE 可见性对 batch/head 相同，先算一个原型节省 mask 构建。
    raw_key = (q_len, kv_len, block_size, device_idx)
    if raw_key not in _block_mask_raw_cache:
        bm = create_block_mask(
            mask_mod_flex,
            B=1,
            H=1,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
            BLOCK_SIZE=block_size,
        )
        # BlockMask 同时带 KV 方向和 Q 方向索引，分别服务 forward 与 backward 的稀疏遍历。
        t = bm.as_tuple()
        _block_mask_raw_cache[raw_key] = {
            "kv": (
                t[_BM_KV_NUM_BLOCKS],
                t[_BM_KV_INDICES],
                t[_BM_FULL_KV_NUM_BLOCKS],
                t[_BM_FULL_KV_INDICES],
            ),
            "q": (
                t[_BM_Q_NUM_BLOCKS],
                t[_BM_Q_INDICES],
                t[_BM_FULL_Q_NUM_BLOCKS],
                t[_BM_FULL_Q_INDICES],
            ),
        }

    # 第二层才按真实 B/H 扩展连续存储，避免每次 kernel 调用重复 expand+contiguous。
    cache_key = (q_len, kv_len, bsz, num_heads, direction, block_size, device_idx)
    if cache_key not in _block_sparse_cache:
        cnt, idx, f_cnt, f_idx = _block_mask_raw_cache[raw_key][direction]
        _block_sparse_cache[cache_key] = BlockSparseTensorsTorch(
            mask_block_cnt=cnt.expand(bsz, num_heads, -1).contiguous(),
            mask_block_idx=idx.expand(bsz, num_heads, -1, -1).contiguous(),
            full_block_cnt=f_cnt.expand(bsz, num_heads, -1).contiguous()
            if f_cnt is not None
            else None,
            full_block_idx=f_idx.expand(bsz, num_heads, -1, -1).contiguous()
            if f_idx is not None
            else None,
            block_size=block_size,
        )
    return _block_sparse_cache[cache_key]


class _EagleMaskedFlashAttnFunc(torch.autograd.Function):
    """Autograd wrapper for flash_attn fwd/bwd with mask_mod + block sparsity.

    The public flash_attn_func does not pass mask_mod to its backward, making
    gradients incorrect for non-causal custom masks. This wrapper correctly
    forwards mask_mod to both _flash_attn_fwd and _flash_attn_bwd.

    Block-sparse iteration lets the kernel skip tiles that are entirely masked
    out, which is critical for the EAGLE3 pattern at long sequences where the
    mask is very sparse (causal + diagonal suffix).
    """

    # 调用底层 FA4 时传入自定义 mask 和前向块索引，保存 Q/K/V/out/LSE 与反向所需的 mask 元数据。
    @staticmethod
    def forward(
        ctx, q, k, v, mask_mod_cute, mask_mod_flex, softmax_scale, max_seq_len, aux_tensors=None
    ):
        bsz, q_len, num_heads, _ = q.shape
        kv_len = k.shape[1]

        # 前向按每个 Q 块查会访问的 KV 块，跳过完全屏蔽的 tile，稀疏 suffix 长时尤为重要。
        fwd_block_sparse = _get_block_sparse(
            mask_mod_flex,
            q_len,
            kv_len,
            bsz,
            num_heads,
            q.device,
            direction="kv",
        )

        # causal=False 因为完整可见性由 mask_mod 给出；若再套普通 causal，会错误删掉物理位置靠后的链键。
        out, lse, *_rest = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=False,
            mask_mod=mask_mod_cute,
            block_sparse_tensors=fwd_block_sparse,
            return_lse=True,
            aux_tensors=aux_tensors,
        )

        # 反向访问方向相反，需要每个 KV 块对应的 Q 块列表；不能直接复用 forward 元数据。
        bwd_block_sparse = _get_block_sparse(
            mask_mod_flex,
            q_len,
            kv_len,
            bsz,
            num_heads,
            q.device,
            direction="q",
        )

        # 保存底层反向需要的张量，Python mask/稀疏元数据保存在 ctx 属性上，不参与求导。
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.mask_mod_cute = mask_mod_cute
        ctx.bwd_block_sparse = bwd_block_sparse
        ctx.aux_tensors = aux_tensors
        return out

    # 将与前向相同的 mask_mod 传给底层 backward，返回 Q/K/V 梯度；仅前向加 mask 会训练出错误梯度。
    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout.contiguous(),
            lse,
            softmax_scale=ctx.softmax_scale,
            causal=False,
            mask_mod=ctx.mask_mod_cute,
            block_sparse_tensors=ctx.bwd_block_sparse,
            aux_tensors=ctx.aux_tensors,
        )
        return dq, dk, dv, None, None, None, None, None


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # 分配 Q/K/V/O 与 RoPE；通常输入 2H，但 attn_norm 路径已先压回 H。
    # head_dim 可显式配置，注意力内部宽度不要求等于残差 hidden_size。
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        # attn_norm layers project norm(emb) | norm(hidden) back to hidden_size before attention.
        in_size = self.hidden_size * (1 if getattr(config, "attn_norm", False) else 2)
        self.q_proj = nn.Linear(in_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(in_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(in_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    # 把 GQA 的完整 head_dim 交共享 RoPE 构造器，模型配置决定频率和缩放算法。
    def _init_rope(self):
        self.rotary_emb = build_rotary_embedding(
            self.config, self.head_dim, self.max_position_embeddings
        )

    # 无缓存时做普通因果 SDPA；训练缓存为 [B,H,depth,S,D]，第 0 块做三角矩阵乘。
    # 后续块只计算同槽点积并拼到分数最后维，再对前缀+所有链节点统一 softmax，避免跨训练链泄漏。
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        # 阶段一：线性层作用于最后维，得到 Q 的 Hq*D 和 K/V 的 Hkv*D。
        # view+transpose 仅改布局为 [B,H,S,D]，Q/K 头数不同是 GQA 的共享设计。
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # 阶段二：无缓存普通 attention 与多深度训练分开处理；训练包装器通常显式传 use_cache=True。
        # 两者的参数层相同，但缓存路径必须额外保留各深度同槽键。
        if not use_cache:
            # Standard path — no caching
            if isinstance(self.rotary_emb, LlamaMutiRotaryEmbedding):
                cos, sin = self.rotary_emb(query_states, position_ids)
                cos, sin = cos.to(query_states.device), sin.to(query_states.device)
                query_states, key_states = apply_multimodal_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                    self.config.rope_scaling["mrope_section"],
                )
            else:
                cos, sin = self.rotary_emb(query_states, seq_len=q_len)
                cos, sin = cos.to(query_states.device), sin.to(query_states.device)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids
                )

            # 此实现显式把 KV 复制到 Q 头数以便普通 SDPA；Flex 路径交给 enable_gqa 处理，布局不可混用。
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
                dropout_p=0.0,
            )

        else:
            # Cached path — cache_keys shape: [bsz, num_heads, num_cached, seq_len, head_dim]
            # 缓存第 2 维是已完成的深度数。第 d 次调用给 position_ids 加 d，保证同一链的时间坐标递增。
            lck = 0 if cache_keys is None else cache_keys.shape[2]

            if isinstance(self.rotary_emb, LlamaMutiRotaryEmbedding):
                cos, sin = self.rotary_emb(query_states, position_ids + lck)
                cos, sin = cos.to(query_states.device), sin.to(query_states.device)
                query_states, key_states = apply_multimodal_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                    self.config.rope_scaling["mrope_section"],
                )
            else:
                cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
                cos, sin = cos.to(query_states.device), sin.to(query_states.device)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids + lck
                )

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            # Append to tensor cache: [bsz, num_heads, num_cached, seq_len, head_dim]
            if cache_keys is None:
                cache_keys = key_states.unsqueeze(2)
                cache_values = value_states.unsqueeze(2)
            else:
                cache_keys = torch.cat([cache_keys, key_states.unsqueeze(2)], dim=2)
                cache_values = torch.cat([cache_values, value_states.unsqueeze(2)], dim=2)

            lck = cache_keys.shape[2]
            # 设 S=4、当前深度为2：query 槽1可看 k0 的位置0..1，以及 k1[1]、k2[1]。
            # 它不能看 k1[0] 或 k2[0]，这些属于其他训练起点的预测链。
            k0 = cache_keys[:, :, 0]
            v0 = cache_values[:, :, 0]

            # causal
            attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(self.head_dim)

            attn_weights = attn_weights + attention_mask

            # 阶段三：第 i 轮只做 query*ki 的逐元素乘再沿 D 求和，输出 [B,H,S]，而非 S×S 矩阵。
            # 每条链只有一个新候选键，所以分数张量的最后宽度为 S+(depth-1)。
            for i in range(1, lck):
                ki = cache_keys[:, :, i]
                attn_weightsi = (query_states * ki).sum(-1) / math.sqrt(self.head_dim)
                attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

            # upcast attention to fp32
            # 阶段四：前缀和链键在同一个最后维上归一化，float32 softmax 降低指数/求和误差。
            # 转回 Q dtype 后，前缀用矩阵乘 V0，链节点用逐槽标量权重乘 Vi。
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            attn_weights0 = attn_weights[..., :q_len]

            attn_output = torch.matmul(attn_weights0, v0)

            for i in range(1, lck):
                vi = cache_values[:, :, i]
                attn_weightsi = attn_weights[..., q_len + i - 1]
                attn_outputi = attn_weightsi[..., None] * vi
                attn_output = attn_output + attn_outputi

        # 将所有 heads 拼回 [B,S,Hq*D] 后做 o_proj 回残差宽度 H；head_dim 显式配置时二者可不同。
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)

        attn_output = self.o_proj(attn_output)

        return attn_output, cache_keys, cache_values


class LlamaFlexAttention(LlamaAttention):
    """
    Attention layer implemented with flex attention. We keep the parameters consistent with LlamaAttention.
    The used parameters are:
        - hidden_states: input hidden states
        - attention_mask: attention mask not expanded, straight from data loader.
        - position_ids: position ids
        - cache_keys/cache_values: tensor caches for storing past key and value states.
    """

    # 每轮沿 KV 序列维追加 S 个键，lck=past_seen_tokens//S，RoPE 坐标加的是深度而非总缓存长度。
    # 通过解析 causal+suffix BlockMask 和 enable_gqa 直接处理共享 KV，避免显式重复 heads。
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        # cache_keys shape: [bsz, num_kv_heads, cache_seq_len, head_dim] (concatenated along seq dim)
        # 这份缓存是 [B,Hkv,depth*S,D]，不同于 SDPA 的五维缓存。
        # 读取长度后除以当前 S 才得到展开深度，RoPE 不能直接偏移 past_seen_tokens。
        past_seen_tokens = cache_keys.shape[2] if cache_keys is not None else 0

        # Q/K/V 投影与 SDPA 共用权重，因此后端差异集中在排列、mask 表达和 kernel，不改变参数数学含义。
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        lck = past_seen_tokens // q_len
        if isinstance(self.rotary_emb, LlamaMutiRotaryEmbedding):
            cos, sin = self.rotary_emb(query_states, position_ids + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                self.config.rope_scaling["mrope_section"],
            )
        else:
            cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            # Keep positions ids aligned when padding so the KV cache is unaffected.
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids + lck
            )

        # Concatenate along sequence dimension: [bsz, num_kv_heads, total_seq_len, head_dim]
        # 追加整个 S 长度的新轮；这些键仍保留梯度链，后续 loss 可以回传到更早深度的预测特征。
        if cache_keys is not None:
            key_cache = torch.cat([cache_keys, key_states], dim=2)
            value_cache = torch.cat([cache_values, value_states], dim=2)
        else:
            key_cache = key_states
            value_cache = value_states

        flex_attention_func = flex_attention if q_len <= 128 else compile_friendly_flex_attention

        # 解析结构不依赖样本内容，B 维参与广播，H=1 让所有头共享同一可见关系。
        # 此 EAGLE mask 不编码 packed 样本边界，不能自行把多样本塞在一行后期望隔离。
        block_mask = eagle3_block_mask(
            Q_LEN=q_len,
            KV_LEN=key_cache.shape[-2],
            B=bsz,
            H=1,  # Rely on broadcast
            device=query_states.device,
            lck=lck,
        )
        # 传入 contiguous 的 K/V，kernel 内部按 GQA 头组读取，避免 Python repeat_kv 导致的大量重复数据。
        attn_output = flex_attention_func(
            query=query_states,
            key=key_cache.contiguous(),
            value=value_cache.contiguous(),
            block_mask=block_mask,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)
        return attn_output, key_cache, value_cache


class LlamaFlashAttention(LlamaAttention):
    """
    Attention layer implemented with flash attention. We keep the parameters consistent with LlamaAttention.
    The used parameters are:
        - hidden_states: input hidden states
        - position_ids: position ids
        - cache_keys/cache_values: tensor caches for storing past key and value states
    """

    # 复用 GQA 参数，并在实例化时检查标准 FA 的前后向与 padding 工具，缺失依赖立即报错。
    def __init__(self, config):
        super().__init__(config)
        if (
            _std_flash_attn_forward is None
            or _std_flash_attn_backward is None
            or _std_flash_unpad_input is None
            or _std_flash_pad_input is None
            or _std_flash_attn_varlen_backward is None
        ):
            _raise_standard_flash_attn_unavailable()

    # 使用 [B,depth,S,Hkv,D] 缓存，依据 attention_mask 与当前深度计算有效长度。
    # 交给自定义 merge 统一前缀与链节点的归一化，输出再合并 heads 过 o_proj。
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        # 标准 FA 使用 [B,S,H,D]，与 SDPA 的 [B,H,S,D] 不同；后面 RoPE 因此 unsqueeze_dim=2。
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # cache 的第 1 维是 depth，当前新 token 的实际时间位置为原始 position_ids+lck。
        lck = 0 if cache_keys is None else cache_keys.shape[1]
        if isinstance(self.rotary_emb, LlamaMutiRotaryEmbedding):
            cos, sin = self.rotary_emb(query_states, position_ids + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                self.config.rope_scaling["mrope_section"],
                unsqueeze_dim=2,
            )
        else:
            cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids + lck, unsqueeze_dim=2
            )

        # 每深度单独存 [S,Hkv,D]，自定义 merge 从第0块读取前缀，其他块按同槽读取链键。
        if cache_keys is not None:
            cache_keys = torch.cat([cache_keys, key_states.unsqueeze(1)], dim=1)
            cache_values = torch.cat([cache_values, value_states.unsqueeze(1)], dim=1)
        else:
            cache_keys = key_states.unsqueeze(1)
            cache_values = value_states.unsqueeze(1)

        assert attention_mask is not None, "LlamaFlashAttention cached path requires attention_mask"
        # 随深度增加，每条样本可监督的尾部缩短；clamp 避免短样本在深层得到负长度。
        # 这里假定 mask 对应有效前缀长度，不能用任意中间空洞 mask 替代。
        valid_lengths = attention_mask.sum(dim=-1, dtype=torch.long) - lck
        valid_lengths = valid_lengths.clamp_(0, q_len)

        # 把完整缓存和有效长度交给自定义 autograd，确保正反向共用同一混合分母。
        attn_output = _FlashCachedMergeFunc.apply(
            query_states,
            cache_keys,
            cache_values,
            valid_lengths,
            1.0 / math.sqrt(self.head_dim),
        )

        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)
        return attn_output, cache_keys, cache_values


class LlamaFlashAttentionMasked(LlamaAttention):
    """[EXPERIMENTAL] Attention layer using flash-attention cute DSL mask_mod for EAGLE3.

    Status: experimental.  Known limitations:
      - Requires flash-attention with cute DSL (flash_attn.cute) and cutlass-dsl.
      - SM90 (H100/H200): validated.
      - SM80 (A100): validated.
      - SM100+ (Blackwell): uses simplified mask_mod (no backward block-sparse).
      - Variable-length batches: correctness is maintained (loss masking protects
        gradients), but attention quality at padding positions is not guaranteed.

    This implementation concatenates all KV blocks into a single tensor and passes
    the full EAGLE3 causal+suffix attention pattern to a single flash_attn kernel via
    cute.jit mask_mod, eliminating the nested logsumexp gradient errors of
    LlamaFlashAttention.

    The used parameters are:
        - hidden_states: input hidden states
        - position_ids: position ids
        - cache_keys/cache_values: tensor caches [bsz, num_cached, seq_len, num_kv_heads, head_dim]
        - attention_mask: bool mask [bsz, seq_len] for padding
    """

    # 先把查询长度补齐到模式要求的 bucket，再按 bucket 追加每深度缓存并展平所有轮。
    # FA4 使用同槽 mask 一次计算，输出裁回原长度；缓存仍保留补齐形状以维持各轮 stride 一致。
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, orig_q_len, _ = hidden_states.size()

        if not _has_cute_dsl or _flash_attn_fwd is None:
            raise RuntimeError(
                "LlamaFlashAttentionMasked requires flash-attention cute interface "
                "and the cutlass DSL (pip install cutlass-dsl)"
            )

        # Snap q_len so every batch in the same bucket shares the same compiled
        # flash_attn kernel.  "seqlen_po2" requires power-of-2 Q_LEN;
        # all other modes only need _SNAP_Q alignment.
        q_len = _snap_q_len(orig_q_len, _effective_mask_mode())
        # bucket 中的额外行只是形状占位：真实输出末尾裁掉，未来深度缓存仍按补齐宽度取模。
        pad_sz = q_len - orig_q_len
        if pad_sz > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_sz))
            if position_ids is not None:
                extra_ids = (
                    torch.arange(orig_q_len, q_len, dtype=torch.long, device=hidden_states.device)
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )
                position_ids = torch.cat([position_ids, extra_ids], dim=1)

        # 先 padding 再投影，所有 Q/K/V 轮宽一致；否则旧轮模数与当前轮模数不一致会串链。
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # FA layout: [bsz, seq_len, heads, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # RoPE 前进按深度计数，内存补齐宽度不改变真实 token 的时间步增量。
        lck = 0 if cache_keys is None else cache_keys.shape[1]
        if isinstance(self.rotary_emb, LlamaMutiRotaryEmbedding):
            cos, sin = self.rotary_emb(query_states, position_ids + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                self.config.rope_scaling["mrope_section"],
                unsqueeze_dim=2,
            )
        else:
            cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids + lck, unsqueeze_dim=2
            )

        # Append to tensor cache: [bsz, num_cached, seq_len, num_kv_heads, head_dim]
        if cache_keys is not None:
            cache_keys = torch.cat([cache_keys, key_states.unsqueeze(1)], dim=1)
            cache_values = torch.cat([cache_values, value_states.unsqueeze(1)], dim=1)
        else:
            cache_keys = key_states.unsqueeze(1)
            cache_values = value_states.unsqueeze(1)

        # Flatten all KV cache blocks: [bsz, num_cached * seq_len, num_kv_heads, head_dim]
        # 展平 depth×S 后，各轮同槽键间隔正好 q_len；mask 的取模公式依赖这一连续布局。
        k_all = cache_keys.reshape(bsz, -1, self.num_key_value_heads, self.head_dim)
        v_all = cache_values.reshape(bsz, -1, self.num_key_value_heads, self.head_dim)

        max_seq_len = q_len
        mask_mod_cute, mask_mod_flex, aux_tensors = _build_eagle3_mask_pair(
            q_len,
            k_all.shape[1],
            bsz,
            lck,
            hidden_states.device,
        )

        # 此 wrapper 同时传递前向/反向 mask 与块稀疏索引，避免只有 forward 正确、backward 漏掉屏蔽。
        attn_output = _EagleMaskedFlashAttnFunc.apply(
            query_states,
            k_all,
            v_all,
            mask_mod_cute,
            mask_mod_flex,
            1.0 / math.sqrt(self.head_dim),
            max_seq_len,
            aux_tensors,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)
        if pad_sz > 0:
            attn_output = attn_output[:, :orig_q_len, :]
        return attn_output, cache_keys, cache_values


# 固定 dropout/window/softcap 等选项，调用标准定长 FA，返回输出与 LSE 给多分支归一化合并。
def _standard_flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, *_rest = _std_flash_attn_forward(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )
    return out, lse


# 向标准 FA backward 传全局合并后的 out/LSE 及预分配梯度缓冲，保持前缀项的正确 softmax 导数。
def _standard_flash_attn_backward_call(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> None:
    _std_flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        0.0,
        softmax_scale,
        causal,
        -1,
        -1,
        0.0,
        None,
        False,
        None,
    )


# 按 attention_mask 去掉 padding 后调用 varlen FA，再把输出和 LSE 放回 [B,S,...] 布局。
def _standard_flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = _std_flash_unpad_input(q, attention_mask)
    k_unpad, _, cu_seqlens_k, max_seqlen_k, _ = _std_flash_unpad_input(k, attention_mask)
    v_unpad, _, _, _, _ = _std_flash_unpad_input(v, attention_mask)
    out_unpad, lse_unpad, _ = _std_flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=True,
    )
    out = _std_flash_pad_input(out_unpad, indices_q, q.shape[0], q.shape[1])
    lse_padded = _std_flash_pad_input(
        lse_unpad.transpose(0, 1), indices_q, q.shape[0], q.shape[1]
    ).transpose(1, 2)
    return out, lse_padded, indices_q, cu_seqlens_q


# 对 Q/K/V、dout、合并 out/LSE 使用一致去 padding 索引，反传后将梯度 pad 回原张量并 copy 到缓冲。
def _standard_flash_attn_varlen_backward_call(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    attention_mask: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> None:
    # Q 与 K 各自保留还原索引，但使用同一有效范围；cu_seqlens 描述去 padding 后每个样本的边界。
    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = _std_flash_unpad_input(q, attention_mask)
    k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = _std_flash_unpad_input(k, attention_mask)
    v_unpad, _, _, _, _ = _std_flash_unpad_input(v, attention_mask)
    dout_unpad, _, _, _, _ = _std_flash_unpad_input(dout, attention_mask)
    out_unpad, _, _, _, _ = _std_flash_unpad_input(out, attention_mask)
    lse_unpad, _, _, _, _ = _std_flash_unpad_input(softmax_lse.transpose(1, 2), attention_mask)
    # LSE 从 [B,H,S] 换到去 padding 的 [H,total_tokens]，布局正确与数值正确同等重要。
    lse_unpad = lse_unpad.transpose(0, 1).contiguous()

    dq_unpad = torch.empty_like(q_unpad)
    dk_unpad = torch.empty_like(k_unpad)
    dv_unpad = torch.empty_like(v_unpad)
    _std_flash_attn_varlen_backward(
        dout_unpad,
        q_unpad,
        k_unpad,
        v_unpad,
        out_unpad,
        lse_unpad,
        dq_unpad,
        dk_unpad,
        dv_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        softmax_scale,
        causal,
        -1,
        -1,
        0.0,
        None,
        False,
        None,
    )
    # 反向结果恢复 padding 布局后写入调用方给的缓冲，后面可继续加各 suffix 分支梯度。
    dq.copy_(_std_flash_pad_input(dq_unpad, indices_q, q.shape[0], q.shape[1]))
    dk.copy_(_std_flash_pad_input(dk_unpad, indices_k, k.shape[0], k.shape[1]))
    dv.copy_(_std_flash_pad_input(dv_unpad, indices_k, v.shape[0], v.shape[1]))


class _FlashCachedMergeFunc(torch.autograd.Function):
    # 前缀用 varlen 因果 FA，后续每个链节点是单独一项分数，用 logsumexp 合并分母。
    # 输出是各分支输出乘 exp(LSE_i-LSE_all) 的和，不能简单平均；padding 输出清零。
    @staticmethod
    def forward(ctx, q, cache_k, cache_v, valid_lengths, softmax_scale: float):
        bsz, q_len, num_heads, head_dim = q.shape
        num_blocks = cache_k.shape[1]
        num_kv_heads = cache_k.shape[3]
        num_groups = num_heads // num_kv_heads
        # 把 Q 头拆成 [Hkv,groups]，每个 KV 头的值能广播给同组多个 Q，无需物化重复 KV。
        q_expanded = q.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        valid_lengths = valid_lengths.to(device=q.device, dtype=torch.long).clamp_(0, q_len)
        valid_mask = (
            torch.arange(q_len, device=q.device).unsqueeze(0) < valid_lengths.unsqueeze(1)
        ).view(bsz, q_len, 1, 1)
        attention_mask = valid_mask.view(bsz, q_len)

        k0 = cache_k[:, 0].contiguous()
        v0 = cache_v[:, 0].contiguous()
        # 阶段一：真实有效前缀用高效 FA 算出归一化输出 out0 与对数分母 LSE0。
        # 只有 out0 不够，因为后面新增链键时必须知道前缀原本占多少概率质量。
        out0, lse0_kernel, _, _ = _standard_flash_attn_varlen_forward(
            q.contiguous(),
            k0,
            v0,
            attention_mask,
            softmax_scale=softmax_scale,
            causal=True,
        )
        out0_expanded = out0.view(bsz, q_len, num_kv_heads, num_groups, head_dim).float()
        neg_inf = torch.tensor(float("-inf"), device=q.device, dtype=torch.float32)
        lse0 = lse0_kernel.transpose(1, 2).view(bsz, q_len, num_kv_heads, num_groups).float()
        # 被 padding 排除的分支质量用 -inf 表示 exp(-inf)=0；最终再清理全无效行的未定义混合值。
        lse0 = torch.where(valid_mask, lse0, neg_inf)
        lse_terms = [lse0]
        attn_terms = [out0_expanded]
        # 阶段二：每条链每深度只有一个键，因此该分支输出就是 Vi，本身不需 softmax 矩阵。
        for i in range(1, num_blocks):
            ki = cache_k[:, i].unsqueeze(-2).float()
            vi = cache_v[:, i].unsqueeze(-2).float()
            lse_i = (q_expanded.float() * ki).sum(-1) * softmax_scale
            lse_terms.append(torch.where(valid_mask, lse_i, neg_inf))
            attn_terms.append(vi.expand_as(out0_expanded))

        # 若前缀未归一化质量为 3、某链键质量为 1，合并输出应为 0.75*out_prefix+0.25*v_chain。
        # 用两个分支各自 softmax 后直接相加会让该链键错误获得与整个前缀相同权重。
        merged_lse = torch.logsumexp(torch.stack(lse_terms, dim=-1), dim=-1)
        out = sum(
            term * torch.exp(lse - merged_lse).unsqueeze(-1)
            for term, lse in zip(attn_terms, lse_terms)
        )
        out = torch.where(valid_mask.unsqueeze(-1), out, 0.0)
        merged_lse = torch.where(valid_mask, merged_lse, 0.0)
        # 保存 float32 合并结果和全局 LSE 给 backward；只保存各子 attention 的独立输出会丢失跨分支梯度。
        ctx.save_for_backward(q, cache_k, cache_v, out, merged_lse, valid_lengths)
        ctx.softmax_scale = softmax_scale
        return out.to(q.dtype).reshape_as(q)

    # 前缀 FA 使用合并输出/LSE 反传；链分数梯度为 w_i*dout·(v_i-out)。
    # GQA 对共享 K/V 的梯度沿组维求和，最后转回原 dtype；忽略 LSE 合并导数会产生错误训练梯度。
    @staticmethod
    def backward(ctx, grad_out):
        q, cache_k, cache_v, out, merged_lse, valid_lengths = ctx.saved_tensors
        bsz, q_len, num_heads, head_dim = q.shape
        num_blocks = cache_k.shape[1]
        num_kv_heads = cache_k.shape[3]
        num_groups = num_heads // num_kv_heads
        scale = ctx.softmax_scale
        if grad_out.ndim == 3:
            grad_out = grad_out.view(bsz, q_len, num_heads, head_dim)
        valid_lengths = valid_lengths.to(device=q.device, dtype=torch.long)
        valid_mask = (
            torch.arange(q_len, device=q.device).unsqueeze(0) < valid_lengths.unsqueeze(1)
        ).view(bsz, q_len, 1, 1)
        attention_mask = valid_mask.view(bsz, q_len)
        # 先把占位行上游梯度清零，避免 padding 行的无效 softmax 中间值影响真实参数。
        grad_out = torch.where(valid_mask, grad_out, 0.0)

        grad_out_f = grad_out.float().view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        q_f = q.float()
        q_expanded = q_f.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        out_f = out.float()
        out_expanded = out_f.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        out_q = out.to(q.dtype).reshape(bsz, q_len, num_heads, head_dim)

        # 用 float32 累加多个深度贡献；Q 同时影响前缀和每个链键，不能只保留某一分支梯度。
        dq = torch.zeros_like(q_f)
        dcache_k = torch.zeros_like(cache_k.float())
        dcache_v = torch.zeros_like(cache_v.float())

        dq0 = torch.zeros_like(q)
        dk0 = torch.zeros_like(cache_k[:, 0])
        dv0 = torch.zeros_like(cache_v[:, 0])
        merged_lse_kernel = merged_lse.reshape(bsz, q_len, num_heads).transpose(1, 2).contiguous()
        # 阶段一：仍让 FA kernel 算高成本前缀梯度，但传入的是合并后的输出和总分母。
        # FA 的公式据此自动包含链节点对前缀归一化的竞争影响。
        _standard_flash_attn_varlen_backward_call(
            grad_out.contiguous(),
            q.contiguous(),
            cache_k[:, 0].contiguous(),
            cache_v[:, 0].contiguous(),
            out_q.contiguous(),
            merged_lse_kernel,
            attention_mask,
            dq0,
            dk0,
            dv0,
            softmax_scale=scale,
            causal=True,
        )
        dq += dq0.float()
        dcache_k[:, 0] += dk0.float()
        dcache_v[:, 0] += dv0.float()

        # 阶段二：显式计算每个单键链分支对 Q/K/V 的导数，再与前缀导数相加。
        for i in range(1, num_blocks):
            ki = cache_k[:, i].float().unsqueeze(-2)
            vi = cache_v[:, i].float().unsqueeze(-2)
            lse_i = (q_expanded * ki).sum(-1) * scale
            wi = torch.where(valid_mask, torch.exp(lse_i - merged_lse), 0.0)
            d_out_i = grad_out_f * wi.unsqueeze(-1)
            # v_i-out 是 softmax 分母导数带来的扣除项，保证分支质量增加时其他分支权重同步下降。
            d_lse_i = wi * (grad_out_f * (vi.expand_as(out_expanded) - out_expanded)).sum(-1)
            dq += (d_lse_i.unsqueeze(-1) * scale * ki).reshape_as(q)
            # 一个 KV 头被多个 query heads 共享，对组维求和把所有头对同一参数的贡献合并。
            dcache_k[:, i] += (d_lse_i.unsqueeze(-1) * scale * q_expanded).sum(dim=3)
            dcache_v[:, i] += d_out_i.sum(dim=3)

        return dq.to(q.dtype), dcache_k.to(cache_k.dtype), dcache_v.to(cache_v.dtype), None, None


class _USPFlashCachedMergeFunc(torch.autograd.Function):
    # 从 Q 与 [B,depth,S,Hkv,D] 缓存统一提取尺寸及 GQA 组数，避免前反向拆形状不一致。
    @staticmethod
    def _merge_dims(q: torch.Tensor, cache_k: torch.Tensor):
        bsz, q_len, num_heads, head_dim = q.shape
        num_blocks = cache_k.shape[1]
        num_kv_heads = cache_k.shape[3]
        num_groups = num_heads // num_kv_heads
        return bsz, q_len, num_heads, head_dim, num_blocks, num_kv_heads, num_groups

    # 将 [B,S,Hkv,groups] 的 LSE 重排为 kernel 需要的连续 [B,Hq,S]。
    @staticmethod
    def _kernel_lse(lse: torch.Tensor, bsz: int, q_len: int, num_heads: int) -> torch.Tensor:
        return lse.reshape(bsz, q_len, num_heads).transpose(1, 2).contiguous()

    # 用于不带 padding 的 USP 分片，定长 FA 算因果前缀，显式链项通过同一 logsumexp 分母混合。
    @staticmethod
    def forward(ctx, q, cache_k, cache_v, softmax_scale: float):
        bsz, q_len, num_heads, head_dim, num_blocks, num_kv_heads, num_groups = (
            _USPFlashCachedMergeFunc._merge_dims(q, cache_k)
        )
        q_expanded = q.view(bsz, q_len, num_kv_heads, num_groups, head_dim)

        k0 = cache_k[:, 0].contiguous()
        v0 = cache_v[:, 0].contiguous()
        assert _std_flash_attn_forward is not None and _std_flash_attn_backward is not None, (
            "USP cached merge requires standard flash-attn forward/backward "
            f"(standard import error: {_std_flash_attn_import_error!r})"
        )
        # 定长 USP 分片已由上游安排边界，不使用 varlen 去 padding；首轮仍是因果前缀。
        out0, lse0_kernel = _standard_flash_attn_forward(
            q.contiguous(),
            k0,
            v0,
            softmax_scale=softmax_scale,
            causal=True,
        )
        out0_expanded = out0.view(bsz, q_len, num_kv_heads, num_groups, head_dim).float()
        lse0 = lse0_kernel.transpose(1, 2).reshape(bsz, q_len, num_kv_heads, num_groups).float()
        lse_terms = [lse0]
        attn_terms = [out0_expanded]
        # 每个后续深度只加入当前序列槽自己的 K/V，与 Ulysses 分头后的 Q 组广播匹配。
        for i in range(1, num_blocks):
            ki = cache_k[:, i].unsqueeze(-2).float()
            vi = cache_v[:, i].unsqueeze(-2).float()
            lse_terms.append((q_expanded.float() * ki).sum(-1) * softmax_scale)
            attn_terms.append(vi.expand_as(out0_expanded))

        # 稳定计算 sum(exp(LSE_i)) 的对数；各分支权重来自同一分母，总质量为 1。
        merged_lse = torch.logsumexp(torch.stack(lse_terms, dim=-1), dim=-1)
        out = sum(
            term * torch.exp(lse - merged_lse).unsqueeze(-1)
            for term, lse in zip(attn_terms, lse_terms)
        )
        # 缓存合并后的浮点高精度统计，反向标准 FA 与解析 suffix 都要使用它。
        ctx.save_for_backward(q, cache_k, cache_v, out, merged_lse)
        ctx.softmax_scale = softmax_scale
        return out.to(q.dtype).reshape_as(q)

    # 以前向合并的 out/LSE 计算前缀梯度，再累加同槽链的解析梯度；共享 KV 头须归并各 Q 组贡献。
    @staticmethod
    def backward(ctx, grad_out):
        q, cache_k, cache_v, out, merged_lse = ctx.saved_tensors
        bsz, q_len, num_heads, head_dim, num_blocks, num_kv_heads, num_groups = (
            _USPFlashCachedMergeFunc._merge_dims(q, cache_k)
        )
        scale = ctx.softmax_scale

        grad_out_f = grad_out.float().view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        q_f = q.float()
        q_expanded = q_f.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        out_f = out.float()
        out_expanded = out_f.view(bsz, q_len, num_kv_heads, num_groups, head_dim)

        # 先建立 float32 梯度累计槽，避免每个 suffix 分支相加时过早舍入到 bf16。
        dq = torch.zeros_like(q_f)
        dcache_k = torch.zeros_like(cache_k.float())
        dcache_v = torch.zeros_like(cache_v.float())

        merged_lse_kernel = _USPFlashCachedMergeFunc._kernel_lse(merged_lse, bsz, q_len, num_heads)
        dq0 = torch.empty_like(q)
        dk0 = torch.empty_like(cache_k[:, 0])
        dv0 = torch.empty_like(cache_v[:, 0])
        # 前缀反向读合并 out/LSE，反映所有链键参与后的总 softmax，而不是单独前缀 softmax。
        _standard_flash_attn_backward_call(
            grad_out.contiguous(),
            q.contiguous(),
            cache_k[:, 0].contiguous(),
            cache_v[:, 0].contiguous(),
            out.to(q.dtype).reshape_as(q).contiguous(),
            merged_lse_kernel,
            dq0,
            dk0,
            dv0,
            softmax_scale=scale,
            causal=True,
        )
        dq += dq0.float()
        dcache_k[:, 0] += dk0.float()
        dcache_v[:, 0] += dv0.float()

        # 链梯度由 w*dout 和 w*dout·(v-out) 两项组成，分别对应 value 与 score 的导数。
        for i in range(1, num_blocks):
            ki = cache_k[:, i].float().unsqueeze(-2)
            vi = cache_v[:, i].float().unsqueeze(-2)
            lse_i = (q_expanded * ki).sum(-1) * scale
            wi = torch.exp(lse_i - merged_lse)
            d_out_i = grad_out_f * wi.unsqueeze(-1)
            d_lse_i = wi * (grad_out_f * (vi.expand_as(out_expanded) - out_expanded)).sum(-1)
            dq += (d_lse_i.unsqueeze(-1) * scale * ki).reshape_as(q)
            dcache_k[:, i] += (d_lse_i.unsqueeze(-1) * scale * q_expanded).sum(dim=3)
            dcache_v[:, i] += d_out_i.sum(dim=3)

        return dq.to(q.dtype), dcache_k.to(cache_k.dtype), dcache_v.to(cache_v.dtype), None


# 用 logaddexp 稳定合并两个分支分母，再按 exp(oldLSE-newLSE) 加权输出，避免直接指数溢出。
def _update_ring_out_and_lse(
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.float()
    block_lse = block_lse.float()
    if out is None or lse is None:
        return block_out, block_lse
    new_lse = torch.logaddexp(lse, block_lse)
    out = out * torch.exp(lse - new_lse).unsqueeze(-1) + block_out * torch.exp(
        block_lse - new_lse
    ).unsqueeze(-1)
    return out, new_lse


class _USPRingFlashCachedMergeFunc(torch.autograd.Function):
    # 先用 ring attention 得到跨 rank 的完整因果前缀输出，再仅添加本链的 suffix 节点。
    # 每次合并都更新输出与 LSE，因此 ring 与本地链使用同一个归一化分母。
    @staticmethod
    def forward(ctx, q, cache_k, cache_v, ring_group: dist.ProcessGroup, softmax_scale: float):
        from yunchang.ring.ring_flash_attn import ring_flash_attn_forward

        bsz, q_len, num_heads, head_dim, num_blocks, num_kv_heads, num_groups = (
            _USPFlashCachedMergeFunc._merge_dims(q, cache_k)
        )
        q_expanded = q.view(bsz, q_len, num_kv_heads, num_groups, head_dim)

        # 阶段一：ring 各 rank 交换首轮前缀 KV，得到跨分片因果注意力输出；不是只看本地 chunk。
        out_ring, lse_ring = ring_flash_attn_forward(
            ring_group,
            q.contiguous(),
            cache_k[:, 0].contiguous(),
            cache_v[:, 0].contiguous(),
            softmax_scale=softmax_scale,
            dropout_p=0.0,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
        )
        # 不同 ring 返回布局可能为 [B,H,S]，先统一成 [B,S,H] 再拆 GQA 组维。
        if lse_ring.dim() == 3 and lse_ring.shape[1] == num_heads:
            lse_ring = lse_ring.transpose(1, 2)
        acc_out = out_ring.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        acc_lse = lse_ring.reshape(bsz, q_len, num_kv_heads, num_groups)

        # 阶段二：预测链的同槽节点在本 rank 已具备，无需对整个 suffix 再进行全序列 ring 交换。
        for i in range(1, num_blocks):
            ki = cache_k[:, i].unsqueeze(-2).float()
            vi = cache_v[:, i].unsqueeze(-2).float()
            lse_i = (q_expanded.float() * ki).sum(-1) * softmax_scale
            out_i = vi.expand(bsz, q_len, num_kv_heads, num_groups, head_dim)
            acc_out, acc_lse = _update_ring_out_and_lse(acc_out, acc_lse, out_i, lse_i)

        ctx.save_for_backward(q, cache_k, cache_v, acc_out, acc_lse)
        # 反向必须复用同一进程组与前向 softmax_scale，以保持 collective 顺序和数学契约一致。
        ctx.ring_group = ring_group
        ctx.softmax_scale = softmax_scale
        return acc_out.to(q.dtype).reshape_as(q)

    # ring backward 接收合并后的 out/LSE，把跨分片前缀梯度算正确，再补本地同槽链梯度。
    @staticmethod
    def backward(ctx, grad_out):
        from yunchang.ring.ring_flash_attn import ring_flash_attn_backward

        q, cache_k, cache_v, out, merged_lse = ctx.saved_tensors
        bsz, q_len, num_heads, head_dim, num_blocks, num_kv_heads, num_groups = (
            _USPFlashCachedMergeFunc._merge_dims(q, cache_k)
        )
        scale = ctx.softmax_scale

        if grad_out.ndim == 3:
            grad_out = grad_out.view(bsz, q_len, num_heads, head_dim)

        grad_out_f = grad_out.float().view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        q_f = q.float()
        q_expanded = q_f.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        out_expanded = out.float().view(bsz, q_len, num_kv_heads, num_groups, head_dim)

        dcache_k = torch.zeros_like(cache_k.float())
        dcache_v = torch.zeros_like(cache_v.float())
        out_q = out.to(q.dtype).reshape_as(q)
        # 先把 GQA 分组 LSE 合并头维并转成 kernel 布局，数值仍是跨 ring 前缀与本地链的总分母。
        merged_lse_kernel = _USPFlashCachedMergeFunc._kernel_lse(merged_lse, bsz, q_len, num_heads)

        # ring kernel 使用总输出/分母计算各 rank 前缀贡献；通信部分不另造局部分母。
        dq, dk0, dv0 = ring_flash_attn_backward(
            ctx.ring_group,
            grad_out.contiguous(),
            q.contiguous(),
            cache_k[:, 0].contiguous(),
            cache_v[:, 0].contiguous(),
            out_q.contiguous(),
            merged_lse_kernel,
            softmax_scale=scale,
            dropout_p=0.0,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
        )
        dq = dq.float()
        # 第0轮 KV 梯度由 ring 通信内核完成跨分片汇总；后续轮只在本地逐槽更新对应缓存梯度。
        dcache_k[:, 0] = dk0.float()
        dcache_v[:, 0] = dv0.float()

        # 在 ring 返回的 dQ 上追加本地链贡献；KV 的第0块来自 ring，其余块来自逐槽解析导数。
        for i in range(1, num_blocks):
            ki = cache_k[:, i].float().unsqueeze(-2)
            vi = cache_v[:, i].float().unsqueeze(-2)
            lse_i = (q_expanded * ki).sum(-1) * scale
            wi = torch.exp(lse_i - merged_lse)
            d_out_i = grad_out_f * wi.unsqueeze(-1)
            d_lse_i = wi * (grad_out_f * (vi.expand_as(out_expanded) - out_expanded)).sum(-1)
            dq += (d_lse_i.unsqueeze(-1) * scale * ki).reshape_as(q)
            dcache_k[:, i] += (d_lse_i.unsqueeze(-1) * scale * q_expanded).sum(dim=3)
            dcache_v[:, i] += d_out_i.sum(dim=3)

        return dq.to(q.dtype), dcache_k.to(cache_k.dtype), dcache_v.to(cache_v.dtype), None, None


class LlamaUSPFlashAttention(LlamaAttention):
    """USP attention with Ulysses all-to-all and local flash attention."""

    # 要求分布式与 ring/Ulysses 组已初始化，保存 all-to-all 算子及 scatter heads/gather sequence 维号。
    def __init__(self, config):
        super().__init__(config)
        if not dist.is_initialized():
            raise RuntimeError(
                "LlamaUSPFlashAttention requires torch.distributed to be initialized first."
            )
        try:
            from yunchang.comm import SeqAllToAll4D
        except ImportError as exc:
            raise RuntimeError(
                "USP requires yunchang; please install it to use USP attention."
            ) from exc

        self._SeqAllToAll4D = SeqAllToAll4D
        self.ring_pg = get_sp_ring_group()
        self.ulysses_pg = get_sp_ulysses_group()
        if self.ring_pg is None or self.ulysses_pg is None:
            raise RuntimeError("USP requires sp ring/ulysses groups to be initialized.")
        self.sp_ring_degree = dist.get_world_size(self.ring_pg)
        self.sp_ulysses_degree = dist.get_world_size(self.ulysses_pg)
        self.scatter_idx = 2
        self.gather_idx = 1
        self.use_sync = False

    # Q/K/V 先经 Ulysses all-to-all，从短序列多头变成长序列少头，再按全局位置旋转与缓存。
    # ring_degree>1 时前缀用环通信，否则本地合并；输出反向 all-to-all 恢复本 rank 序列与完整头。
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        del attention_mask, use_cache
        bsz, q_len, _ = hidden_states.size()
        local_q_len = q_len

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        # scatter_idx=2 拆头，gather_idx=1 拼序列；总元素数不变，但每 rank 计算的注意力视野变大。
        query_states = self._SeqAllToAll4D.apply(
            self.ulysses_pg,
            query_states,
            self.scatter_idx,
            self.gather_idx,
            self.use_sync,
        )

        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        # KV 必须做与 Q 相同的通信布局转换，确保位置与共享头分组仍对应。
        key_states = self._SeqAllToAll4D.apply(
            self.ulysses_pg,
            key_states,
            self.scatter_idx,
            self.gather_idx,
            self.use_sync,
        )

        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = self._SeqAllToAll4D.apply(
            self.ulysses_pg,
            value_states,
            self.scatter_idx,
            self.gather_idx,
            self.use_sync,
        )

        q_len = query_states.shape[1]
        # 例如本地 S=256、Ulysses=2、ring=2：当前 rank 经 all-to-all 得到 512 个位置的一部分头，RoPE 表覆盖全局1024位置。
        global_q_len = local_q_len * self.sp_ring_degree * self.sp_ulysses_degree

        lck = 0 if cache_keys is None else cache_keys.shape[1]
        cos, sin = self.rotary_emb(query_states, seq_len=global_q_len + lck)
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids + lck, unsqueeze_dim=2
        )

        # 缓存保存的是通信后的头分片布局，后续深度追加到同一 shard，而不是重新回到原始本地布局。
        if cache_keys is not None:
            cache_keys = torch.cat([cache_keys, key_states.unsqueeze(1)], dim=1)
            cache_values = torch.cat([cache_values, value_states.unsqueeze(1)], dim=1)
        else:
            cache_keys = key_states.unsqueeze(1)
            cache_values = value_states.unsqueeze(1)

        softmax_scale = 1.0 / math.sqrt(self.head_dim)
        # Ulysses 组之外若还有序列分片，需 ring 补齐全局因果前缀；仅 Ulysses 时本地 FA 已可覆盖。
        if self.sp_ring_degree > 1:
            attn_output = _USPRingFlashCachedMergeFunc.apply(
                query_states,
                cache_keys,
                cache_values,
                self.ring_pg,
                softmax_scale,
            )
        else:
            attn_output = _USPFlashCachedMergeFunc.apply(
                query_states,
                cache_keys,
                cache_values,
                softmax_scale,
            )

        # 输出的 all-to-all 交换 scatter/gather 维号，恢复短序列全头布局，再在本地应用输出线性层。
        attn_output = self._SeqAllToAll4D.apply(
            self.ulysses_pg,
            attn_output,
            self.gather_idx,
            self.scatter_idx,
            self.use_sync,
        )
        attn_output = attn_output.reshape(bsz, local_q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)
        return attn_output, cache_keys, cache_values


# 对需要的长度 bucket 运行真实前后向以填充编译缓存，避免首个训练 step 承担 JIT 延迟。
# closure/simple 按 bucket 预热，运行时长度模式只需代表形状；无可用 CuTe 依赖则直接返回。
def warmup_flash_attention_masked(
    q_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    device=None,
    warmup_buckets: Optional[list] = None,
) -> None:
    """Pre-compile flash_attn cute DSL kernels for LlamaFlashAttentionMasked.

    Runs a fwd+bwd warmup to populate the in-process compile caches
    (_flash_attn_fwd.compile_cache / _flash_attn_bwd.compile_cache),
    so training step 0 avoids compilation.

    SM90:  Each Q_LEN bucket produces a distinct compiled kernel (closure-captured
           Q_LEN). All lck values within a bucket share the same compile_key, so
           one fwd+bwd call per bucket suffices.
    SM100+: A single kernel handles all Q_LEN values (seqlen_po2 reads Q_LEN from
            seqlen_info at runtime). Only one fwd+bwd call is needed regardless
            of the number of buckets.

    Args:
        warmup_buckets: Q_LEN values to pre-compile. Default (None) warms up
            only the max bucket (SM90) or a single representative (SM100+).
            For SM90 variable-length training, pass explicit bucket sizes.
    """
    if not _has_cute_dsl or _flash_attn_fwd is None:
        return
    if device is None:
        device = torch.cuda.current_device()

    mode = _effective_mask_mode()
    # Modes with closure-captured Q_LEN need one warmup per bucket;
    # runtime-Q_LEN modes ("seqlen_po2", "seqlen", "dynamic") need only one.
    # 闭包捕获 Q_LEN 会进入 kernel cache key，运行时 seqlen/aux 模式则可跨长度共享编译结果。
    needs_per_bucket = mode in ("closure", "simple")
    if needs_per_bucket:
        max_bucket = _snap_to_multiple(q_len, _SNAP_Q)
        if warmup_buckets is None:
            buckets = sorted({_SNAP_Q, max_bucket})
        else:
            buckets = sorted(
                {_SNAP_Q, max_bucket} | {_snap_to_multiple(b, _SNAP_Q) for b in warmup_buckets}
            )
    else:
        # Single compiled kernel for all Q_LEN values.
        buckets = [_SNAP_Q]

    softmax_scale = 1.0 / math.sqrt(head_dim)

    print_with_rank(
        f"Warming up flash_attn cute DSL kernels "
        f"({len(buckets)} Q_LEN bucket(s): {buckets}, mode={mode}, sm_major={_cuda_sm_major})..."
    )

    # 使用代表性 dtype/head_dim/GQA 头数组合触发真实编译，后续训练若换这些静态维仍可能产生新 kernel。
    for warmup_q_len in buckets:
        mask_mod_cute, mask_mod_flex, aux_tensors = _build_eagle3_mask_pair(
            warmup_q_len,
            warmup_q_len,
            1,
            0,
            device,
        )

        q_t = torch.randn(
            1, warmup_q_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True
        )
        k_t = torch.randn(1, warmup_q_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_t = torch.randn(1, warmup_q_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        out = _EagleMaskedFlashAttnFunc.apply(
            q_t, k_t, v_t, mask_mod_cute, mask_mod_flex, softmax_scale, warmup_q_len, aux_tensors
        )
        # 不仅 forward 要预热，训练还会第一次触发 backward；此处用 sum 构造简单上游梯度完成两者编译。
        out.sum().backward()
        del q_t, k_t, v_t, out

    torch.cuda.synchronize()
    print_with_rank(f"flash_attn cute DSL warmup complete ({len(buckets)} bucket(s)).")


class LlamaMLP(nn.Module):
    # 创建 gate/up/down 三投影并从 config.hidden_act 取激活，支持普通 dense FFN 的参数布局。
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    # 一般执行 down(act(gate(x))*up(x))；pretraining_tp>1 时按中间维切权重再拼/求和。
    # 这里是本地张量切片模拟分块计算，并未自行发起分布式 TP collective。
    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    # 初始化一维 gamma=1 与 eps，保存均方归一化参数，供 backbone 与 fused loss 读取。
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    # float32 累加最后维平方均值，乘 rsqrt 后回原 dtype 再乘 gamma，不执行中心化减均值。
    @torch.compile(dynamic=True)
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# 分别 norm token embedding 与历史隐藏态，沿最后维拼接成 2H。
# 普通层保留原 hidden 作残差；attn_norm 层经 eh_proj 压回 H，并以该投影作为新的残差起点。
def fuse_layer_input(layer, input_emb: torch.Tensor, hidden_states: torch.Tensor):
    """(attention input, residual) of a draft layer.

    The input is norm(embedding) | norm(hidden). An ``attn_norm`` layer projects it back to
    hidden_size through ``eh_proj``, whose output is then also the residual.
    """
    fused = torch.cat((layer.input_layernorm(input_emb), layer.hidden_norm(hidden_states)), dim=-1)
    if not hasattr(layer, "eh_proj"):
        return fused, hidden_states
    residual = layer.eh_proj(fused)
    return layer.pre_attention_layernorm(residual), residual


class LlamaDecoderLayer(nn.Module):
    # 选择 attention 后端并创建 norm/MLP（可替换 MoE）；当前滑窗只允许 SDPA。
    # attn_norm 与 post_mlp_norm 为显式架构变化，参数存在与否决定输入融合和后续反馈契约。
    def __init__(self, config, attention_backend: str = "sdpa"):
        super().__init__()
        self.hidden_size = config.hidden_size
        if draft_sliding_window(config) is not None and attention_backend != "sdpa":
            raise ValueError(
                "Sliding-window drafts are only masked on attention_backend='sdpa' "
                f"(dense or anchored), got {attention_backend!r}"
            )

        if attention_backend == "sdpa":
            self.self_attn = LlamaAttention(config=config)
        elif attention_backend == "flex_attention":
            print_with_rank("Using flex attention on draft model training!")
            self.self_attn = LlamaFlexAttention(config=config)
        elif attention_backend == "fa4":
            self.self_attn = LlamaFlashAttentionMasked(config=config)
        elif attention_backend == "fa":
            self.self_attn = LlamaFlashAttention(config=config)
        elif attention_backend == "usp":
            self.self_attn = LlamaUSPFlashAttention(config=config)
        else:
            raise ValueError(f"Unknown attention backend {attention_backend}")

        self.attention_backend = attention_backend
        self.mlp = Eagle4MoE(config) if getattr(config, "enable_moe", False) else LlamaMLP(config)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if getattr(config, "attn_norm", False):
            self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
            self.pre_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if getattr(config, "enable_post_mlp_norm", False):
            self.post_mlp_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # 融合两路输入后执行 attention 残差与 MLP 残差，返回未做 final norm 的层输出和更新后的缓存。
    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        hidden_states, residual = fuse_layer_input(self, input_emb, hidden_states)
        # Self Attention
        hidden_states, cache_keys, cache_values = self.self_attn(
            hidden_states=hidden_states,
            cache_keys=cache_keys,
            cache_values=cache_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, cache_keys, cache_values


class LlamaForCausalLMEagle3(Eagle3DraftModel):
    config_class = LlamaConfig

    # 构建全词表 embedding、单层 draft、K 路教师 fc 融合及 final norm/head；可选择输出裁剪或低秩头。
    # 裁剪映射先注册占位缓冲；efficient head 候选重评分要求完整词表，因此不允许与裁剪共用。
    def __init__(self, config, attention_backend="sdpa") -> None:
        super().__init__(config)
        self.config = config

        self.target_vocab_size = config.vocab_size
        self.vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.midlayer = LlamaDecoderLayer(config, attention_backend=attention_backend)

        # 教师与草稿宽度可不同，fc 输入按教师宽度×路数，输出按草稿残差宽度。
        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)

        self.fc = torch.nn.Linear(
            target_hidden_size * self.num_aux_hidden_states,
            config.hidden_size,
            bias=False,
        )
        use_fc_norm = getattr(config, "fc_norm", None)
        if use_fc_norm:
            self.fc_norm = nn.ModuleList(
                [
                    LlamaRMSNorm(target_hidden_size, eps=config.rms_norm_eps)
                    for _ in range(self.num_aux_hidden_states)
                ]
            )
        else:
            self.fc_norm = None

        # final norm/head 用于词表预测；norm_output 则决定 hidden 反馈，不改变 head 必须先 norm 的约定。
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_output = getattr(config, "norm_output", False)
        self.lm_head = build_lm_head(
            config, config.hidden_size, self.vocab_size, self.target_vocab_size
        )

        if getattr(config, "efficient_lm_head_config", None) and self.vocab_size != (
            self.target_vocab_size
        ):
            raise ValueError(
                "efficient_lm_head_config rescores candidates with the full-vocab lm_head; "
                "train without draft vocab pruning"
            )
        # 只在裁剪时注册映射缓冲；t2d 全 True 是初始化占位，后续需与真正 head 行序一起配置。
        if self.vocab_size != self.target_vocab_size:
            self.register_buffer("t2d", torch.ones(self.target_vocab_size, dtype=torch.bool))
            self.register_buffer("d2t", torch.zeros(self.vocab_size, dtype=torch.int64))

    # 查全 target 词表 embedding，给每个训练深度提供已对齐的真实 token 特征。
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    # 检查 K*H_target 输入宽度，按需分别 norm 各教师层；默认 float32 做 fc 后转回权重 dtype。
    # 环境变量可关闭此精度提升，避免多路特征投影在低精度下额外积累误差。
    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # eagle 3 requires hidden states from 3 layers
        expected_size = self.fc.in_features
        if hidden_states.size(-1) != expected_size:
            raise ValueError(
                f"Target hidden states size mismatch: {hidden_states.size(-1)} != expected: {expected_size}"
            )
        if self.fc_norm is not None:
            chunks = hidden_states.chunk(self.num_aux_hidden_states, dim=-1)
            hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)],
                dim=-1,
            )
        if os.environ.get("TORCHSPEC_EAGLE3_PROJ_FP32", "1") in {"0", "false", "False"}:
            return self.fc(hidden_states.to(self.fc.weight.dtype))
        proj = F.linear(
            hidden_states.to(torch.float32),
            self.fc.weight.to(torch.float32),
            None if self.fc.bias is None else self.fc.bias.to(torch.float32),
        )
        return proj.to(self.fc.weight.dtype)

    # final norm 后执行 dense 或 low-rank head，输出与 fused loss 使用的 norm/head 参数一致。
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_hidden_states = self.norm(hidden_states)
        return self.lm_head(norm_hidden_states)

    # 按统一 EAGLE 接口委托 midlayer，原样传递缓存、mask、位置和 use_cache，输出保留后端缓存布局。
    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.midlayer(
            input_emb=input_embeds,
            hidden_states=hidden_states,
            cache_keys=cache_keys,
            cache_values=cache_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
