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

# TokenSpeed 适配器使用 prefill 隐状态捕获，把后端已发布的键转成 TorchSpec 样本元数据。
# 当前实现明确限定单节点、文本、非 decode，并要求保存 last_hidden_states。

"""Ray actor wrapper for TokenSpeed offline hidden-state materialization."""

from __future__ import annotations

from typing import Any

import ray
import torch
from omegaconf import DictConfig, OmegaConf

from torchspec.inference.engine.base import InferenceEngine
from torchspec.ray.ray_actor import RayActor
from torchspec.transfer.mooncake.eagle_store import HIDDEN_STATES_STORAGE_DTYPE
from torchspec.utils.logging import logger, setup_file_logging

_PROTECTED_ENGINE_KEYS = frozenset(
    {
        "model",
        "attn_tp_size",
        "base_gpu_id",
        "nprocs_per_node",
        "nnodes",
        "gpu_memory_utilization",
        "enable_spec_training_mooncake",
        "eagle3_layers_to_capture",
        "enable_prefix_caching",
        "disable_kvstore",
        "enforce_eager",
        "disable_overlap_schedule",
        "disable_prefill_graph",
        "skip_server_warmup",
        "chunked_prefill_size",
        "max_model_len",
        "max_prefill_tokens",
    }
)


# 封装 TokenSpeed Engine 的启动和训练数据协议，张量写入发生在后端内部。
class TokenSpeedEngine(InferenceEngine, RayActor):
    """Use TokenSpeed eager prefill to publish TorchSpec Mooncake records."""

    # 记录 actor 拓扑及空引擎、空模型维度；捕获层列表在读取目标模型配置后填充。
    def __init__(
        self,
        args,
        rank: int,
        base_gpu_id: int | None = None,
        num_gpus_per_engine: int = 1,
        node_rank: int = 0,
        engine_group: int = 0,
    ) -> None:
        self.args = args
        self.rank = rank
        self.base_gpu_id = base_gpu_id
        self.num_gpus_per_engine = num_gpus_per_engine
        self.node_rank = node_rank
        self.local_gpu_id = None
        self._engine = None
        self._mooncake_config = None
        self._hidden_size = None
        self.aux_hidden_state_layer_ids: list[int] = []
        setup_file_logging("inference", self.rank, group=engine_group)

    # 校验单节点和训练模式，设置 GPU 与 Mooncake 环境，再读取隐藏维度及辅助层并创建 Engine。
    # 托管参数会覆盖或过滤 extra_args；TP 必须等于该引擎实际占用 GPU 数，避免拓扑与资源分配脱节。
    def init(
        self,
        mooncake_config=None,
        dist_init_addr: str | None = None,
        pre_allocated_port: int | None = None,
    ) -> None:
        del dist_init_addr, pre_allocated_port
        if self.node_rank != 0 or getattr(self.args, "tokenspeed_nnodes", 1) != 1:
            raise NotImplementedError("The initial TokenSpeed integration supports one node only")
        if not getattr(self.args, "store_last_hidden_states", True):
            raise NotImplementedError(
                "TokenSpeed currently requires inference.store_last_hidden_states=true"
            )
        if getattr(self.args, "train_with_decode", False):
            raise NotImplementedError(
                "TokenSpeed currently supports offline prefill, not train_with_decode"
            )

        # 在导入并启动实际 Engine 前定位当前 GPU；后端子进程继承正确设备配置。
        if self.base_gpu_id is not None:
            self.local_gpu_id = self.setup_gpu(self.base_gpu_id)
        else:
            self.local_gpu_id = self.setup_gpu()

        self._mooncake_config = mooncake_config
        if mooncake_config is None:
            raise ValueError("TokenSpeed hidden-state capture requires Mooncake")
        mooncake_config.local_hostname = self.get_node_ip()
        mooncake_config.export_env()

        from torchspec.transfer.mooncake.utils import (
            check_mooncake_master_available,
        )

        check_mooncake_master_available(
            mooncake_config.master_server_address,
            mooncake_config.metadata_server,
        )

        from transformers import AutoConfig

        # 只读模型配置取得 H 和层总数，以后返回 shape 无需把巨大张量重新拉回 actor。
        model_config = AutoConfig.from_pretrained(
            self.args.target_model_path,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
            cache_dir=getattr(self.args, "model_download_dir", None),
        )
        model_config = getattr(model_config, "text_config", model_config)
        self._hidden_size = int(model_config.hidden_size)
        if self.args.aux_hidden_states_layers is not None:
            self.aux_hidden_state_layer_ids = list(self.args.aux_hidden_states_layers)
        else:
            num_layers = int(model_config.num_hidden_layers)
            self.aux_hidden_state_layer_ids = [
                1,
                num_layers // 2 - 1,
                num_layers - 4,
            ]

        tp_size = self.num_gpus_per_engine
        configured_tp = getattr(self.args, "tokenspeed_tp_size", tp_size)
        if configured_tp != tp_size:
            raise ValueError(
                f"tokenspeed.tp_size ({configured_tp}) must equal "
                f"inference_num_gpus_per_engine ({tp_size})"
            )

        extra_args = getattr(self.args, "tokenspeed_extra_args", None)
        if isinstance(extra_args, DictConfig):
            extra = OmegaConf.to_container(extra_args, resolve=True)
        else:
            extra = dict(extra_args or {})
        # 模型路径、TP、捕获开关等由集成层统一管理，避免 extra_args 改变写入协议却仍返回旧元数据。
        blocked = extra.keys() & _PROTECTED_ENGINE_KEYS
        if blocked:
            logger.warning(
                "TokenSpeed extra_args contains managed keys that will be ignored: %s",
                sorted(blocked),
            )
            extra = {key: value for key, value in extra.items() if key not in blocked}

        max_seq_length = int(getattr(self.args, "max_seq_length", 8192))
        engine_kwargs = {
            "log_level": "warning",
            **extra,
            "model": self.args.target_model_path,
            "attn_tp_size": tp_size,
            "base_gpu_id": self.local_gpu_id,
            "nprocs_per_node": tp_size,
            "nnodes": 1,
            "gpu_memory_utilization": getattr(self.args, "tokenspeed_mem_fraction_static", 0.8),
            "enable_spec_training_mooncake": True,
            "eagle3_layers_to_capture": ",".join(
                str(layer_id) for layer_id in self.aux_hidden_state_layer_ids
            ),
            "max_model_len": max_seq_length,
            "max_prefill_tokens": max_seq_length,
            "trust_remote_code": getattr(self.args, "trust_remote_code", True),
            "download_dir": getattr(self.args, "model_download_dir", None),
        }

        from tokenspeed.runtime.entrypoints.engine import Engine

        # 真正的模型及执行资源在这里创建；之前的错误检查尽量放在昂贵加载之前。
        self._engine = Engine(**engine_kwargs)
        logger.info(
            "TokenSpeedEngine rank %d initialized: model=%s tp=%d aux_layers=%s",
            self.rank,
            self.args.target_model_path,
            tp_size,
            self.aux_hidden_state_layer_ids,
        )

    # 文本和 token 输入必须二选一；token 输入规范成一维，执行 max_new_tokens=0 的全序列捕获。
    # 每条结果必须恰有一个键；shape 使用实际 prompt_tokens，辅助层沿最后一维拼接，未添加 batch 维。
    def generate(
        self,
        data_id: str | list[str],
        input_ids_ref: ray.ObjectRef | list[torch.Tensor] | None = None,
        packed_loss_mask_list: list[str] | None = None,
        formatted_prompts: list[str] | None = None,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        multimodal_inputs: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        del data_id, packed_loss_mask_list, return_last_hidden_states, return_logits
        if self._engine is None:
            raise RuntimeError("TokenSpeedEngine is not initialized")
        if any(bool(item) for item in (multimodal_inputs or [])):
            raise NotImplementedError(
                "The initial TokenSpeed integration does not support multimodal inputs"
            )
        if (input_ids_ref is None) == (formatted_prompts is None):
            raise ValueError("Exactly one of input_ids_ref or formatted_prompts must be set")

        input_ids_list: list[list[int]] | None = None
        # 文本由后端分词，长度只能以后端结果为准；token 路径可保留输入长度作为后备。
        if formatted_prompts is not None:
            request_kwargs = {"prompt": formatted_prompts}
            expected_lengths = None
        else:
            resolved = (
                ray.get(input_ids_ref)
                if isinstance(input_ids_ref, ray.ObjectRef)
                else input_ids_ref
            )
            input_ids_list = []
            for ids in resolved:
                if ids.dim() == 2 and ids.shape[0] == 1:
                    ids = ids.squeeze(0)
                if ids.dim() != 1:
                    raise ValueError(f"Unexpected input_ids shape: {ids.shape}")
                input_ids_list.append(ids.tolist())
            request_kwargs = {"input_ids": input_ids_list}
            expected_lengths = [len(ids) for ids in input_ids_list]

        # max_new_tokens=0 表示不续写，只让已知序列经过模型并捕获各位置表示。
        results = self._engine.generate(
            **request_kwargs,
            sampling_params={"max_new_tokens": 0, "temperature": 0},
            return_hidden_states=True,
        )
        if isinstance(results, dict):
            results = [results]

        outputs = []
        for index, result in enumerate(results):
            keys = result.get("meta_info", {}).get("spec_training_mooncake_store_keys", [])
            # 当前协议是一条请求一个对象组，缺键或多键都拒绝，避免默默错配样本。
            if len(keys) != 1:
                raise RuntimeError(
                    f"TokenSpeed did not return exactly one Mooncake key for result {index}: {keys}"
                )
            seq_len = result["meta_info"].get("prompt_tokens")
            if seq_len is None and expected_lengths is not None:
                seq_len = expected_lengths[index]
            if seq_len is None:
                raise RuntimeError("TokenSpeed did not report prompt_tokens")
            # 元数据只保留每样本 S、层拼接宽度 K×H 和真实存储 dtype，网络返回体无需携带隐藏张量。
            outputs.append(
                {
                    "mooncake_key": keys[0],
                    "tensor_shapes": {
                        "hidden_states": (
                            seq_len,
                            len(self.aux_hidden_state_layer_ids) * self._hidden_size,
                        ),
                        "input_ids": (seq_len,),
                        "last_hidden_states": (seq_len, self._hidden_size),
                    },
                    "tensor_dtypes": {
                        "hidden_states": HIDDEN_STATES_STORAGE_DTYPE,
                        "input_ids": torch.long,
                        "last_hidden_states": HIDDEN_STATES_STORAGE_DTYPE,
                    },
                }
            )
        return outputs

    # 当前仅检查 Engine 实例存在，不发起推理请求。
    def health_check(self, timeout: float = 5.0) -> bool:
        del timeout
        return self._engine is not None

    # 调用 TokenSpeed 自身的关闭入口，再清空本地引用。
    def shutdown(self) -> None:
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
        logger.info("TokenSpeedEngine rank %d shutdown complete", self.rank)

    # 汇报 actor 位置与读取到的 hidden_size，帮助核对元数据维度。
    def get_status(self) -> dict:
        return {
            "rank": self.rank,
            "initialized": self._engine is not None,
            "base_gpu_id": self.base_gpu_id,
            "hidden_size": self._hidden_size,
        }
