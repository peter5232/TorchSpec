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

# Hugging Face 后端的远程外壳：处理 actor 的 GPU 定位、日志和 ObjectRef，模型计算交给 HFRunner。
# 这条路径最接近常规 PyTorch 前向，适合先理解 teacher 数据如何变成训练样本。

"""
HF Ray actor engine for distributed deployment across multiple nodes.

This wraps HFRunner as a Ray actor with placement group support.

Accepts pre-tokenized input_ids and loss_mask from batch preprocessing.
"""

from typing import Any

import ray
import torch

from torchspec.inference.engine.base import InferenceEngine
from torchspec.ray.ray_actor import RayActor
from torchspec.utils.logging import logger, setup_file_logging


# 把普通 HFRunner 包装成 Ray 可部署对象；实例创建和真正加载权重分开。
class HFEngine(InferenceEngine, RayActor):
    """Ray actor wrapper for HFRunner with distributed deployment support.

    This allows HF engines to be deployed across multiple nodes using Ray
    placement groups.

    Accepts pre-tokenized input_ids and loss_mask instead of raw prompts.
    """

    # 保存 rank、GPU 基址及尚未初始化的 runner；日志按引擎组区分，构造阶段不加载权重。
    def __init__(
        self,
        args,
        rank: int,
        base_gpu_id: int | None = None,
        engine_group: int = 0,
    ):
        self.args = args
        self.rank = rank
        self.base_gpu_id = base_gpu_id
        self._engine = None
        self._mooncake_config = None
        setup_file_logging("inference", self.rank, group=engine_group)

    # 先将物理 GPU 映射到 actor 可见设备，再检查 Mooncake 可达性并加载目标模型及捕获层配置。
    def init(self, mooncake_config=None) -> None:
        """Initialize the HFRunner on the allocated GPU.

        This is called after the Ray actor is scheduled on a node.

        Args:
            mooncake_config: MooncakeConfig object for distributed storage.
        """
        from torchspec.inference.engine.hf_runner import HFRunner

        if self.base_gpu_id is not None:
            local_gpu_id = self.setup_gpu(self.base_gpu_id)
            logger.info(
                f"HFEngine rank {self.rank}: base_gpu_id={self.base_gpu_id}, "
                f"using local GPU {local_gpu_id}"
            )

        self._mooncake_config = mooncake_config

        if mooncake_config is not None:
            from torchspec.transfer.mooncake.utils import (
                check_mooncake_master_available,
            )

            check_mooncake_master_available(
                mooncake_config.master_server_address, mooncake_config.metadata_server
            )

        self._engine = HFRunner.from_pretrained(
            pretrained_model_name_or_path=self.args.target_model_path,
            max_seq_length=getattr(self.args, "max_seq_length", 8192),
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
            aux_hidden_states_layers=getattr(self.args, "aux_hidden_states_layers", None),
            mooncake_config=mooncake_config,
        )

        logger.info(f"HFEngine rank {self.rank}: initialized from {self.args.target_model_path}")

    # 解引用输入张量列表，合并批次标识并交给 runner；packed loss mask 原样随样本传递。
    # 此实现没有使用文本、多模态和返回选项；实际输出由 runner 的完整 teacher 前向决定。
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
        """Generate training data from pre-tokenized inputs.

        Args:
            data_id: Data ID(s) for the batch.
            input_ids_ref: Ray ObjectRef or list of input_ids tensors.
            packed_loss_mask_list: List of packed loss_mask strings (e.g. "2,3,2,2,1").
            formatted_prompts: Unused (accepted for interface compatibility).
            return_last_hidden_states: Unused (accepted for interface compatibility).
            return_logits: Unused (accepted for interface compatibility).
            multimodal_inputs: Unused (accepted for interface compatibility).

        Returns:
            List of dicts with mooncake_key or tensors.
        """
        if self._engine is None:
            raise RuntimeError("HFEngine not initialized. Call init() first.")

        assert input_ids_ref is not None, "input_ids_ref must not be None"
        assert packed_loss_mask_list is not None, "packed_loss_mask_list must not be None"

        if isinstance(input_ids_ref, ray.ObjectRef):
            input_ids_list = ray.get(input_ids_ref)
        else:
            input_ids_list = input_ids_ref

        if isinstance(data_id, list):
            batch_data_id = ",".join(data_id)
        else:
            batch_data_id = data_id

        logger.info(
            f"HFEngine rank {self.rank}: processing data_id={batch_data_id}, shapes: {[ids.shape for ids in input_ids_list]}"
        )

        result = self._engine.generate(
            data_id=batch_data_id,
            input_ids_list=input_ids_list,
            packed_loss_mask_list=packed_loss_mask_list,
        )
        return result

    # 只判断 runner 是否已创建；timeout 在此实现中不触发额外探测。
    def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the engine is healthy."""
        return self._engine is not None

    # 先让 runner 回收存储和模型，再清空引用，使后续生成明确报未初始化。
    def shutdown(self) -> None:
        """Clean up resources."""
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
        logger.info(f"HFEngine rank {self.rank}: shutdown complete")

    # 汇报 actor 的 rank、初始化标志和 GPU 基址，供编排层查看部署位置。
    def get_status(self) -> dict:
        """Get engine status."""
        return {
            "rank": self.rank,
            "initialized": self._engine is not None,
            "base_gpu_id": self.base_gpu_id,
        }
