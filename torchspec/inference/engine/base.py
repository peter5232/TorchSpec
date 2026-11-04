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

# 定义训练数据生产者的共同生命周期；调用方通过 Ray 调度，具体后端负责前向和张量发布。
# generate 在这里表示生产 teacher 训练数据，不保证每个实现都生成新 token。

"""Abstract base class for inference engines."""

from abc import ABC, abstractmethod
from typing import Any

import ray
import torch


# 抽象接口统一初始化、数据生产、状态查询和回收；GPU 放置能力由具体类混入 RayActor 提供。
class InferenceEngine(ABC):
    """Abstract interface for inference engines.

    Concrete implementations (HFEngine, SglEngine) must also inherit from
    RayActor for distributed deployment via Ray placement groups.
    """

    # 模型和传输资源应在 actor 被调度到目标节点后创建，避免在驱动进程提前占用 GPU。
    @abstractmethod
    def init(self, mooncake_config=None) -> None:
        """Initialize the engine on the allocated GPU.

        Called after the Ray actor is scheduled on a node.

        Args:
            mooncake_config: MooncakeConfig object for distributed storage.
        """

    # 输入可以是 token 张量引用或已格式化文本；结果通常是每条样本的 Mooncake 键及解释字节所需的元数据。
    # 这是后端共享的签名，各实现支持的参数不同，应继续阅读具体 generate 的校验和返回字段。
    @abstractmethod
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
        """Generate training data from inputs.

        Accepts either pre-tokenized input_ids or formatted prompt strings.

        Args:
            data_id: Data ID(s) for the batch.
            input_ids_ref: Ray ObjectRef or list of input_ids tensors.
            packed_loss_mask_list: List of packed loss_mask strings.
            formatted_prompts: List of chat-template-formatted prompt strings.
            return_last_hidden_states: Whether to return last hidden states.
            return_logits: Whether to return target logits.
            multimodal_inputs: List of multimodal input dicts.

        Returns:
            List of dicts with mooncake_key or tensors.
        """

    # 向上层暴露当前引擎的存储配置；未设置时返回 None，不创建新的客户端连接。
    def get_mooncake_config(self):
        """Get the mooncake configuration.

        Subclasses must set ``self._mooncake_config`` in their ``__init__``.
        """
        return getattr(self, "_mooncake_config", None)

    # 约定轻量健康查询接口；具体实现多只检查初始化状态，不能等同一次真实模型前向测试。
    @abstractmethod
    def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the engine is healthy."""

    # 约定释放模型、后台进程和存储连接的入口，由具体后端决定清理顺序。
    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""

    # 返回可经 Ray 传输的小字典供监控使用，不返回模型权重或隐状态张量。
    @abstractmethod
    def get_status(self) -> dict:
        """Get engine status."""
