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

# 中文导读：Ray actor 共用的节点、GPU 和分布式 rendezvous 地址工具。
# Ray 负责把进程放到节点；PyTorch distributed 还需要主节点地址/端口建立通信，两层不能混淆。

import os
import random

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from torchspec.utils.logging import logger
from torchspec.utils.misc import _to_local_gpu_id, get_current_node_ip, get_free_port


# 在存活 Ray 节点中查 IP 并生成硬亲和约束（soft=False）；找不到时立即报错而非换机器。
def node_affinity_for_ip(ip: str, name: str = None) -> NodeAffinitySchedulingStrategy:
    """Return a NodeAffinitySchedulingStrategy pinned to the live Ray node with the given IP.

    Args:
        ip: Node IP address to pin to.
        name: Optional actor name for log messages.

    Returns:
        NodeAffinitySchedulingStrategy with soft=False.

    Raises:
        RuntimeError: If no live Ray node with that IP is found.
    """
    for node in ray.nodes():
        if node.get("Alive", False) and node.get("NodeManagerAddress") == ip:
            node_id = node["NodeID"]
            label = f"{name} " if name else ""
            logger.info(f"Pinning {label}actor to node {ip} (id={node_id[:8]}...)")
            return NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)

    live_ips = [n["NodeManagerAddress"] for n in ray.nodes() if n.get("Alive", False)]
    raise RuntimeError(f"No live Ray node with IP {ip!r} found. Live nodes: {live_ips}")


# 提供基础进程工具；实际模型和 process group 初始化由具体训练/推理 actor 完成。
class RayActor:
    """Base class for all torchspec Ray actors."""

    # 取得当前 actor 所在节点的可用 IP，不是提交作业的 driver 地址。
    @staticmethod
    def get_node_ip() -> str:
        """Get current node IP address."""
        return get_current_node_ip()

    # 在当前节点寻找空闲端口或连续端口段；寻找与之后绑定并非一个原子操作。
    @staticmethod
    def find_free_port(start_port=10000, consecutive=1) -> int:
        """Find available port(s) on current node."""
        return get_free_port(start_port=start_port, consecutive=consecutive)

    # 将物理卡号转换到 CUDA_VISIBLE_DEVICES 定义的进程可见编号。
    # 例如进程只看到物理卡 5，torch.cuda 使用的索引可能是 0。
    @staticmethod
    def resolve_local_gpu_id(physical_gpu_id: int) -> int:
        """Convert physical GPU ID to node-local GPU ID."""
        return _to_local_gpu_id(physical_gpu_id)

    # 从显式卡号或 Ray 分配结果选卡，设置当前 CUDA device 和 LOCAL_RANK，返回本地编号。
    def setup_gpu(self, base_gpu_id: int | None = None) -> int:
        """Resolve GPU, set CUDA device, set LOCAL_RANK env var.

        Args:
            base_gpu_id: Physical GPU ID. If None, auto-detect from ray.get_gpu_ids().

        Returns:
            Local GPU ID.
        """
        if base_gpu_id is None:
            gpu_ids = ray.get_gpu_ids()
            base_gpu_id = int(float(gpu_ids[0])) if gpu_ids else 0
        local_gpu_id = self.resolve_local_gpu_id(base_gpu_id)
        torch.cuda.set_device(local_gpu_id)
        os.environ["LOCAL_RANK"] = str(local_gpu_id)
        return local_gpu_id

    # 跟随已有主地址或为首 rank 选择本节点空闲端口，保存 rendezvous 信息供其余 rank 使用。
    def setup_master(self, master_addr=None, master_port=None, port_range=(10000, 11000)):
        """Resolve master address/port for distributed communication.

        If master_addr is provided, use it directly. Otherwise auto-resolve.
        Stores result in self.master_addr, self.master_port.
        """
        if master_addr:
            self.master_addr = master_addr
            self.master_port = master_port
        else:
            self.master_addr = self.get_node_ip()
            self.master_port = self.find_free_port(start_port=random.randint(*port_range))

    # 返回初始化过的通信主地址/端口，让创建组的进程传给后续 actor。
    def get_master_addr_and_port(self):
        """Return (master_addr, master_port) tuple."""
        return self.master_addr, self.master_port
