# 中文导读：用“从队列取数据并写盘”替代训练消费者，证明生产与消费是通过协议解耦的。
# 这是 CPU actor；它从 Mooncake 拉取张量到 CPU，写完后移除存储中的临时对象。

"""Queue consumer that saves inference outputs instead of training on them."""

from __future__ import annotations

import dataclasses

import ray
import torch

from torchspec.offline.dataset import OfflineDataset
from torchspec.ray.ray_actor import RayActor
from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore


# 常驻磁盘消费者，拥有一个数据清单写者和独立 Mooncake 客户端。
@ray.remote(num_cpus=1, num_gpus=0)
class OfflineSavingActor:
    """Consume the normal training queues and persist their tensors."""

    # 打开已有输出目录，建立不贡献存储段、不启用 GPU direct 的 CPU 读取客户端。
    def __init__(self, output_dir, mooncake_config) -> None:
        self.dataset = OfflineDataset(output_dir)
        config = dataclasses.replace(
            mooncake_config,
            local_hostname=RayActor.get_node_ip(),
            global_segment_size=0,
            async_put_pool_size=0,
            enable_gpu_direct=False,
        )
        self.store = EagleMooncakeStore(config)
        self.store.setup(device=torch.device("cpu"))
        self.queues = {}

    # 保存训练/评估两个队列句柄，后续按 split 选择正确数据流。
    def set_queues(self, train_queue, eval_queue) -> None:
        self.queues = {"train": train_queue, "eval": eval_queue}

    # 精确取 count 条并保存，返回新增文件数（重复 ID 不重复记数）。
    # PP 分层 manifest 当前不支持，提前拒绝以免误读不存在的基础 key 并漏删碎片。
    # 成功 get 后无论写盘是否成功都在 finally 释放 Mooncake 临时张量。
    def save_from_queue(self, split: str, count: int = 1) -> int:
        """Save exactly ``count`` items and return the number newly written."""
        written = 0
        for _ in range(count):
            sample = self.queues[split].get()
            if not sample.data_id:
                raise ValueError("Offline saving requires TrainSample.data_id")
            if (sample.metadata or {}).get("vllm_pp_layer_manifest") is not None:
                # Under vLLM pipeline parallelism each stage writes its own
                # per-layer fragments and the base key is never written, so a
                # plain get() here would miss the data and then "clean up" a
                # key that does not exist while every fragment leaks.
                raise NotImplementedError(
                    "Offline saving does not support vLLM pipeline parallelism yet; "
                    f"sample {sample.data_id} carries a per-layer PP manifest. "
                    "Generate offline data with vllm_pp_size=1."
                )
            dtypes = {
                key: getattr(torch, value.replace("torch.", ""))
                if isinstance(value, str)
                else value
                for key, value in (sample.tensor_dtypes or {}).items()
            }
            output = self.store.get(
                key=sample.mooncake_key,
                shapes=sample.tensor_shapes,
                dtypes=dtypes,
                device=torch.device("cpu"),
            )
            try:
                written += self.dataset.append(
                    split,
                    data_id=sample.data_id,
                    tensors=output.to_tensor_dict(),
                    packed_loss_mask=sample.packed_loss_mask,
                    metadata=sample.metadata,
                )
            finally:
                self.store.remove_eagle3_tensors(
                    sample.mooncake_key,
                    has_last_hidden_states="last_hidden_states" in sample.tensor_shapes,
                    has_target="target" in sample.tensor_shapes,
                )
        return written

    # 报告磁盘索引中各 split 的总数，而非本次调用刚写入的数量。
    def counts(self) -> dict[str, int]:
        return {split: self.dataset.count(split) for split in ("train", "eval")}

    # 关闭存储客户端，归还注册缓冲和相关资源。
    def close(self) -> None:
        self.store.close()
