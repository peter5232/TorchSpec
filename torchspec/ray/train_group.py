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

# 中文导读：训练 actor 的主进程端代理，批量发起远程调用并管理 rank→actor 的映射。
# 它不是 nn.Module；真正的 forward/backward 位于每个 actor 的 trainer 中。
# 涉及 FSDP 聚合的保存与训练必须让所有 rank 参与，即便最终只有一个 rank 写完整权重。

import json
import os
from typing import Optional, Type

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from torchspec.utils.distributed import _build_usp_group_ranks
from torchspec.utils.env import get_torchspec_env_vars


# 将一个分布式训练组封装成接口集合，隐藏逐个 actor 发 RPC 的重复代码。
class RayTrainGroup:
    """
    A group of ray actors for training
    Functions start with 'async' should return list of object refs

    Args:
        args (Namespace): Arguments for the training group.
        num_nodes (int): Number of nodes for this training group.
        num_gpus_per_node (int): Number of gpus for this training group.
        pg (PlacementGroup, optional): Placement group to schedule training workers on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each training worker.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
        resources (Dict[str, float], optional): Custom resources to allocate for each training worker.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Number of custom resources to allocate for each node.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        training_class (type, optional): The training class to use. Defaults to None.
    """

    # 记录拓扑、角色和 actor 类，再依据 placement group 为每个 rank 创建常驻 worker。
    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int], list[int]],
        num_gpus_per_actor: float = 1,
        role: str = "training",
        training_class: Optional[Type] = None,
    ) -> None:
        self.args = args
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role
        self._training_class = training_class

        # Allocate the GPUs for training workers
        self._allocate_gpus_for_training(pg, num_gpus_per_actor)

    # 将环境变量传入 worker，按稳定 bundle 顺序创建 rank；rank 0 先选通信地址，再传给其余 rank。
    # num_gpus_per_actor 是 Ray 逻辑资源比例，不自动限制模型的显存分配或 CUDA 算力。
    def _allocate_gpus_for_training(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices, _reordered_gpu_ids = pg

        train_env_vars = self.args.train_env_vars
        if isinstance(train_env_vars, str):
            train_env_vars = json.loads(train_env_vars) if train_env_vars else {}

        env_vars = {
            **get_torchspec_env_vars(),
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get(
                "NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1"
            ),
            **train_env_vars,
        }
        if "TORCHINDUCTOR_CACHE_DIR" in os.environ:
            env_vars["TORCHINDUCTOR_CACHE_DIR"] = os.environ["TORCHINDUCTOR_CACHE_DIR"]
        env_vars.setdefault(
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
            os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800"),
        )
        env_vars.setdefault(
            "TORCHINDUCTOR_FX_GRAPH_CACHE",
            os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE", "1"),
        )

        TrainRayActor = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(
            self._training_class
        )

        # Create worker actors
        self._actor_handlers = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self._actor_handlers.append(actor)

    # 向所有 worker 同时发模型/优化器初始化请求，返回 refs 以便 driver 与其他初始化重叠。
    def async_init(self, args, role, mooncake_config=None, with_ref=False):
        """
        Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
        """
        self.args = args
        return [
            actor.init.remote(args, role, mooncake_config=mooncake_config, with_ref=with_ref)
            for actor in self._actor_handlers
        ]

    # 向全部 rank 发同一个 step 和累积批次数，再等待所有结果；一次调用对应一次 optimizer step。
    def train_from_queue(self, step: int, num_batches: int):
        """Do one optimizer step by fetching data from queues"""
        return ray.get(
            [actor.train_from_queue.remote(step, num_batches) for actor in self._actor_handlers]
        )

    # 让各 worker 显式取回最后一批延迟指标，避免训练结束时丢掉尾步日志。
    def flush_pending_train_metrics(self):
        """Materialize the final one-step-delayed metrics on every trainer."""
        return ray.get(
            [actor.flush_pending_train_metrics.remote() for actor in self._actor_handlers]
        )

    # 所有 rank 一起进入 checkpoint 保存；force_sync 作为兼容参数原样传递。
    # 当前 Trainer.save_model 并未使用它切换保存方式，不能仅凭这个参数推断发生了额外等待。
    def save_model(self, step, force_sync=False):
        """Save training model"""
        return ray.get(
            [actor.save_model.remote(step, force_sync=force_sync) for actor in self._actor_handlers]
        )

    # 所有 rank 参与 FSDP 参数聚合并导出供推理加载的权重格式；不能只调用 rank 0。
    def save_draft_model_for_serving(self, output_dir: str):
        """Save draft model in HuggingFace format for serving weight update.

        This is a collective FSDP2 operation - all actors must participate.
        """
        return ray.get(
            [
                actor.save_draft_model_for_serving.remote(output_dir)
                for actor in self._actor_handlers
            ]
        )

    # 把同一 d2t/t2d 发给所有草稿副本，保证各 rank 的输出列表示同一个 token。
    def set_vocab_buffers(self, d2t, t2d):
        """Set vocab mapping buffers on draft models on all actors."""
        return ray.get([actor.set_vocab_buffers.remote(d2t, t2d) for actor in self._actor_handlers])

    # 按 rank 配对训练队列并把存储连接与取样数量传下去；strict zip 防止漏绑 worker。
    def set_train_queues(self, queues, mooncake_config, per_dp_rank_batch_size: int = 1):
        """Set training data queues for queue-based training.

        Args:
            queues: List of Ray Queues, one per DP rank.
            mooncake_config: MooncakeConfig object. Each actor initializes its own store.
            per_dp_rank_batch_size: Number of samples per DP rank per training step.
        """
        queues = self._expand_queues_for_usp(queues)
        return ray.get(
            [
                actor.set_train_queue.remote(
                    queue,
                    mooncake_config=mooncake_config,
                    per_dp_rank_batch_size=per_dp_rank_batch_size,
                )
                for actor, queue in zip(self._actor_handlers, queues, strict=True)
            ]
        )

    # 与训练队列绑定相同的过程，但使用独立评估消费者，避免不同用途的样本混用。
    def set_eval_queues(self, queues, mooncake_config, per_dp_rank_batch_size: int = 1):
        """Set eval data queues — mirrors set_train_queues."""
        queues = self._expand_queues_for_usp(queues)
        return ray.get(
            [
                actor.set_eval_queue.remote(
                    queue,
                    mooncake_config=mooncake_config,
                    per_dp_rank_batch_size=per_dp_rank_batch_size,
                )
                for actor, queue in zip(self._actor_handlers, queues, strict=True)
            ]
        )

    # 已有每 actor 一个队列时直接返回；若旧接口只给 DP 队列，则按 SP 分组映射到各 rank。
    # 这里只复制句柄，不广播队列元素；当前控制器通常提供完整 DP×SP 队列并逐队列投递。
    def _expand_queues_for_usp(self, queues):
        actor_count = len(self._actor_handlers)
        if len(queues) == actor_count:
            return queues

        attention_backend = getattr(self.args, "attention_backend", None)
        if attention_backend != "usp":
            raise ValueError(
                f"Number of queues ({len(queues)}) must match number of actors ({actor_count})"
            )

        sp_size = getattr(self.args, "sp_ulysses_size", 1) * getattr(self.args, "sp_ring_size", 1)
        if sp_size <= 0 or actor_count % sp_size != 0:
            raise ValueError(f"Invalid USP topology: actor_count={actor_count}, sp_size={sp_size}")
        dp_size = actor_count // sp_size
        if len(queues) != dp_size:
            raise ValueError(
                f"USP expects {dp_size} queues for {actor_count} actors with sp_size={sp_size}, "
                f"got {len(queues)}"
            )
        draft_sp_groups, _, _ = _build_usp_group_ranks(
            world_size=actor_count,
            sp_ulysses_size=getattr(self.args, "sp_ulysses_size", 1),
            sp_ring_size=getattr(self.args, "sp_ring_size", 1),
        )
        rank_to_dp_idx = {
            rank: dp_idx for dp_idx, ranks in enumerate(draft_sp_groups) for rank in ranks
        }
        return [queues[rank_to_dp_idx[rank]] for rank in range(actor_count)]

    # 让每个 worker 从独立验证队列消费 count 条并存到 CPU 缓存，阻塞直到全组完成。
    def cache_eval_samples(self, count: int):
        """Tell every actor to drain ``count`` individual samples from its eval queue into CPU cache."""
        return ray.get([actor.cache_eval_samples.remote(count) for actor in self._actor_handlers])

    # 等待全组提交评估缓存保存的 RPC；Trainer 内部另起后台写盘线程，RPC 返回不代表落盘完成。
    # 后续 load/close 等路径会等待内部保存 Future；这里与下方 async 版本的区别只在 RPC 是否等待。
    def save_eval_cache(self, cache_dir: str):
        """Persist CPU-cached eval batches to disk on every actor (blocking)."""
        return ray.get([actor.save_eval_cache.remote(cache_dir) for actor in self._actor_handlers])

    # 只提交保存并返回 refs，调用返回不代表磁盘文件已经写完。
    def async_save_eval_cache(self, cache_dir: str) -> list[ray.ObjectRef]:
        """Fire-and-forget disk save — returns refs the caller can optionally await."""
        return [actor.save_eval_cache.remote(cache_dir) for actor in self._actor_handlers]

    # 各 rank 读取自己的缓存并返回条数列表；上层必须逐 rank 判断是否命中，不能只看第一个。
    def load_eval_cache(self, cache_dir: str):
        """Load eval batches from disk on every actor. Returns list of counts (0 = miss)."""
        return ray.get([actor.load_eval_cache.remote(cache_dir) for actor in self._actor_handlers])

    # 全组从缓存执行只前向的评估，等待汇总结果返回。
    def run_eval(self):
        """Run forward-only eval on every actor using the CPU-cached batches."""
        return ray.get([actor.eval_from_cache.remote() for actor in self._actor_handlers])
