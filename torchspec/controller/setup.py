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

# 中文导读：把已经创建的推理引擎、训练组和控制器连起来。
# 本模块只确定连接和步数，不执行网络 forward。队列传递样本描述，Mooncake 承载张量。

"""Pipeline setup: mooncake config, training steps calculation, async training setup."""

import math

import ray

from torchspec.utils.env import get_torchspec_env_vars
from torchspec.utils.logging import logger


# 将扁平 mooncake_* 参数交给统一配置类解析，使各 actor 连接同一存储服务。
def build_mooncake_config(args):
    """Build MooncakeConfig from flat args namespace."""
    from torchspec.config.mooncake_config import MooncakeConfig

    return MooncakeConfig.from_flat_args(args)


# 复用或新建控制器，创建推理调度 actor，再将训练/评估队列传给训练组。
# 训练队列按派生 batch 大小读取；评估队列一次读取单条，随后由 eval_micro_batch_size 重新组批。
# mooncake_config 是连接参数而不是跨进程共享的 store 对象，每个 actor 自行初始化客户端。
def setup_async_training_with_engines(
    args, train_group, mooncake_config, inference_engines, controller=None
):
    """Setup async training with distributed inference engines (e.g., Eagle3).

    The engines are Ray actors responsible for storing tensors in mooncake and returning keys.
    AsyncInferenceManager forwards the keys to the controller.

    Args:
        args: Configuration arguments.
        train_group: Training group.
        mooncake_config: MooncakeConfig object. Each actor initializes its own store.
        inference_engines: List of Ray actor engine handles for distributed generation.
        controller: Optional pre-created AsyncTrainingController. If None, a new one is created.
    """
    from torchspec.controller.inference_manager import AsyncInferenceManager
    from torchspec.controller.training_controller import AsyncTrainingController

    dp_size = (
        getattr(args, "dp_size", None) or args.training_num_nodes * args.training_num_gpus_per_node
    )

    if controller is None:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        driver_node_id = ray.get_runtime_context().get_node_id()
        controller = AsyncTrainingController.options(
            runtime_env={"env_vars": get_torchspec_env_vars()},
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=driver_node_id, soft=False),
        ).remote(args, dp_size)

    max_concurrent = getattr(args, "max_concurrent_batches", 1)
    inference_manager = AsyncInferenceManager.remote(
        args,
        controller,
        inference_engines=inference_engines,
        max_concurrent_batches=max_concurrent,
    )

    train_queues = ray.get(controller.get_train_queues.remote())
    train_group.set_train_queues(
        train_queues, mooncake_config, per_dp_rank_batch_size=args.per_dp_rank_batch_size
    )

    eval_queues = ray.get(controller.get_eval_queues.remote())
    # eval_from_cache re-collates individual samples with eval_micro_batch_size,
    # so the fetcher must yield unbatched (batch_size=1) entries.
    train_group.set_eval_queues(eval_queues, mooncake_config, per_dp_rank_batch_size=1)

    return controller, inference_manager


# 把样本数换算为 optimizer 步数，而不是推理批次数或梯度累积次数。
# 例如 1000 条、global_batch=32，整数除法得到每 epoch 31 步；未凑满的尾部不单独加一步。
# 显式 num_train_steps 优先，显式 lr_total_steps 也保留；小于一个 global batch 时至少设 1 步。
def auto_calculate_training_steps(args, dataset_size: int):
    """Auto-calculate num_train_steps and lr_total_steps based on dataset size if not explicitly set.

    All step counts are in optimizer steps (not dispatches).
    steps_per_epoch = dataset_size // global_batch_size
    where global_batch_size = per_dp_rank_batch_size * dp_size * draft_accumulation_steps.

    If num_train_steps is set by user, num_epochs is calculated from it.
    Otherwise: lr_total_steps = steps_per_epoch * num_epochs
    """

    global_batch_size = args.global_batch_size
    steps_per_epoch = dataset_size // global_batch_size

    # 至少 1 步只是调度数值兜底，不会复制样本补出完整 global batch。
    # 太小的数据集能否凑齐窗口仍取决于控制器投递/重载规则，不能凭这个值认定输入已经足够。
    if steps_per_epoch == 0:
        logger.warning(
            f"Dataset size ({dataset_size}) < global_batch_size ({global_batch_size}). "
            f"Setting steps_per_epoch to 1."
        )
        steps_per_epoch = 1

    args.steps_per_epoch = steps_per_epoch

    current_num_train_steps = getattr(args, "num_train_steps", None)
    current_lr_total_steps = getattr(args, "lr_total_steps", None)

    # 例数据 1000 条、global_batch=32，steps_per_epoch=31；显式总步数 40 得 num_epochs=2，
    # 但最后一个 epoch 只跑 9 步。显式 lr_total_steps=100 仍保留，训练结束不代表调度器走到末尾。
    if current_num_train_steps is not None:
        args.num_epochs = math.ceil(current_num_train_steps / steps_per_epoch)
        logger.info(
            f"Setting num_epochs to {args.num_epochs} based on num_train_steps={current_num_train_steps}!"
        )
        if current_lr_total_steps is None:
            args.lr_total_steps = current_num_train_steps
    else:
        num_epochs = getattr(args, "num_epochs", 1)
        calculated_total_steps = num_epochs * steps_per_epoch
        args.num_train_steps = calculated_total_steps
        if current_lr_total_steps is None:
            args.lr_total_steps = calculated_total_steps

    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    logger.info(
        f"Training steps (optimizer steps): num_train_steps={args.num_train_steps}, "
        f"lr_total_steps={args.lr_total_steps} "
        f"(dataset_size={dataset_size}, global_batch_size={global_batch_size}, "
        f"per_dp_rank_batch_size={args.per_dp_rank_batch_size}, "
        f"accumulation_steps={accumulation_steps}, "
        f"steps_per_epoch={steps_per_epoch}, num_epochs={args.num_epochs})"
    )
