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

# 多进程 W&B 接入：主进程建 run，其他 actor 通过 run id 关联；指标显式绑定各自 step 轴。

import logging
import os
from copy import deepcopy

import wandb

logger = logging.getLogger(__name__)


# args.wandb_mode 优先于环境 WANDB_MODE，只有值为 offline 才返回 True。
def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


# 未启用时清 run id；启用时配置 mode/login、group/name 和脱敏 config，创建 run 后回写 id。
# 在线使用 shared primary，离线本地写入；目录按需创建，公共指标轴随后统一注册。
def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    # 离线模式显式跳过 login，日志保存在本地；online/shared 则可能连接 W&B 服务。
    # 这个函数执行的是 tracking 副作用，调用它不是单纯构造一个本地配置对象。
    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 6 length string with characters
    # group 组织同一实验的运行，run_name 决定界面显示名，run id 用于后续 actor 关联。
    # 随机后缀可避免多次实验混名，但不同概念不能仅靠同名来推断它们属于同一 run。
    base_group = args.wandb_group or args.wandb_project or "default"
    if args.wandb_random_suffix:
        group = base_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = base_group
        run_name = base_group

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    # 把实际创建的 id 写回 args，编排方随后把它传给 secondary。
    # 先初始化 secondary 再创建 primary 会因为无 id 而直接返回，不会自动延迟重试。
    args.wandb_run_id = wandb.run.id


# 深拷贝参数并清除 wandb_key，只收集白名单环境变量，返回适合记录的配置而不修改原 args。
def _compute_config_for_logging(args):
    output = deepcopy(args.__dict__)
    # The API key may be supplied through YAML for explicit, reproducible
    # login, but it must never be copied into W&B run configuration.
    output["wandb_key"] = None

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    return output


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
# 无主 run id 就不创建；否则使用同 id 连接 shared secondary，禁止 secondary 更新结束状态。
# 可把 SGLang 的 engine_metrics endpoint 配入统计采集；离线则使用独立 offline 设置。
def init_wandb_secondary(args, router_addr=None):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    # 此分支仅配置 OpenMetrics endpoint 的采集，不把推理指标伪装成训练 loss。
    # router_addr 必须可由该进程访问；它与训练 rank 的分布式 rendezvous 地址不是同一个概念。
    if args.sglang_enable_metrics and router_addr is not None:
        logger.info(f"Forward SGLang metrics at {router_addr} to WandB.")
        settings_kwargs |= dict(
            x_stats_open_metrics_endpoints={
                "sgl_engine": f"{router_addr}/engine_metrics",
            },
            x_stats_open_metrics_filters={
                "sgl_engine.*": {},
            },
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": _compute_config_for_logging(args),
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    # secondary 复用主 run id，并以 x_update_finish_state=False 避免某个工作进程退出就结束整个 run。
    # 这与 primary 负责实验生命周期的分工一致；离线模式下则使用离线 settings。
    wandb.init(**init_kwargs)

    _init_wandb_common()


# 训练、性能、控制器指标绑 train/step，推理/多轮指标绑 inference/step，评估绑 eval/step。
# 异步系统各部分进度不一致，因此不能把所有曲线都绑定 W&B 自动递增的单一横轴。
def _init_wandb_common():
    # Training metrics
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")

    # Performance metrics (tied to training step)
    wandb.define_metric("perf/*", step_metric="train/step")

    # Data pipeline metrics
    wandb.define_metric("data/*", step_metric="train/step")

    # Model health metrics
    wandb.define_metric("model/*", step_metric="train/step")

    # Learning rate schedule metrics
    wandb.define_metric("lr/*", step_metric="train/step")

    # Inference metrics
    wandb.define_metric("inference/step")
    wandb.define_metric("inference/*", step_metric="inference/step")

    # Controller metrics (async pipeline)
    wandb.define_metric("controller/*", step_metric="train/step")

    # Multi-turn and passrate metrics
    wandb.define_metric("multi_turn/*", step_metric="inference/step")
    wandb.define_metric("passrate/*", step_metric="inference/step")

    # Evaluation metrics
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
