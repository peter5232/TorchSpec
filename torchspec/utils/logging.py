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

# 日志和 tracking 入口：控制台按 actor 区分，文件按 role/node/rank 分开；W&B/TensorBoard 管理指标输出。

import datetime
import logging
import os

import torch.distributed as dist

from torchspec.utils import wandb as wandb_utils

_LOG_FORMAT = "[%(asctime)s] %(filename)s:%(lineno)d %(levelname)s %(message)s"
_tb_writer = None


# 读取 TORCHSPEC_LOG_LEVEL 并查 logging 常量；这是环境配置解析，不会改变模型计算。
def _get_logger_level():
    level_str = os.getenv("TORCHSPEC_LOG_LEVEL", "INFO").upper()
    try:
        log_level = getattr(logging, level_str)
    except ValueError:
        logging.warning("Invalid log level: %s, defaulting to WARNING", level_str)
    return log_level


# 按 actor 选 logger，移除旧 handler 并禁用父 logger 传播，安装统一格式的 StreamHandler 后返回。
def setup_logger(log_level=None, actor_name=None, ip_addr=None):
    logger_name = "TorchSpec" if actor_name is None else f"TorchSpec-{actor_name}"
    _logger = logging.getLogger(logger_name)
    _logger.handlers.clear()
    _logger.propagate = False
    if log_level is None:
        log_level = _get_logger_level()
    _logger.setLevel(log_level)
    handler = logging.StreamHandler()
    if ip_addr is None:
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    else:
        rank = os.environ.get("RANK", 0)
        handler.setFormatter(
            logging.Formatter(
                f"[%(asctime)s{ip_addr} RANK:{rank}] %(filename)s:%(lineno)d %(levelname)s %(message)s"
            )
        )
    handler.setLevel(log_level)
    _logger.addHandler(handler)
    return _logger


logger = setup_logger()


# 未配置目录时不操作；否则替换旧 FileHandler，按 role/IP/rank/时间戳组织文件路径。
# 目录或 NFS 不可写仅记录 warning，不让日志失败终止训练。
def setup_file_logging(
    role: str,
    rank: int | str,
    group: int = 0,
    log_dir: str | None = None,
) -> None:
    """Add a FileHandler to the module-level logger for per-role/per-node/per-rank file logging."""
    if log_dir is None:
        log_dir = os.environ.get("TORCHSPEC_LOG_DIR")
    if log_dir is None:
        return

    # 遍历 handler 列表副本，移除旧文件输出时不会跳过后续元素。
    # 先 close 再 remove 释放文件描述符；console handler 保留，文件日志只是附加观测通道。
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)

    try:
        from torchspec.utils.misc import get_current_node_ip

        node_ip = get_current_node_ip()
    except Exception:
        node_ip = "unknown"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 文件路径同时带 role/node/group/rank/time，可区分多机上同 rank 编号的不同角色。
    # 这对异步作业很重要：teacher、trainer、controller 不共享同一推进节奏。
    dir_path = os.path.join(log_dir, role, node_ip)
    filename = f"{role}_g{group}_rank{rank}_{timestamp}.log"
    filepath = os.path.join(dir_path, filename)

    try:
        os.makedirs(dir_path, exist_ok=True)
        fh = logging.FileHandler(filepath)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT))
        fh.setLevel(logger.level)
        logger.addHandler(fh)
        logger.info(f"File logging enabled: {filepath}")
    except OSError:
        logger.warning(f"Could not set up file logging at {filepath} — NFS may not be available")


# 分布式已初始化时加 rank 前缀，否则标记 non-distributed，便于统一查看单进程调试日志。
def print_with_rank(message):
    if dist.is_available() and dist.is_initialized():
        logger.info(f"rank {dist.get_rank()}: {message}")
    else:
        logger.info(f"non-distributed: {message}")


# 只在全局 rank 0 输出消息；调用前必须已初始化 distributed。
def print_on_rank0(message):
    if dist.get_rank() == 0:
        logger.info(message)


# primary 创建主 W&B run 和可选 TensorBoard writer；secondary 加入已有 W&B run。
def init_tracking(args, primary: bool = True, **kwargs):
    global _tb_writer
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
        if getattr(args, "use_tensorboard", False) and getattr(args, "output_dir", None):
            from torch.utils.tensorboard import SummaryWriter

            tb_log_dir = os.path.join(args.output_dir, "runs")
            os.makedirs(tb_log_dir, exist_ok=True)
            _tb_writer = SummaryWriter(log_dir=tb_log_dir)
            logger.info(f"TensorBoard writer initialized at {tb_log_dir}")
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


# 读取模块级 SummaryWriter；未启用或关闭后返回 None。
def get_tb_writer():
    """Return the module-level TensorBoard SummaryWriter, or None if not initialized."""
    return _tb_writer


# 先 flush 再关闭 TensorBoard writer，并清空全局引用，避免残留待写事件。
def close_tb_writer():
    """Flush and close the TensorBoard writer."""
    global _tb_writer
    if _tb_writer is not None:
        _tb_writer.flush()
        _tb_writer.close()
        _tb_writer = None
