# 中文导读：导出新版磁盘回放协议与 norm 元数据配置器；它不等于旧 .ckpt 数据集接口。

"""Offline target-data materialization and training support."""

from torchspec.offline.dataset import (
    OFFLINE_SCHEMA_VERSION,
    OfflineDataset,
    configure_offline_args,
)

__all__ = [
    "OFFLINE_SCHEMA_VERSION",
    "OfflineDataset",
    "configure_offline_args",
]
