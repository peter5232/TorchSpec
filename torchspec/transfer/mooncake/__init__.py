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

# Mooncake 子包的公共入口：轻量工具直接导出，配置和存储类按属性访问时再导入。
# 这样配置模块也能引用工具，而不会与依赖配置的存储实现形成循环导入。

from torchspec.transfer.mooncake.helpers import calculate_eagle3_buffer_size
from torchspec.transfer.mooncake.utils import (
    MooncakeMaster,
    check_mooncake_master_available,
    launch_mooncake_master,
    resolve_mooncake_master_bin,
)


# 只对三个延迟公开的类执行实际导入；未知名称抛 AttributeError，让正常 Python 属性检查语义保持不变。
def __getattr__(name):
    # Lazy imports to avoid circular dependency with config.mooncake_config
    if name == "MooncakeConfig":
        from torchspec.config.mooncake_config import MooncakeConfig

        return MooncakeConfig
    if name == "MooncakeHiddenStateStore":
        from torchspec.transfer.mooncake.store import MooncakeHiddenStateStore

        return MooncakeHiddenStateStore
    if name == "EagleMooncakeStore":
        from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore

        return EagleMooncakeStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MooncakeConfig",
    "MooncakeMaster",
    "MooncakeHiddenStateStore",
    "EagleMooncakeStore",
    "calculate_eagle3_buffer_size",
    "resolve_mooncake_master_bin",
    "check_mooncake_master_available",
    "launch_mooncake_master",
]
