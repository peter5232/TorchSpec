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

# 集中导出推理接口和可用后端，方便调用方选择实现。
# 可选后端通过捕获 ImportError 隔离缺少依赖的环境；模型加载仍在各引擎 init 中执行。


from torchspec.inference.engine.base import InferenceEngine
from torchspec.inference.engine.hf_engine import HFEngine
from torchspec.inference.engine.offline_replay_engine import OfflineReplayEngine

__all__ = [
    "InferenceEngine",
    "HFEngine",
    "OfflineReplayEngine",
]

try:
    from torchspec.inference.engine.hf_runner import HFRunner  # noqa: F401

    __all__.append("HFRunner")
except ImportError as _e:
    from torchspec.utils.logging import logger as _logger

    _logger.debug("HFRunner not available: %s", _e)

# Lazy imports: SGLang/vLLM are optional — HF-only training (e.g. single-GPU
# DFlash) should not require these heavy dependencies to be installed.
try:
    from torchspec.inference.engine.sgl_engine import SglEngine  # noqa: F401

    __all__.append("SglEngine")
except ImportError as _e:
    from torchspec.utils.logging import logger as _logger

    _logger.debug("SglEngine not available: %s", _e)

try:
    from torchspec.inference.engine.vllm_engine import VllmEngine  # noqa: F401

    __all__.append("VllmEngine")
except ImportError as _e:
    from torchspec.utils.logging import logger as _logger

    _logger.debug("VllmEngine not available: %s", _e)

try:
    from torchspec.inference.engine.trtllm_engine import TrtllmEngine  # noqa: F401

    __all__.append("TrtllmEngine")
except ImportError as _e:
    from torchspec.utils.logging import logger as _logger

    _logger.debug("TrtllmEngine not available: %s", _e)
