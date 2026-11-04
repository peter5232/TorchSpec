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

# 中文导读：名称到 renderer 类的注册表，把模型特定格式与通用数据加载器解耦。
# 缓存版本也是契约的一部分：相同消息在新规则下产生不同 ID/mask 时应使旧缓存失效。

from typing import Any, List

from torchspec.data.renderers.base import DEFAULT_CACHE_VERSION, ConversationRenderer


# 存放类而非实例；每个 worker 可用自己的 tokenizer 构造 renderer。
class RendererRegistry:
    """Registry of renderer classes instantiated with a model tokenizer."""

    # 建立空的名称→实现类映射，具体注册由包初始化完成。
    def __init__(self):
        self.renderers: dict[str, type[ConversationRenderer]] = {}

    # 默认拒绝同名覆盖；只有明确 override 才替换，避免导入顺序悄悄改变渲染规则。
    def register(
        self,
        name: str,
        renderer_cls: type[ConversationRenderer],
        override: bool = False,
    ) -> None:
        assert override or name not in self.renderers, (
            f"Renderer for the model type {name} has already been registered"
        )
        self.renderers[name] = renderer_cls

    # 按名称取类，不实例化；不存在的名字会触发 KeyError。
    def get(self, name: str) -> type[ConversationRenderer]:
        return self.renderers[name]

    # 取得类后传入 tokenizer 创建独立实例。
    def create(self, name: str, tokenizer: Any) -> ConversationRenderer:
        return self.get(name)(tokenizer)

    # 读取实现声明的缓存版本，未声明则使用协议默认版本。
    def cache_version(self, name: str) -> str:
        return getattr(self.get(name), "CACHE_VERSION", DEFAULT_CACHE_VERSION)

    # 列举可选名称，供配置校验和错误提示使用。
    def get_all_renderer_names(self) -> List[str]:
        return list(self.renderers.keys())


RENDERER_REGISTRY = RendererRegistry()
