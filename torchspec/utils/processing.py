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

# Tokenizer 与 assistant 边界适配：为 loss mask 提供与实际 renderer/chat template 一致的 token 标记。

from transformers import AutoTokenizer

from torchspec.utils.logging import logger


# 默认使用 fast tokenizer，但保留调用方显式 use_fast 值及其他参数，返回 Hugging Face tokenizer。
def load_tokenizer(name_or_path: str, **kwargs):
    kwargs.setdefault("use_fast", True)
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


# renderer 优先提供精确边界；否则按 chat template 编码 assistant header 与结束标记。
# 去掉 header 尾换行后匹配，再返回需要跳过的换行 token 数，避免 BPE 把连续换行合并导致匹配失败。
# 没有可用模板边界时返回 (None,None,0)，由 mask 解析方决定后续策略。
def get_assistant_token_ids(
    args,
) -> tuple[list[int] | None, list[int] | None, int]:
    """Derive assistant_header_ids, end_token_ids, and skip_after_header.

    Returns:
        (header_ids, end_ids, skip_after_header) where skip_after_header is the
        number of trailing-newline tokens stripped from the header to avoid BPE
        merge issues. Pass this to ``compute_assistant_loss_mask`` so it skips
        the formatting newline after the role name.
    """
    from torchspec.data.renderers import RENDERER_REGISTRY
    from torchspec.data.template import TEMPLATE_REGISTRY

    # renderer 可能包含模型特定的角色格式，优先向它询问边界，避免用通用模板猜 token 序列。
    # 若 renderer 没实现匹配接口或返回空边界，显式失败比生成错误监督 mask 更安全。
    renderer_name = getattr(args, "renderer", None)
    if renderer_name:
        tokenizer = load_tokenizer(args.target_model_path, trust_remote_code=True)
        renderer = RENDERER_REGISTRY.create(renderer_name, tokenizer)
        get_match_tokens = getattr(renderer, "get_assistant_token_ids", None)
        if get_match_tokens is None:
            raise ValueError(
                f"Dataset renderer {renderer_name!r} does not support dynamic "
                "assistant-token matching"
            )
        header_ids, end_ids, skip_after_header = get_match_tokens()
        if not header_ids or not end_ids:
            raise ValueError(
                f"Dataset renderer {renderer_name!r} returned an empty assistant matcher"
            )
        logger.info(
            f"Renderer assistant loss mask token IDs ({renderer_name}): "
            f"header={header_ids}, end={end_ids}, "
            f"skip_after_header={skip_after_header}"
        )
        return header_ids, end_ids, skip_after_header

    chat_template_name = getattr(args, "chat_template", None)
    if not chat_template_name:
        return None, None, 0

    template = TEMPLATE_REGISTRY.get(chat_template_name)
    if not template.assistant_header or not template.end_of_turn_token:
        return None, None, 0

    tokenizer = load_tokenizer(args.target_model_path, trust_remote_code=True)

    # 分别编码完整 header 和去尾换行的 header，利用长度差估计需要跳过的格式 token。
    # 例如正文以换行开头，BPE 可能把 header 末尾换行与正文换行合并，
    # 直接匹配完整 header token 会失败；匹配不含尾换行的核心角色标记更稳定。
    full_ids = tokenizer.encode(template.assistant_header, add_special_tokens=False)
    header_text = template.assistant_header.rstrip("\n")
    stripped_ids = tokenizer.encode(header_text, add_special_tokens=False)
    # BPE tokenizers may merge the header's trailing \n with content-leading
    # newlines into a single multi-char token, breaking subsequence matching.
    # We tokenize without the trailing \n and tell the mask function to skip
    # the newline token(s) that follow.
    skip_after_header = len(full_ids) - len(stripped_ids)

    end_token_ids = tokenizer.encode(template.end_of_turn_token, add_special_tokens=False)
    logger.info(
        f"Assistant loss mask token IDs: header={stripped_ids}, end={end_token_ids}, "
        f"skip_after_header={skip_after_header}"
    )
    return stripped_ids, end_token_ids, skip_after_header
