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

# 中文导读：聊天文本的“协议层”，决定角色边界、思考内容、工具调用以及监督区间。
# format 把消息结构变成模型熟悉的文本；parse 再得到 [T] token 和同长 loss_mask。
# 不同家族控制符不同，不能用字符串包含 assistant 就断定训练区间；更不能把字符索引当 token 索引。
# 此处仅处理格式与标签，不执行大模型推理。模板改变会改变训练分布，因此需要配套测试。

import json
import re
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
from transformers import PreTrainedTokenizer

from torchspec.data.template import ChatTemplate

if TYPE_CHECKING:
    from typing import Any

    Conversation = List[Dict[str, Any]]

__all__ = [
    "GeneralParser",
    "HarmonyParser",
    "KimiK25Parser",
    "MiniMaxParser",
    "create_parser",
    "has_thinking_content",
    "has_unbalanced_thinking_tags",
]

_HAS_THINKING_RE = re.compile(r"<think>(?!\s*</think>)")


# 识别含闭标签但开头无 <think> 的保存文本，处理开标签已在生成提示里而未被保存的情况。
def _has_dropped_think_opener(content: str) -> bool:
    return "</think>" in content and not content.lstrip().startswith("<think>")


# 只检查 assistant 的真实内容和独立 reasoning 字段，空 <think></think> 不算思考。
# 必须在格式化前调用，因为格式化器可能自动插入空标签，污染判断。
def has_thinking_content(conversation: list) -> bool:
    """Detect whether any assistant message contains real thinking content.

    Checks for non-empty <think> blocks in message content (including the
    dropped-opener ``{reasoning}</think>{answer}`` shape that recovery restores)
    and for separate thinking/thinking_content/reasoning_content/reasoning fields
    on the message dict (covers preserved_thinking outputs from engines).
    Must be called on the raw conversation BEFORE formatting, since
    formatters (e.g. KimiK25Parser) inject empty <think></think> tags.
    """
    for msg in conversation:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and (
            _HAS_THINKING_RE.search(content) or _has_dropped_think_opener(content)
        ):
            return True
        for field in ("thinking", "thinking_content", "reasoning_content", "reasoning"):
            if msg.get(field):
                return True
    return False


# 比较开闭标签数量，用于发现格式化后可疑样本；数量相等不等于嵌套和语义一定合法。
def has_unbalanced_thinking_tags(formatted_text: str) -> bool:
    return formatted_text.count("<think>") != formatted_text.count("</think>")


# 两阶段协议抽象：format 可供引擎延迟分词，parse 用于本地预处理并生成监督 mask。
class Parser(ABC):
    # 持有 tokenizer 和模板对象；它们共同决定控制符与 token 编码。
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    # 抽象接口：消息列表→格式化文本。具体模型家族必须实现角色与特殊 token 的排列。
    @abstractmethod
    def format(self, conversation: "Conversation", **kwargs) -> str:
        """Apply chat template to conversation messages, return formatted text.

        This is the first phase: formatting only, no tokenization.
        Use this when sending prompts to an inference engine that
        handles tokenization internally.
        """
        pass

    # 抽象接口：返回 token ID 与监督 mask；调用方依赖二者位置一一对应。
    @abstractmethod
    def parse(
        self, conversation: "Conversation", max_length: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Format, tokenize, and compute loss mask.

        This is the full two-phase pipeline for offline dataset building.
        Returns (input_ids, loss_mask) tensors.
        """
        pass

    # preformatted=True 时直接使用传入文本，避免重复套模板；否则调用具体 format。
    def _prepare_text(self, conversation: "Conversation", preformatted: bool, **kwargs) -> str:
        if preformatted:
            return conversation
        return self.format(conversation, **kwargs)

    # 当前实现按 pad_token_id 的真假值判断并回退到 unk；这也会把数值 0 当作假值。
    def _ensure_pad_token(self):
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

    # 先编码全文，再用 regex 的第一捕获组找到 assistant 字符区间，分别编码区间前缀换算 token 边界。
    # 例如前缀编码长 10、包含回答的前缀长 14，就监督 [10:14]；最后裁到截断后的 token 长度。
    # 这是 prefix 编码边界法，不是逐字符与 token 一一映射；last_turn_only 仅保留最后一个匹配。
    def _tokenize_with_loss_mask(
        self,
        text: str,
        max_length: int,
        assistant_pattern: str,
        last_turn_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text and compute loss mask via encode-prefix character mapping.

        Args:
            last_turn_only: If True, only compute loss for the last assistant turn.
        """
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        matches = list(re.finditer(assistant_pattern, text))
        if last_turn_only and matches:
            matches = matches[-1:]

        for match in matches:
            content_start_char = match.start(1)
            content_end_char = match.end(1)

            # 字符长度和 token 长度不能互换：中文一个词可能是多个 token，英文多个字符也可能合成一个 token。
            # 这里通过重新编码前缀求 token 边界；截断之后的不存在位置不能再写进 mask。
            prefix_ids = self.tokenizer.encode(text[:content_start_char], add_special_tokens=False)
            full_ids = self.tokenizer.encode(text[:content_end_char], add_special_tokens=False)

            start_token_idx = len(prefix_ids)
            end_token_idx = len(full_ids)

            actual_start = min(start_token_idx, len(input_ids))
            actual_end = min(end_token_idx, len(input_ids))

            if actual_start < actual_end:
                loss_mask[actual_start:actual_end] = 1

        return input_ids, loss_mask


# 优先调用 tokenizer 自带 chat template，再依注册模板的 assistant/end 标记识别监督区间。
class GeneralParser(Parser):
    # 从模板缓存系统提示及用户/助手分隔符，后续格式化和正则匹配共享这些设置。
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)
        self.system_prompt = chat_template.system_prompt
        self.user_message_separator = f"{chat_template.end_of_turn_token}"
        self.assistant_message_separator = f"{chat_template.assistant_header}"

    # 默认不添加生成头，允许调用者显式覆盖；tokenize=False 仅生成文本。
    def _apply_chat_template(self, messages, **kwargs) -> str:
        kwargs.setdefault("add_generation_prompt", False)
        conversation = self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        return conversation

    # 先决定系统提示，再校验 user/assistant/tool 的基本顺序；不合法时截断到已接受前缀。
    # 模板不可用时退回简化字符串拼接，该回退只覆盖 system/user/assistant 三种角色。
    def format(self, conversation: "Conversation", **kwargs) -> str:
        messages = []

        if conversation[0]["role"] == "system":
            warnings.warn(
                "The first message is from system, we will use the system prompt from the data and ignore the system prompt from the template"
            )
            messages.append({"role": "system", "content": conversation[0]["content"]})
            conversation = conversation[1:]
        else:
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

        for j, sentence in enumerate(conversation):
            role = sentence["role"]
            if j == 0:
                if role != "user":
                    warnings.warn(
                        f"Conversation must start with a 'user' role, but found '{role}'. Conversation truncated."
                    )
                    break
            else:
                prev_role = conversation[j - 1]["role"]
                if role == "tool" and prev_role not in ["assistant", "tool"]:
                    warnings.warn(
                        f"A 'tool' message must follow an 'assistant' or 'tool' message, but was preceded by '{prev_role}'. Conversation truncated."
                    )
                    break
                if role == "assistant" and prev_role not in ["user", "tool"]:
                    warnings.warn(
                        f"An 'assistant' message must follow a 'user' or 'tool' message, but was preceded by '{prev_role}'. Conversation truncated."
                    )
                    break
            messages.append(sentence)

        try:
            return self._apply_chat_template(messages, **kwargs)
        except (ValueError, TypeError):
            warnings.warn("Tokenizer does not have a chat_template, using fallback rendering.")
            add_generation_prompt = kwargs.get("add_generation_prompt", False)
            parts = []
            bos_token = getattr(self.tokenizer, "bos_token", None)
            user_header = self.chat_template.user_header or ""
            assistant_header = self.chat_template.assistant_header or ""
            end_of_turn = self.chat_template.end_of_turn_token or ""

            if bos_token:
                parts.append(bos_token)

            for msg in messages:
                if msg["role"] == "system":
                    parts.append(msg["content"])
                elif msg["role"] == "user":
                    parts.append(f"{user_header}{msg['content']}")
                elif msg["role"] == "assistant":
                    parts.append(f"{assistant_header}{msg['content']}{end_of_turn}")

            if add_generation_prompt:
                parts.append(assistant_header)

            return "".join(parts)

    # 匹配 assistant header 之后直到轮次结束或文本尾部；此正则把结束标记放进捕获组。
    # 因此它与 Kimi/MiniMax 的 mask 边界不完全相同，不能认为所有 parser 都监督相同特殊 token。
    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        last_turn_only: bool = False,
        **kwargs,
    ) -> Dict[str, List[torch.Tensor]]:
        text = self._prepare_text(conversation, preformatted, **kwargs)
        self._ensure_pad_token()

        assistant_pattern = (
            re.escape(self.assistant_message_separator)
            + r"([\s\S]*?(?:"
            + re.escape(self.chat_template.end_of_turn_token)
            + "|$))"
        )
        return self._tokenize_with_loss_mask(
            text,
            max_length,
            assistant_pattern,
            last_turn_only=last_turn_only,
        )


# 将带 channel 的消息渲染为 Harmony 控制符格式，使用 offset mapping 计算监督区间。
class HarmonyParser(Parser):
    # 初始化支持的 reasoning level 名称和默认 low 设置。
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)
        self.reasoning_levels = ["low", "medium", "high"]
        self.default_reasoning_level = "low"

    # 按角色和 channel 追加对应消息；system/effort 分支会重建起始文本，未知角色直接报错。
    def build_single_turn_prompt(
        self,
        prompt_text: str,
        role: str,
        content: str,
    ) -> str:
        if role == "system":
            prompt_text = f"<|start|>system<|message|>{content}<|end|>"
        elif role == "assistant_reasoning_effort":
            prompt_text = f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-06-28\n\nReasoning: {content.lower()}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        elif role == "user":
            prompt_text += f"<|start|>user<|message|>{content}<|end|>"
        elif role == "assistant_analysis":
            prompt_text += f"<|start|>assistant<|channel|>analysis<|message|>{content}<|end|>"
        elif role == "assistant_commentary":
            prompt_text += f"<|start|>assistant<|channel|>commentary<|message|>{content}<|end|>"
        elif role == "assistant_final":
            prompt_text += f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>"
        else:
            raise ValueError(f"Unknown role: {role}")
        return prompt_text

    # 缺少首条系统/effort 消息时补默认信息，按顺序追加各轮，按需添加 assistant analysis 生成头。
    def format(self, conversation: "Conversation", **kwargs) -> str:
        add_generation_prompt = kwargs.pop("add_generation_prompt", False)
        prompt_text = ""
        for j, message in enumerate(conversation):
            if (
                j == 0
                and message["role"] != "system"
                and message["role"] != "assistant_reasoning_effort"
            ):
                prompt_text = self.build_single_turn_prompt(
                    prompt_text,
                    "assistant_reasoning_effort",
                    self.default_reasoning_level,
                )
            prompt_text = self.build_single_turn_prompt(
                prompt_text, message["role"], message["content"]
            )
        if add_generation_prompt:
            prompt_text += "<|start|>assistant<|channel|>analysis<|message|>"
        return prompt_text

    # 使用 tokenizer 的字符 offset 将 assistant 到下一 user 前的片段映射为 token mask。
    # 与通用 prefix 编码法不同，这个路径依赖 tokenizer 提供 offset_mapping。
    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        last_turn_only: bool = False,
        **kwargs,
    ) -> List[torch.Tensor]:
        text = self._prepare_text(conversation, preformatted, **kwargs)
        self._ensure_pad_token()

        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        offsets = encoding.offset_mapping[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        pattern = re.compile(r"<\|start\|>assistant([\s\S]*?)(?=<\|start\|>user<\|message\|>|$)")

        matches = list(pattern.finditer(text))
        if last_turn_only and matches:
            matches = matches[-1:]

        for match in matches:
            start_char = match.start(1)
            end_char = match.end(1)

            for idx, (ts, te) in enumerate(offsets):
                if ts >= start_char and te <= end_char:
                    loss_mask[idx] = 1

        return input_ids, loss_mask


# 通用 parser 的变体：先渲染历史和生成提示，再拼最后一轮回答，保留思考模型的提示前缀。
class ThinkingParser(GeneralParser):
    # 复用通用 parser 的分隔符和系统提示初始化。
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)

    # 要求最后一条为 assistant，历史渲染时添加生成头，再拼原回答和轮次结束 token。
    # 这样避免 tokenizer 在普通历史格式化时删除最后回答中需要监督的思考内容。
    def _apply_chat_template(self, messages, **kwargs) -> str:
        # ThinkingParser always generates prompt before appending last assistant content
        kwargs.pop("add_generation_prompt", None)
        if messages[-1]["role"] == "assistant":
            conversation_history = self.tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
                **kwargs,
            )
            conversation = (
                conversation_history
                + messages[-1]["content"]
                + self.chat_template.end_of_turn_token
            )
            return conversation
        else:
            raise Exception(f"The last message is not assistant but {messages[-1]['role']}")

    # 模板声明 enable_thinking 时显式传递该选项，然后复用通用角色处理。
    def format(self, conversation: "Conversation", **kwargs) -> str:
        if self.chat_template.enable_thinking:
            kwargs["enable_thinking"] = True
        return super().format(conversation, **kwargs)


# 手工实现 Kimi K2.5 控制符，处理图像占位符、工具调用和末轮思考保留。
class KimiK25Parser(Parser):
    """
    Parser for Kimi-K2.5 model with manual string formatting.
    The default system prompt might cause confusion to users and unexpected behaviours, so we remove it.
    The token <|media_start|> is incorrect; it has been replaced with <|media_begin|> in the chat template. We also remove the system prompt from the chat template.

    Handles:
    - Converting <|image|> placeholders to Kimi media token structure
    - Preserving existing <think>...</think> blocks in assistant responses
    - Generating loss mask for assistant content (including thinking)
    """

    MEDIA_TOKEN = "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
    USER_HEADER = "<|im_user|>user<|im_middle|>"
    ASSISTANT_HEADER = "<|im_assistant|>assistant<|im_middle|>"
    SYSTEM_HEADER = "<|im_system|>system<|im_middle|>"
    TOOL_HEADER = "<|im_system|>tool<|im_middle|>"
    END_TOKEN = "<|im_end|>"
    IMAGE_PLACEHOLDER = "<|image|>"
    TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>"
    TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>"
    TOOL_CALL_BEGIN = "<|tool_call_begin|>"
    TOOL_CALL_END = "<|tool_call_end|>"
    TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"

    # 保存 tokenizer 与模板，家族控制符由类常量给定。
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)

    # 仅展开 user 文本中的图像占位符；不把 assistant 输出中的相似字符串当作新图片输入。
    def _format_content(self, content: str, role: str) -> str:
        """Format message content, replacing image placeholders for user messages."""
        if role == "user":
            return content.replace(self.IMAGE_PLACEHOLDER, self.MEDIA_TOKEN + "\n")
        return content

    THINK_PATTERN = re.compile(r"<think>[\s\S]*?</think>")

    # 删除完整 think 区块，用于历史 assistant 回答；正文仍保留。
    def _strip_thinking(self, content: str) -> str:
        return self.THINK_PATTERN.sub("", content)

    # 对带闭标签但缺生成前缀的内容补开标签，使后续 strip/监督能识别完整范围。
    @staticmethod
    def _recover_missing_think_open(content: str) -> str:
        if _has_dropped_think_opener(content):
            return "<think>" + content
        return content

    _REASONING_FIELDS = ("thinking", "thinking_content", "reasoning_content", "reasoning")

    # 按兼容字段优先级取首个非空 reasoning 内容，支持多种数据导出格式。
    @classmethod
    def _reasoning_from_field(cls, msg: dict) -> str:
        for field in cls._REASONING_FIELDS:
            value = msg.get(field)
            if value:
                return value
        return ""

    # 正文尚无 think 标签时才合并独立 reasoning，避免把同一段思考拼两次。
    @classmethod
    def _merge_reasoning_field(cls, msg: dict, content: str) -> str:
        reasoning = cls._reasoning_from_field(msg)
        if reasoning and "<think>" not in content and "</think>" not in content:
            return f"<think>{reasoning}</think>{content}"
        return content

    # 将结构化调用列表序列化为 Kimi 原生 section/call/argument 控制符，维持训练时工具协议。
    def _format_tool_calls(self, tool_calls: list) -> str:
        """Format structured tool_calls into Kimi-native inline tokens."""
        tc_parts = []
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            tc_args = tc.get("function", {}).get("arguments", "")
            tc_parts.append(
                f"{self.TOOL_CALL_BEGIN}{tc_id}"
                f"{self.TOOL_CALL_ARGUMENT_BEGIN}{tc_args}"
                f"{self.TOOL_CALL_END}"
            )
        return self.TOOL_CALLS_SECTION_BEGIN + "".join(tc_parts) + self.TOOL_CALLS_SECTION_END

    # 找到最后一条 assistant：历史轮删除思考，末轮合并/恢复思考；无思考时补空区块。
    # 同时处理工具回包标识和图像占位符，延迟分词时可保留媒体占位不展开。
    def format(self, conversation: "Conversation", **kwargs) -> str:
        """Build conversation string with Kimi-K2.5 tokens."""
        add_generation_prompt = kwargs.pop("add_generation_prompt", False)
        expand_media_tokens = kwargs.pop("expand_media_tokens", True)
        parts = []

        last_assistant_idx = max(
            (i for i, msg in enumerate(conversation) if msg["role"] == "assistant"),
            default=-1,
        )

        for idx, msg in enumerate(conversation):
            role = msg["role"]
            content = msg.get("content", "")

            if not isinstance(content, str):
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") in ("image", "image_url"):
                            if expand_media_tokens:
                                text_parts.append(self.MEDIA_TOKEN + "\n")
                            else:
                                text_parts.append(self.IMAGE_PLACEHOLDER)
                    content = "".join(text_parts)
                else:
                    content = str(content)
            else:
                if expand_media_tokens:
                    content = self._format_content(content, role)

            if role == "assistant" and idx != last_assistant_idx:
                content = self._recover_missing_think_open(content)
                content = self._strip_thinking(content)

            if role == "system":
                parts.append(f"{self.SYSTEM_HEADER}{content}{self.END_TOKEN}")
            elif role == "user":
                parts.append(f"{self.USER_HEADER}{content}{self.END_TOKEN}")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    content += self._format_tool_calls(tool_calls)
                if idx == last_assistant_idx:
                    content = self._merge_reasoning_field(msg, content)
                    content = self._recover_missing_think_open(content)
                if not content.startswith("<think>"):
                    content = "<think></think>" + content
                parts.append(f"{self.ASSISTANT_HEADER}{content}{self.END_TOKEN}")
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                if tool_call_id:
                    content = f"## Return of {tool_call_id}\n{content}"
                parts.append(f"{self.TOOL_HEADER}{content}{self.END_TOKEN}")

        if add_generation_prompt:
            parts.append(f"{self.ASSISTANT_HEADER}<think>")

        return "".join(parts)

    # 只捕获 assistant header 与 END_TOKEN 之间的内容，不把外层角色头和结束符算入监督。
    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        last_turn_only: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self._prepare_text(conversation, preformatted, **kwargs)
        self._ensure_pad_token()

        assistant_pattern = (
            re.escape(self.ASSISTANT_HEADER) + r"([\s\S]*?)" + re.escape(self.END_TOKEN)
        )
        return self._tokenize_with_loss_mask(
            text,
            max_length,
            assistant_pattern,
            last_turn_only=last_turn_only,
        )


# MiniMax 消息协议：保留最后 user 之后的思考，将连续 tool 回包合在同一个 tool 区块。
class MiniMaxParser(Parser):
    """Parser for MiniMax-M2 model with manual string formatting.

    Handles:
    - Interleaved thinking: preserves <think>...</think> only in the last
      assistant turn (after the last user message), strips it from earlier turns.
    - Tool calls with <minimax:tool_call> XML format.
    - Consecutive tool responses grouped under a single ]~b]tool header.
    - Multimodal content with vision token placeholders.
    """

    BOS = "]~!b["
    ROLE_PREFIX = "]~b]"
    END_TOKEN = "[e~["
    SYSTEM_HEADER = "]~!b[]~b]system\n"
    USER_HEADER = "]~b]user\n"
    ASSISTANT_HEADER = "]~b]ai\n"
    TOOL_HEADER = "]~b]tool"
    TOOL_CALL_BEGIN = "<minimax:tool_call>"
    TOOL_CALL_END = "</minimax:tool_call>"
    MEDIA_TOKEN = "]<]start of image[>[" + "]<]vision pad[>[" + "]<]end of image[>["
    IMAGE_PLACEHOLDER = "<image>"

    THINK_PATTERN = re.compile(r"<think>[\s\S]*?</think>")

    # 保存 tokenizer/模板，并允许模板覆盖本实例使用的图片占位符。
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)
        if chat_template.image_placeholder:
            self.IMAGE_PLACEHOLDER = chat_template.image_placeholder

    # 只在 user 文本中将图片占位符替换为家族的视觉 token 结构。
    def _format_content(self, content: str, role: str) -> str:
        if role == "user":
            return content.replace(self.IMAGE_PLACEHOLDER, self.MEDIA_TOKEN + "\n")
        return content

    # 移除完整 think 区块并去掉开头换行；这是文本辅助函数，不处理 token 边界。
    def _strip_thinking(self, content: str) -> str:
        return self.THINK_PATTERN.sub("", content).lstrip("\n")

    # 以首个闭标签切分思考和答案，也兼容缺失开标签的生成结果，返回两个字符串。
    def _extract_thinking(self, content: str) -> Tuple[str, str]:
        """Split content into (reasoning, remaining) following MiniMax jinja logic."""
        if "</think>" in content:
            before, after = content.split("</think>", 1)
            reasoning = before.split("<think>")[-1].strip("\n")
            remaining = after.strip("\n")
            return reasoning, remaining
        return "", content

    # 把函数名和参数构成 XML 风格 invoke/parameter；JSON 参数字符串先解析，解析失败保存在 raw 字段。
    def _format_tool_calls(self, tool_calls: list) -> str:
        """Format structured tool_calls into MiniMax XML tags."""
        tc_parts = []
        for tc in tool_calls:
            func = tc.get("function", tc)
            name = func.get("name", "")
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {"raw": args}
            invoke_lines = [f'<invoke name="{name}">']
            if isinstance(args, dict):
                for k, v in args.items():
                    val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    invoke_lines.append(f'<parameter name="{k}">{val}</parameter>')
            invoke_lines.append("</invoke>")
            tc_parts.append("\n".join(invoke_lines))
        return "\n" + self.TOOL_CALL_BEGIN + "\n" + "\n".join(tc_parts) + "\n" + self.TOOL_CALL_END

    # 系统提示取数据首条或模板默认；最后 user 之前的思考不再输出，之后的思考保留。
    # 相邻 tool 消息共享一个头和结束符，避免训练文本与模型原生工具对话格式不一致。
    def format(self, conversation: "Conversation", **kwargs) -> str:
        add_generation_prompt = kwargs.pop("add_generation_prompt", False)
        expand_media_tokens = kwargs.pop("expand_media_tokens", True)
        parts = []

        messages = list(conversation)
        system_content = None
        if messages and messages[0]["role"] == "system":
            system_content = messages[0]["content"]
            messages = messages[1:]

        if system_content is None:
            system_content = self.chat_template.system_prompt or "You are a helpful assistant."

        parts.append(f"{self.SYSTEM_HEADER}{system_content}{self.END_TOKEN}\n")

        last_user_idx = max(
            (i for i, msg in enumerate(messages) if msg["role"] == "user"),
            default=-1,
        )

        for idx, msg in enumerate(messages):
            role = msg["role"]
            content = msg.get("content", "") or ""

            if not isinstance(content, str):
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") in ("image", "image_url"):
                            if expand_media_tokens:
                                text_parts.append(self.MEDIA_TOKEN + "\n")
                            else:
                                text_parts.append(self.IMAGE_PLACEHOLDER)
                    content = "".join(text_parts)
                else:
                    content = str(content)
            else:
                if expand_media_tokens:
                    content = self._format_content(content, role)

            if role == "user":
                parts.append(f"{self.USER_HEADER}{content}{self.END_TOKEN}\n")
            elif role == "assistant":
                reasoning = msg.get("reasoning_content", "")
                if not reasoning:
                    reasoning, content = self._extract_thinking(content)

                assistant_text = ""
                if reasoning and idx > last_user_idx:
                    assistant_text += f"<think>\n{reasoning}\n</think>\n\n"

                if content:
                    assistant_text += content

                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    assistant_text += self._format_tool_calls(tool_calls)

                parts.append(f"{self.ASSISTANT_HEADER}{assistant_text}{self.END_TOKEN}\n")
            elif role == "tool":
                if idx == 0 or messages[idx - 1]["role"] != "tool":
                    parts.append(self.TOOL_HEADER)
                parts.append(f"\n<response>{content}</response>")
                if idx == len(messages) - 1 or messages[idx + 1]["role"] != "tool":
                    parts.append(f"{self.END_TOKEN}\n")

        if add_generation_prompt:
            parts.append(f"{self.ASSISTANT_HEADER}<think>\n")

        return "".join(parts)

    # 匹配 ai 头之后、轮次结束前的正文并编码监督 mask，外层协议控制符不在捕获组内。
    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        last_turn_only: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self._prepare_text(conversation, preformatted, **kwargs)
        self._ensure_pad_token()

        assistant_pattern = (
            re.escape(self.ASSISTANT_HEADER) + r"([\s\S]*?)" + re.escape(self.END_TOKEN)
        )
        return self._tokenize_with_loss_mask(
            text,
            max_length,
            assistant_pattern,
            last_turn_only=last_turn_only,
        )


# 按模板的 parser_type 选择具体类，未知类型立即报错；模型差异集中在这一层而不散入训练器。
def create_parser(tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate) -> Parser:
    """Create the appropriate parser based on chat_template.parser_type."""
    if chat_template.parser_type == "general":
        return GeneralParser(tokenizer, chat_template)
    elif chat_template.parser_type == "thinking":
        return ThinkingParser(tokenizer, chat_template)
    elif chat_template.parser_type == "openai-harmony":
        return HarmonyParser(tokenizer, chat_template)
    elif chat_template.parser_type == "kimi-k25":
        return KimiK25Parser(tokenizer, chat_template)
    elif chat_template.parser_type == "minimax-m2":
        return MiniMaxParser(tokenizer, chat_template)
    else:
        raise ValueError(f"Invalid parser type: {chat_template.parser_type}")
