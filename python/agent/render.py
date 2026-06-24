from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .dataset import AgentExample


@dataclass(frozen=True)
class ReusePrompt:
    system: str
    tools: str
    conversation: str


def render_full_prompt(tokenizer: Any, example: AgentExample) -> str:
    """Render the native agent prompt: system prompt and tools share one message."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": example.system_prompt},
        *example.messages,
    ]
    return tokenizer.apply_chat_template(
        messages,
        tools=example.tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_reuse_prompt(tokenizer: Any, example: AgentExample) -> ReusePrompt:
    """Render system and tools as two independently reusable system messages."""
    system = tokenizer.apply_chat_template(
        [{"role": "system", "content": example.system_prompt}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    tools = tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}],
        tools=example.tools,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    conversation = tokenizer.apply_chat_template(
        example.messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return ReusePrompt(system=system, tools=tools, conversation=conversation)


def count_tool_tokens(tokenizer: Any, example: AgentExample) -> int:
    """Count tokens in the independently rendered tool-definition message."""
    tool_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}],
        tools=example.tools,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return len(tokenizer.encode(tool_prompt, add_special_tokens=False))
