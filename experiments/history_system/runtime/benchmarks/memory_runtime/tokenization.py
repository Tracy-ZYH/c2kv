"""The observable SGLang function-tool serialization used by the 1088 profile."""
from __future__ import annotations

TOOL_SCHEMA_PROFILE = "sglang-function-full-v1"


def serving_tools(tools):
    """Match SGLang protocol.Tool/Function.model_dump(), c2kv_tools_dump=full.

    The server drops unknown function fields (including benchmark response
    schemas) and emits the four declared fields in model-definition order.
    Passing the original client JSON directly to a tokenizer changes the
    common tool prologue even when the chat messages are identical.
    """
    if not tools:
        return None
    output = []
    for tool in tools:
        if tool.get("type", "function") != "function":
            raise ValueError("The 1088 tool profile accepts function tools only")
        function = tool["function"]
        output.append({"type": "function", "function": {
            "description": function.get("description"),
            "name": function["name"],
            "parameters": function.get("parameters"),
            "strict": function.get("strict", False),
        }})
    return output
