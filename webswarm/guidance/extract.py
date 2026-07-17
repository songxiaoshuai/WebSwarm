"""Trace-rendering utilities used by guidance.

subtask_experience needs to read scout-subtask execution traces without placing complete
system prompts or reasoning content into the extraction prompt. This module compresses
message history into an auditable plain-text trace readable by the LLM.
"""

from __future__ import annotations

import json
from typing import Any


# Per-message content limit to prevent tool output from exhausting the LLM context.
_MAX_MSG_CHARS = 3000

# Total trace-block limit; when exceeded, retain the **tail** because later actions carry more information.
_MAX_TRACE_CHARS = 24000


def _truncate_middle(text: str, limit: int) -> str:
    """Truncate the middle of overlong text while preserving its beginning and end."""
    if len(text) <= limit:
        return text
    half = max(1, (limit - 40) // 2)
    return (
        text[:half]
        + f"\n... <truncated {len(text) - 2 * half} chars> ...\n"
        + text[-half:]
    )


def _msg_content_to_str(content: Any) -> str:
    """Convert both OpenAI- and Claude-style message.content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for piece in content:
            if isinstance(piece, dict):
                t = piece.get("text") or piece.get("content") or ""
                if t:
                    parts.append(str(t))
            elif isinstance(piece, str):
                parts.append(piece)
        return "\n".join(parts)
    return str(content)


def _render_trace(messages: list[dict]) -> str:
    """Render sub-agent messages as a compact plain-text trace.

    Rendering rules: skip system messages, discard reasoning_content, and retain only tool
    names, arguments, and summarized tool returns so the scout's hidden reasoning is not
    transferred to later agents as fact.
    """
    if not messages:
        return "(empty trace)"

    lines: list[str] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")

        if role == "system":
            continue

        if role == "user":
            text = _msg_content_to_str(m.get("content"))
            lines.append(f"[{i}] USER: {_truncate_middle(text, _MAX_MSG_CHARS)}")

        elif role == "assistant":
            text = _msg_content_to_str(m.get("content"))
            tcs = m.get("tool_calls") or []
            if text:
                lines.append(
                    f"[{i}] ASSISTANT: {_truncate_middle(text, _MAX_MSG_CHARS)}"
                )
            for tc in tcs:
                fn = (tc or {}).get("function") or {}
                name = fn.get("name", "<?>")
                args = fn.get("arguments", "")
                if not isinstance(args, str):
                    try:
                        args = json.dumps(args, ensure_ascii=False)
                    except Exception:
                        args = str(args)
                lines.append(
                    f"[{i}] TOOL_CALL {name}({_truncate_middle(args, 600)})"
                )

        elif role == "tool":
            name = m.get("name", "<?>")
            text = _msg_content_to_str(m.get("content"))
            lines.append(
                f"[{i}] TOOL_RESULT {name}: {_truncate_middle(text, _MAX_MSG_CHARS)}"
            )

        else:
            text = _msg_content_to_str(m.get("content"))
            lines.append(f"[{i}] {role}: {_truncate_middle(text, _MAX_MSG_CHARS)}")

    rendered = "\n".join(lines)

    if len(rendered) > _MAX_TRACE_CHARS:
        rendered = (
            f"<... {len(rendered) - _MAX_TRACE_CHARS} chars truncated from start ...>\n"
            + rendered[-_MAX_TRACE_CHARS:]
        )
    return rendered
