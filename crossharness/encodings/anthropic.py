"""Anthropic Messages encoding: structured tool_use / tool_result blocks.

This is the dialect Claude Code speaks. Calls are structured API objects,
not text:

    {"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_abc", "name": "Read",
         "input": {"file_path": "notes.md"}}
    ]}

Results return inside a *user* message and correlate explicitly by id:

    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_abc",
         "content": "...", "is_error": false}
    ]}

Contrast with Hermes: arguments are a native JSON object (no string
double-encoding), ids are mandatory, and nothing lives in free text.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, List

from ..ir import ParseError, ToolCall, ToolResult

ENCODING = "anthropic"


def parse_calls(message: "dict[str, Any]") -> List[ToolCall]:
    """Extract tool calls from an Anthropic-style assistant message."""
    if not isinstance(message, dict):
        raise ParseError("Expected an assistant message dict")
    calls: List[ToolCall] = []
    for block in message.get("content", []) or []:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        call_id = block.get("id")
        name = block.get("name")
        args = block.get("input", {})
        if not call_id or not name:
            raise ParseError("tool_use block missing id or name", snippet=json.dumps(block)[:200])
        if not isinstance(args, dict):
            raise ParseError("tool_use input must be an object", snippet=json.dumps(block)[:200])
        calls.append(
            ToolCall(id=call_id, name=name, args=args, origin=ENCODING, raw=block)
        )
    return calls


def emit_calls(calls: Iterable[ToolCall]) -> "dict[str, Any]":
    """Render IR calls as an Anthropic-style assistant message."""
    content = [
        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}
        for c in calls
    ]
    return {"role": "assistant", "content": content}


def parse_results(message: "dict[str, Any]") -> List[ToolResult]:
    """Extract tool results from an Anthropic-style user message."""
    if not isinstance(message, dict):
        raise ParseError("Expected a user message dict")
    results: List[ToolResult] = []
    for block in message.get("content", []) or []:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        call_id = block.get("tool_use_id")
        if not call_id:
            raise ParseError(
                "tool_result block missing tool_use_id", snippet=json.dumps(block)[:200]
            )
        results.append(
            ToolResult(
                call_id=call_id,
                content=block.get("content"),
                is_error=bool(block.get("is_error", False)),
                origin=ENCODING,
                raw=block,
            )
        )
    return results


def emit_results(results: Iterable[ToolResult]) -> "dict[str, Any]":
    """Render IR results as an Anthropic-style user message."""
    content = []
    for r in results:
        block: "dict[str, Any]" = {
            "type": "tool_result",
            "tool_use_id": r.call_id,
            "content": r.content,
        }
        if r.is_error:
            block["is_error"] = True
        content.append(block)
    return {"role": "user", "content": content}
