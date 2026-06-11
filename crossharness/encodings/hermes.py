"""Hermes-format encoding: in-band text tool calls.

NousResearch Hermes models (2 Pro / 3) emit tool calls as literal text in
the generation stream, delimited by XML-ish tags:

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Berlin"}}
    </tool_call>

and consume results the same way:

    <tool_response>
    {"name": "get_weather", "content": {...}}
    </tool_response>

Two properties make this dialect maximally distant from structured-API
dialects like Anthropic's:

  * there is no call id at all -- correlation is positional, so a
    translator must synthesize ids and keep the books itself;
  * the call lives inside free text -- prose can surround it, and the
    arguments arrive as part of a JSON blob that some models double-encode.
"""
from __future__ import annotations

import json
import re
from typing import Iterable, Iterator, List, Optional

from ..ir import ParseError, ToolCall, ToolResult, sequential_id_generator

ENCODING = "hermes"

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_calls(text: str, idgen: Optional[Iterator[str]] = None) -> List[ToolCall]:
    """Extract tool calls from raw Hermes-format model text.

    Surrounding prose is ignored (models often narrate before calling).
    Ids are synthesized deterministically because the format has none.
    """
    if idgen is None:
        idgen = sequential_id_generator()
    calls: List[ToolCall] = []
    for match in _TOOL_CALL_RE.finditer(text):
        blob = match.group(1)
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Malformed JSON inside <tool_call>: {exc}", snippet=blob[:200])
        name = payload.get("name")
        if not name or not isinstance(name, str):
            raise ParseError("Tool call missing string 'name' field", snippet=blob[:200])
        args = payload.get("arguments", {})
        if isinstance(args, str):
            # Some models double-encode arguments as a JSON string
            # (an OpenAI-dialect habit bleeding through). Tolerate it.
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                raise ParseError(f"Arguments string is not valid JSON: {exc}", snippet=args[:200])
        if not isinstance(args, dict):
            raise ParseError("Arguments must decode to an object", snippet=blob[:200])
        calls.append(
            ToolCall(
                id=next(idgen),
                name=name,
                args=args,
                origin=ENCODING,
                raw=match.group(0),
                id_synthesized=True,
            )
        )
    return calls


def emit_calls(calls: Iterable[ToolCall]) -> str:
    """Render IR calls as Hermes in-band text (reverse direction)."""
    chunks = []
    for call in calls:
        payload = {"name": call.name, "arguments": call.args}
        chunks.append(f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>")
    return "\n".join(chunks)


def emit_results(results: Iterable[ToolResult]) -> str:
    """Render IR results as Hermes <tool_response> text for the model.

    Hermes correlates positionally; the name is echoed because the
    reference format includes it and models are trained to expect it.
    """
    chunks = []
    for result in results:
        payload = {"name": result.name, "content": result.content}
        if result.is_error:
            payload["error"] = True
        chunks.append(
            f"<tool_response>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_response>"
        )
    return "\n".join(chunks)


def strip_calls(text: str) -> str:
    """Return the prose around any tool calls (useful for display)."""
    return _TOOL_CALL_RE.sub("", text).strip()
