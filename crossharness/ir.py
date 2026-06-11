"""Canonical intermediate representation (IR) for tool calls and results.

The IR is deliberately small. It captures the four things every harness
dialect must agree on, regardless of how it encodes them on the wire:

  1. identity      -- which call is which (id, synthesized when the origin
                      format has none, e.g. Hermes in-band text)
  2. the tool      -- the surface name as emitted, plus the canonical
                      operation it resolves to once a harness profile has
                      mapped it (e.g. "read_file" -> "fs.read")
  3. the arguments -- always a parsed dict in the IR, regardless of whether
                      the wire format carries an object (Anthropic) or a
                      JSON-encoded string (OpenAI) or tag-delimited text
                      (Hermes)
  4. provenance    -- the original raw bytes, kept for auditability so a
                      translation is always inspectable after the fact
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class ToolCall:
    """A single tool invocation, normalized out of any wire format."""

    id: str
    name: str
    args: "dict[str, Any]"
    origin: str  # encoding that produced this, e.g. "hermes" | "anthropic"
    raw: Union[str, dict, None] = None
    id_synthesized: bool = False
    canonical: Optional[str] = None  # canonical op, set by surface mapping


@dataclass
class ToolResult:
    """The outcome of a tool invocation, normalized out of any wire format."""

    call_id: str
    content: Any
    is_error: bool = False
    name: str = ""  # tool name, carried for formats (Hermes) that echo it
    origin: str = ""
    raw: Union[str, dict, None] = None


class ParseError(ValueError):
    """Raised when wire bytes cannot be parsed into the IR."""

    def __init__(self, message: str, snippet: str = ""):
        super().__init__(message)
        self.snippet = snippet


def sequential_id_generator(prefix: str = "call"):
    """Deterministic id source for formats that carry no call ids.

    Determinism matters: translated transcripts must be reproducible
    byte-for-byte so eval runs and fixtures can be diffed.
    """
    n = 0
    while True:
        n += 1
        yield f"{prefix}_{n:03d}"
