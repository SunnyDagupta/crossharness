"""A minimal Claude-Code-style harness executor.

This is the *target* side of the demo: a harness that speaks the Anthropic
structured dialect and exposes Claude Code's file tool surface (Read,
Write, LS) with Claude Code's argument names (file_path).

Faithfulness matters more than features here, in two specific ways:

  * It is STRICT. Real harnesses do not fuzzy-match tool names; a call to
    ``read_file`` when the surface is ``Read`` is an error, exactly as it
    would be in production. The eval's middle condition depends on this
    honesty.
  * Execution is REAL. Tools operate on an actual directory on disk
    (jailed to a sandbox root) -- nothing is mocked, so a passing task is
    evidence the translated call genuinely drove the harness.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from .encodings import anthropic
from .ir import ToolCall, ToolResult

MAX_READ_BYTES = 64 * 1024


class SandboxExecutor:
    """Executes Anthropic-dialect tool calls against a jailed directory."""

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tools: "Dict[str, Dict[str, Any]]" = {
            "Read": {"required": ["file_path"], "fn": self._read},
            "Write": {"required": ["file_path", "content"], "fn": self._write},
            "LS": {"required": ["path"], "fn": self._ls},
        }

    # -- public API ----------------------------------------------------------

    def execute_message(self, assistant_message: "dict[str, Any]") -> "dict[str, Any]":
        """Take an Anthropic-style assistant message, run every tool_use
        block, and return the Anthropic-style user message of tool_results.

        A message containing no tool_use blocks yields a results message
        with no blocks -- which is exactly what happens when a model speaks
        an alien dialect at a structured harness: nothing is recognized.
        """
        calls = anthropic.parse_calls(assistant_message)
        results = [self._execute(call) for call in calls]
        return anthropic.emit_results(results)

    # -- internals -------------------------------------------------------------

    def _execute(self, call: ToolCall) -> ToolResult:
        spec = self.tools.get(call.name)
        if spec is None:
            available = ", ".join(sorted(self.tools))
            return ToolResult(
                call_id=call.id,
                content=f"Unknown tool: {call.name}. Available tools: {available}",
                is_error=True,
                name=call.name,
            )
        missing = [k for k in spec["required"] if k not in call.args]
        if missing:
            return ToolResult(
                call_id=call.id,
                content=(
                    f"Invalid arguments for {call.name}: missing "
                    + ", ".join(missing)
                    + f". Got: {sorted(call.args)}"
                ),
                is_error=True,
                name=call.name,
            )
        try:
            content = spec["fn"](call.args)
            return ToolResult(call_id=call.id, content=content, name=call.name)
        except PermissionError as exc:
            return ToolResult(call_id=call.id, content=str(exc), is_error=True, name=call.name)
        except FileNotFoundError as exc:
            return ToolResult(
                call_id=call.id, content=f"Not found: {exc}", is_error=True, name=call.name
            )
        except OSError as exc:
            return ToolResult(call_id=call.id, content=str(exc), is_error=True, name=call.name)

    def _jail(self, raw_path: str) -> Path:
        """Resolve a path strictly inside the sandbox root."""
        candidate = Path(raw_path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise PermissionError(f"Path escapes sandbox: {raw_path}")
        return resolved

    def _read(self, args: "dict[str, Any]") -> str:
        path = self._jail(str(args["file_path"]))
        if not path.is_file():
            raise FileNotFoundError(str(args["file_path"]))
        data = path.read_bytes()[:MAX_READ_BYTES]
        return data.decode("utf-8", errors="replace")

    def _write(self, args: "dict[str, Any]") -> str:
        path = self._jail(str(args["file_path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(args["content"])
        path.write_text(content, encoding="utf-8")
        rel = os.path.relpath(path, self.root)
        return f"OK: wrote {len(content.encode('utf-8'))} bytes to {rel}"

    def _ls(self, args: "dict[str, Any]") -> str:
        path = self._jail(str(args["path"]))
        if not path.is_dir():
            raise FileNotFoundError(str(args["path"]))
        entries = []
        for child in sorted(path.iterdir()):
            entries.append(child.name + ("/" if child.is_dir() else ""))
        return "\n".join(entries) if entries else "(empty directory)"
