"""Harness surface profiles: the layer nobody else ships.

Wire-format translation (objects vs strings, tags vs blocks, id synthesis)
is commodity -- LiteLLM and vLLM's tool-call parsers already do it. What no
existing layer does is translate between harness *surfaces*: the tool
names, argument schemas, and conventions a model was actually trained
against.

A model fine-tuned on generic function-calling data calls
``read_file({"path": ...})``. Claude Code exposes ``Read({"file_path":
...})``. Perfect syntax, wrong dialect -- the harness rejects it. A surface
profile declares how one harness's tool surface maps onto a small canonical
vocabulary, so any two profiled harnesses can be bridged through it.

Profiles are data, not code: adding a harness is writing one table.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from .ir import ToolCall

# Canonical operations for the Phase 1 file-ops world. Deliberately tiny.
CANONICAL_OPS = ("fs.read", "fs.write", "fs.list")


@dataclass(frozen=True)
class ToolSpec:
    """How one harness names a canonical operation and its arguments."""

    surface_name: str
    canonical: str
    # surface argument name -> canonical argument name
    arg_map: "Dict[str, str]" = field(default_factory=dict)
    # Alternative surface names that resolve to this spec. Models that fall
    # back to trained generic conventions do not fall back uniformly --
    # ``read``, ``cat``, ``open_file`` all occur in the wild. The alias
    # table is itself measurement-driven data: Phase 2a live captures feed
    # it. Emission always uses the primary surface_name.
    aliases: "Tuple[str, ...]" = ()

    def inverted_args(self) -> "Dict[str, str]":
        return {v: k for k, v in self.arg_map.items()}


class HarnessProfile:
    """The tool surface of one harness, expressed against canonical ops."""

    def __init__(self, name: str, tools: "List[ToolSpec]"):
        self.name = name
        self.tools = list(tools)
        self._by_surface = {}
        for tool in self.tools:
            self._by_surface[tool.surface_name] = tool
            for alias in tool.aliases:
                self._by_surface.setdefault(alias, tool)
        self._by_canonical = {t.canonical: t for t in self.tools}

    def spec_for_surface(self, surface_name: str) -> Optional[ToolSpec]:
        return self._by_surface.get(surface_name)

    def spec_for_canonical(self, canonical: str) -> Optional[ToolSpec]:
        return self._by_canonical.get(canonical)

    def to_canonical(self, call: ToolCall) -> "Tuple[ToolCall, bool]":
        """Lift a surface-level call into canonical naming.

        Returns (call, mapped). When the tool is not in this profile the
        call passes through untouched and ``mapped`` is False -- the
        receiving harness will reject it, which is the honest behavior
        (a translator should not invent mappings it does not have).
        """
        spec = self.spec_for_surface(call.name)
        if spec is None:
            return call, False
        new_args = {}
        for key, value in call.args.items():
            new_args[spec.arg_map.get(key, key)] = value
        return (
            replace(call, name=spec.canonical, args=new_args, canonical=spec.canonical),
            True,
        )

    def from_canonical(self, call: ToolCall) -> "Tuple[ToolCall, bool]":
        """Lower a canonical call into this harness's surface naming."""
        spec = self.spec_for_canonical(call.canonical or call.name)
        if spec is None:
            return call, False
        inverted = spec.inverted_args()
        new_args = {}
        for key, value in call.args.items():
            new_args[inverted.get(key, key)] = value
        return (
            replace(call, name=spec.surface_name, args=new_args, canonical=spec.canonical),
            True,
        )


# ---------------------------------------------------------------------------
# Phase 1 profiles
# ---------------------------------------------------------------------------

# The surface a generic function-calling fine-tune actually speaks:
# snake_case verbs, "path"-style argument names. This is what Hermes-format
# training data overwhelmingly looks like.
HERMES_GENERIC = HarnessProfile(
    "hermes-generic",
    [
        ToolSpec(
            "read_file",
            "fs.read",
            {"path": "path", "file_path": "path", "filename": "path"},
            aliases=("read", "open_file", "view_file", "cat", "get_file", "read_text_file"),
        ),
        ToolSpec(
            "write_file",
            "fs.write",
            {"path": "path", "file_path": "path", "filename": "path", "content": "content", "text": "content", "data": "content"},
            aliases=("write", "create_file", "save_file", "save", "write_text_file"),
        ),
        ToolSpec(
            "list_directory",
            "fs.list",
            {"path": "path", "directory": "path", "dir": "path"},
            aliases=("list_dir", "ls", "list_files", "dir", "browse_directory"),
        ),
    ],
)

# The surface Claude Code exposes: PascalCase tools, file_path arguments.
CLAUDE_CODE = HarnessProfile(
    "claude-code",
    [
        ToolSpec("Read", "fs.read", {"file_path": "path"}),
        ToolSpec("Write", "fs.write", {"file_path": "path", "content": "content"}),
        ToolSpec("LS", "fs.list", {"path": "path"}),
    ],
)

PROFILES = {p.name: p for p in (HERMES_GENERIC, CLAUDE_CODE)}
