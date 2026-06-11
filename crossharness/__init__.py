"""crossharness: a cross-harness portability shim for tool-calling models.

Translates tool-calling dialects between agent harnesses at inference
time -- both the wire encoding (in-band text vs structured blocks, id
synthesis, argument encoding) and the harness surface (tool names and
schemas) -- so a model fluent in one harness's conventions can drive
another without retraining.
"""
from .ir import ParseError, ToolCall, ToolResult, sequential_id_generator
from .shim import CallRecord, Shim
from .surfaces import CLAUDE_CODE, HERMES_GENERIC, PROFILES, HarnessProfile, ToolSpec
from .executor import SandboxExecutor

__version__ = "0.2.0"

__all__ = [
    "ParseError",
    "ToolCall",
    "ToolResult",
    "sequential_id_generator",
    "Shim",
    "CallRecord",
    "HarnessProfile",
    "ToolSpec",
    "HERMES_GENERIC",
    "CLAUDE_CODE",
    "PROFILES",
    "SandboxExecutor",
]
