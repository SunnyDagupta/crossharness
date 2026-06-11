"""The shim: encoding translation composed with surface mapping.

Direction model -> harness:
    raw model text (Hermes in-band)  -> IR -> surface map -> Anthropic message
Direction harness -> model:
    Anthropic tool_result message    -> IR -> name restore -> Hermes text

The shim keeps a per-conversation ledger of every call it has translated,
keyed by synthesized id, so results can be correlated back to the exact
surface name the *model* used -- the model must see its own dialect echoed
back, or the next turn degrades.

``syntactic_only=True`` disables surface mapping while keeping encoding
translation. That mode exists so the eval can isolate the contribution of
each layer; it is also a faithful stand-in for what existing format
adapters (LiteLLM-style) do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .encodings import anthropic, hermes
from .ir import ToolCall, ToolResult, sequential_id_generator
from .surfaces import HarnessProfile


@dataclass
class CallRecord:
    """Bookkeeping for one translated call, kept for result back-translation."""

    model_call: ToolCall    # as the model emitted it (its own surface)
    harness_call: ToolCall  # as delivered to the harness (post-mapping)
    mapped: bool            # whether surface mapping found a translation


class Shim:
    """Bidirectional translator between a Hermes-speaking model and an
    Anthropic-style (Claude Code dialect) harness."""

    def __init__(
        self,
        model_profile: HarnessProfile,
        harness_profile: HarnessProfile,
        syntactic_only: bool = False,
    ):
        self.model_profile = model_profile
        self.harness_profile = harness_profile
        self.syntactic_only = syntactic_only
        self._idgen = sequential_id_generator()
        self.ledger: "Dict[str, CallRecord]" = {}
        self.warnings: "List[str]" = []

    # -- model -> harness ---------------------------------------------------

    def model_to_harness(self, model_text: str) -> "Tuple[dict, List[CallRecord]]":
        """Translate raw Hermes-format model text into an Anthropic-style
        assistant message ready for the harness. Returns the message and the
        records for this turn."""
        model_calls = hermes.parse_calls(model_text, idgen=self._idgen)
        records: "List[CallRecord]" = []
        for call in model_calls:
            if self.syntactic_only:
                harness_call, mapped = call, False
            else:
                canonical, lifted = self.model_profile.to_canonical(call)
                if lifted:
                    harness_call, mapped = self.harness_profile.from_canonical(canonical)
                else:
                    harness_call, mapped = call, False
                if not mapped:
                    self.warnings.append(
                        f"No surface mapping for tool '{call.name}' "
                        f"({self.model_profile.name} -> {self.harness_profile.name}); "
                        "passing through verbatim"
                    )
            record = CallRecord(model_call=call, harness_call=harness_call, mapped=mapped)
            self.ledger[call.id] = record
            records.append(record)
        message = anthropic.emit_calls([r.harness_call for r in records])
        return message, records

    # -- harness -> model ---------------------------------------------------

    def harness_to_model(self, results_message: "dict[str, Any]") -> str:
        """Translate an Anthropic-style tool_result message back into Hermes
        <tool_response> text, restoring the tool names the model used."""
        results = anthropic.parse_results(results_message)
        restored: "List[ToolResult]" = []
        for result in results:
            record = self.ledger.get(result.call_id)
            name = record.model_call.name if record else result.name
            restored.append(
                ToolResult(
                    call_id=result.call_id,
                    content=result.content,
                    is_error=result.is_error,
                    name=name,
                    origin=result.origin,
                    raw=result.raw,
                )
            )
        return hermes.emit_results(restored)
