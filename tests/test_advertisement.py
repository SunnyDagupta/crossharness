"""Tests for Phase F advertisement-axis additions."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2a.advertisement import (
    ADVERTISEMENT_PROMPTS,
    ADVERTISEMENT_TOOL_SETS,
    EXECUTOR_TOOLS,
    SYSTEM_PROMPT_NONE,
    SYSTEM_PROMPT_LARGE,
    ADVERTISED_TOOLS_FULL,
    ADVERTISED_TOOLS_NONE,
    ADVERTISED_TOOLS_LARGE,
    DISTRACTOR_TOOLS,
)
from phase2a.tasks import SYSTEM_PROMPT
from eval_live import classify_call, _call_classes
from analyze import aggregate, rule_of_three, two_proportion_se


# ---------------------------------------------------------------------------
# Advertisement prompt selection
# ---------------------------------------------------------------------------

def test_full_uses_tasks_system_prompt():
    assert ADVERTISEMENT_PROMPTS["full"] is None


def test_none_prompt_has_no_tool_schema():
    prompt = SYSTEM_PROMPT_NONE
    assert "<tools>" not in prompt
    assert "Read" not in prompt
    assert "Write" not in prompt
    assert "LS" not in prompt


def test_none_prompt_retains_wire_format_instruction():
    prompt = SYSTEM_PROMPT_NONE
    assert "<tool_call>" in prompt
    assert "function name" in prompt.lower()


def test_large_prompt_has_fifteen_tools():
    prompt = SYSTEM_PROMPT_LARGE
    # Count JSON objects with "name" in the tools block
    import json
    lines = [l.strip() for l in prompt.split("\n") if l.strip().startswith('{"name":')]
    assert len(lines) == 15


def test_large_prompt_real_tools_at_positions_4_9_14():
    import json
    lines = [l.strip() for l in SYSTEM_PROMPT_LARGE.split("\n") if l.strip().startswith('{"name":')]
    names = [json.loads(l)["name"] for l in lines]
    assert names[3] == "Read"
    assert names[8] == "Write"
    assert names[13] == "LS"


def test_advertised_sets_match_prompts():
    assert ADVERTISED_TOOLS_FULL == frozenset({"Read", "Write", "LS"})
    assert ADVERTISED_TOOLS_NONE == frozenset()
    assert "Read" in ADVERTISED_TOOLS_LARGE
    assert "web_search" in ADVERTISED_TOOLS_LARGE
    assert len(ADVERTISED_TOOLS_LARGE) == 15


def test_executor_tools_is_real_tools():
    assert EXECUTOR_TOOLS == frozenset({"Read", "Write", "LS"})


def test_distractor_tools_excludes_executor():
    assert DISTRACTOR_TOOLS.isdisjoint(EXECUTOR_TOOLS)
    assert len(DISTRACTOR_TOOLS) == 12


def test_advertisement_tool_sets_keyed_correctly():
    assert ADVERTISEMENT_TOOL_SETS["full"] == ADVERTISED_TOOLS_FULL
    assert ADVERTISEMENT_TOOL_SETS["none"] == ADVERTISED_TOOLS_NONE
    assert ADVERTISEMENT_TOOL_SETS["large"] == ADVERTISED_TOOLS_LARGE


# ---------------------------------------------------------------------------
# Distractor classification
# ---------------------------------------------------------------------------

def test_classify_harness_native():
    # "Read" maps on the Claude Code surface
    result = classify_call("Read", ADVERTISED_TOOLS_FULL)
    assert result == "harness_native"


def test_classify_advertised_distractor():
    # "web_search" is in ADVERTISED_TOOLS_LARGE but not in EXECUTOR_TOOLS
    result = classify_call("web_search", ADVERTISED_TOOLS_LARGE)
    assert result == "advertised_distractor"


def test_classify_mappable_generic():
    # "read_file" maps via HERMES_GENERIC but is not a Claude Code name and not advertised
    result = classify_call("read_file", ADVERTISED_TOOLS_NONE)
    assert result == "mappable_generic"


def test_classify_unmapped():
    result = classify_call("totally_unknown_tool_xyz", ADVERTISED_TOOLS_NONE)
    assert result == "unmapped"


def test_harness_native_beats_distractor():
    # "Read" is in ADVERTISED_TOOLS_LARGE but should still classify as harness_native
    result = classify_call("Read", ADVERTISED_TOOLS_LARGE)
    assert result == "harness_native"


def test_distractor_not_mapped_when_not_advertised():
    # "web_search" is a distractor only when advertised; with no advertisement it is unmapped
    result = classify_call("web_search", ADVERTISED_TOOLS_NONE)
    assert result == "unmapped"


# ---------------------------------------------------------------------------
# First-call vs post-first-call split in aggregate()
# ---------------------------------------------------------------------------

def _make_row(condition, passed, calls, advertisement="full", family="E"):
    return {
        "task": "test_task",
        "family": family,
        "condition": condition,
        "passed": passed,
        "turns": 2,
        "calls": calls,
        "malformed_turns": 0,
        "advertisement": advertisement,
    }


def test_first_call_split_basic():
    rows = [
        _make_row("full", True, [
            {"class": "harness_native", "turn": 1},
            {"class": "mappable_generic", "turn": 2},
        ]),
        _make_row("full", False, [
            {"class": "harness_native", "turn": 1},
        ]),
        _make_row("none", False, []),
        _make_row("syntactic", True, [{"class": "harness_native", "turn": 1}]),
    ]
    agg = aggregate(rows)
    fd = agg["full"]
    assert fd["first_call_total"] == 2
    assert fd["first_call_deviation"] == 0
    assert fd["post_call_total"] == 1
    assert fd["post_call_deviation"] == 1


def test_first_call_split_with_distractor():
    # distractor counts are excluded from deviation tallies
    rows = [
        _make_row("full", True, [
            {"class": "advertised_distractor", "turn": 1},
            {"class": "harness_native", "turn": 2},
        ]),
        _make_row("none", False, []),
        _make_row("syntactic", False, []),
    ]
    agg = aggregate(rows)
    fd = agg["full"]
    assert fd["calls_distractor"] == 1
    assert fd["calls_deviation"] == 0  # distractor excluded
    assert fd["first_call_deviation"] == 0  # distractor excluded
    assert fd["post_call_deviation"] == 0


def test_invocation_rate_computed():
    rows = [
        _make_row("full", True,  [{"class": "harness_native", "turn": 1}]),
        _make_row("full", False, []),  # no calls — not invoked
        _make_row("none", False, []),
        _make_row("syntactic", True, [{"class": "harness_native", "turn": 1}]),
    ]
    agg = aggregate(rows)
    assert agg["full"]["invoked"] == 1
    assert agg["full"]["n_invocable"] == 2
    assert agg["syntactic"]["invoked"] == 1
    assert agg["syntactic"]["n_invocable"] == 1


def test_backward_compat_old_string_calls():
    # Old JSONL format: calls is a list of strings, not dicts.
    # Strings are all assigned turn=1 by _call_records(), so all count as first-turn.
    rows = [
        _make_row("full", True,  ["harness_native", "harness_native"]),
        _make_row("none", False, []),
        _make_row("syntactic", True, ["harness_native"]),
    ]
    agg = aggregate(rows)
    assert agg["full"]["calls_native"] == 2
    assert agg["full"]["calls_deviation"] == 0
    # both string calls land on turn=1, so both are "first" calls
    assert agg["full"]["first_call_total"] == 2
    assert agg["full"]["post_call_total"] == 0


def test_call_classes_helper_normalizes():
    row_str  = {"calls": ["harness_native", "mappable_generic"]}
    row_dict = {"calls": [{"class": "harness_native", "turn": 1}, {"class": "mappable_generic", "turn": 2}]}
    assert _call_classes(row_str) == ["harness_native", "mappable_generic"]
    assert _call_classes(row_dict) == ["harness_native", "mappable_generic"]


# ---------------------------------------------------------------------------
# Math helpers (regression guard)
# ---------------------------------------------------------------------------

def test_rule_of_three_zero_events():
    assert abs(rule_of_three(100) - 0.03) < 1e-9


def test_two_proportion_se_symmetric():
    se1 = two_proportion_se(0.5, 50, 0.7, 50)
    se2 = two_proportion_se(0.7, 50, 0.5, 50)
    assert abs(se1 - se2) < 1e-9
