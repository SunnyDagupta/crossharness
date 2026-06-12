#!/usr/bin/env python3
"""Phase 2a live runner: measure dialect failure modes on a real model.

    export OPENROUTER_API_KEY=...          # or any OpenAI-compatible endpoint
    python3 eval_live.py --model nousresearch/hermes-3-llama-3.1-8b
    python3 eval_live.py --advertisement none    # cell A: no tool advertisement
    python3 eval_live.py --advertisement large   # cell B: 15-tool catalog
    python3 eval_live.py --smoke                 # no network, validates pipeline

Drives a live Hermes-format model through the 24-task suite under three
conditions (none / syntactic / full), classifies every emitted call, and
reports the recovery decomposition defined in docs/phase2a-protocol.md.

Stdlib only. Honors proxy environment variables via urllib defaults.

Deployment notes
----------------
Ollama (tested 0.30.6): the /v1/chat/completions endpoint hangs
indefinitely without ``"stream": false`` in the request body. This file
always sends that flag. Switch to the native /api/chat endpoint if you
prefer to drop the OpenAI-compat layer entirely.

Ollama evicts models from memory after ~5 minutes of inactivity. The
client sends keep_alive=-1 for local endpoints so the model stays loaded
between cells. Override with OLLAMA_KEEP_ALIVE env var or --no-keep-alive.
The client retries transport errors (TimeoutError, OSError) up to three
times with 10/20 s backoff.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crossharness import CLAUDE_CODE, HERMES_GENERIC, ParseError, SandboxExecutor, Shim
from crossharness.encodings import anthropic, hermes
from phase2a.tasks import SYSTEM_PROMPT, TASKS, LiveTask
from phase2a.advertisement import (
    ADVERTISEMENT_PROMPTS,
    ADVERTISEMENT_TOOL_SETS,
    EXECUTOR_TOOLS,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.join(HERE, "sandbox_live")
RESULTS_DIR = os.path.join(HERE, "results")

CONDITIONS = ("none", "syntactic", "full")


# ---------------------------------------------------------------------------
# Model clients
# ---------------------------------------------------------------------------

class OpenAICompatClient:
    """Minimal chat-completions client over urllib."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float,
        keep_alive: "Optional[int]" = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.keep_alive = keep_alive

    def chat(self, messages: "List[dict]") -> str:
        payload: "Dict[str, Any]" = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 768,
            "stream": False,
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503) and attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise
            except (TimeoutError, OSError, urllib.error.URLError) as exc:
                if attempt < 2:
                    time.sleep(10.0 * (attempt + 1))
                    continue
                raise
        return ""


class ScriptedClient:
    """Deterministic stand-in for --smoke: validates the whole pipeline
    (loop, translation, execution, feedback, classification, reporting)
    with no network. Emits generic-surface calls for the first E task."""

    def __init__(self):
        self.turn = 0

    def chat(self, messages: "List[dict]") -> str:
        self.turn += 1
        if self.turn == 1:
            return (
                "Reading the config first.\n"
                '<tool_call>\n{"name": "read_file", "arguments": {"path": "config.json"}}\n</tool_call>'
            )
        if self.turn == 2:
            return (
                "Got it, the port is 8080.\n"
                '<tool_call>\n{"name": "write_file", "arguments": {"path": "port.txt", "content": "8080"}}\n</tool_call>'
            )
        return "The task is complete."


# ---------------------------------------------------------------------------
# Call classification
# ---------------------------------------------------------------------------

def classify_call(name: str, advertised_set: "frozenset[str]") -> str:
    """Classify a single tool call by surface type.

    Buckets (in priority order):
      harness_native      -- name resolves on the Claude Code surface
      advertised_distractor -- in advertised_set but not in the executor
                              (semantic confusion; out of shim scope)
      mappable_generic    -- name resolves via the generic Hermes profile
      unmapped            -- no mapping found
    """
    if CLAUDE_CODE.spec_for_surface(name) is not None:
        return "harness_native"
    if name in advertised_set and name not in EXECUTOR_TOOLS:
        return "advertised_distractor"
    if HERMES_GENERIC.spec_for_surface(name) is not None:
        return "mappable_generic"
    return "unmapped"


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------

def run_sample(
    task: LiveTask,
    condition: str,
    client,
    max_turns: int,
    tool_role: str,
    system_prompt: str,
    advertised_set: "frozenset[str]",
) -> "Dict[str, Any]":
    if os.path.isdir(SANDBOX):
        shutil.rmtree(SANDBOX)
    executor = SandboxExecutor(SANDBOX)
    if task.seed:
        task.seed(SANDBOX)

    shim = Shim(HERMES_GENERIC, CLAUDE_CODE, syntactic_only=(condition == "syntactic"))
    messages: "List[dict]" = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.prompt},
    ]
    transcript: "List[dict]" = []
    call_records: "List[Dict[str, Any]]" = []
    conversation: "List[Dict[str, Any]]" = []  # full turn-by-turn text for analysis
    malformed_turns = 0
    turns_used = 0

    for _ in range(max_turns):
        turns_used += 1
        model_text = client.chat(messages)
        messages.append({"role": "assistant", "content": model_text})
        conversation.append({"turn": turns_used, "role": "model", "text": model_text})

        if condition == "none":
            # Raw text to a structured harness: nothing is recognized.
            # Classification still runs so deviation is observable here too.
            try:
                for call in hermes.parse_calls(model_text):
                    call_records.append({
                        "class": classify_call(call.name, advertised_set),
                        "turn": turns_used,
                    })
            except ParseError:
                malformed_turns += 1
            break

        try:
            harness_message, records = shim.model_to_harness(model_text)
        except ParseError:
            malformed_turns += 1
            break  # a real adapter failing to parse delivers nothing

        if not records:
            break  # model emitted no tool call; it believes it is done

        for record in records:
            call_records.append({
                "class": classify_call(record.model_call.name, advertised_set),
                "turn": turns_used,
            })

        results_message = executor.execute_message(harness_message)
        transcript.append(results_message)
        response_text = shim.harness_to_model(results_message)
        messages.append({"role": tool_role, "content": response_text})
        conversation.append({"turn": turns_used, "role": "tool", "text": response_text})

        if task.check(SANDBOX, transcript):
            break  # early success; do not burn turns

    passed = task.check(SANDBOX, transcript)
    return {
        "task": task.name,
        "family": task.family,
        "condition": condition,
        "passed": passed,
        "turns": turns_used,
        "calls": call_records,
        "malformed_turns": malformed_turns,
        "conversation": conversation,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _call_classes(row: dict) -> "List[str]":
    """Normalize old list[str] and new list[dict] call formats."""
    return [
        c if isinstance(c, str) else c["class"]
        for c in row.get("calls", [])
    ]


def aggregate(rows: "List[dict]") -> None:
    by_condition: "Dict[str, List[dict]]" = {c: [] for c in CONDITIONS}
    for row in rows:
        by_condition[row["condition"]].append(row)

    print()
    print("phase 2a live results")
    print("-" * 60)
    rates: "Dict[str, float]" = {}
    for condition in CONDITIONS:
        cond_rows = by_condition[condition]
        if not cond_rows:
            continue
        passed = sum(r["passed"] for r in cond_rows)
        rates[condition] = passed / len(cond_rows)
        print(f"{condition:<12} task success {passed}/{len(cond_rows)} ({rates[condition]:.0%})")

    full_calls = [c for r in by_condition["full"] for c in _call_classes(r)]
    if full_calls:
        from collections import Counter
        cc = Counter(full_calls)
        deviating = sum(v for k, v in cc.items() if k != "harness_native")
        print()
        print(f"calls observed under full condition: {len(full_calls)}")
        for bucket in ("harness_native", "mappable_generic", "advertised_distractor", "unmapped"):
            if cc[bucket]:
                print(f"  {bucket:<24} {cc[bucket]}")
        print(f"  deviation rate:   {deviating / len(full_calls):.0%}")

    if all(c in rates for c in CONDITIONS):
        print()
        print(f"wire-format recovery (syntactic - none): {rates['syntactic'] - rates['none']:+.0%}")
        print(f"surface recovery (full - syntactic):     {rates['full'] - rates['syntactic']:+.0%}")
        print()
        print("Interpretation guide: docs/phase2a-protocol.md")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_COMPAT_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_COMPAT_MODEL", "nousresearch/hermes-3-llama-3.1-8b"))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--tasks", default="", help="comma-separated task name filter")
    parser.add_argument("--tool-role", choices=("user", "tool"), default="user",
                        help="chat role used to deliver tool responses (default: user, most provider-compatible)")
    parser.add_argument("--advertisement", choices=("full", "none", "large"), default="full",
                        help="tool advertisement cell: full (default), none, or large")
    parser.add_argument("--no-keep-alive", action="store_true",
                        help="disable keep_alive=-1 for local Ollama endpoints")
    parser.add_argument("--smoke", action="store_true", help="no-network pipeline validation")
    args = parser.parse_args()

    conditions = [c for c in args.conditions.split(",") if c in CONDITIONS]
    tasks = TASKS
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",")}
        tasks = [t for t in TASKS if t.name in wanted]

    adv_prompt_override = ADVERTISEMENT_PROMPTS[args.advertisement]
    system_prompt = adv_prompt_override if adv_prompt_override is not None else SYSTEM_PROMPT
    advertised_set = ADVERTISEMENT_TOOL_SETS[args.advertisement]

    rows: "List[dict]" = []

    if args.smoke:
        task = next(t for t in TASKS if t.name == "e1_json_port")
        row = run_sample(
            task, "full", ScriptedClient(), args.max_turns, args.tool_role,
            system_prompt, advertised_set,
        )
        rows.append(row)
        print(json.dumps(row, indent=2))
        smoke_classes = [c["class"] if isinstance(c, dict) else c for c in row["calls"]]
        ok = row["passed"] and smoke_classes == ["mappable_generic", "mappable_generic"]
        print("smoke:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_COMPAT_API_KEY", "")
    if not api_key and "localhost" not in args.base_url and "127.0.0.1" not in args.base_url:
        print("No API key found (OPENROUTER_API_KEY / OPENAI_COMPAT_API_KEY).", file=sys.stderr)
        sys.exit(2)

    is_local = "localhost" in args.base_url or "127.0.0.1" in args.base_url
    env_keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE")
    if args.no_keep_alive:
        keep_alive: "Optional[int]" = None
    elif env_keep_alive is not None:
        keep_alive = int(env_keep_alive)
    elif is_local:
        keep_alive = -1
    else:
        keep_alive = None

    client = OpenAICompatClient(args.base_url, args.model, api_key, args.temperature,
                                keep_alive=keep_alive)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"live_{stamp}.jsonl")

    total = len(tasks) * len(conditions) * args.samples
    done = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for task in tasks:
            for condition in conditions:
                for _ in range(args.samples):
                    row = run_sample(
                        task, condition, client, args.max_turns, args.tool_role,
                        system_prompt, advertised_set,
                    )
                    row["model"] = args.model
                    row["advertisement"] = args.advertisement
                    rows.append(row)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    done += 1
                    status = "PASS" if row["passed"] else "FAIL"
                    print(f"[{done}/{total}] {task.name:<20} {condition:<10} {status}")

    print(f"\nwrote {out_path}")
    aggregate(rows)


if __name__ == "__main__":
    main()
