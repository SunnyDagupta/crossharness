#!/usr/bin/env python3
"""Regenerate fixtures from a live OpenAI-compatible endpoint (optional).

The shipped fixtures are constructed by hand to represent documented
failure modes deterministically. This script exists so they can also be
captured from a real Hermes-format model, which is the honest path to
Phase 2 numbers: same tasks, live outputs, measured frequencies.

Usage:
    export OPENAI_COMPAT_BASE_URL=http://localhost:8000/v1   # e.g. vLLM
    export OPENAI_COMPAT_MODEL=NousResearch/Hermes-3-Llama-3.1-8B
    export OPENAI_COMPAT_API_KEY=...                          # if required
    python3 fixtures/capture.py

Writes fixtures/captured_task*.txt next to the constructed ones. Inspect
before use; live captures replace nothing automatically.

Stdlib only (urllib). Honors HTTPS_PROXY et al. via urllib defaults.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

BASE_URL = os.environ.get("OPENAI_COMPAT_BASE_URL", "").rstrip("/")
MODEL = os.environ.get("OPENAI_COMPAT_MODEL", "")
API_KEY = os.environ.get("OPENAI_COMPAT_API_KEY", "")

HERE = os.path.dirname(os.path.abspath(__file__))

# Tools are advertised Hermes-style: JSON schemas in the system prompt,
# calls expected back inside <tool_call> tags. The advertised surface here
# is the HARNESS surface (Claude Code names) -- whether the model adheres
# to it or falls back to its trained generic surface is exactly the
# behavior worth capturing.
SYSTEM_PROMPT = """\
You are a function calling AI model. You are provided with function
signatures within <tools></tools> XML tags. You may call one or more
functions to assist with the user query. For each function call return a
json object with function name and arguments within <tool_call></tool_call>
XML tags.

<tools>
{"name": "Read", "description": "Read a file from the working directory", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}
{"name": "Write", "description": "Write content to a file in the working directory", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}}
{"name": "LS", "description": "List a directory in the working directory", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
</tools>"""

TASK_PROMPTS = {
    "captured_task1_write_haiku.txt": [
        "Write a haiku about translation between languages to the file haiku.txt."
    ],
    "captured_task2_config_extract.txt": [
        "Read config.json and then write just the port number to out.txt.",
    ],
    "captured_task3_find_and_read.txt": [
        "Find the markdown file in the docs directory and read its contents.",
    ],
}


def chat(messages: "list[dict]") -> str:
    body = json.dumps(
        {"model": MODEL, "messages": messages, "temperature": 0.2, "max_tokens": 512}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"] or ""


def main() -> None:
    if not BASE_URL or not MODEL:
        print(
            "Set OPENAI_COMPAT_BASE_URL and OPENAI_COMPAT_MODEL to capture "
            "fixtures from a live endpoint.",
            file=sys.stderr,
        )
        sys.exit(2)

    for filename, user_turns in TASK_PROMPTS.items():
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        captured_turns = []
        for user_turn in user_turns:
            messages.append({"role": "user", "content": user_turn})
            assistant_text = chat(messages)
            captured_turns.append(assistant_text.strip())
            messages.append({"role": "assistant", "content": assistant_text})
        out_path = os.path.join(HERE, filename)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Captured live from {MODEL} via {BASE_URL}\n")
            fh.write("\n---TURN---\n".join(captured_turns) + "\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
