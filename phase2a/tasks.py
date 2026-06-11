"""Phase 2a task suite: 24 file-ops tasks in 4 parameterized families.

Design rules:
  * every task seeds deterministically and checks programmatically;
  * no task prompt mentions tool names -- the system prompt advertises
    the harness surface, and whether the model adheres is the measurement;
  * checkers accept outcomes, not call sequences.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class LiveTask:
    name: str
    family: str
    prompt: str
    seed: Optional[Callable[[str], None]]
    check: Callable[[str, list], bool]


# Tools are advertised Hermes-style with the HARNESS surface (Claude Code
# names). Nothing hints at translation or at generic alternatives.
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


def _write(root: str, rel: str, content: str) -> None:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path) or root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def _read(root: str, rel: str) -> Optional[str]:
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _last_result_contains(transcript: list, needle: str) -> bool:
    from crossharness.encodings import anthropic

    for results_message in reversed(transcript):
        results = anthropic.parse_results(results_message)
        if not results:
            continue
        for result in results:
            if not result.is_error and needle in str(result.content):
                return True
    return False


TASKS: "List[LiveTask]" = []


# --- Family W: direct writes (6) -------------------------------------------

_W_SPECS = [
    ("w1_status", "status.txt", "Create a file status.txt containing exactly the word: ready", "ready"),
    ("w2_version", "VERSION", "Write the version string 2.0.0 to a file named VERSION.", "2.0.0"),
    ("w3_greeting", "greeting.txt", "Write a one-line friendly greeting to greeting.txt. Include the word welcome.", "welcome"),
    ("w4_marker", "build/marker.txt", "Create build/marker.txt containing the word: complete", "complete"),
    ("w5_csvrow", "row.csv", "Write a single CSV line with the values alpha,beta,gamma to row.csv.", "alpha,beta,gamma"),
    ("w6_json", "settings.json", 'Write a JSON object {"debug": true} to settings.json.', '"debug"'),
]

for name, filename, prompt, needle in _W_SPECS:
    def _check_w(root: str, transcript: list, _f=filename, _n=needle) -> bool:
        content = _read(root, _f)
        return content is not None and _n.lower() in content.lower()

    TASKS.append(LiveTask(name, "W", prompt, None, _check_w))


# --- Family E: read, extract, write (8) -------------------------------------

def _seed_json_port(root: str) -> None:
    _write(root, "config.json", json.dumps({"service": "gateway", "port": 8080, "tls": False}, indent=2))


def _seed_json_nested(root: str) -> None:
    _write(root, "app.json", json.dumps({"app": {"name": "relay", "owner": {"team": "infra-7"}}}, indent=2))


def _seed_csv(root: str) -> None:
    _write(root, "inventory.csv", "item,count\nbolts,42\nwashers,18\n")


def _seed_ini(root: str) -> None:
    _write(root, "server.conf", "host = 10.0.0.5\nport = 9000\nworkers = 4\n")


def _seed_log(root: str) -> None:
    _write(root, "run.log", "INFO boot ok\nERROR disk full on /dev/sda2\nINFO retry scheduled\n")


def _seed_md(root: str) -> None:
    _write(root, "spec.md", "# Carrier Pigeon Protocol\n\nA short specification.\n")


def _seed_env(root: str) -> None:
    _write(root, "deploy.env", "REGION=eu-west-1\nREPLICAS=6\n")


def _seed_txt_key(root: str) -> None:
    _write(root, "note.txt", "The access code for the vault is 7351, do not share it.\n")


_E_SPECS = [
    ("e1_json_port", _seed_json_port, "Read config.json and write just the port number to port.txt.", "port.txt", "8080"),
    ("e2_json_team", _seed_json_nested, "Read app.json and write the owner team name to team.txt.", "team.txt", "infra-7"),
    ("e3_csv_count", _seed_csv, "Read inventory.csv and write the count of bolts to bolts.txt.", "bolts.txt", "42"),
    ("e4_ini_host", _seed_ini, "Read server.conf and write the host value to host.txt.", "host.txt", "10.0.0.5"),
    ("e5_log_error", _seed_log, "Read run.log and write the full ERROR line to error.txt.", "error.txt", "disk full"),
    ("e6_md_title", _seed_md, "Read spec.md and write its top-level title text to title.txt.", "title.txt", "Carrier Pigeon Protocol"),
    ("e7_env_region", _seed_env, "Read deploy.env and write the REGION value to region.txt.", "region.txt", "eu-west-1"),
    ("e8_txt_code", _seed_txt_key, "Read note.txt and write the access code to code.txt.", "code.txt", "7351"),
]

for name, seed, prompt, out_file, needle in _E_SPECS:
    def _check_e(root: str, transcript: list, _f=out_file, _n=needle) -> bool:
        content = _read(root, _f)
        return content is not None and _n.lower() in content.lower()

    TASKS.append(LiveTask(name, "E", prompt, seed, _check_e))


# --- Family L: navigate, find, read (5) --------------------------------------

_L_SENTINELS = {
    "l1": "SENTINEL ALPHA: the docs file was truly read.",
    "l2": "SENTINEL BRAVO: yearly report contents.",
    "l3": "SENTINEL CHARLIE: the only python file.",
    "l4": "SENTINEL DELTA: deepest note found.",
    "l5": "SENTINEL ECHO: newest changelog entry.",
}


def _seed_l1(root: str) -> None:
    _write(root, "docs/readme.md", _L_SENTINELS["l1"])
    _write(root, "docs/data.bin", "not this one")


def _seed_l2(root: str) -> None:
    _write(root, "reports/2024.txt", "old report")
    _write(root, "reports/2025.txt", _L_SENTINELS["l2"])


def _seed_l3(root: str) -> None:
    _write(root, "src/util.js", "// js")
    _write(root, "src/main.py", _L_SENTINELS["l3"])
    _write(root, "src/style.css", "/* css */")


def _seed_l4(root: str) -> None:
    _write(root, "a/b/c/note.txt", _L_SENTINELS["l4"])
    _write(root, "a/decoy.txt", "decoy")


def _seed_l5(root: str) -> None:
    _write(root, "changes/CHANGELOG-v1.md", "v1 notes")
    _write(root, "changes/CHANGELOG-v2.md", _L_SENTINELS["l5"])


_L_SPECS = [
    ("l1_md_in_docs", _seed_l1, "Find the markdown file in the docs directory and read its contents.", "l1"),
    ("l2_latest_report", _seed_l2, "In the reports directory, read the most recent year's report.", "l2"),
    ("l3_python_file", _seed_l3, "Find the Python file in src and read it.", "l3"),
    ("l4_deep_note", _seed_l4, "There is a note.txt somewhere under the directory a. Find it and read it.", "l4"),
    ("l5_latest_changelog", _seed_l5, "Read the highest-version changelog in the changes directory.", "l5"),
]

for name, seed, prompt, key in _L_SPECS:
    def _check_l(root: str, transcript: list, _k=key) -> bool:
        return _last_result_contains(transcript, _L_SENTINELS[_k])

    TASKS.append(LiveTask(name, "L", prompt, seed, _check_l))


# --- Family M: multi-step (5) -------------------------------------------------

def _seed_m1(root: str) -> None:
    _write(root, "first.txt", "carrier")
    _write(root, "second.txt", "pigeon")


def _seed_m2(root: str) -> None:
    _write(root, "numbers.txt", "17\n25\n")


def _seed_m3(root: str) -> None:
    _write(root, "draft.txt", "the deadline is friday")


def _seed_m4(root: str) -> None:
    _write(root, "users.json", json.dumps([{"name": "mara", "active": True}, {"name": "jude", "active": False}]))


def _seed_m5(root: str) -> None:
    _write(root, "parts/header.txt", "BEGIN")
    _write(root, "parts/footer.txt", "END")


def _check_m1(root: str, transcript: list) -> bool:
    content = _read(root, "combined.txt")
    return content is not None and "carrier" in content and "pigeon" in content


def _check_m2(root: str, transcript: list) -> bool:
    content = _read(root, "sum.txt")
    return content is not None and "42" in content


def _check_m3(root: str, transcript: list) -> bool:
    content = _read(root, "final.txt")
    return content is not None and "DEADLINE IS FRIDAY" in content.upper() and "the deadline is friday" not in content


def _check_m4(root: str, transcript: list) -> bool:
    content = _read(root, "active.txt")
    return content is not None and "mara" in content.lower() and "jude" not in content.lower()


def _check_m5(root: str, transcript: list) -> bool:
    content = _read(root, "document.txt")
    if content is None:
        return False
    return content.find("BEGIN") != -1 and content.find("END") > content.find("BEGIN")


_M_SPECS = [
    ("m1_combine", _seed_m1, "Read first.txt and second.txt and write both words, space separated, to combined.txt.", _check_m1),
    ("m2_sum", _seed_m2, "Read numbers.txt, add the two numbers, and write the sum to sum.txt.", _check_m2),
    ("m3_uppercase", _seed_m3, "Read draft.txt and write the same sentence in all uppercase to final.txt.", _check_m3),
    ("m4_filter", _seed_m4, "Read users.json and write the names of only the active users to active.txt.", _check_m4),
    ("m5_assemble", _seed_m5, "Read parts/header.txt and parts/footer.txt and write a document.txt with the header text first and footer text last.", _check_m5),
]

for name, seed, prompt, check in _M_SPECS:
    TASKS.append(LiveTask(name, "M", prompt, seed, check))


assert len(TASKS) == 24, f"expected 24 tasks, have {len(TASKS)}"
