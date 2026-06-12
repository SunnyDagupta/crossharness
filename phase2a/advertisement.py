"""Advertisement-axis prompt variants and advertised-tool sets.

Three advertisement conditions (constants held: model, tasks, temperature,
max-turns, samples):

  full  -- 3 real tools advertised in the trained Hermes format (cell 1)
  none  -- wire-format instruction only; no tool names or schemas
  large -- same preamble; 15 tools advertised; real tools at positions 4/9/14

Alias table is frozen for the entire axis (see docs/advertisement-axis-protocol.md).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Cell A -- no advertisement
# ---------------------------------------------------------------------------
# Keeps the wire-format instruction so the cell isolates surface knowledge.
# Total silence about tool calling would conflate format discovery with
# invocation willingness (two variables at once).

SYSTEM_PROMPT_NONE = """\
You are a function calling AI model working against files in a working
directory. You may call one or more functions to assist with the user
query. For each function call return a json object with function name
and arguments within <tool_call></tool_call> XML tags, like:
<tool_call>
{"name": "<function-name>", "arguments": <args-json-object>}
</tool_call>
Call whatever functions are appropriate for the task."""

# ---------------------------------------------------------------------------
# Cell B -- large catalog (15 tools; real tools at positions 4, 9, 14)
# ---------------------------------------------------------------------------
# Preamble is byte-identical to SYSTEM_PROMPT in phase2a/tasks.py.
# Distractors are advertised but not implemented; executor stays strict.

SYSTEM_PROMPT_LARGE = """\
You are a function calling AI model. You are provided with function
signatures within <tools></tools> XML tags. You may call one or more
functions to assist with the user query. For each function call return a
json object with function name and arguments within <tool_call></tool_call>
XML tags.

<tools>
{"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
{"name": "exec_command", "description": "Execute a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}
{"name": "http_request", "description": "Make an HTTP request", "parameters": {"type": "object", "properties": {"method": {"type": "string"}, "url": {"type": "string"}}, "required": ["method", "url"]}}
{"name": "Read", "description": "Read a file from the working directory", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}
{"name": "message_send", "description": "Send a message to a channel", "parameters": {"type": "object", "properties": {"channel": {"type": "string"}, "text": {"type": "string"}}, "required": ["channel", "text"]}}
{"name": "calendar_create", "description": "Create a calendar event", "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "start": {"type": "string"}}, "required": ["title", "start"]}}
{"name": "image_generate", "description": "Generate an image from a text prompt", "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}}
{"name": "db_query", "description": "Execute a database query", "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}
{"name": "Write", "description": "Write content to a file in the working directory", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}}
{"name": "sessions_list", "description": "List active sessions", "parameters": {"type": "object", "properties": {}, "required": []}}
{"name": "cron_schedule", "description": "Schedule a cron job", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}, "task": {"type": "string"}}, "required": ["expression", "task"]}}
{"name": "email_send", "description": "Send an email", "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}
{"name": "screenshot_capture", "description": "Capture a screenshot of the screen", "parameters": {"type": "object", "properties": {}, "required": []}}
{"name": "LS", "description": "List a directory in the working directory", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
{"name": "browser_open", "description": "Open a URL in the browser", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}
</tools>"""

# ---------------------------------------------------------------------------
# Advertised-tool sets (frozen for the axis; used by classifier)
# ---------------------------------------------------------------------------

# Names explicitly mentioned in each cell's system prompt.
ADVERTISED_TOOLS_FULL: "frozenset[str]" = frozenset({"Read", "Write", "LS"})

ADVERTISED_TOOLS_NONE: "frozenset[str]" = frozenset()

ADVERTISED_TOOLS_LARGE: "frozenset[str]" = frozenset({
    "web_search", "exec_command", "http_request",
    "Read",
    "message_send", "calendar_create", "image_generate", "db_query",
    "Write",
    "sessions_list", "cron_schedule", "email_send", "screenshot_capture",
    "LS",
    "browser_open",
})

# Tools the executor actually implements (Read/Write/LS strict surface).
EXECUTOR_TOOLS: "frozenset[str]" = frozenset({"Read", "Write", "LS"})

# Distractors: advertised in large cell but not implemented.
DISTRACTOR_TOOLS: "frozenset[str]" = ADVERTISED_TOOLS_LARGE - EXECUTOR_TOOLS

ADVERTISEMENT_PROMPTS = {
    "full": None,       # sentinel: use SYSTEM_PROMPT from tasks.py
    "none": SYSTEM_PROMPT_NONE,
    "large": SYSTEM_PROMPT_LARGE,
}

ADVERTISEMENT_TOOL_SETS = {
    "full": ADVERTISED_TOOLS_FULL,
    "none": ADVERTISED_TOOLS_NONE,
    "large": ADVERTISED_TOOLS_LARGE,
}
