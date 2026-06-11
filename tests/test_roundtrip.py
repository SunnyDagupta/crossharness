"""Round-trip and behavioral tests for the translation core.

Run from the repo root:

    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crossharness import (
    CLAUDE_CODE,
    HERMES_GENERIC,
    ParseError,
    SandboxExecutor,
    Shim,
    sequential_id_generator,
)
from crossharness.encodings import anthropic, hermes


HERMES_TURN = """\
Reading the config now.
<tool_call>
{"name": "read_file", "arguments": {"path": "config.json"}}
</tool_call>
And listing the docs directory.
<tool_call>
{"name": "list_directory", "arguments": {"path": "docs"}}
</tool_call>"""


class TestHermesEncoding(unittest.TestCase):
    def test_parses_calls_with_surrounding_prose(self):
        calls = hermes.parse_calls(HERMES_TURN)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(calls[0].args, {"path": "config.json"})
        self.assertEqual(calls[1].name, "list_directory")

    def test_id_synthesis_is_deterministic(self):
        a = hermes.parse_calls(HERMES_TURN)
        b = hermes.parse_calls(HERMES_TURN)
        self.assertEqual([c.id for c in a], ["call_001", "call_002"])
        self.assertEqual([c.id for c in a], [c.id for c in b])
        self.assertTrue(all(c.id_synthesized for c in a))

    def test_shared_idgen_does_not_collide_across_turns(self):
        gen = sequential_id_generator()
        first = hermes.parse_calls(HERMES_TURN, idgen=gen)
        second = hermes.parse_calls(HERMES_TURN, idgen=gen)
        ids = [c.id for c in first + second]
        self.assertEqual(len(ids), len(set(ids)))

    def test_tolerates_double_encoded_arguments(self):
        text = '<tool_call>{"name": "read_file", "arguments": "{\\"path\\": \\"a.txt\\"}"}</tool_call>'
        calls = hermes.parse_calls(text)
        self.assertEqual(calls[0].args, {"path": "a.txt"})

    def test_malformed_json_raises_parse_error(self):
        with self.assertRaises(ParseError):
            hermes.parse_calls("<tool_call>{not json}</tool_call>")

    def test_no_calls_in_plain_prose(self):
        self.assertEqual(hermes.parse_calls("Just thinking out loud here."), [])


class TestRoundTrip(unittest.TestCase):
    def test_hermes_to_anthropic_to_hermes_preserves_arguments(self):
        calls = hermes.parse_calls(HERMES_TURN)
        message = anthropic.emit_calls(calls)
        reparsed = anthropic.parse_calls(message)
        self.assertEqual(
            [(c.name, c.args) for c in calls],
            [(c.name, c.args) for c in reparsed],
        )
        text = hermes.emit_calls(reparsed)
        final = hermes.parse_calls(text)
        self.assertEqual(
            [(c.name, c.args) for c in calls],
            [(c.name, c.args) for c in final],
        )

    def test_result_id_correlation_survives_round_trip(self):
        calls = hermes.parse_calls(HERMES_TURN)
        message = anthropic.emit_calls(calls)
        parsed = anthropic.parse_calls(message)
        from crossharness.ir import ToolResult

        results = [
            ToolResult(call_id=c.id, content=f"result for {c.name}", name=c.name)
            for c in parsed
        ]
        results_message = anthropic.emit_results(results)
        reparsed = anthropic.parse_results(results_message)
        self.assertEqual([r.call_id for r in reparsed], [c.id for c in calls])


class TestSurfaceMapping(unittest.TestCase):
    def test_generic_to_claude_code(self):
        call = hermes.parse_calls(
            '<tool_call>{"name": "read_file", "arguments": {"path": "x.txt"}}</tool_call>'
        )[0]
        canonical, lifted = HERMES_GENERIC.to_canonical(call)
        self.assertTrue(lifted)
        self.assertEqual(canonical.canonical, "fs.read")
        surfaced, mapped = CLAUDE_CODE.from_canonical(canonical)
        self.assertTrue(mapped)
        self.assertEqual(surfaced.name, "Read")
        self.assertEqual(surfaced.args, {"file_path": "x.txt"})

    def test_unknown_tool_passes_through_unmapped(self):
        call = hermes.parse_calls(
            '<tool_call>{"name": "send_email", "arguments": {"to": "a@b.c"}}</tool_call>'
        )[0]
        passed, lifted = HERMES_GENERIC.to_canonical(call)
        self.assertFalse(lifted)
        self.assertEqual(passed.name, "send_email")
        self.assertEqual(passed.args, {"to": "a@b.c"})

    def test_unknown_argument_passes_through_with_known_renamed(self):
        call = hermes.parse_calls(
            '<tool_call>{"name": "read_file", "arguments": {"path": "x", "encoding": "utf-8"}}</tool_call>'
        )[0]
        canonical, _ = HERMES_GENERIC.to_canonical(call)
        surfaced, _ = CLAUDE_CODE.from_canonical(canonical)
        self.assertEqual(surfaced.args, {"file_path": "x", "encoding": "utf-8"})

    def test_aliases_resolve_to_primary_surface(self):
        for alias, expected_target in (
            ("cat", "Read"),
            ("save_file", "Write"),
            ("ls", "LS"),
            ("list_dir", "LS"),
        ):
            call = hermes.parse_calls(
                f'<tool_call>{{"name": "{alias}", "arguments": {{"path": "x"}}}}</tool_call>'
            )[0]
            canonical, lifted = HERMES_GENERIC.to_canonical(call)
            self.assertTrue(lifted, f"alias {alias} did not lift")
            surfaced, mapped = CLAUDE_CODE.from_canonical(canonical)
            self.assertTrue(mapped)
            self.assertEqual(surfaced.name, expected_target)

    def test_alternate_argument_spellings_normalize(self):
        call = hermes.parse_calls(
            '<tool_call>{"name": "write_file", "arguments": {"filename": "a.txt", "text": "hi"}}</tool_call>'
        )[0]
        canonical, _ = HERMES_GENERIC.to_canonical(call)
        surfaced, _ = CLAUDE_CODE.from_canonical(canonical)
        self.assertEqual(surfaced.name, "Write")
        self.assertEqual(surfaced.args, {"file_path": "a.txt", "content": "hi"})


class TestShim(unittest.TestCase):
    def test_full_shim_maps_surface(self):
        shim = Shim(HERMES_GENERIC, CLAUDE_CODE)
        message, records = shim.model_to_harness(
            '<tool_call>{"name": "write_file", "arguments": {"path": "n.txt", "content": "hi"}}</tool_call>'
        )
        block = message["content"][0]
        self.assertEqual(block["name"], "Write")
        self.assertEqual(block["input"], {"file_path": "n.txt", "content": "hi"})
        self.assertTrue(records[0].mapped)

    def test_syntactic_only_passes_names_verbatim(self):
        shim = Shim(HERMES_GENERIC, CLAUDE_CODE, syntactic_only=True)
        message, records = shim.model_to_harness(
            '<tool_call>{"name": "write_file", "arguments": {"path": "n.txt", "content": "hi"}}</tool_call>'
        )
        self.assertEqual(message["content"][0]["name"], "write_file")
        self.assertFalse(records[0].mapped)

    def test_results_restore_the_models_surface_name(self):
        shim = Shim(HERMES_GENERIC, CLAUDE_CODE)
        _, records = shim.model_to_harness(
            '<tool_call>{"name": "read_file", "arguments": {"path": "x.txt"}}</tool_call>'
        )
        results_message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": records[0].model_call.id,
                    "content": "file body",
                }
            ],
        }
        text = shim.harness_to_model(results_message)
        self.assertIn('"name": "read_file"', text)
        self.assertIn("<tool_response>", text)
        self.assertIn("file body", text)

    def test_unmappable_tool_warns_and_passes_through(self):
        shim = Shim(HERMES_GENERIC, CLAUDE_CODE)
        message, records = shim.model_to_harness(
            '<tool_call>{"name": "send_email", "arguments": {"to": "a@b.c"}}</tool_call>'
        )
        self.assertEqual(message["content"][0]["name"], "send_email")
        self.assertFalse(records[0].mapped)
        self.assertTrue(shim.warnings)


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.executor = SandboxExecutor(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _execute_single(self, name, args):
        message = {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": name, "input": args}],
        }
        results = anthropic.parse_results(self.executor.execute_message(message))
        return results[0]

    def test_write_then_read(self):
        write = self._execute_single("Write", {"file_path": "a.txt", "content": "hello"})
        self.assertFalse(write.is_error)
        read = self._execute_single("Read", {"file_path": "a.txt"})
        self.assertFalse(read.is_error)
        self.assertEqual(read.content, "hello")

    def test_unknown_tool_is_strict_error(self):
        result = self._execute_single("read_file", {"path": "a.txt"})
        self.assertTrue(result.is_error)
        self.assertIn("Unknown tool", str(result.content))

    def test_missing_argument_is_schema_error(self):
        result = self._execute_single("Read", {"path": "a.txt"})
        self.assertTrue(result.is_error)
        self.assertIn("missing file_path", str(result.content))

    def test_path_escape_is_rejected(self):
        result = self._execute_single("Read", {"file_path": "../../etc/hosts"})
        self.assertTrue(result.is_error)
        self.assertIn("escapes sandbox", str(result.content))

    def test_plain_text_message_yields_no_results(self):
        message = {"role": "assistant", "content": [{"type": "text", "text": "<tool_call>...</tool_call>"}]}
        results = anthropic.parse_results(self.executor.execute_message(message))
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
