#!/usr/bin/env python3
"""CPU-only checks of allowlist.py on synthetic trace files.

    python3 -I tests/test_allowlist.py
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "metallib_slim"
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("allowlist", LIB / "allowlist.py")
A = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = A
_spec.loader.exec_module(A)

CLASSES = ["natural", "g13g", "g15s"]


def trace_text(names, summary=None, header=True) -> str:
    lines = [{"event": "header", "pid": 5, "variant": "default-library-names"}] if header else []
    lines += [{"event": "call", "kind": "default_kernel", "label": n} for n in names]
    lines.append({"event": "summary", "pid": 5, "lines": len(names), "dropped": 0, "write_failures": 0, "lookups": 99, "distinct": len(names)} | (summary or {}))
    return "".join(json.dumps(line) + "\n" for line in lines)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def trace(self, label, names, pid=5, text=None) -> Path:
        folder = self.root / "traces" / label
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"trace.{pid}.jsonl"
        path.write_text(text or trace_text(names))
        return path

    def test_union_of_every_label_with_the_trace_hashes_and_a_complete_flag(self):
        traces = {"natural": [self.trace("natural", ["a", "b"])], "g13g": [self.trace("g13g", ["b", "c"])], "g15s": [self.trace("g15s", ["d"])]}
        result = A.build(traces, CLASSES, "decl", False)
        self.assertEqual(result["names"], ["a", "b", "c", "d"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["pendingJobs"], [])
        self.assertEqual(result["jobs"]["g13g"]["distinctNames"], 2)
        self.assertEqual(result["jobs"]["g13g"]["traceFiles"][0]["sha256"], hashlib.sha256(traces["g13g"][0].read_bytes()).hexdigest())
        self.assertEqual(result["declarationSHA256"], "decl")
        self.assertEqual(result["namesSHA256"], hashlib.sha256(b"a\nb\nc\nd").hexdigest())

    def test_a_directory_holds_every_process_trace_of_a_label(self):
        self.trace("natural", ["a"], pid=1)
        self.trace("natural", ["b", "a"], pid=2)
        traces = A.parse_traces([f"natural={self.root / 'traces/natural'}"])
        self.assertEqual([p.name for p in traces["natural"]], ["trace.1.jsonl", "trace.2.jsonl"])
        result = A.build(traces, ["natural"], "decl", False)
        self.assertEqual(result["names"], ["a", "b"])
        self.assertEqual(len(result["jobs"]["natural"]["traceFiles"]), 2)
        (self.root / "empty").mkdir()
        for bad in ("natural", "=x", f"natural={self.root / 'empty'}", f"natural={self.root / 'missing.jsonl'}"):
            with self.subTest(bad), self.assertRaises(A.AllowlistError):
                A.parse_traces([bad])
        path = self.trace("g13g", ["a"])
        with self.assertRaises(A.AllowlistError):
            A.parse_traces([f"g13g={path}", f"g13g={path}"])

    def test_a_missing_label_is_an_error_unless_partial_is_allowed_and_then_it_is_pending(self):
        traces = {"natural": [self.trace("natural", ["a"])]}
        with self.assertRaises(A.AllowlistError):
            A.build(traces, CLASSES, "decl", False)
        partial = A.build(traces, CLASSES, "decl", True)
        self.assertFalse(partial["complete"])
        self.assertEqual(partial["pendingJobs"], ["g13g", "g15s"])

    def test_an_incomplete_or_malformed_trace_is_rejected_never_skipped(self):
        for label, text in {"dropped": trace_text(["a"], {"dropped": 2}), "write failure": trace_text(["a"], {"write_failures": 1}),
                            "no header": trace_text(["a"], header=False), "count mismatch": trace_text(["a"], {"lines": 5}),
                            "duplicate": trace_text(["a", "a"], {"lines": 2, "distinct": 1}),
                            "no summary": "".join(trace_text(["a"]).splitlines(True)[:-1]),
                            "wrong kind": trace_text(["a"]).replace("default_kernel", "pipeline")}.items():
            with self.subTest(label):
                records = [json.loads(line) for line in text.splitlines()]
                self.assertTrue(A.trace_names(records)[1], label)
                with self.assertRaises(A.AllowlistError):
                    A.build({"natural": [self.trace("natural", [], text=text)]}, ["natural"], "decl", False)
        with self.assertRaises(A.AllowlistError):
            A.build({"natural": [self.trace("natural", [], text="{not json\n")]}, ["natural"], "decl", False)
        with self.assertRaises(A.AllowlistError):
            A.build({"natural": [self.trace("natural", [])]}, ["natural"], "decl", False)  # no kernel name at all

    def test_a_base_allowlist_keeps_its_names_and_jobs_and_the_new_names_are_listed(self):
        base = A.build({"natural": [self.trace("natural", ["a", "b"])]}, ["natural"], "decl", False)
        extended = A.build({"g13g": [self.trace("g13g", ["b", "c"])]}, ["natural", "g13g"], "decl2", False, base)
        self.assertEqual(extended["names"], ["a", "b", "c"])
        self.assertEqual(sorted(extended["jobs"]), ["g13g", "natural"])
        self.assertEqual(extended["newNamesBeyondBase"], ["c"])
        self.assertEqual(extended["baseNamesSHA256"], base["namesSHA256"])
        self.assertEqual((extended["pendingJobs"], extended["complete"]), ([], True))  # a label of the base counts as traced
        self.assertEqual(A.build({"g13g": [self.trace("g13g", ["c"])]}, ["g15s"], "d", True, base)["pendingJobs"], ["g15s"])

    def test_freezing_writes_a_hash_that_verify_checks_and_a_frozen_file_is_not_rewritten(self):
        allowlist = {"schema": 1, "complete": True, "pendingJobs": [], "jobs": {}, "nameCount": 2, "names": ["a", "b"],
                     "namesSHA256": hashlib.sha256(b"a\nb").hexdigest()}
        out = self.root / "allowlist.json"
        digest = A.write(allowlist, out)
        self.assertEqual((self.root / "allowlist.sha256").read_text().strip(), digest)
        self.assertEqual(A.verify(out)["names"], ["a", "b"])
        out.write_text(out.read_text().replace('"a"', '"z"'))
        with self.assertRaises(A.AllowlistError):
            A.verify(out)

    def test_the_command_line_builds_verifies_and_never_rewrites(self):
        declaration = self.root / "declaration.json"
        declaration.write_text('{"reduction": {"baseFiles": []}}\n')
        self.trace("natural", ["a"])
        self.trace("g13g", ["b"])
        out = self.root / "allowlist.json"
        argv = ["build", "--declaration", str(declaration), "--out", str(out), "--trace", f"natural={self.root / 'traces/natural'}",
                "--trace", f"g13g={self.root / 'traces/g13g'}", "--expect", "natural", "--expect", "g13g"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(A.cli(argv), 0)
            self.assertEqual(A.cli(["verify", str(out)]), 0)
            self.assertEqual(A.cli(argv), 1)  # frozen
        allowlist = json.loads(out.read_text())
        self.assertEqual((allowlist["names"], allowlist["declarationSHA256"]), (["a", "b"], hashlib.sha256(declaration.read_bytes()).hexdigest()))


if __name__ == "__main__":
    unittest.main()
