#!/usr/bin/env python3
"""Checks of the `metallib-slim` entry point: dispatch, help and exit codes (no Metal, no Xcode).

    python3 -I tests/test_cli.py
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRY = ROOT / "metallib-slim"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", "-B", str(ENTRY), *args], capture_output=True, text=True)


class EntryPointTests(unittest.TestCase):
    def test_help_lists_every_command_and_no_command_is_a_usage_error(self):
        result = run("--help")
        self.assertEqual(result.returncode, 0)
        for command in ("prepare-source", "prepare-trace-source", "allowlist build", "allowlist verify", "record", "scan", "reduce", "names",
                        "verify", "repro", "check-macho"):
            self.assertIn(command, result.stdout)
        self.assertEqual(run().returncode, 2)
        self.assertEqual(run("no-such-command").returncode, 2)

    def test_every_subcommand_has_help_named_after_the_entry_point(self):
        for args, usage in ((["prepare-source"], "metallib-slim prepare-source"), (["prepare-trace-source"], "metallib-slim prepare-trace-source"),
                            (["allowlist", "build"], "metallib-slim allowlist build"), (["record"], "metallib-slim record"),
                            (["scan"], "metallib-slim scan"), (["reduce"], "metallib-slim reduce"), (["verify"], "metallib-slim verify"),
                            (["repro"], "metallib-slim repro")):
            with self.subTest(args):
                result = run(*args, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(usage, result.stdout)
        self.assertEqual(run("names", "--help").returncode, 0)
        self.assertEqual(run("names").returncode, 2)
        self.assertEqual(run("check-macho").returncode, 2)

    def test_errors_are_reported_on_stderr_with_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            allowlist = Path(tmp) / "allowlist.json"
            names = ["a", "b"]
            text = json.dumps({"schema": 1, "complete": True, "nameCount": 2, "names": names,
                               "namesSHA256": hashlib.sha256("\n".join(names).encode()).hexdigest()}, indent=1, sort_keys=True) + "\n"
            allowlist.write_text(text)
            result = run("allowlist", "verify", str(allowlist))
            self.assertEqual(result.returncode, 1)
            self.assertIn("not frozen", result.stderr)
            allowlist.with_suffix(".sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
            result = run("allowlist", "verify", str(allowlist))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("2 names", result.stdout)
            result = run("scan", "--build-log", str(Path(tmp) / "missing.log"), "--core", tmp, "--declaration", str(allowlist), "--out", tmp)
            self.assertEqual(result.returncode, 1)
            self.assertIn("reduce_library:", result.stderr)


if __name__ == "__main__":
    unittest.main()
