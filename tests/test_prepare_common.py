#!/usr/bin/env python3
"""CPU-only checks of prepare_common.py: zero-fuzz edits, tree digests, the clone guard, the stock-checkout check (on a fabricated git checkout)
and the expected-patch rule.

    python3 -I tests/test_prepare_common.py
"""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "metallib_slim"
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("prepare_common", LIB / "prepare_common.py")
P = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = P
_spec.loader.exec_module(P)

EDITS = [("one", "alpha\n", "alpha\nhooked one\n"), ("two", "beta\n", "hooked two\nbeta\n")]


def run_git(path: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


class ApplyEditsTests(unittest.TestCase):
    def test_every_anchor_applies_once_and_reports_it(self):
        result, report = P.apply_edits("alpha\nbeta\n", EDITS)
        self.assertEqual(result, "alpha\nhooked one\nhooked two\nbeta\n")
        self.assertEqual(report, [{"edit": "one", "anchorOccurrences": 1}, {"edit": "two", "anchorOccurrences": 1}])

    def test_a_missing_or_duplicated_anchor_produces_nothing(self):
        with self.assertRaisesRegex(P.PrepareError, "'two'.*0 times"):
            P.apply_edits("alpha\n", EDITS)
        with self.assertRaisesRegex(P.PrepareError, "'one'.*2 times"):
            P.apply_edits("alpha\nalpha\nbeta\n", EDITS)

    def test_the_edit_digest_follows_the_edits(self):
        self.assertEqual(P.edits_digest(EDITS), P.edits_digest([list(e) for e in EDITS]))
        self.assertNotEqual(P.edits_digest(EDITS), P.edits_digest(EDITS[:1]))


class TreeDigestTests(unittest.TestCase):
    def tree(self, root: Path, files: dict):
        for relative, text in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def test_digest_ignores_git_and_the_skipped_file_and_sees_other_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp, "a"), Path(tmp, "b")
            base = {"x/one.txt": "1", "two.txt": "2", "hooked.cpp": "original"}
            self.tree(a, base)
            self.tree(b, {**base, "hooked.cpp": "instrumented", ".git": "gitdir: x", "sub/.git": "y"} | {"sub/keep.txt": "k"})
            self.tree(a, {"sub/keep.txt": "k"})
            self.assertEqual(P.tree_digest(a, skip={"hooked.cpp"})[0], P.tree_digest(b, skip={"hooked.cpp"})[0])
            self.assertNotEqual(P.tree_digest(a)[0], P.tree_digest(b)[0])
            (b / "two.txt").write_text("changed")
            self.assertNotEqual(P.tree_digest(a, skip={"hooked.cpp"})[0], P.tree_digest(b, skip={"hooked.cpp"})[0])

    def test_file_diff_sees_changed_added_and_removed_files(self):
        self.assertEqual(P.file_diff({"a": "1", "b": "2"}, {"a": "1", "b": "3", "c": "4"}), ["b", "c"])
        self.assertEqual(P.file_diff({"a": "1", "b": "2"}, {"a": "1"}), ["b"])
        self.assertEqual(P.file_diff({"a": "1"}, {"a": "1"}), [])

    def test_clone_refuses_an_existing_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "src").mkdir()
            Path(tmp, "src/f").write_text("x")
            P.clone(Path(tmp, "src"), Path(tmp, "dst"))
            self.assertEqual(Path(tmp, "dst/f").read_text(), "x")
            with self.assertRaises(P.PrepareError):
                P.clone(Path(tmp, "src"), Path(tmp, "dst"))


class VerifyStockTests(unittest.TestCase):
    """A fabricated mlx-swift checkout: a git repository with the MLX core and mlx-c as nested repositories (gitlinks)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.checkout = Path(self.tmp.name) / "mlx-swift"
        self.heads = {}
        for key, relative in (("mlx-core", P.CORE), ("mlx-c", P.MLX_C)):
            path = self.checkout / relative
            path.mkdir(parents=True)
            run_git(path, "init", "-q")
            (path / "README").write_text(key)
            run_git(path, "add", "README")
            run_git(path, "commit", "-q", "-m", key)
            self.heads[key] = run_git(path, "rev-parse", "HEAD")
        (self.checkout / "Package.swift").write_text("// package\n")
        run_git(self.checkout, "init", "-q")
        run_git(self.checkout, "add", "Package.swift", P.CORE, P.MLX_C)
        run_git(self.checkout, "commit", "-q", "-m", "swift")
        self.heads["mlx-swift"] = run_git(self.checkout, "rev-parse", "HEAD")

    def test_a_clean_checkout_at_the_pins_passes_and_returns_the_commits(self):
        self.assertEqual(P.verify_stock(self.checkout), self.heads)
        self.assertEqual(P.verify_stock(self.checkout, dict(self.heads)), self.heads)
        self.assertEqual(P.verify_stock(self.checkout, {"mlx-core": self.heads["mlx-core"]}), self.heads)

    def test_another_commit_or_an_unknown_pin_is_refused(self):
        with self.assertRaisesRegex(P.PrepareError, "stock mlx-c is"):
            P.verify_stock(self.checkout, {"mlx-c": "0" * 40})
        with self.assertRaises(P.PrepareError):
            P.verify_stock(self.checkout, {"mlx": self.heads["mlx-swift"]})

    def test_a_local_modification_anywhere_is_refused(self):
        for relative in ("Package.swift", f"{P.CORE}/README", f"{P.MLX_C}/README"):
            with self.subTest(relative):
                path = self.checkout / relative
                original = path.read_text()
                path.write_text(original + "changed\n")
                with self.assertRaisesRegex(P.PrepareError, "local modifications"):
                    P.verify_stock(self.checkout)
                path.write_text(original)
        P.verify_stock(self.checkout)


class ExpectedPatchTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.expected = Path(self.dir.name) / "x.patch"

    def test_equal_patch_passes_and_a_different_or_missing_one_is_an_error_that_writes_nothing(self):
        self.expected.write_text("patch A\n")
        self.assertEqual(P.compare_expected(self.expected, "patch A\n", "x"), "equal")
        with self.assertRaises(P.PrepareError):
            P.compare_expected(self.expected, "patch B\n", "x")
        self.assertEqual(self.expected.read_text(), "patch A\n")
        missing = Path(self.dir.name) / "missing.patch"
        with self.assertRaises(P.PrepareError):
            P.compare_expected(missing, "patch A\n", "x")
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
