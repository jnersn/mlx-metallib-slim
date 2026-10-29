#!/usr/bin/env python3
"""CPU-only checks of prepare_source.py: the exact-anchor Package.swift edit, the kernel selector, and the proof that nothing else changes.

    python3 -I tests/test_prepare_source.py
    MLX_SWIFT_CHECKOUT=<SourcePackages/checkouts/mlx-swift> python3 -I tests/test_prepare_source.py     # also the real-checkout cases

The end-to-end cases run on a small fabricated checkout (the pin check is replaced by a stub there, because a fabricated tree has no git history);
the cases that read a REAL mlx-swift checkout run only when MLX_SWIFT_CHECKOUT names one, and never write to it.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "metallib_slim"
PATCHES = HERE.parent / "patches"
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("prepare_source", LIB / "prepare_source.py")
S = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = S
_spec.loader.exec_module(S)
P = S.P
REAL_CHECKOUT = Path(os.environ.get("MLX_SWIFT_CHECKOUT", "/nonexistent/mlx-swift"))
# mlx-swift commit -> the patches/ directory whose patches were generated from it
PATCH_DIRS = {"19601207e9a0de51e03ee6ec0c3c5f3784275075": "mlx-swift-0.32.3"}
PINS = {"mlx-swift": "a", "mlx-core": "b", "mlx-c": "c"}


def real_patch_dir():
    """The patches/ directory of the real checkout's commit, or None."""
    if not (REAL_CHECKOUT / "Package.swift").exists():
        return None
    head = subprocess.run(["git", "-C", str(REAL_CHECKOUT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return PATCHES / PATCH_DIRS[head] if head in PATCH_DIRS else None

# A faithful miniature of the pinned Package.swift: the noMetal list at the top (with its own `kernels` line and a jit_kernels.cpp entry), the
# Cmlx target with the anchor lines, and the package declaration that closes the block.
PACKAGE = '''// swift-tools-version: 6.3
import PackageDescription

let noMetalCmlxExcludes = [
    "mlx/mlx/backend/metal/jit_kernels.cpp",
    "mlx/mlx/backend/metal/kernels",  // Exclude kernels directory
    "mlx/mlx/backend/metal/jit",  // Exclude jit directory
]

let cmlx = Target.target(
    name: "Cmlx",
    exclude: platformExcludes + [
        // mlx files that are not part of the build
        "mlx/ACKNOWLEDGMENTS.md",

        // see PrepareMetalShaders -- don't build the kernels in place
        "mlx/mlx/backend/metal/kernels",
        "mlx/mlx/backend/metal/nojit_kernels.cpp",

        // distributed backends: enable ring, disable MPI + NCCL + JACCL
        "mlx/mlx/distributed/mpi/mpi.cpp",
    ],
)

let package = Package(
    name: "mlx-swift",
)
'''
DEPENDENCY = '.package(url: "https://github.com/ml-explore/mlx-swift", revision: mlxSwiftRevision),'
CONSUMER_PACKAGE = f'''// swift-tools-version: 6.0
import PackageDescription
let mlxSwiftRevision = "0000000000000000000000000000000000000000"
let package = Package(
    name: "Consumer",
    dependencies: [
        {DEPENDENCY}
    ],
)
'''
CORE = "Source/Cmlx/mlx/mlx/backend/metal"


def make_checkout(root: Path, package: str = PACKAGE) -> Path:
    checkout = root / "stock"
    (checkout / CORE).mkdir(parents=True)
    (checkout / "Package.swift").write_text(package)
    (checkout / CORE / "jit_kernels.cpp").write_text("// jit\n")
    (checkout / CORE / "nojit_kernels.cpp").write_text("// nojit\n")
    (checkout / "Source/Cmlx/mlx-generated").mkdir(parents=True)
    (checkout / "Source/Cmlx/mlx-generated/unary.cpp").write_text("// preamble\n")
    for path in checkout.rglob("*"):
        if path.is_file():
            path.chmod(0o444)   # SwiftPM checkouts are read-only
    return checkout


def make_package(root: Path, manifest: str = CONSUMER_PACKAGE) -> Path:
    package = root / "consumer"
    (package / "Sources/Model").mkdir(parents=True)
    (package / "Sources/Worker").mkdir(parents=True)
    (package / "Sources/tool").mkdir(parents=True)
    (package / "Tests/T").mkdir(parents=True)
    (package / "Package.swift").write_text(manifest)
    (package / "Package.resolved").write_text("{}\n")
    (package / "Sources/Model/A.swift").write_text("let a = 1\n")
    (package / "Sources/tool/main.swift").write_text("print(1)\n")
    (package / "Tests/T/T.swift").write_text("// t\n")
    shared = root / "shared-contract.swift"
    shared.write_text("let contract = 2\n")
    (package / "Sources/Worker/Contract.swift").symlink_to(shared)   # a file shared with another target through a symlink
    return package


class PatchTextTests(unittest.TestCase):
    def test_the_anchor_occurs_exactly_once_and_the_control_has_no_edit(self):
        self.assertEqual(PACKAGE.count(S.EDITS[0][1]), 1)
        self.assertEqual(S.apply_edits(PACKAGE, S.edits_for("jitcontrol")), (PACKAGE, []))

    def test_the_edit_swaps_the_two_kernel_sources_and_nothing_else(self):
        after, report = S.apply_edits(PACKAGE, S.edits_for("precompiled"))
        self.assertEqual(report, [{"edit": "cmlx-select-nojit-kernels", "anchorOccurrences": 1}])
        self.assertEqual(S.kernel_selection(PACKAGE), S.EXPECTED_SELECTION["jitcontrol"])
        self.assertEqual(S.kernel_selection(after), S.EXPECTED_SELECTION["precompiled"])
        removed = [line for line in PACKAGE.splitlines() if line not in after.splitlines()]
        added = [line for line in after.splitlines() if line not in PACKAGE.splitlines()]
        self.assertEqual(removed, ['        "mlx/mlx/backend/metal/nojit_kernels.cpp",'])
        self.assertEqual(added, ['        "mlx/mlx/backend/metal/jit_kernels.cpp",'])

    def test_the_noMetal_list_and_everything_outside_the_cmlx_block_are_untouched(self):
        after, _ = S.apply_edits(PACKAGE, S.edits_for("precompiled"))
        before_head = PACKAGE[:PACKAGE.index(S.CMLX_START)]
        self.assertEqual(after[:after.index(S.CMLX_START)], before_head)
        self.assertIn('"mlx/mlx/backend/metal/jit_kernels.cpp",\n    "mlx/mlx/backend/metal/kernels",  // Exclude kernels directory', before_head)
        self.assertEqual(after[after.index(S.CMLX_END):], PACKAGE[PACKAGE.index(S.CMLX_END):])

    def test_a_changed_or_duplicated_anchor_produces_nothing(self):
        with self.assertRaises(S.PrepareError):
            S.apply_edits(PACKAGE.replace("nojit_kernels.cpp", "nojit_kernels_v2.cpp"), S.edits_for("precompiled"))
        with self.assertRaises(S.PrepareError):
            S.apply_edits(PACKAGE + S.EDITS[0][1], S.edits_for("precompiled"))

    def test_the_selector_is_about_the_cmlx_block_only_and_refuses_ambiguity(self):
        # jit_kernels.cpp also appears in the noMetal list above the block: that must not count as "excluded" in the Cmlx target.
        self.assertEqual(S.kernel_selection(PACKAGE)["jit_kernels.cpp"], "compiled")
        both = PACKAGE.replace('        "mlx/mlx/backend/metal/nojit_kernels.cpp",\n', '        "mlx/mlx/backend/metal/nojit_kernels.cpp",\n        "mlx/mlx/backend/metal/jit_kernels.cpp",\n')
        self.assertEqual(S.kernel_selection(both), {"jit_kernels.cpp": "excluded", "nojit_kernels.cpp": "excluded"})
        with self.assertRaises(S.PrepareError):
            S.kernel_selection("let nothing = 1\n")

    def test_file_diff_sees_changed_added_and_removed_files(self):
        self.assertEqual(S.file_diff({"a": "1", "b": "2"}, {"a": "1", "b": "3", "c": "4"}), ["b", "c"])
        self.assertEqual(S.file_diff({"a": "1", "b": "2"}, {"a": "1"}), ["b"])
        self.assertEqual(S.file_diff({"a": "1"}, {"a": "1"}), [])


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._verify = P.verify_stock
        P.verify_stock = lambda checkout, pins=None: dict(PINS)   # a fabricated tree has no git history; the real pin check is exercised by --verify-only in the driver

    def tearDown(self):
        P.verify_stock = self._verify
        for path in self.root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                path.chmod(0o644)
        self.tmp.cleanup()

    def prepare(self, variant: str, package: str = PACKAGE, **extra) -> dict:
        checkout = make_checkout(self.root / variant, package)
        consumer = make_package(self.root / variant)
        return S.prepare(variant, checkout, consumer, self.root / variant / "out", self.root / variant / "patch", self.root / variant / "manifest.json", **extra)

    def test_precompiled_changes_package_swift_alone_and_leaves_every_other_file_byte_identical(self):
        manifest = self.prepare("precompiled")
        self.assertEqual(sorted(manifest["changedFiles"]), ["Package.swift"])
        self.assertTrue(manifest["otherFilesIdentical"])
        self.assertEqual(manifest["kernelSelection"], {"before": S.EXPECTED_SELECTION["jitcontrol"], "after": S.EXPECTED_SELECTION["precompiled"]})
        self.assertEqual(manifest["patchLines"], {"removed": 1, "added": 1})
        self.assertEqual(manifest["stockTree"]["files"], manifest["copyTree"]["files"])
        self.assertEqual(manifest["unchangedFiles"], manifest["stockTree"]["files"] - 1)
        self.assertNotEqual(manifest["stockTree"]["digest"], manifest["copyTree"]["digest"])
        stock, out = self.root / "precompiled/stock", self.root / "precompiled/out/mlx-swift"
        for relative in (f"{CORE}/jit_kernels.cpp", f"{CORE}/nojit_kernels.cpp", "Source/Cmlx/mlx-generated/unary.cpp"):
            self.assertEqual((stock / relative).read_bytes(), (out / relative).read_bytes(), relative)
        self.assertEqual(manifest["kernelSourceSHA256"]["jit_kernels.cpp"], P.sha256_file(stock / CORE / "jit_kernels.cpp"))
        self.assertEqual((stock / "Package.swift").read_text(), PACKAGE, "the stock checkout is never written")
        patch = (self.root / "precompiled/patch").read_text()
        self.assertIn('-        "mlx/mlx/backend/metal/nojit_kernels.cpp",', patch)
        self.assertIn('+        "mlx/mlx/backend/metal/jit_kernels.cpp",', patch)
        self.assertEqual(P.sha256_bytes(patch.encode()), manifest["patchSHA256"])
        self.assertEqual(json.loads((self.root / "precompiled/manifest.json").read_text()), manifest)

    def test_the_control_copy_is_identical_to_the_stock_checkout_in_every_file(self):
        manifest = self.prepare("jitcontrol")
        self.assertEqual(manifest["changedFiles"], {})
        self.assertEqual(manifest["stockTree"]["digest"], manifest["copyTree"]["digest"])
        self.assertEqual(manifest["patchLines"], {"removed": 0, "added": 0})
        self.assertEqual((self.root / "jitcontrol/patch").read_text(), "")
        self.assertEqual(manifest["kernelSelection"]["after"], S.EXPECTED_SELECTION["jitcontrol"])

    def test_both_variants_share_the_package_path_and_dereference_symlinks(self):
        for variant in S.VARIANTS:
            manifest = self.prepare(variant)
            package = self.root / variant / "out/package"
            text = (package / "Package.swift").read_text()
            self.assertIn(f'.package(path: "{self.root / variant / "out/mlx-swift"}"),', text)
            self.assertNotIn(DEPENDENCY, text)
            self.assertEqual(text.replace(f'.package(path: "{self.root / variant / "out/mlx-swift"}"),', DEPENDENCY), CONSUMER_PACKAGE)
            contract = package / "Sources/Worker/Contract.swift"
            self.assertFalse(contract.is_symlink(), "the package copy holds the symlink target's bytes")
            self.assertEqual(contract.read_text(), "let contract = 2\n")
            self.assertEqual(manifest["package"]["packageSourcesEqualOriginal"], {"Sources": True, "Tests": True})
            self.assertEqual(manifest["package"]["dependencyLine"], DEPENDENCY)
            self.assertEqual(sorted(manifest["package"]["targetSources"]), ["Model", "Worker", "tool"])
            self.assertEqual(manifest["package"]["targetSources"]["Worker"], P.tree_digest(package / "Sources/Worker")[0])
            self.assertEqual(manifest["package"]["originalPackageSwiftSHA256"], P.sha256_file(self.root / variant / "consumer/Package.swift"))

    def test_the_expected_patch_must_match_byte_for_byte(self):
        generated = self.prepare("precompiled")
        self.assertIsNone(generated["expectedPatch"])
        expected = self.root / "expected.patch"
        expected.write_text((self.root / "precompiled/patch").read_text())
        checkout = self.root / "precompiled/stock"
        consumer = self.root / "precompiled/consumer"
        self.assertEqual(S.prepare("precompiled", checkout, consumer, self.root / "again", expect_patch=expected)["expectedPatch"], "equal")
        expected.write_text(expected.read_text().replace("jit_kernels", "jit_kernelz"))
        with self.assertRaises(S.PrepareError):
            S.prepare("precompiled", checkout, consumer, self.root / "wrong", expect_patch=expected)
        self.assertFalse((self.root / "wrong/package").exists(), "nothing is built on a patch mismatch")
        with self.assertRaises(S.PrepareError):
            S.prepare("precompiled", checkout, consumer, self.root / "missing", expect_patch=self.root / "no-such.patch")

    def test_the_consumer_package_needs_exactly_one_mlx_swift_dependency(self):
        for label, manifest in {"none": CONSUMER_PACKAGE.replace(DEPENDENCY, ""), "two": CONSUMER_PACKAGE.replace(DEPENDENCY, DEPENDENCY + "\n        " + DEPENDENCY)}.items():
            with self.subTest(label):
                checkout = make_checkout(self.root / label)
                consumer = make_package(self.root / label, manifest)
                with self.assertRaises(S.PrepareError):
                    S.prepare("precompiled", checkout, consumer, self.root / label / "out")
        self.assertEqual(P.dependency_anchor('  .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),\n'),
                         '.package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),')

    def test_the_two_variants_differ_in_nothing_but_the_package_swift_and_its_hash(self):
        control, pre = self.prepare("jitcontrol"), self.prepare("precompiled")
        self.assertEqual(control["stockTree"], pre["stockTree"])
        self.assertEqual(control["package"]["targetSources"], pre["package"]["targetSources"])
        self.assertEqual(control["kernelSourceSHA256"], pre["kernelSourceSHA256"])
        self.assertNotEqual(control["copyTree"]["digest"], pre["copyTree"]["digest"])

    def test_a_pinned_source_that_is_not_the_expected_one_produces_nothing(self):
        checkout = make_checkout(self.root / "bad", PACKAGE.replace("nojit_kernels.cpp", "other.cpp"))
        consumer = make_package(self.root / "bad")
        with self.assertRaises(S.PrepareError):
            S.prepare("precompiled", checkout, consumer, self.root / "bad/out", None, None)
        self.assertFalse((self.root / "bad/out/package").exists())

    def test_an_existing_output_directory_is_refused(self):
        checkout = make_checkout(self.root / "twice")
        consumer = make_package(self.root / "twice")
        S.prepare("jitcontrol", checkout, consumer, self.root / "twice/out", None, None)
        with self.assertRaises(S.PrepareError):
            S.prepare("jitcontrol", checkout, consumer, self.root / "twice/out", None, None)


class RealCheckoutTests(unittest.TestCase):
    """Read only: the anchor is in the REAL pinned Package.swift exactly once and the committed patch is what the edit produces."""

    @unittest.skipUnless((REAL_CHECKOUT / "Package.swift").exists(), "no real checkout (MLX_SWIFT_CHECKOUT)")
    def test_the_real_package_swift_has_the_anchor_once_and_the_expected_selection(self):
        text = (REAL_CHECKOUT / "Package.swift").read_text()
        self.assertEqual(text.count(S.EDITS[0][1]), 1)
        self.assertEqual(S.kernel_selection(text), S.EXPECTED_SELECTION["jitcontrol"])
        after, _ = S.apply_edits(text, S.edits_for("precompiled"))
        self.assertEqual(S.kernel_selection(after), S.EXPECTED_SELECTION["precompiled"])
        self.assertEqual(after[:after.index(S.CMLX_START)], text[:text.index(S.CMLX_START)], "the Linux and noMetal sections are untouched")
        self.assertEqual(after[after.index(S.CMLX_END):], text[text.index(S.CMLX_END):])

    @unittest.skipUnless(real_patch_dir(), "no checkout at a revision with committed patches")
    def test_the_committed_patch_is_exactly_the_edit_of_the_real_package_swift(self):
        import difflib
        text = (REAL_CHECKOUT / "Package.swift").read_text()
        after, _ = S.apply_edits(text, S.edits_for("precompiled"))
        diff = "".join(difflib.unified_diff(text.splitlines(True), after.splitlines(True), "a/Package.swift", "b/Package.swift"))
        self.assertEqual((real_patch_dir() / "package-nojit.patch").read_bytes(), diff.encode())

    @unittest.skipUnless((REAL_CHECKOUT / "Source/Cmlx/mlx/mlx/backend/metal/nojit_kernels.cpp").exists(), "no real checkout (MLX_SWIFT_CHECKOUT)")
    def test_the_real_nojit_source_defines_every_kernel_getter_the_jit_source_defines(self):
        # Both files must provide every get_*_kernel the metal backend calls, or the nojit build would not link (the build proves it fully).
        import re
        metal = REAL_CHECKOUT / CORE
        names = lambda name: set(re.findall(r"^\S[^\n(]*\b(get_\w+_kernel)\(", (metal / name).read_text(), re.M))
        jit, nojit = names("jit_kernels.cpp"), names("nojit_kernels.cpp")
        self.assertTrue(jit)
        self.assertLessEqual(jit, nojit, f"missing from nojit_kernels.cpp: {sorted(jit - nojit)}")


if __name__ == "__main__":
    unittest.main()
