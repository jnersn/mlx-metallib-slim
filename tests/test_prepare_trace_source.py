#!/usr/bin/env python3
"""CPU-only checks of prepare_trace_source.py: the trace hook anchors, what the hook may and may not do, and the traced copy on a fabricated
checkout. No build, no GPU, never writes to a stock checkout.

    python3 -I tests/test_prepare_trace_source.py
    MLX_SWIFT_CHECKOUT=<SourcePackages/checkouts/mlx-swift> python3 -I tests/test_prepare_trace_source.py     # also the real-checkout cases
"""
import difflib
import importlib.util
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
_spec = importlib.util.spec_from_file_location("prepare_trace_source", LIB / "prepare_trace_source.py")
T = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = T
_spec.loader.exec_module(T)
P, S = T.P, T.S
REAL_CHECKOUT = Path(os.environ.get("MLX_SWIFT_CHECKOUT", "/nonexistent/mlx-swift"))
PATCH_DIRS = {"19601207e9a0de51e03ee6ec0c3c5f3784275075": "mlx-swift-0.32.3"}

DEVICE = ("// Copyright\n\n#include <cstdlib>\n#include <sstream>\n\n#include <fmt/format.h>\n\nnamespace mlx::core::metal {\n\n"
          "std::pair<MTL::Library*, NS::Error*> load_library_from_path(\n    MTL::Device* device,\n    const char* path) {\n  return {};\n}\n\n"
          "MTL::ComputePipelineState* Device::get_kernel(\n    const std::string& base_name,\n    MTL::Library* mtl_lib,\n    const std::string& hash_name) {\n"
          "  const auto& kname = hash_name.empty() ? base_name : hash_name;\n  {\n    // Multiple readers allowed\n  }\n}\n}\n")
PACKAGE = '''let cmlx = Target.target(
    name: "Cmlx",
    exclude: [
        // see PrepareMetalShaders -- don't build the kernels in place
        "mlx/mlx/backend/metal/kernels",
        "mlx/mlx/backend/metal/nojit_kernels.cpp",
    ],
)

let package = Package(
    name: "mlx-swift",
)
'''
CONSUMER = 'let package = Package(\n    dependencies: [\n        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.32.3"),\n    ],\n)\n'


def real_patch_dir():
    if not (REAL_CHECKOUT / T.DEVICE).exists():
        return None
    head = subprocess.run(["git", "-C", str(REAL_CHECKOUT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return PATCHES / PATCH_DIRS[head] if head in PATCH_DIRS else None


class EditTests(unittest.TestCase):
    def test_every_anchor_applies_once_and_the_hook_runs_before_the_cache_lookup(self):
        hooked, report = T.apply_edits(DEVICE)
        self.assertEqual([r["anchorOccurrences"] for r in report], [1, 1, 1])
        self.assertEqual(hooked.count("reduced_trace::note(base_name);"), 1)
        self.assertLess(hooked.index("reduced_trace::note(base_name);"), hooked.index("// Multiple readers allowed"))
        self.assertIn("if (mtl_lib == default_library_.get()) {", hooked)
        self.assertLess(hooked.index("namespace reduced_trace"), hooked.index("load_library_from_path("))
        for header in ("<fcntl.h>", "<unistd.h>", "<mutex>", "<set>"):
            self.assertIn(f"#include {header}", hooked)

    def test_the_hook_writes_a_header_a_summary_and_never_stdout(self):
        block = T.TRACE_BLOCK
        for needle in ('\\"event\\":\\"header\\"', '\\"event\\":\\"summary\\"', "write_failures", "dropped", "O_EXCL", "default_kernel",
                       f'std::getenv("{T.TRACE_ENV}")'):
            self.assertIn(needle, block)
        stripped = block.replace("snprintf(", "")  # snprintf into a buffer is fine
        for forbidden in ("std::cout", "std::cerr", "printf(", "stdout", "stderr"):
            self.assertNotIn(forbidden, stripped)

    def test_the_hook_never_changes_options_precision_or_optimisation(self):
        added = "".join(replacement for _, _, replacement in T.EDITS)
        for forbidden in ("setOptimizationLevel", "LibraryOptimizationLevel", "setFastMath", "setMathMode", "setMathFloatingPointFunctions",
                          "setPreprocessorMacros", "MTLCompileOptions"):
            self.assertNotIn(forbidden, added, forbidden)

    def test_a_changed_or_repeated_anchor_produces_nothing(self):
        with self.assertRaises(T.PrepareError):
            T.apply_edits(DEVICE.replace("const auto& kname", "const auto& kname2"))
        with self.assertRaises(T.PrepareError):
            T.apply_edits(DEVICE + DEVICE)
        once, _ = T.apply_edits(DEVICE)
        with self.assertRaises(T.PrepareError):
            T.apply_edits(once)  # applying twice fails instead of stacking hooks

    def test_the_hook_does_not_touch_anything_but_its_three_places(self):
        hooked, _ = T.apply_edits(DEVICE)
        restored = hooked
        for _name, anchor, replacement in reversed(T.EDITS):
            restored = restored.replace(replacement, anchor)
        self.assertEqual(restored, DEVICE)


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._verify = P.verify_stock
        P.verify_stock = lambda checkout, pins=None: {"mlx-swift": "a", "mlx-core": "b", "mlx-c": "c"}  # a fabricated tree has no git history
        self.stock = self.root / "stock"
        (self.stock / "Source/Cmlx/mlx/mlx/backend/metal").mkdir(parents=True)
        (self.stock / "Package.swift").write_text(PACKAGE)
        (self.stock / T.DEVICE).write_text(DEVICE)
        for name in ("jit_kernels.cpp", "nojit_kernels.cpp", "other.cpp"):
            (self.stock / "Source/Cmlx/mlx/mlx/backend/metal" / name).write_text(f"// {name}\n")
        for path in self.stock.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        self.consumer = self.root / "consumer"
        (self.consumer / "Sources/App").mkdir(parents=True)
        (self.consumer / "Package.swift").write_text(CONSUMER)
        (self.consumer / "Sources/App/main.swift").write_text("print(1)\n")

    def tearDown(self):
        P.verify_stock = self._verify
        for path in self.root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                path.chmod(0o644)
        self.tmp.cleanup()

    def test_the_traced_copy_differs_in_package_swift_and_device_cpp_only_and_the_patch_can_be_pinned(self):
        result = T.prepare(self.stock, self.consumer, self.root / "out")
        self.assertEqual(result["trace"]["changedFiles"], ["Package.swift", T.DEVICE])
        self.assertEqual(result["traceEnvironment"], T.TRACE_ENV)
        self.assertIsNone(result["tracePatch"])
        copy = self.root / "out/mlx-swift"
        self.assertIn("reduced_trace::note(base_name);", (copy / T.DEVICE).read_text())
        self.assertEqual((self.stock / T.DEVICE).read_text(), DEVICE, "the stock checkout is never written")
        self.assertEqual(S.kernel_selection((copy / "Package.swift").read_text()), S.EXPECTED_SELECTION["precompiled"])
        patch = (self.root / "out/device-names-trace.patch").read_text()
        self.assertTrue(patch.startswith("--- a/mlx/backend/metal/device.cpp\n+++ b/mlx/backend/metal/device.cpp\n"))
        self.assertEqual(P.sha256_bytes(patch.encode()), result["trace"]["patchSHA256"])
        self.assertTrue((self.root / "out/manifest-trace.json").is_file())
        expected = self.root / "expected.patch"
        expected.write_text(patch)
        self.assertEqual(T.prepare(self.stock, self.consumer, self.root / "again", expect_trace_patch=expected)["tracePatch"], "equal")
        expected.write_text(patch.replace("reduced_trace", "other_trace"))
        with self.assertRaises(T.PrepareError):
            T.prepare(self.stock, self.consumer, self.root / "wrong", expect_trace_patch=expected)


class RealCheckoutTests(unittest.TestCase):
    """Read only: the anchors are in the REAL device.cpp exactly once and the committed patch is what the hook produces."""

    @unittest.skipUnless((REAL_CHECKOUT / T.DEVICE).exists(), "no real checkout (MLX_SWIFT_CHECKOUT)")
    def test_every_anchor_occurs_exactly_once_in_the_real_device_cpp(self):
        text = (REAL_CHECKOUT / T.DEVICE).read_text(encoding="utf-8")
        for name, anchor, _ in T.EDITS:
            self.assertEqual(text.count(anchor), 1, name)

    @unittest.skipUnless(real_patch_dir(), "no checkout at a revision with committed patches")
    def test_the_committed_trace_patch_is_exactly_the_hook_of_the_real_device_cpp(self):
        text = (REAL_CHECKOUT / T.DEVICE).read_text(encoding="utf-8")
        hooked, _ = T.apply_edits(text)
        diff = "".join(difflib.unified_diff(text.splitlines(True), hooked.splitlines(True), "a/mlx/backend/metal/device.cpp", "b/mlx/backend/metal/device.cpp"))
        self.assertEqual((real_patch_dir() / "device-names-trace.patch").read_bytes(), diff.encode())


if __name__ == "__main__":
    unittest.main()
