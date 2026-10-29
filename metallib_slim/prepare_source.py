#!/usr/bin/env python3
"""Prepares an isolated, reproducible no-JIT SOURCE copy of a pinned mlx-swift checkout and a consumer package that builds against it. Fails closed.

    prepare_source.py --stock-checkout <SourcePackages/checkouts/mlx-swift> --package-dir <your package> --out <dir>
        [--variant precompiled|jitcontrol] [--expect-patch <package-nojit.patch>] [--patch-out <file>] [--manifest-out <file>]
        [--mlx-swift-revision R] [--mlx-core-revision R] [--mlx-c-revision R]
    prepare_source.py --stock-checkout <dir> --verify-only [--mlx-swift-revision R ...]

What it does:
  1. Verifies the STOCK checkout has no local modification (and is at the given revisions). It never writes there.
  2. Clones the checkout to <out>/mlx-swift (APFS copy-on-write).
     * precompiled (default): ONE exact-anchor edit to the COPY's Package.swift: the Cmlx target excludes jit_kernels.cpp instead of
       nojit_kernels.cpp, so SwiftPM compiles nojit_kernels.cpp (kernels are looked up in a precompiled mlx.metallib) and no longer compiles
       jit_kernels.cpp (kernels built from source at run time). Everything else is kept on purpose: the generated preambles in mlx-generated/ that
       compiled (fused) kernels, gather/scatter and the other source-built kernels need, and the Linux/noMetal exclude lists.
     * jitcontrol: the copy is left UNMODIFIED, so a control build goes through the same prepared-package path.
  3. Proves, per file (not by a digest that skips the edited file), that the copy differs from the stock checkout in EXACTLY the expected set of
     files: Package.swift alone for precompiled, none for jitcontrol; and that the edit changed exactly the two intended lines.
  4. Builds <out>/package: the consumer package's Package.swift with its mlx-swift dependency (`.package(url: "https://github.com/ml-explore/mlx-swift",
     ...),`) replaced by a path dependency on the copy, plus its Package.resolved and dereferenced copies of Sources/ and Tests/, and proves the
     copied sources hash-equal the originals and the manifest differs in that one line only.
  5. Writes <out>/manifest.json (and --manifest-out) and the unified diff <out>/package-nojit.patch (and --patch-out; empty for jitcontrol).
     With --expect-patch the diff must equal that file byte for byte (pin the reviewed patch this way).

A changed anchor means the pinned source is not the one this edit was written for, and nothing is produced.
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if "prepare_common" in sys.modules:
    P = sys.modules["prepare_common"]
else:
    _spec = importlib.util.spec_from_file_location("prepare_common", HERE / "prepare_common.py")
    P = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = P
    _spec.loader.exec_module(P)

PrepareError = P.PrepareError
VARIANTS = ("precompiled", "jitcontrol")
TARGET = "Package.swift"
JIT = '"mlx/mlx/backend/metal/jit_kernels.cpp",'
NOJIT = '"mlx/mlx/backend/metal/nojit_kernels.cpp",'
CMLX_START = "let cmlx = Target.target("
CMLX_END = "let package = Package("

# (name, exact anchor, replacement). The anchor must occur exactly once in the pinned Package.swift. The comment line makes it unique: the
# `kernels` entry also occurs in noMetalCmlxExcludes (with another comment), which this edit does not touch.
EDITS = [
    ("cmlx-select-nojit-kernels",
     "        // see PrepareMetalShaders -- don't build the kernels in place\n"
     '        "mlx/mlx/backend/metal/kernels",\n'
     '        "mlx/mlx/backend/metal/nojit_kernels.cpp",\n',
     "        // see PrepareMetalShaders -- don't build the kernels in place\n"
     '        "mlx/mlx/backend/metal/kernels",\n'
     '        "mlx/mlx/backend/metal/jit_kernels.cpp",\n'),
]


def edits_for(variant: str) -> list:
    return EDITS if variant == "precompiled" else []


def apply_edits(source: str, edits: list) -> tuple:
    """Zero fuzz: every anchor must occur exactly once, otherwise nothing is produced."""
    return P.apply_edits(source, edits, "Package.swift")


def cmlx_block(source: str) -> str:
    start, end = source.find(CMLX_START), source.find(CMLX_END)
    if start < 0 or end <= start or source.count(CMLX_START) != 1 or source.count(CMLX_END) != 1:
        raise PrepareError("Package.swift: the Cmlx target block was not found exactly once")
    return source[start:end]


def kernel_selection(source: str) -> dict:
    """Which of the two kernel sources the Cmlx target EXCLUDES (the other one is compiled)."""
    block = cmlx_block(source)
    return {"jit_kernels.cpp": "excluded" if block.count(JIT) == 1 else ("compiled" if block.count(JIT) == 0 else "ambiguous"),
            "nojit_kernels.cpp": "excluded" if block.count(NOJIT) == 1 else ("compiled" if block.count(NOJIT) == 0 else "ambiguous")}


EXPECTED_SELECTION = {
    "jitcontrol": {"jit_kernels.cpp": "compiled", "nojit_kernels.cpp": "excluded"},
    "precompiled": {"jit_kernels.cpp": "excluded", "nojit_kernels.cpp": "compiled"},
}

file_diff = P.file_diff


def prepare(variant: str, stock: Path, package_dir: Path, out: Path, patch_out: Path | None = None, manifest_out: Path | None = None,
            pins: dict | None = None, expect_patch: Path | None = None) -> dict:
    found_pins = P.verify_stock(stock, pins)
    stock_file = stock / TARGET
    before_text = stock_file.read_text(encoding="utf-8")
    before_sha = P.sha256_bytes(before_text.encode())
    if kernel_selection(before_text) != EXPECTED_SELECTION["jitcontrol"]:
        raise PrepareError(f"the pinned Package.swift does not select the JIT kernels as expected: {kernel_selection(before_text)}")
    after_text, report = apply_edits(before_text, edits_for(variant))
    if kernel_selection(after_text) != EXPECTED_SELECTION[variant]:
        raise PrepareError(f"{variant}: the kernel selection after the edit is {kernel_selection(after_text)}")

    copy = out / "mlx-swift"
    P.clone(stock, copy)
    if after_text != before_text:
        # SwiftPM sources are read-only; change permissions only on the isolated copy.
        (copy / TARGET).chmod((copy / TARGET).stat().st_mode | 0o200)
        (copy / TARGET).write_text(after_text, encoding="utf-8")
    if P.sha256_file(stock_file) != before_sha:
        raise PrepareError("the stock checkout changed during preparation")

    stock_digest, stock_files = P.tree_digest(stock)
    copy_digest, copy_files = P.tree_digest(copy)
    changed = file_diff(stock_files, copy_files)
    expected_changed = [TARGET] if variant == "precompiled" else []
    if changed != expected_changed:
        raise PrepareError(f"the copy differs from the stock checkout in {changed}, expected exactly {expected_changed}")

    diff = "".join(difflib.unified_diff(before_text.splitlines(True), after_text.splitlines(True), "a/Package.swift", "b/Package.swift"))
    removed = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    added = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    expected_lines = (1, 1) if variant == "precompiled" else (0, 0)
    if (len(removed), len(added)) != expected_lines:
        raise PrepareError(f"the patch changes {len(removed)} removed / {len(added)} added lines, expected {expected_lines}")
    (out / "package-nojit.patch").write_text(diff, encoding="utf-8")
    patch_check = P.compare_expected(expect_patch, diff, "Package.swift patch") if expect_patch else None

    package = P.build_package(package_dir, copy, out / "package")

    core = Path(P.CORE) / "mlx/backend/metal"
    manifest = {
        "schema": 1,
        "variant": variant,
        "purpose": ("Unmodified copy of the pinned mlx-swift checkout (control); no edit." if variant == "jitcontrol" else
                    "Copy of the pinned mlx-swift checkout whose Package.swift compiles nojit_kernels.cpp instead of jit_kernels.cpp (one exact-anchor edit)."),
        "pins": found_pins,
        "stockCheckoutClean": True,
        "kernelSelection": {"before": kernel_selection(before_text), "after": kernel_selection(after_text)},
        "kernelSourceSHA256": {name: copy_files[(core / name).as_posix()] for name in ("jit_kernels.cpp", "nojit_kernels.cpp")},
        "changedFiles": {path: {"stock": stock_files.get(path), "copy": copy_files.get(path)} for path in changed},
        "packageSwiftSHA256": {"stock": before_sha, "copy": P.sha256_file(copy / TARGET)},
        "editsSHA256": P.edits_digest(edits_for(variant)),
        "edits": report,
        "patchSHA256": P.sha256_bytes(diff.encode()),
        "patchLines": {"removed": len(removed), "added": len(added)},
        "expectedPatch": patch_check,
        "stockTree": {"digest": stock_digest, "files": len(stock_files)},
        "copyTree": {"digest": copy_digest, "files": len(copy_files)},
        "unchangedFiles": len(stock_files) - len([p for p in changed if p in stock_files]),
        "otherFilesIdentical": True,
        "package": package,
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (out / "manifest.json").write_text(text, encoding="utf-8")
    for destination, content in ((patch_out, diff), (manifest_out, text)):
        if destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
    return manifest


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, default="precompiled")
    parser.add_argument("--stock-checkout", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true", help="only verify the stock checkout is clean (and at the given revisions), print the commits")
    parser.add_argument("--package-dir", type=Path, help="the consumer SwiftPM package that depends on mlx-swift")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expect-patch", type=Path, help="the generated Package.swift patch must equal this file byte for byte")
    parser.add_argument("--patch-out", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    P.add_pin_arguments(parser)
    args = parser.parse_args(argv)
    if args.verify_only:
        print(json.dumps(P.verify_stock(args.stock_checkout.resolve(), P.pins_from(args))))
        return 0
    if not (args.package_dir and args.out):
        parser.error("--package-dir and --out are required")
    manifest = prepare(args.variant, args.stock_checkout.resolve(), args.package_dir.resolve(), args.out.resolve(), args.patch_out, args.manifest_out,
                       P.pins_from(args), args.expect_patch)
    print(json.dumps({"variant": args.variant, "out": str(args.out.resolve()), "changedFiles": sorted(manifest["changedFiles"]),
                      "patchSHA256": manifest["patchSHA256"], "expectedPatch": manifest["expectedPatch"], "kernelSelection": manifest["kernelSelection"]["after"]}))
    return 0


def cli(argv: list | None = None) -> int:
    try:
        return main(argv)
    except (PrepareError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"prepare_source: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli())
