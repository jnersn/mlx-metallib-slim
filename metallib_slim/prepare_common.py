#!/usr/bin/env python3
"""Shared helpers of prepare_source.py and prepare_trace_source.py: the stock-checkout check, the copy-on-write clone, exact-anchor edits, tree
digests and the consumer package that builds against the prepared copy. Every helper fails closed with PrepareError.

The STOCK checkout is a SwiftPM checkout of mlx-swift (with its MLX core and mlx-c submodules) that is never written. The preparation clones it,
edits the copy only, and proves per file which files differ.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

CORE = "Source/Cmlx/mlx"  # the MLX core submodule inside mlx-swift
MLX_C = "Source/Cmlx/mlx-c"
PIN_KEYS = ("mlx-swift", "mlx-core", "mlx-c")
# The consumer package's mlx-swift dependency: `.package(url: "https://github.com/ml-explore/mlx-swift", ...),` on one line.
DEPENDENCY = re.compile(r'\.package\(\s*url:\s*"https://github\.com/ml-explore/mlx-swift(?:\.git)?"[^\n]*\),')


class PrepareError(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def edits_digest(edits: list) -> str:
    return sha256_bytes(json.dumps(edits, sort_keys=True).encode())


def apply_edits(source: str, edits: list, what: str = "source") -> tuple:
    """Zero fuzz: every anchor of `edits` ((name, anchor, replacement), ...) must occur exactly once in `source`, otherwise nothing is produced."""
    report = []
    for name, anchor, replacement in edits:
        count = source.count(anchor)
        if count != 1:
            raise PrepareError(f"edit '{name}': anchor found {count} times (need exactly 1); the pinned {what} is not the one this edit was written for")
        source = source.replace(anchor, replacement)
        report.append({"edit": name, "anchorOccurrences": count})
    return source, report


def tree_digest(root: Path, skip: set = frozenset()) -> tuple:
    """Digest of every file under root (not .git): relative path and content hash, symlinked files read through."""
    files = {}
    for current, directories, names in os.walk(root):
        directories[:] = sorted(d for d in directories if d != ".git")
        for name in sorted(names):
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            if relative in skip or name == ".git":
                continue
            files[relative] = sha256_file(path)
    manifest = "".join(f"{rel}\t{files[rel]}\n" for rel in sorted(files))
    return sha256_bytes(manifest.encode()), files


def file_diff(stock: dict, copy: dict) -> list:
    """Paths whose presence or content differs."""
    return sorted(path for path in set(stock) | set(copy) if stock.get(path) != copy.get(path))


def git(checkout: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True).stdout.strip()


def verify_stock(checkout: Path, pins: dict | None = None) -> dict:
    """The stock checkout must be unmodified and, for every key of `pins` (mlx-swift, mlx-core, mlx-c), at the pinned commit. Returns the commits."""
    found = {
        "mlx-swift": git(checkout, "rev-parse", "HEAD"),
        "mlx-core": git(checkout / CORE, "rev-parse", "HEAD"),
        "mlx-c": git(checkout / MLX_C, "rev-parse", "HEAD"),
    }
    for key, pinned in (pins or {}).items():
        if key not in found:
            raise PrepareError(f"unknown pin {key!r} (known: {', '.join(PIN_KEYS)})")
        if pinned and found[key] != pinned:
            raise PrepareError(f"stock {key} is {found[key]}, pinned {pinned}")
    for label, path in (("mlx-swift", checkout), ("mlx-core", checkout / CORE), ("mlx-c", checkout / MLX_C)):
        dirty = git(path, "status", "--porcelain", "--ignore-submodules=none")
        if dirty:
            raise PrepareError(f"stock {label} checkout has local modifications:\n{dirty}")
    return found


def clone(source: Path, destination: Path) -> None:
    """APFS copy-on-write clone (`cp -c`); never into an existing directory."""
    if destination.exists():
        raise PrepareError(f"{destination} exists; remove it first (a stale copy could be built by mistake)")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-cR", str(source), str(destination)], check=True)


def dependency_anchor(manifest: str) -> str:
    """The one mlx-swift `.package(url: ...)` dependency of a Package.swift."""
    found = DEPENDENCY.findall(manifest)
    if len(found) != 1:
        raise PrepareError(f"Package.swift: the mlx-swift dependency line (.package(url: \"https://github.com/ml-explore/mlx-swift\", ...),) was found {len(found)} times, need exactly 1")
    return found[0]


def build_package(package_dir: Path, copy: Path, out: Path) -> dict:
    """<out>: the consumer package with its mlx-swift dependency replaced by a path dependency on `copy`, its Package.resolved, and dereferenced copies
    of Sources/ (and Tests/ when present). Proves the copied sources hash-equal the originals and the manifest differs in that one line only."""
    if out.exists():
        raise PrepareError(f"{out} exists; remove it first")
    original = (package_dir / "Package.swift").read_text(encoding="utf-8")
    anchor = dependency_anchor(original)
    path_line = f'.package(path: "{copy}"),'
    manifest = original.replace(anchor, path_line)
    if manifest.replace(path_line, anchor) != original:
        raise PrepareError("the package manifest differs from the original in more than the dependency line")
    out.mkdir(parents=True)
    (out / "Package.swift").write_text(manifest, encoding="utf-8")
    if (package_dir / "Package.resolved").is_file():
        shutil.copy2(package_dir / "Package.resolved", out / "Package.resolved")
    directories = [name for name in ("Sources", "Tests") if (package_dir / name).is_dir()]
    if "Sources" not in directories:
        raise PrepareError(f"{package_dir} has no Sources directory")
    for name in directories:
        shutil.copytree(package_dir / name, out / name, symlinks=False)  # symlinks are dereferenced: the copy holds the target's bytes
    equal = {name: tree_digest(package_dir / name)[0] == tree_digest(out / name)[0] for name in directories}
    if not all(equal.values()):
        raise PrepareError(f"copied package sources differ from the originals: {equal}")
    return {
        "dependencyLine": anchor,
        "originalPackageSwiftSHA256": sha256_file(package_dir / "Package.swift"),
        "packageManifestSHA256": sha256_file(out / "Package.swift"),
        "packageResolvedSHA256": sha256_file(out / "Package.resolved") if (out / "Package.resolved").is_file() else None,
        "packageSourcesEqualOriginal": equal,
        "targetSources": {target.name: tree_digest(target)[0] for target in sorted((out / "Sources").iterdir()) if target.is_dir()},
    }


def compare_expected(expected: Path, generated: str, label: str) -> str:
    """The generated patch must equal the expected file byte for byte. The expected file is never written."""
    if not expected.is_file():
        raise PrepareError(f"{label}: expected patch {expected} does not exist")
    if expected.read_bytes() != generated.encode("utf-8"):
        raise PrepareError(f"{label}: the generated patch differs from {expected} (review the generated one in the output directory)")
    return "equal"


def pins_from(args) -> dict:
    return {key: value for key, value in (("mlx-swift", args.mlx_swift_revision), ("mlx-core", args.mlx_core_revision),
                                          ("mlx-c", args.mlx_c_revision)) if value}


def add_pin_arguments(parser) -> None:
    parser.add_argument("--mlx-swift-revision", help="the stock checkout must be at this mlx-swift commit")
    parser.add_argument("--mlx-core-revision", help="... and its MLX core submodule at this commit")
    parser.add_argument("--mlx-c-revision", help="... and its mlx-c submodule at this commit")
