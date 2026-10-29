#!/usr/bin/env python3
"""Builds and freezes the kernel-name allowlist from kernel-name traces, and verifies a frozen one. Fails closed.

    allowlist.py build --declaration <declaration.json> --out <allowlist.json> --trace <label>=<file or directory> [--trace ...]
        [--expect <label> ...] [--base <older allowlist.json>] [--allow-partial]
    allowlist.py verify <allowlist.json>      # file hash == allowlist.sha256, content self-consistent

A trace is the file a traced MLX build (prepare_trace_source.py) writes: a header line, one `call` line per DISTINCT default-library kernel name
(the base exported name, not a specialised hash name), and one summary line at exit. A trace counts only when it is complete: header first, one
summary last, no write failure, no dropped line, the call-line count equals the summary's. Any trace that is not is an error, never skipped.

Each `--trace` names a label (an architecture class such as `g13g`, or any run you want recorded) and a trace file or a directory whose `*.jsonl`
files are all that label's traces (one per traced process). The allowlist is the UNION of the names of every trace (and of `--base`, when given).
Every `--expect` label without a trace (here or in the base allowlist's jobs) is PENDING: without --allow-partial that is an error; with it the file says `complete: false` and lists the
pending labels, and a library reduced from it covers only the traced labels (reduce_library.py refuses it unless told otherwise).

The allowlist records the SHA-256 of the declaration it was built for, and is frozen by writing <name>.sha256 (its own SHA-256) beside it. A frozen
allowlist is never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

NAME = re.compile(r"^[A-Za-z0-9_]+$")


class AllowlistError(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list:
    out = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AllowlistError(f"{path}:{number}: {error}") from error
    return out


def trace_names(records: list) -> tuple:
    """(names, problems) of the parsed lines of one trace file."""
    problems = []
    if not records or records[0].get("event") != "header":
        problems.append("first line is not the header")
    summaries = [r for r in records if r.get("event") == "summary"]
    if len(summaries) != 1 or records[-1].get("event") != "summary":
        problems.append("no single final summary line")
    calls = [r for r in records if r.get("event") == "call"]
    names = []
    for call in calls:
        label = call.get("label")
        if call.get("kind") != "default_kernel" or not isinstance(label, str) or not NAME.match(label):
            problems.append(f"unexpected call line {call}")
        else:
            names.append(label)
    if len(set(names)) != len(names):
        problems.append("a name occurs twice (the hook logs distinct names)")
    if summaries:
        summary = summaries[-1]
        if summary.get("write_failures") != 0:
            problems.append(f"write_failures is {summary.get('write_failures')}")
        if summary.get("dropped") != 0:
            problems.append(f"dropped is {summary.get('dropped')}")
        if summary.get("lines") != len(calls) or summary.get("distinct") != len(set(names)):
            problems.append(f"summary lines/distinct {summary.get('lines')}/{summary.get('distinct')} != {len(calls)}/{len(set(names))} call lines")
    return set(names), problems


def trace_files(path: Path) -> list:
    """A trace file, or every `*.jsonl` file of a directory (sorted). An empty directory is an error."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.glob("*.jsonl") if p.is_file())
        if not files:
            raise AllowlistError(f"{path}: no *.jsonl trace file in the directory")
        return files
    if not path.is_file():
        raise AllowlistError(f"{path}: no such trace file or directory")
    return [path]


def label_names(label: str, paths: list) -> dict:
    """The validated names of every trace file of one label: {files: [{file, sha256, distinctNames}], names: [...]}."""
    files, union = [], set()
    for path in paths:
        data = Path(path).read_bytes()
        names, problems = trace_names(read_jsonl(path))
        if problems:
            raise AllowlistError(f"{label}: trace file {path} is not valid: {problems}")
        union |= names
        files.append({"file": Path(path).name, "sha256": sha256_bytes(data), "distinctNames": len(names)})
    return {"files": files, "names": sorted(union)}


def build(traces: dict, expected: list, declaration_sha: str, allow_partial: bool, base: dict | None = None) -> dict:
    """`traces`: label -> list of trace file paths. `expected`: labels that must have a trace. `base`: an older allowlist whose names are kept."""
    jobs = dict((base or {}).get("jobs") or {})
    union = set((base or {}).get("names") or [])
    for label in sorted(traces):
        found = label_names(label, traces[label])
        union |= set(found["names"])
        jobs[label] = {"traceFiles": found["files"], "distinctNames": len(found["names"]),
                       "namesSHA256": sha256_bytes("\n".join(found["names"]).encode())}
    pending = [label for label in expected if label not in jobs]  # a label of the base allowlist counts as traced
    if pending and not allow_partial:
        raise AllowlistError(f"no trace for {pending} (trace them, or pass --allow-partial to record them PENDING)")
    if not union:
        raise AllowlistError("no kernel name found in any trace")
    names = sorted(union)
    allowlist = {"schema": 1, "declarationSHA256": declaration_sha, "complete": not pending, "pendingJobs": pending, "jobs": jobs,
                 "nameCount": len(names), "names": names, "namesSHA256": sha256_bytes("\n".join(names).encode())}
    if base is not None:
        allowlist["baseNamesSHA256"] = base.get("namesSHA256")
        allowlist["newNamesBeyondBase"] = sorted(union - set(base.get("names") or []))
    return allowlist


def write(allowlist: dict, out: Path) -> str:
    text = json.dumps(allowlist, indent=1, sort_keys=True) + "\n"
    out.write_text(text, encoding="utf-8")
    digest = sha256_bytes(text.encode())
    out.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def verify(path: Path) -> dict:
    data = Path(path).read_bytes()
    sha_file = Path(path).with_suffix(".sha256")
    if not sha_file.is_file() or sha_file.read_text().strip() != sha256_bytes(data):
        raise AllowlistError(f"{sha_file.name} is missing or does not match {Path(path).name} (the allowlist is not frozen, or was edited)")
    allowlist = json.loads(data)
    names = allowlist.get("names") or []
    if names != sorted(set(names)) or allowlist.get("nameCount") != len(names) or allowlist.get("namesSHA256") != sha256_bytes("\n".join(names).encode()):
        raise AllowlistError("allowlist names are not sorted, distinct and consistent with their count and hash")
    if any(not NAME.match(n) for n in names):
        raise AllowlistError("allowlist holds a non-identifier name")
    return allowlist


def parse_traces(items: list) -> dict:
    traces = {}
    for item in items:
        label, sep, path = item.partition("=")
        if not sep or not label or not path:
            raise AllowlistError(f"--trace {item!r}: expected <label>=<file or directory>")
        traces.setdefault(label, []).extend(trace_files(Path(path)))
    for label, paths in traces.items():
        if len(set(paths)) != len(paths):
            raise AllowlistError(f"{label}: the same trace file is given twice")
    return traces


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build and freeze an allowlist from traces")
    b.add_argument("--declaration", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--trace", required=True, action="append", metavar="LABEL=PATH", help="trace file or directory of a label (repeatable)")
    b.add_argument("--expect", action="append", default=[], metavar="LABEL", help="a label that must have a trace (repeatable)")
    b.add_argument("--base", type=Path, help="a frozen allowlist whose names and jobs are kept")
    b.add_argument("--allow-partial", action="store_true")
    v = sub.add_parser("verify", help="verify a frozen allowlist")
    v.add_argument("allowlist", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        if args.out.exists() or args.out.with_suffix(".sha256").exists():
            raise AllowlistError(f"{args.out} is already frozen; a frozen allowlist is never rewritten (remove it deliberately to start over)")
        declaration_bytes = args.declaration.read_bytes()
        base = verify(args.base) if args.base else None
        allowlist = build(parse_traces(args.trace), args.expect, sha256_bytes(declaration_bytes), args.allow_partial, base)
        digest = write(allowlist, args.out)
        print(f"allowlist frozen: {allowlist['nameCount']} names, {len(allowlist['jobs'])} labels, pending {allowlist['pendingJobs']}, sha256 {digest}")
        return 0
    allowlist = verify(args.allowlist)
    print(f"allowlist verified: {allowlist['nameCount']} names, complete {allowlist['complete']}")
    return 0


def cli(argv: list | None = None) -> int:
    try:
        return main(argv)
    except (AllowlistError, OSError, KeyError, ValueError) as error:
        print(f"allowlist: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli())
