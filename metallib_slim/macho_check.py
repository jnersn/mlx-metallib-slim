#!/usr/bin/env python3
"""Mach-O and build-log checks for binaries that ship with a reduced metallib. Pure Python; no tool output is trusted.

    macho_check.py sections <binary> [...]          # JSON: every section with size and SHA-256 (file-backed ones), per binary
    macho_check.py payload-equal <a> <b>            # exit 0 when the file-backed sections (code, data, strings) are identical
    macho_check.py no-coverage <binary> [...]       # exit 0 when no LLVM profile / coverage section exists
    macho_check.py log-coverage <xcodebuild.log> [--min-compiles N]   # compile-command audit (flags + response files)
    macho_check.py coverage-proof --out <json> --binary <name>=<path> [--binary ...] [--min-compiles N] <log> [<log> ...]
    macho_check.py payload-report <out.json> <name>=<a>:<b> [<name>=<c>:<d> ...]   # {"payloadEqual": {name: ...}}

`payload-equal`: two signed binaries (different bundle identifiers, hence different code signatures and load commands) hold the same code when
every file-backed section of every segment except __LINKEDIT is equal. Use it to show that the binary you ship is the one you checked.
`no-coverage` fails when a section named __llvm_prf* / __llvm_cov* or a segment named __LLVM_COV exists. `log-coverage` fails when any compile command
or response file carries -fprofile-instr-generate, -fcoverage-mapping, -profile-generate or -profile-coverage-mapping, or when too few compile commands
were seen for the absence to mean anything. `coverage-proof` writes the log audit and the coverage sections of each named binary in one record.
Only the arm64 slice of a universal binary is read.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
from pathlib import Path

MH_MAGIC_64 = 0xFEEDFACF
FAT_MAGIC = 0xCAFEBABE
CPU_TYPE_ARM64 = 0x0100000C
LC_SEGMENT_64 = 0x19
ZEROFILL_TYPES = {0x1, 0xC, 0x12}  # S_ZEROFILL, S_GB_ZEROFILL, S_THREAD_LOCAL_ZEROFILL
COVERAGE_FLAGS = ("-fprofile-instr-generate", "-fcoverage-mapping", "-profile-generate", "-profile-coverage-mapping")


class MachOError(Exception):
    pass


def _name(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii", "replace")


def thin_slice(data: bytes) -> tuple:
    """(offset, size) of the arm64 slice, or of the whole file when it is a thin 64-bit Mach-O."""
    if len(data) < 32:
        raise MachOError("file too small for a Mach-O header")
    if struct.unpack(">I", data[:4])[0] == FAT_MAGIC:
        count = struct.unpack(">I", data[4:8])[0]
        for index in range(count):
            cputype, _, offset, size, _ = struct.unpack(">iiIII", data[8 + 20 * index:28 + 20 * index])
            if cputype & 0xFFFFFFFF == CPU_TYPE_ARM64:
                return offset, size
        raise MachOError("no arm64 slice in the fat file")
    return 0, len(data)


def sections(path: Path) -> list:
    """Every section of the arm64 slice: {segment, section, size, fileBacked, sha256}. Section bytes are read from the file."""
    data = Path(path).read_bytes()
    base, size = thin_slice(data)
    magic, _cpu, _sub, _filetype, ncmds, _sizeofcmds, _flags, _reserved = struct.unpack("<IiiIIIII", data[base:base + 32])
    if magic != MH_MAGIC_64:
        raise MachOError(f"{path}: not a 64-bit little-endian Mach-O (magic {magic:#x})")
    out, offset = [], base + 32
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack("<II", data[offset:offset + 8])
        if cmdsize < 8:
            raise MachOError("corrupt load command")
        if cmd == LC_SEGMENT_64:
            segname = _name(data[offset + 8:offset + 24])
            nsects = struct.unpack("<I", data[offset + 64:offset + 68])[0]
            cursor = offset + 72
            for _s in range(nsects):
                sectname = _name(data[cursor:cursor + 16])
                sect_seg = _name(data[cursor + 16:cursor + 32])
                sect_size = struct.unpack("<Q", data[cursor + 40:cursor + 48])[0]
                sect_offset, _align, _reloff, _nreloc, flags = struct.unpack("<IIIII", data[cursor + 48:cursor + 68])
                backed = (flags & 0xFF) not in ZEROFILL_TYPES and sect_offset != 0
                entry = {"segment": sect_seg or segname, "section": sectname, "size": sect_size, "fileBacked": backed}
                if backed:
                    start = base + sect_offset
                    if start + sect_size > base + size:
                        raise MachOError(f"{sectname}: section leaves the file")
                    entry["sha256"] = hashlib.sha256(data[start:start + sect_size]).hexdigest()
                out.append(entry)
                cursor += 80
        offset += cmdsize
    return out


def coverage_sections(items: list) -> list:
    return [f"{s['segment']},{s['section']}" for s in items if s["segment"] == "__LLVM_COV" or s["section"].startswith(("__llvm_prf", "__llvm_cov"))]


def payload(path: Path) -> list:
    """The comparable part: file-backed sections outside __LINKEDIT (signatures and load commands live elsewhere)."""
    return [(s["segment"], s["section"], s["size"], s["sha256"]) for s in sections(path) if s["fileBacked"] and s["segment"] != "__LINKEDIT"]


def payload_equal(a: Path, b: Path) -> dict:
    left, right = payload(a), payload(b)
    return {"equal": bool(left) and left == right, "sections": len(left), "otherSections": len(right),
            "differing": [f"{x[0]},{x[1]}" for x, y in zip(left, right) if x != y]}


def compile_audit(log_text: str, response_reader=None, min_compiles: int = 1) -> dict:
    """Scans an xcodebuild log. A compile command is a line carrying clang/swiftc/swift-frontend with `-c` / `-frontend`; `@file` response files are read too."""
    reader = response_reader or (lambda path: Path(path).read_text(errors="replace") if Path(path).is_file() else "")
    c_compiles = swift_compiles = 0
    flagged, responses = [], 0
    for number, line in enumerate(log_text.splitlines(), 1):
        is_c = bool(re.search(r"(^|[/\s])clang(\+\+)?\s", line)) and " -c " in line
        is_swift = bool(re.search(r"(swiftc|swift-frontend|swift-driver)", line)) and (" -c " in line or "-frontend" in line or " -emit-" in line)
        if not (is_c or is_swift):
            continue
        c_compiles += is_c
        swift_compiles += is_swift
        text = line
        for token in re.findall(r"@(\S+\.(?:resp|rsp|SwiftFileList))", line):
            responses += 1
            text += "\n" + reader(token)
        hits = sorted(flag for flag in COVERAGE_FLAGS if flag in text)
        if hits:
            flagged.append({"line": number, "flags": hits})
    total = c_compiles + swift_compiles
    return {"cCompileCommands": c_compiles, "swiftCompileCommands": swift_compiles, "responseFilesRead": responses, "coverageFlagLines": flagged,
            "enoughCompiles": total >= min_compiles, "clean": not flagged and total >= min_compiles}


def coverage_proof(logs: list, binaries: dict, min_compiles: int, reader=None) -> dict:
    """The build-log audit (all logs together) and the profile/coverage sections of every named binary ({name: path})."""
    audit = compile_audit("\n".join(Path(log).read_text(errors="replace") for log in logs), response_reader=reader, min_compiles=min_compiles)
    return {"log": audit, "logs": [str(Path(log).name) for log in logs],
            "binaries": {name: coverage_sections(sections(Path(path))) for name, path in binaries.items()}}


def payload_report(pairs: dict) -> dict:
    """{name: (a, b)} -> {"payloadEqual": {name: payload_equal(a, b)}}"""
    return {"payloadEqual": {name: payload_equal(Path(a), Path(b)) for name, (a, b) in pairs.items()}}


def main(argv: list | None = None) -> int:
    """`argv` as sys.argv: the program name first."""
    argv = sys.argv if argv is None else argv
    if len(argv) < 2:
        print(__doc__)
        return 2
    command, rest = argv[1], argv[2:]
    try:
        if command == "sections":
            print(json.dumps({p: sections(Path(p)) for p in rest}, indent=2, sort_keys=True))
            return 0
        if command == "payload-equal" and len(rest) == 2:
            result = payload_equal(Path(rest[0]), Path(rest[1]))
            print(json.dumps(result, sort_keys=True))
            return 0 if result["equal"] else 1
        if command == "no-coverage" and rest:
            report = {p: coverage_sections(sections(Path(p))) for p in rest}
            print(json.dumps(report, sort_keys=True))
            return 0 if not any(report.values()) else 1
        if command == "coverage-proof" and "--out" in rest and "--binary" in rest:
            out, minimum, binaries, logs = None, 1, {}, []
            items = iter(rest)
            for token in items:
                if token == "--out":
                    out = next(items)
                elif token == "--min-compiles":
                    minimum = int(next(items))
                elif token == "--binary":
                    name, _, path = next(items).partition("=")
                    if not name or not path or name in binaries:
                        raise ValueError(f"--binary needs a distinct <name>=<path>, got {name!r}")
                    binaries[name] = path
                else:
                    logs.append(token)
            if not logs:
                raise ValueError("coverage-proof needs at least one build log")
            proof = coverage_proof(logs, binaries, minimum)
            Path(out).write_text(json.dumps(proof, indent=1, sort_keys=True) + "\n")
            print(json.dumps({"log": proof["log"]["clean"], "binaries": proof["binaries"]}, sort_keys=True))
            return 0 if proof["log"]["clean"] and not any(proof["binaries"].values()) else 1
        if command == "payload-report" and len(rest) >= 2:
            pairs = {}
            for item in rest[1:]:
                name, files = item.split("=", 1)
                a, b = files.split(":", 1)
                pairs[name] = (a, b)
            report = payload_report(pairs)
            Path(rest[0]).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
            print(json.dumps({name: entry["equal"] for name, entry in report["payloadEqual"].items()}, sort_keys=True))
            return 0 if all(entry["equal"] for entry in report["payloadEqual"].values()) else 1
        if command == "log-coverage" and rest:
            minimum = int(rest[rest.index("--min-compiles") + 1]) if "--min-compiles" in rest else 1
            result = compile_audit(Path(rest[0]).read_text(errors="replace"), min_compiles=minimum)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["clean"] else 1
    except (MachOError, OSError, ValueError, StopIteration, struct.error) as error:
        print(f"macho_check: {error}", file=sys.stderr)
        return 2
    print(__doc__)
    return 2


def cli(argv: list | None = None) -> int:
    return main(argv)


if __name__ == "__main__":
    sys.exit(cli())
