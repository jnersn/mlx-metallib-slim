#!/usr/bin/env python3
"""CPU-only checks of macho_check.py on synthetic Mach-O files and build logs.

    python3 -I tests/test_macho_check.py
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "metallib_slim"
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("macho_check", LIB / "macho_check.py")
M = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = M
_spec.loader.exec_module(M)


def pad(text: str) -> bytes:
    return text.encode().ljust(16, b"\0")


def section(sectname: str, segname: str, size: int, offset: int, flags: int = 0) -> bytes:
    return pad(sectname) + pad(segname) + struct.pack("<QQIIIIIIII", 0, size, offset, 0, 0, 0, flags, 0, 0, 0)


def macho(sects: list, extra_load_command: bool = False) -> bytes:
    """One segment per (segname, [(sectname, bytes, flags)]) entry; payload bytes follow the load commands."""
    commands = []
    body = b""
    header_size = 32
    seg_sizes = [72 + 80 * len(items) for _, items in sects]
    data_start = header_size + sum(seg_sizes) + (24 if extra_load_command else 0)
    offset = data_start
    for segname, items in sects:
        entries = b""
        for sectname, content, flags in items:
            if (flags & 0xFF) in (1, 0xC, 0x12):
                entries += section(sectname, segname, len(content), 0, flags)
            else:
                entries += section(sectname, segname, len(content), offset, flags)
                body += content
                offset += len(content)
        command = struct.pack("<II", 0x19, 72 + len(entries)) + pad(segname) + struct.pack("<QQQQIIII", 0, 0, 0, 0, 7, 7, len(items), 0) + entries
        commands.append(command)
    if extra_load_command:
        commands.append(struct.pack("<II", 0x1D, 24) + b"\1" * 16)  # a stand-in for LC_CODE_SIGNATURE: differs between signed copies
    blob = b"".join(commands)
    header = struct.pack("<IiiIIIII", M.MH_MAGIC_64, M.CPU_TYPE_ARM64, 0, 2, len(commands), len(blob), 0, 0)
    return header + blob + body


class SectionTests(unittest.TestCase):
    def write(self, name: str, data: bytes) -> Path:
        path = Path(self.dir.name) / name
        path.write_bytes(data)
        return path

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_sections_are_parsed_with_hashes_and_zerofill_is_not_file_backed(self):
        path = self.write("a", macho([("__TEXT", [("__text", b"code", 0)]), ("__DATA", [("__bss", b"\0" * 8, 1), ("__data", b"data", 0)])]))
        items = M.sections(path)
        self.assertEqual([(s["segment"], s["section"], s["fileBacked"]) for s in items], [("__TEXT", "__text", True), ("__DATA", "__bss", False), ("__DATA", "__data", True)])
        self.assertEqual(items[0]["sha256"], hashlib.sha256(b"code").hexdigest())
        self.assertEqual(items[2]["size"], 4)
        self.assertEqual(M.coverage_sections(items), [])

    def test_llvm_profile_and_coverage_sections_are_detected(self):
        path = self.write("c", macho([("__TEXT", [("__text", b"code", 0)]), ("__DATA", [("__llvm_prf_cnts", b"x", 0)]), ("__LLVM_COV", [("__llvm_covmap", b"y", 0)])]))
        self.assertEqual(M.coverage_sections(M.sections(path)), ["__DATA,__llvm_prf_cnts", "__LLVM_COV,__llvm_covmap"])

    def test_payload_equality_ignores_load_commands_and_requires_identical_sections(self):
        layout = [("__TEXT", [("__text", b"code", 0), ("__cstring", b"hello", 0)])]
        a = self.write("a", macho(layout))
        b = self.write("b", macho(layout, extra_load_command=True))  # a differently signed copy
        self.assertTrue(M.payload_equal(a, b)["equal"])
        c = self.write("c", macho([("__TEXT", [("__text", b"cod3", 0), ("__cstring", b"hello", 0)])]))
        result = M.payload_equal(a, c)
        self.assertFalse(result["equal"])
        self.assertEqual(result["differing"], ["__TEXT,__text"])

    def test_a_non_macho_file_is_rejected(self):
        with self.assertRaises(M.MachOError):
            M.sections(self.write("x", b"\0" * 64))


class ProofTests(SectionTests):
    def test_coverage_proof_and_payload_report_assemble_their_records(self):
        clean = self.write("clean", macho([("__TEXT", [("__text", b"code", 0)])]))
        dirty = self.write("dirty", macho([("__TEXT", [("__text", b"code", 0)]), ("__DATA", [("__llvm_prf_cnts", b"x", 0)])]))
        log = Path(self.dir.name) / "build.log"
        log.write_text(LogTests.CLEAN)
        proof = M.coverage_proof([log], {"app": clean, "helper": clean}, 2, reader=lambda path: "-O")
        self.assertTrue(proof["log"]["clean"])
        self.assertEqual(proof["binaries"], {"app": [], "helper": []})
        self.assertEqual(M.coverage_proof([log], {"app": dirty}, 2, reader=lambda path: "-O")["binaries"]["app"], ["__DATA,__llvm_prf_cnts"])
        report = M.payload_report({"app": (clean, clean), "helper": (clean, dirty)})
        self.assertTrue(report["payloadEqual"]["app"]["equal"])
        self.assertFalse(report["payloadEqual"]["helper"]["equal"])

    def test_the_command_line_exit_codes(self):
        clean = self.write("clean", macho([("__TEXT", [("__text", b"code", 0)])]))
        dirty = self.write("dirty", macho([("__TEXT", [("__text", b"cod3", 0)]), ("__DATA", [("__llvm_prf_cnts", b"x", 0)])]))
        log = Path(self.dir.name) / "build.log"
        log.write_text(LogTests.CLEAN.replace("@/dd/W.resp", "-O"))
        out = Path(self.dir.name) / "proof.json"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(M.main(["x", "payload-equal", str(clean), str(clean)]), 0)
            self.assertEqual(M.main(["x", "payload-equal", str(clean), str(dirty)]), 1)
            self.assertEqual(M.main(["x", "no-coverage", str(clean)]), 0)
            self.assertEqual(M.main(["x", "no-coverage", str(clean), str(dirty)]), 1)
            self.assertEqual(M.main(["x", "log-coverage", str(log), "--min-compiles", "2"]), 0)
            self.assertEqual(M.main(["x", "log-coverage", str(log), "--min-compiles", "3"]), 1)
            self.assertEqual(M.main(["x", "coverage-proof", "--out", str(out), "--binary", f"app={clean}", "--min-compiles", "2", str(log)]), 0)
            self.assertEqual(json.loads(out.read_text())["binaries"], {"app": []})
            self.assertEqual(M.main(["x", "coverage-proof", "--out", str(out), "--binary", f"app={dirty}", str(log)]), 1)
            self.assertEqual(M.main(["x", "coverage-proof", "--out", str(out), "--binary", "nameless", str(log)]), 2)
            self.assertEqual(M.main(["x", "sections", str(Path(self.dir.name) / "missing")]), 2)
            self.assertEqual(M.main(["x", "unknown"]), 2)


class LogTests(unittest.TestCase):
    CLEAN = ("CompileC /dd/a.o /src/a.cpp normal arm64 c++\n"
             "    /Applications/Xcode.app/usr/bin/clang -x c++ -O3 -c /src/a.cpp -o /dd/a.o\n"
             "SwiftDriver app\n    builtin-swift-driver -- /Applications/Xcode.app/usr/bin/swiftc -module-name W -O -c @/dd/W.resp\n")

    def test_a_clean_log_passes_and_counts_commands(self):
        audit = M.compile_audit(self.CLEAN, response_reader=lambda path: "-O -whole-module-optimization", min_compiles=2)
        self.assertTrue(audit["clean"])
        self.assertEqual((audit["cCompileCommands"], audit["swiftCompileCommands"], audit["responseFilesRead"]), (1, 1, 1))

    def test_coverage_flags_in_a_command_or_a_response_file_are_found(self):
        direct = self.CLEAN.replace("-O3 -c", "-O3 -fprofile-instr-generate -fcoverage-mapping -c")
        self.assertEqual(M.compile_audit(direct)["coverageFlagLines"][0]["flags"], ["-fcoverage-mapping", "-fprofile-instr-generate"])
        hidden = M.compile_audit(self.CLEAN, response_reader=lambda path: "-profile-generate -profile-coverage-mapping")
        self.assertFalse(hidden["clean"])
        self.assertEqual(hidden["coverageFlagLines"][0]["flags"], ["-profile-coverage-mapping", "-profile-generate"])

    def test_an_empty_log_is_not_a_clean_proof(self):
        audit = M.compile_audit("BUILD SUCCEEDED\n", min_compiles=1)
        self.assertFalse(audit["enoughCompiles"])
        self.assertFalse(audit["clean"])


if __name__ == "__main__":
    unittest.main()
