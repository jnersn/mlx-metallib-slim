#!/usr/bin/env python3
"""CPU-only checks of reduce_library.py with a FAKE metal compiler (no Metal, no Xcode): the command rewriting, the scan/build flow, the failures.

    python3 -I tests/test_reduce_library.py
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "metallib_slim"
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("reduce_library", LIB / "reduce_library.py")
L = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = L
_spec.loader.exec_module(L)

FLAGS = "-x metal -Wall -Wextra -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions -Wmetal-addr-spaces -mmacosx-version-min=26.2"
NAMES = sorted(L.R.REQUIRED_AIR | L.R.NAX_AIR)
BASE = ["rope", "rms_norm", "layer_norm", "scaled_dot_product_attention"]


def instantiation(name: str) -> str:
    return f'template [[host_name("k_" "{name}")]] [[kernel]] decltype(f<int>) f<int>;\n'


def source(name: str) -> str:
    body = f"// kernel file {name}\ntemplate <typename T> [[kernel]] void f(device T* x) {{ x[0] = 1; }}\n"
    return body + instantiation(name) + (instantiation(name + "_b") if name == "unary" else "")


def fake_run(command, log, what):
    """Stands in for `xcrun metal`: preprocess copies the source, compile and link copy their inputs into the output."""
    out = Path(command[command.index("-o") + 1])
    if "-E" in command:
        out.write_text(Path(command[command.index("-E") + 2]).read_text())
    elif "-c" in command:
        out.write_bytes(Path(command[command.index("-c") + 1]).read_bytes())
    else:
        out.write_bytes(b"".join(Path(t).read_bytes() for t in command if t.endswith(".air")))


class Fixture(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.core = self.root / "core"
        kernels = self.core / "mlx/backend/metal/kernels"
        kernels.mkdir(parents=True)
        for name in NAMES:
            (kernels / f"{name}.metal").write_text(source(name))
        lines = [f"cd /w/b/kernels && xcrun -sdk macosx metal {FLAGS} -c {kernels}/{name}.metal -I{self.core} -o {name}.air" for name in NAMES]
        lines.append(f"cd /w/b/kernels && xcrun -sdk macosx metal -mmacosx-version-min=26.2 {' '.join(n + '.air' for n in NAMES)} -o /w/b/kernels/mlx.metallib")
        self.log = self.root / "build.log"
        self.log.write_text("\n".join(lines) + "\n")
        self.declaration = self.root / "declaration.json"
        self.declaration.write_text(json.dumps({"reduction": {"baseFiles": BASE, "retainWholeFiles": []}}))
        self.original_run = L.run
        L.run = fake_run
        self.addCleanup(lambda: setattr(L, "run", self.original_run))

    def allowlist(self, names, complete=True, file="allowlist.json") -> Path:
        path = self.root / file
        text = json.dumps({"schema": 1, "declarationSHA256": hashlib.sha256(self.declaration.read_bytes()).hexdigest(), "complete": complete, "pendingJobs": [] if complete else ["J1"],
                           "jobs": {}, "nameCount": len(names), "names": sorted(names), "namesSHA256": hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()})
        path.write_text(text)
        path.with_suffix(".sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
        return path

    def args(self, out, allowlist=None, **extra):
        return argparse.Namespace(build_log=str(self.log), core=str(self.core), declaration=str(self.declaration), out=str(out), allowlist=str(allowlist) if allowlist else None,
                                  full_build_dir=None, allow_incomplete=False, **{"min_os": "26.2", "nax": True, **extra})


class RewriteTests(Fixture):
    def test_commands_are_replayed_with_only_the_source_and_output_moved(self):
        parsed = L.parse_log(self.log.read_text())
        self.assertEqual(len(parsed["compiles"]), len(NAMES))
        entry = parsed["compiles"][0]
        original = entry["tokens"]
        compile_tokens = L.compile_command(entry, Path("/x/f.metal"), Path("/o/f.air"))
        self.assertEqual([t for t in compile_tokens if t not in ("/x/f.metal", "/o/f.air")], [t for t in original if t not in (entry["src"], entry["air"])])
        pp = L.preprocess_command(entry, Path("/o/f.pp"))
        self.assertEqual(pp[pp.index("-E"):pp.index("-E") + 3], ["-E", "-P", entry["src"]])
        self.assertNotIn("-c", pp)
        self.assertIn("-fno-fast-math", pp)
        airs = {Path(c["air"]).name: Path("/new") / Path(c["air"]).name for c in parsed["compiles"]}
        link = L.link_command(parsed["link"], airs, Path("/new/mlx.metallib"))
        self.assertTrue(all(t.startswith("/new/") for t in link if t.endswith(".air")))
        self.assertEqual(link[link.index("-o") + 1], "/new/mlx.metallib")
        with self.assertRaises(L.ReduceError):
            L.link_command(parsed["link"], {}, Path("/new/x"))

    def test_a_log_that_is_not_the_declared_build_is_rejected(self):
        with self.assertRaises(L.ReduceError):
            L.parse_log(self.log.read_text().replace("-fno-fast-math", "-ffast-math"))
        with self.assertRaises(L.ReduceError):
            L.parse_log("\n".join(self.log.read_text().splitlines()[:-1]))  # no link command

    def test_min_os_and_nax_checks_apply_only_when_asked_for(self):
        text = self.log.read_text()
        older = text.replace("-mmacosx-version-min=26.2", "-mmacosx-version-min=15.0")
        self.assertTrue(L.parse_log(older)["compiles"])  # no deployment-target check without min_os
        with self.assertRaises(L.ReduceError):
            L.parse_log(older, min_os="26.2")
        self.assertTrue(L.parse_log(older, min_os="15.0")["compiles"])
        without_nax = "\n".join(line for line in text.splitlines() if "_nax.metal" not in line).replace(" quantized_nax.air", "").replace(" steel_attention_nax.air", "").replace(" steel_gemm_fused_nax.air", "")
        self.assertTrue(L.parse_log(without_nax, min_os="26.2")["compiles"])
        with self.assertRaises(L.ReduceError):
            L.parse_log(without_nax, min_os="26.2", nax=True)

    def test_kernel_key_is_relative_to_the_pinned_kernels_directory(self):
        self.assertEqual(L.kernel_key(str(self.core / "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal"), self.core), "steel/gemm/kernels/steel_gemm_fused")
        with self.assertRaises(L.ReduceError):
            L.kernel_key("/elsewhere/unary.metal", self.core)


class FlowTests(Fixture):
    def test_scan_indexes_every_instantiation_of_the_filtered_files_only(self):
        out = self.root / "scan"
        self.assertEqual(L.cmd_scan(self.args(out)), 0)
        report = json.loads((out / "scan.json").read_text())
        self.assertEqual(report["retainedFiles"], sorted(BASE))
        self.assertEqual(report["totalInstantiations"], (len(NAMES) - len(BASE)) + 1)
        self.assertEqual(report["nameIndex"]["k_unary_b"], "unary")
        self.assertNotIn("k_rope", report["nameIndex"])

    def test_build_keeps_only_the_allowlist_and_leaves_base_files_original(self):
        out = self.root / "reduced"
        allowlist = self.allowlist(["k_unary", "k_gemv", "rope_plain"])
        self.assertEqual(L.cmd_build(self.args(out, allowlist)), 0)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual((manifest["explicitInstantiationsKept"], manifest["allowlistNamesNotExplicitInstantiations"]), (2, ["rope_plain"]))
        filtered = (out / "pp/unary.filtered.metal").read_text()
        self.assertEqual([i.name for i in L.K.scan(filtered)], ["k_unary"])  # only the selected instantiation remains
        self.assertIn(instantiation("unary"), filtered)  # and its declaration is preserved as written
        self.assertNotIn("unary_b", filtered)
        self.assertEqual(L.K.scan((out / "pp/sort.filtered.metal").read_text()), [])  # an unselected file keeps no instantiation
        self.assertEqual([i.name for i in L.K.scan((out / "pp/gemv.filtered.metal").read_text())], ["k_gemv"])
        self.assertEqual((out / "air/rope.air").read_bytes(), (self.core / "mlx/backend/metal/kernels/rope.metal").read_bytes())
        self.assertEqual(manifest["files"]["rope"]["mode"], "original")
        self.assertEqual(manifest["files"]["unary"]["mode"], "filtered")
        self.assertTrue(manifest["files"]["unary"]["proof"]["otherTextIdentical"])
        self.assertEqual(manifest["metallib"]["sha256"], hashlib.sha256((out / "lib/mlx.metallib").read_bytes()).hexdigest())
        self.assertLess(manifest["metallib"]["bytes"], sum(len(source(n)) for n in NAMES) * 2)

    def test_two_clean_builds_from_the_same_inputs_are_byte_identical_with_a_deterministic_compiler(self):
        allowlist = self.allowlist(["k_unary"])
        hashes = []
        for name in ("one", "two"):
            L.cmd_build(self.args(self.root / name, allowlist))
            hashes.append(json.loads((self.root / name / "manifest.json").read_text())["metallib"]["sha256"])
        self.assertEqual(hashes[0], hashes[1])

    def test_failures_are_closed(self):
        allowlist = self.allowlist(["k_unary"])
        (self.root / "exists").mkdir()
        with self.assertRaises(L.ReduceError):
            L.cmd_build(self.args(self.root / "exists", allowlist))  # never builds into an existing directory
        with self.assertRaises(L.ReduceError):
            L.cmd_build(self.args(self.root / "partial", self.allowlist(["k_unary"], complete=False, file="partial.json")))
        bad = self.core / "mlx/backend/metal/kernels/sort.metal"
        bad.write_text('[[host_name("x")]] [[kernel]] void sort_def() {}\n')
        with self.assertRaises(L.ReduceError) as caught:
            L.cmd_build(self.args(self.root / "bad", allowlist))
        self.assertIn("retain it whole", str(caught.exception))
        other = self.root / "other.json"
        other.write_text("{}")
        with self.assertRaises(L.ReduceError):  # an allowlist derived for another declaration
            L.cmd_build(argparse.Namespace(**{**vars(self.args(self.root / "x", allowlist)), "declaration": str(other)}))


class ReproTests(unittest.TestCase):
    def manifest(self, sha, air="a1"):
        return {"metallib": {"sha256": sha, "bytes": 10}, "files": {"unary": {"airSHA256": air}}, "allowlistSHA256": "al", "declarationSHA256": "d", "fullBuildLogSHA256": "l"}

    def test_identical_builds_pass_and_differing_builds_are_REPRO_UNRESOLVED_with_the_differing_air_named(self):
        same = L.repro_report(self.manifest("x"), self.manifest("x"), "/fixed")
        self.assertEqual((same["verdict"], same["differingAirFiles"]), ("IDENTICAL", []))
        different = L.repro_report(self.manifest("x"), self.manifest("y", air="a2"), "/fixed")
        self.assertEqual((different["verdict"], different["identical"], different["differingAirFiles"]), ("REPRO UNRESOLVED", False, ["unary"]))
        self.assertIn("different path is not claimed", different["note"])


class VerifyTests(unittest.TestCase):
    def test_every_allowlist_name_must_be_present_and_nothing_may_be_new(self):
        ok = L.verify_names(["a", "b", "extra"], ["a", "b", "c", "extra"], ["a", "b"])
        self.assertTrue(ok["ok"])
        self.assertEqual((ok["untracedExtraCount"], ok["removedFromFull"]), (1, 1))
        self.assertFalse(L.verify_names(["a"], ["a", "b"], ["a", "b"])["ok"])  # b missing from the reduced library
        self.assertEqual(L.verify_names(["a"], ["a", "b"], ["a", "b"])["allowlistMissingFromReduced"], ["b"])
        self.assertFalse(L.verify_names(["a", "b"], ["a", "b"], ["a", "zz"])["ok"])  # a traced name the full library never had
        self.assertFalse(L.verify_names(["a", "b", "new"], ["a", "b"], ["a"])["ok"])  # the reduced library holds something the full one lacks


if __name__ == "__main__":
    unittest.main()
