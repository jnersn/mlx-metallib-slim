#!/usr/bin/env python3
"""CPU-only checks of metallib_record.py on synthetic CMake caches and logs (no CMake, no Metal).

    python3 -I tests/test_metallib_record.py
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "metallib_slim"
_spec = importlib.util.spec_from_file_location("metallib_record", LIB / "metallib_record.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

FLAGS = "-x metal -Wall -Wextra -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions -Wmetal-addr-spaces -mmacosx-version-min=26.2"
CONFIGURE = """-- Building MLX for arm64 processor on Darwin
-- Building on macOS 27.0.1
-- Building with macOS 27.0 SDK
-- Building for macOS 26.2
-- Metal version: 400
-- Configuring done
"""


ALL_AIR = M.REQUIRED_AIR | M.NAX_AIR


def cache(**over) -> str:
    values = {**M.expected_cache("26.2"), "CMAKE_HOME_DIRECTORY": "/work/core", "CMAKE_GENERATOR": "Unix Makefiles", "CMAKE_OSX_SYSROOT": "/sdk"}
    values.update(over)
    return "# comment\n// help text\n" + "".join(f"{key}:STRING={value}\n" for key, value in values.items())


def build_log(names=None, flags=FLAGS, link=True) -> str:
    names = sorted(ALL_AIR) if names is None else names
    lines = ["[  1%] Building mlx-metallib", "make[3]: Entering directory"]
    for name in names:
        lines.append(f"cd /work/b/kernels && xcrun -sdk macosx metal {flags} -c /work/core/mlx/backend/metal/kernels/{name}.metal -I/work/core -o {name}.air")
    if link:
        lines.append(f"cd /work/b/kernels && xcrun -sdk macosx metal -mmacosx-version-min=26.2 {' '.join(n + '.air' for n in names)} -o /work/b/kernels/mlx.metallib")
    return "\n".join(lines) + "\n"


class ParseTests(unittest.TestCase):
    def test_cache_parsing_and_the_declared_options(self):
        parsed = M.parse_cache(cache())
        self.assertEqual(M.check_cache(parsed, "/work/core", "26.2"), [])
        self.assertEqual(parsed["MLX_METAL_JIT"], "OFF")

    def test_each_wrong_option_or_source_is_reported(self):
        for key, bad in [("MLX_METAL_JIT", "ON"), ("CMAKE_BUILD_TYPE", "Debug"), ("CMAKE_OSX_DEPLOYMENT_TARGET", "15.0"), ("CMAKE_OSX_ARCHITECTURES", "x86_64"),
                         ("MLX_BUILD_GGUF", "ON"), ("MLX_BUILD_TESTS", "ON"), ("MLX_BUILD_EXAMPLES", "ON"), ("MLX_BUILD_PYTHON_BINDINGS", "ON")]:
            problems = M.check_cache(M.parse_cache(cache(**{key: bad})), "/work/core", "26.2")
            self.assertTrue(any(key in problem for problem in problems), key)
        self.assertTrue(M.check_cache(M.parse_cache(cache(CMAKE_HOME_DIRECTORY="/elsewhere/core")), "/work/core", "26.2"), "built from another source tree")

    def test_the_configure_log_must_show_the_nax_capable_toolchain(self):
        info = M.parse_configure_log(CONFIGURE)
        self.assertEqual(info, {"metalVersion": 400, "sdkVersion": "27.0", "deploymentTarget": "26.2", "withoutNaxWarning": False})
        self.assertEqual(M.check_configure(info, "26.2", nax=True), [])
        self.assertTrue(M.check_configure(M.parse_configure_log(CONFIGURE + "CMake Warning: Building without NAX kernels\n"), "26.2", nax=True))
        self.assertTrue(M.check_configure(M.parse_configure_log(CONFIGURE.replace("Metal version: 400", "Metal version: 320")), "26.2", nax=True))
        self.assertTrue(M.check_configure(M.parse_configure_log(CONFIGURE.replace("27.0 SDK", "26.1 SDK")), "26.2", nax=True))
        self.assertTrue(M.check_configure(M.parse_configure_log(CONFIGURE.replace("for macOS 26.2", "for macOS 15.0")), "26.2", nax=True))
        self.assertTrue(M.check_configure(M.parse_configure_log(""), "26.2", nax=True))

    def test_without_nax_only_the_deployment_target_is_checked(self):
        older = CONFIGURE.replace("Metal version: 400", "Metal version: 320").replace("27.0 SDK", "15.5 SDK").replace("for macOS 26.2", "for macOS 15.0")
        self.assertEqual(M.check_configure(M.parse_configure_log(older + "Building without NAX kernels\n"), "15.0"), [])
        self.assertTrue(M.check_configure(M.parse_configure_log(older), "26.2"))
        names = sorted(M.REQUIRED_AIR)
        problems, _, _ = M.check_build(M.parse_build_log(build_log(names=names, flags=FLAGS.replace("26.2", "15.0"))), "15.0")
        self.assertEqual(problems, [])
        problems, _, _ = M.check_build(M.parse_build_log(build_log(names=names)), "26.2", nax=True)
        self.assertTrue(any("quantized_nax" in p for p in problems))

    def test_the_build_log_gives_compiles_and_exactly_one_link(self):
        parsed = M.parse_build_log(build_log())
        self.assertEqual(len(parsed["compiles"]), len(ALL_AIR))
        self.assertEqual(len(parsed["links"]), 1)
        problems, names, flag_sets = M.check_build(parsed, "26.2", nax=True)
        self.assertEqual(problems, [])
        self.assertEqual(names, sorted(ALL_AIR))
        self.assertEqual(len(flag_sets), 1)
        self.assertIn("-fno-fast-math", flag_sets[0])

    def test_flag_violations_are_reported(self):
        cases = {
            "missing -fno-fast-math": FLAGS.replace("-fno-fast-math ", ""),
            "fast math on": FLAGS + " -ffast-math",
            "metal fast math mode": FLAGS + " -fmetal-math-mode=fast",
            "wrong deployment target": FLAGS.replace("26.2", "15.0"),
        }
        for name, flags in cases.items():
            problems, _, _ = M.check_build(M.parse_build_log(build_log(flags=flags)), "26.2", nax=True)
            self.assertTrue(problems, name)

    def test_two_flag_sets_a_missing_kernel_or_a_missing_link_are_reported(self):
        log = build_log() + f"cd /x && xcrun -sdk macosx metal {FLAGS} -O3 -c /x/extra.metal -o extra.air\n"
        problems, _, flag_sets = M.check_build(M.parse_build_log(log), "26.2", nax=True)
        self.assertEqual(len(flag_sets), 2)
        self.assertTrue(any("different flag sets" in p for p in problems))
        problems, _, _ = M.check_build(M.parse_build_log(build_log(names=sorted(ALL_AIR - {"quantized_nax"}))), "26.2", nax=True)
        self.assertTrue(any("quantized_nax" in p for p in problems))
        problems, _, _ = M.check_build(M.parse_build_log(build_log(names=sorted(ALL_AIR - {"sort"}))), "26.2")
        self.assertTrue(any("sort" in p for p in problems))
        problems, _, _ = M.check_build(M.parse_build_log(build_log(link=False)), "26.2", nax=True)
        self.assertTrue(any("link" in p for p in problems))
        problems, _, _ = M.check_build(M.parse_build_log("nothing useful\n"), "26.2", nax=True)
        self.assertTrue(any("verbose" in p for p in problems))


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.build = Path(self.tmp.name) / "build"
        (self.build / "mlx/backend/metal/kernels").mkdir(parents=True)
        self.core = Path(self.tmp.name) / "core"
        self.core.mkdir()
        (self.build / "CMakeCache.txt").write_text(cache(CMAKE_HOME_DIRECTORY=str(self.core)))
        self.metallib = self.build / "mlx/backend/metal/kernels/mlx.metallib"
        self.metallib.write_bytes(b"MTLB" + b"steel_gemm_fused_nax_nn_float16 steel_attention_nax attention_nax_dsplit affine_qmm_t_nax_x" + bytes(100))

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, **over):
        args = {"configure_log": CONFIGURE, "build_log": build_log(), "cmake_version": "cmake version 4.4.5", "nax": True}
        args.update(over)
        return M.record(self.build, self.core, args["configure_log"], args["build_log"], args["cmake_version"], "26.2", args["nax"])

    def test_a_declared_build_is_recorded_with_the_metallib_hash(self):
        manifest = self.record()
        self.assertEqual(manifest["metallib"]["sha256"], M.sha256_file(self.metallib))
        self.assertEqual(manifest["metallib"]["bytes"], self.metallib.stat().st_size)
        self.assertTrue(all(manifest["metallib"]["nameMarkers"].values()))
        self.assertEqual(manifest["cacheValues"]["MLX_METAL_JIT"], "OFF")
        self.assertEqual(manifest["kernelCompileCommands"], len(ALL_AIR))
        self.assertEqual((manifest["minimumOS"], manifest["nax"]), ("26.2", True))
        self.assertEqual(manifest["target"], "mlx-metallib")
        json.dumps(manifest)

    def test_a_metallib_without_the_nax_kernels_is_refused_only_with_nax(self):
        self.metallib.write_bytes(b"MTLB" + b"steel_gemm_fused_nn_float16" + bytes(100))
        with self.assertRaises(M.RecordError):
            self.record()
        self.assertEqual(self.record(nax=False)["metallib"]["nameMarkers"], {})

    def test_each_failed_precondition_refuses_the_record(self):
        (self.build / "CMakeCache.txt").write_text(cache(CMAKE_HOME_DIRECTORY=str(self.core), MLX_METAL_JIT="ON"))
        with self.assertRaises(M.RecordError):
            self.record()
        (self.build / "CMakeCache.txt").write_text(cache(CMAKE_HOME_DIRECTORY=str(self.core)))
        with self.assertRaises(M.RecordError):
            self.record(configure_log=CONFIGURE + "Building without NAX kernels\n")
        with self.assertRaises(M.RecordError):
            self.record(build_log=build_log(flags=FLAGS + " -ffast-math"))

    def test_exactly_one_metallib_must_exist(self):
        (self.build / "other").mkdir()
        (self.build / "other/mlx.metallib").write_bytes(b"x")
        with self.assertRaises(M.RecordError):
            self.record()
        self.metallib.unlink()
        (self.build / "other/mlx.metallib").unlink()
        with self.assertRaises(M.RecordError):
            self.record()


if __name__ == "__main__":
    unittest.main()
