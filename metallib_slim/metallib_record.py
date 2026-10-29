#!/usr/bin/env python3
"""Records and verifies the FULL precompiled MLX metallib build (the CMake target `mlx-metallib`, MLX_METAL_JIT=OFF). Fails closed.

    metallib_record.py --build-dir <cmake binary dir> --source-core <copy>/Source/Cmlx/mlx --configure-log <file> --build-log <file>
        --cmake-version "<text>" --min-os <deployment target> [--nax] --out <json>

Build the MLX core of the prepared source copy with CMake (Release, arm64, -DCMAKE_OSX_DEPLOYMENT_TARGET=<min-os>, -DMLX_METAL_JIT=OFF, GGUF, tests,
examples and Python bindings off), only the target `mlx-metallib`, and keep the verbose configure and build logs. This tool reads the CMake cache
and both logs and writes a manifest:
  * the cache holds exactly the declared options and was configured from the prepared core copy;
  * every kernel compile command is recorded; each carries -fno-fast-math and -mmacosx-version-min=<min-os> and none enables fast math; the
    normalised flag set is one;
  * exactly one mlx.metallib exists; its size and SHA-256 are recorded.
With --nax the build must also contain MLX's NAX kernels (the Metal 4 kernels MLX uses on newer GPUs when built for macOS 26.2 or later): the
configure log shows a Metal 4 toolchain, a macOS 26.2 or later SDK and NO "Building without NAX kernels" warning, the NAX kernel objects are
compiled, and NAX kernel names are inside the metallib.

reduce_library.py uses parse_build_log and check_build of this module to accept a build log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

EXPECTED_CACHE = {
    "CMAKE_BUILD_TYPE": "Release",
    "CMAKE_OSX_ARCHITECTURES": "arm64",
    "MLX_METAL_JIT": "OFF",
    "MLX_BUILD_METAL": "ON",
    "MLX_BUILD_GGUF": "OFF",  # mlx/io would FetchContent gguflib (unavailable offline); GGUF I/O is not part of mlx-metallib
    "MLX_BUILD_TESTS": "OFF",
    "MLX_BUILD_EXAMPLES": "OFF",
    "MLX_BUILD_PYTHON_BINDINGS": "OFF",
    "MLX_BUILD_BENCHMARKS": "OFF",
}
# Kernel objects a complete mlx-metallib build compiles (a log that lacks one is not the whole build), and the NAX ones (--nax).
REQUIRED_AIR = {"steel_gemm_fused", "steel_attention", "quantized", "fp_quantized", "binary", "unary", "copy", "reduce", "softmax", "rms_norm",
                "layer_norm", "rope", "scaled_dot_product_attention", "gemv", "sort", "scan"}
NAX_AIR = {"steel_gemm_fused_nax", "steel_attention_nax", "quantized_nax"}
NAX_MARKERS = [b"steel_gemm_fused_nax_", b"attention_nax", b"affine_qmm_t_nax"]
NAX_MIN_SDK = (26, 2)
FAST_MATH = ("-ffast-math", "-funsafe-math-optimizations", "-fmetal-math-mode=fast", "-fmetal-math-mode=relaxed")


class RecordError(Exception):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cache(text: str) -> dict:
    """CMakeCache.txt entries `KEY:TYPE=VALUE` (comments and blank lines skipped)."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("//", "#")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        key, value = line.split("=", 1)
        out[key.split(":", 1)[0]] = value
    return out


def expected_cache(min_os: str) -> dict:
    return {**EXPECTED_CACHE, "CMAKE_OSX_DEPLOYMENT_TARGET": min_os}


def check_cache(cache: dict, source_core: str, min_os: str) -> list:
    problems = [f"cache {key} is {cache.get(key)!r}, expected {want!r}" for key, want in expected_cache(min_os).items() if cache.get(key) != want]
    if Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve() != Path(source_core).resolve():
        problems.append(f"CMAKE_HOME_DIRECTORY {cache.get('CMAKE_HOME_DIRECTORY')!r} is not the prepared core copy {source_core!r}")
    return problems


def version_tuple(text: str) -> tuple:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def parse_configure_log(text: str) -> dict:
    metal = re.search(r"Metal version: (\d+)", text)
    sdk = re.search(r"Building with macOS ([\d.]+) SDK", text)
    target = re.search(r"Building for macOS ([\d.]+)", text)
    return {"metalVersion": int(metal.group(1)) if metal else None, "sdkVersion": sdk.group(1) if sdk else None,
            "deploymentTarget": target.group(1) if target else None, "withoutNaxWarning": "Building without NAX kernels" in text}


def check_configure(info: dict, min_os: str, nax: bool = False) -> list:
    problems = []
    if nax:
        if info["withoutNaxWarning"]:
            problems.append("configure log says 'Building without NAX kernels': the metallib would lack the NAX kernels the JIT build uses")
        if info["metalVersion"] is None or info["metalVersion"] < 400:
            problems.append(f"Metal version {info['metalVersion']} (need >= 400 for the NAX kernels)")
        if info["sdkVersion"] is None or version_tuple(info["sdkVersion"]) < NAX_MIN_SDK:
            problems.append(f"macOS SDK {info['sdkVersion']} (need >= {'.'.join(map(str, NAX_MIN_SDK))} for the NAX kernels)")
    if info["deploymentTarget"] != min_os:
        problems.append(f"configure log deployment target {info['deploymentTarget']!r}, expected {min_os}")
    return problems


def parse_build_log(text: str) -> dict:
    """The kernel compile commands (`xcrun -sdk macosx metal ... -c SRC ... -o X.air`) and the link command (`... X.air ... -o mlx.metallib`)."""
    compiles, links = [], []
    for line in text.splitlines():
        if "xcrun -sdk macosx metal" not in line:
            continue
        command = line[line.index("xcrun -sdk macosx metal"):]
        try:
            tokens = shlex.split(command)
        except ValueError:
            continue
        if "-c" in tokens and "-o" in tokens:
            compiles.append(tokens)
        elif "-o" in tokens and tokens[tokens.index("-o") + 1].endswith("mlx.metallib"):
            links.append(tokens)
    return {"compiles": compiles, "links": links}


def normalise(tokens: list) -> tuple:
    """The flag set of a compile command without the per-kernel source and output."""
    out, skip = [], False
    for index, token in enumerate(tokens):
        if skip:
            skip = False
        elif token in ("-c", "-o"):
            skip = True
        else:
            out.append(token)
    return tuple(out)


def check_build(parsed: dict, min_os: str | None = None, nax: bool = False) -> tuple:
    """(problems, kernel object names, flag sets). `min_os`: every compile must carry -mmacosx-version-min=<min_os> (unchecked when None)."""
    problems = []
    compiles = parsed["compiles"]
    if not compiles:
        problems.append("no kernel compile command found in the build log (was it built with --verbose?)")
    names = set()
    for tokens in compiles:
        air = Path(tokens[tokens.index("-o") + 1]).name
        names.add(air[:-4] if air.endswith(".air") else air)
        if "-fno-fast-math" not in tokens:
            problems.append(f"{air}: -fno-fast-math missing")
        if min_os is not None and f"-mmacosx-version-min={min_os}" not in tokens:
            problems.append(f"{air}: -mmacosx-version-min={min_os} missing")
        if any(token.startswith(FAST_MATH) for token in tokens):
            problems.append(f"{air}: a fast-math option is set")
    flag_sets = {normalise(tokens) for tokens in compiles}
    if len(flag_sets) > 1:
        problems.append(f"{len(flag_sets)} different flag sets across kernel compiles")
    missing = sorted((REQUIRED_AIR | (NAX_AIR if nax else set())) - names)
    if missing:
        problems.append(f"expected kernel objects missing from the build: {missing}")
    if len(parsed["links"]) != 1:
        problems.append(f"expected exactly one metallib link command, found {len(parsed['links'])}")
    return problems, sorted(names), sorted(flag_sets)


def find_metallib(build_dir: Path) -> Path:
    found = sorted(path for path in build_dir.rglob("mlx.metallib") if path.is_file())
    if len(found) != 1:
        raise RecordError(f"expected exactly one mlx.metallib under {build_dir}, found {len(found)}")
    return found[0]


def check_metallib(path: Path, nax: bool = False) -> tuple:
    data = path.read_bytes()
    markers = {marker.decode(): marker in data for marker in (NAX_MARKERS if nax else [])}
    problems = [f"kernel name {name!r} not found inside the metallib" for name, present in markers.items() if not present]
    if len(data) == 0:
        problems.append("the metallib is empty")
    return problems, markers, len(data)


def record(build_dir: Path, source_core: Path, configure_log: str, build_log: str, cmake_version: str, min_os: str, nax: bool = False) -> dict:
    cache = parse_cache((build_dir / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace"))
    configure = parse_configure_log(configure_log)
    parsed = parse_build_log(build_log)
    build_problems, air_names, flag_sets = check_build(parsed, min_os, nax)
    metallib = find_metallib(build_dir)
    metallib_problems, markers, size = check_metallib(metallib, nax)
    problems = check_cache(cache, str(source_core), min_os) + check_configure(configure, min_os, nax) + build_problems + metallib_problems
    if problems:
        raise RecordError("precompiled metallib build is not the declared one:\n  " + "\n  ".join(problems))
    return {
        "schema": 1,
        "cmakeVersion": cmake_version,
        "generator": cache.get("CMAKE_GENERATOR"),
        "target": "mlx-metallib",
        "minimumOS": min_os,
        "nax": nax,
        "sourceCore": str(source_core),
        "cacheValues": {key: cache.get(key) for key in sorted(EXPECTED_CACHE) + ["CMAKE_GENERATOR", "CMAKE_HOME_DIRECTORY", "CMAKE_OSX_SYSROOT",
                                                                              "FETCHCONTENT_FULLY_DISCONNECTED", "FETCHCONTENT_SOURCE_DIR_METAL_CPP",
                                                                              "FETCHCONTENT_SOURCE_DIR_JSON", "FETCHCONTENT_SOURCE_DIR_FMT"]},
        "configure": configure,
        "kernelObjects": air_names,
        "kernelCompileCommands": len(parsed["compiles"]),
        "kernelFlagSet": [list(flags) for flags in flag_sets],
        "linkCommand": parsed["links"][0],
        "metallib": {"path": str(metallib), "bytes": size, "sha256": sha256_file(metallib), "nameMarkers": markers},
        "configureLogSHA256": hashlib.sha256(configure_log.encode()).hexdigest(),
        "buildLogSHA256": hashlib.sha256(build_log.encode()).hexdigest(),
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--source-core", required=True, type=Path)
    parser.add_argument("--configure-log", required=True, type=Path)
    parser.add_argument("--build-log", required=True, type=Path)
    parser.add_argument("--cmake-version", required=True)
    parser.add_argument("--min-os", required=True, help="the deployment target the build was configured for, such as 26.2")
    parser.add_argument("--nax", action="store_true", help="the build must contain MLX's NAX kernels")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = record(args.build_dir, args.source_core, args.configure_log.read_text(encoding="utf-8", errors="replace"),
                      args.build_log.read_text(encoding="utf-8", errors="replace"), args.cmake_version, args.min_os, args.nax)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"metallib recorded: {manifest['metallib']['bytes']} bytes sha256 {manifest['metallib']['sha256']}, {manifest['kernelCompileCommands']} kernel compiles")
    return 0


def cli(argv: list | None = None) -> int:
    try:
        return main(argv)
    except (RecordError, OSError) as error:
        print(f"metallib_record: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli())
