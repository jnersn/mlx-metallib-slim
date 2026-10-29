#!/usr/bin/env python3
"""Builds the REDUCED precompiled MLX Metal library from the FULL build's own recorded commands. Fails closed.

    reduce_library.py scan   --build-log <cmake-build.log> --core <copy>/Source/Cmlx/mlx --declaration <declaration.json> --out <dir> [--min-os V] [--nax]
    reduce_library.py reduce --build-log <cmake-build.log> --core <copy>/Source/Cmlx/mlx --declaration <declaration.json>
        --allowlist <allowlist.json> --out <dir> [--full-build-dir <cmake dir>] [--allow-incomplete] [--min-os V] [--nax]
    reduce_library.py verify --manifest <manifest.json> --reduced-names <json> --full-names <json> --allowlist <allowlist.json> --out <json>
    reduce_library.py repro  --a <manifest A> --b <manifest B> --fixed-path <dir> --out <json>

The full build (CMake target mlx-metallib, MLX_METAL_JIT=OFF, built with --verbose) logged every `xcrun -sdk macosx metal ... -c SRC ... -o X.air`
command and the link command. This tool replays them unchanged except:
  * base files (declaration reduction.baseFiles + retainWholeFiles) are compiled from the ORIGINAL source with the original command (only `-o` moves);
  * every other kernel file is preprocessed with the same command and `-E -P`, filtered by kernel_filter.py (only unselected explicit instantiations are
    removed; an independent proof is recorded), and the filtered text is compiled with the same flags;
  * the link command gets the new AIR paths and `-o` moves.
`scan` (CPU only, no allowlist) proves the parser accepts every kernel file of the pinned source and writes the name -> file index; run it before the
declaration is frozen. `reduce` (alias `build`) writes <out>/lib/mlx.metallib and <out>/manifest.json. `verify` compares the library's reflected function names (metallib_names.swift) with the allowlist and the full library.
`repro` compares the manifests of two clean builds. --min-os and --nax add the checks of metallib_record.py to the accepted build log.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # needed by @dataclass under `from __future__ import annotations` (Python 3.9)
    spec.loader.exec_module(module)
    return module


K = _load("kernel_filter", HERE / "kernel_filter.py")
A = _load("allowlist", HERE / "allowlist.py")
R = _load("metallib_record", HERE / "metallib_record.py")  # parse_build_log, normalise, flag checks

KERNELS = Path("mlx/backend/metal/kernels")
METAL = "xcrun -sdk macosx metal"


class ReduceError(Exception):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- the recorded commands -------------------------------------------------

def parse_log(text: str, min_os: str | None = None, nax: bool = False) -> dict:
    """compiles: [{tokens, src, air}] in log order; link: {tokens, airs, out}. Fails closed unless exactly one link and one flag set are found."""
    parsed = R.parse_build_log(text)
    problems, _names, flag_sets = R.check_build(parsed, min_os, nax)
    if problems:
        raise ReduceError("the full build log is not the declared build:\n  " + "\n  ".join(problems))
    compiles = []
    for tokens in parsed["compiles"]:
        compiles.append({"tokens": tokens, "src": tokens[tokens.index("-c") + 1], "air": tokens[tokens.index("-o") + 1]})
    link_tokens = parsed["links"][0]
    airs = [t for t in link_tokens if t.endswith(".air")]
    if sorted(Path(a).name for a in airs) != sorted(Path(c["air"]).name for c in compiles):
        raise ReduceError("the link command's AIR files are not exactly the compiled ones")
    return {"compiles": compiles, "link": {"tokens": link_tokens, "airs": airs}, "flagSet": [list(f) for f in flag_sets]}


def replace_output(tokens: list, output: str) -> list:
    out = list(tokens)
    out[out.index("-o") + 1] = str(output)
    return out


def compile_command(entry: dict, source: Path, air: Path) -> list:
    tokens = list(entry["tokens"])
    tokens[tokens.index("-c") + 1] = str(source)
    return replace_output(tokens, str(air))


def preprocess_command(entry: dict, pp: Path) -> list:
    tokens = list(entry["tokens"])
    index = tokens.index("-c")
    tokens[index:index + 2] = ["-E", "-P", entry["src"]]
    return replace_output(tokens, str(pp))


def link_command(link: dict, airs: dict, output: Path) -> list:
    """`airs`: AIR file name -> new path. Every AIR token must map."""
    tokens = []
    for token in link["tokens"]:
        if token.endswith(".air"):
            if Path(token).name not in airs:
                raise ReduceError(f"link input {token} has no reduced AIR")
            tokens.append(str(airs[Path(token).name]))
        else:
            tokens.append(token)
    return replace_output(tokens, str(output))


def kernel_key(src: str, core: Path) -> str:
    """`unary`, `steel/gemm/kernels/steel_gemm_fused`, ... relative to the pinned kernels directory, without suffix."""
    path = Path(src)
    root = (Path(core) / KERNELS)
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ReduceError(f"{src} is not under {root}") from error
    return relative.with_suffix("").as_posix()


def retained_keys(declaration: dict) -> set:
    reduction = declaration["reduction"]
    return set(reduction["baseFiles"]) | set(reduction.get("retainWholeFiles", []))


def run(command: list, log: Path, what: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    log.write_text(f"$ {' '.join(command)}\n--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}\nexit {result.returncode}\n", encoding="utf-8")
    if result.returncode != 0:
        raise ReduceError(f"{what} failed (exit {result.returncode}); log {log}\n{result.stderr[-1500:]}")


# ---- scan / build ----------------------------------------------------------

def preprocess_all(parsed: dict, core: Path, declaration: dict, out: Path) -> dict:
    """Preprocesses every non-retained kernel file and scans it. Returns key -> {entry, pp (path), text, found}."""
    retained = retained_keys(declaration)
    (out / "pp").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    files, seen = {}, {}
    for entry in parsed["compiles"]:
        key = kernel_key(entry["src"], core)
        if key in retained:
            files[key] = {"entry": entry, "retained": True}
            continue
        pp = out / "pp" / (key.replace("/", "__") + ".pp.metal")
        run(preprocess_command(entry, pp), out / "logs" / f"pp-{key.replace('/', '__')}.log", f"preprocessing {key}")
        text = pp.read_text(encoding="utf-8")
        try:
            found = K.scan(text)
        except K.FilterError as error:
            raise ReduceError(f"{key}: the kernel filter cannot parse this file ({error}); retain it whole (declaration reduction.retainWholeFiles, before the freeze) or stop") from error
        for item in found:
            if item.name in seen:
                raise ReduceError(f"kernel name {item.name} is instantiated in both {seen[item.name]} and {key}")
            seen[item.name] = key
        files[key] = {"entry": entry, "retained": False, "pp": pp, "text": text, "found": found}
    missing = retained - set(files)
    if missing:
        raise ReduceError(f"declared base/retained files are not in the full build: {sorted(missing)}")
    return files


def scan_report(files: dict) -> dict:
    trimmed = {k: v for k, v in files.items() if not v["retained"]}
    return {"schema": 1,
            "files": {k: {"ppBytes": len(v["text"].encode("utf-8")), "ppSHA256": sha256_text(v["text"]), "instantiations": len(v["found"]),
                          "names": [i.name for i in v["found"]]} for k, v in sorted(trimmed.items())},
            "retainedFiles": sorted(k for k, v in files.items() if v["retained"]),
            "totalInstantiations": sum(len(v["found"]) for v in trimmed.values()),
            "totalPreprocessedBytes": sum(len(v["text"].encode("utf-8")) for v in trimmed.values()),
            "nameIndex": {i.name: k for k, v in sorted(trimmed.items()) for i in v["found"]}}


def log_checks(args) -> dict:
    return {"min_os": getattr(args, "min_os", None), "nax": getattr(args, "nax", False)}


def cmd_scan(args) -> int:
    parsed = parse_log(Path(args.build_log).read_text(errors="replace"), **log_checks(args))
    declaration = json.loads(Path(args.declaration).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = scan_report(preprocess_all(parsed, Path(args.core), declaration, out))
    (out / "scan.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"scan ok: {len(report['files'])} filtered files, {report['totalInstantiations']} explicit instantiations, {len(report['retainedFiles'])} retained files, "
          f"{report['totalPreprocessedBytes']} preprocessed bytes (ignored output: {out})")
    return 0


def cmd_build(args) -> int:
    declaration_bytes = Path(args.declaration).read_bytes()
    declaration = json.loads(declaration_bytes)
    allowlist = A.verify(Path(args.allowlist))
    if not allowlist["complete"] and not args.allow_incomplete:
        raise ReduceError(f"the allowlist is incomplete (pending {allowlist['pendingJobs']}); pass --allow-incomplete to build a library covering only the traced jobs")
    if allowlist["declarationSHA256"] != hashlib.sha256(declaration_bytes).hexdigest():
        raise ReduceError("the allowlist was derived for another declaration")
    keep = set(allowlist["names"])
    out = Path(args.out)
    if out.exists():
        raise ReduceError(f"{out} exists: a reduced build always starts clean (remove it deliberately)")
    out.mkdir(parents=True)
    (out / "air").mkdir()
    (out / "lib").mkdir()
    parsed = parse_log(Path(args.build_log).read_text(errors="replace"), **log_checks(args))
    core = Path(args.core)
    started = time.time()
    files = preprocess_all(parsed, core, declaration, out)
    index = scan_report(files)["nameIndex"]
    unmapped = sorted(keep - set(index))
    airs, per_file = {}, {}
    for key, item in sorted(files.items()):
        entry = item["entry"]
        air = out / "air" / Path(entry["air"]).name
        record = {"source": entry["src"], "air": air.name}
        if item["retained"]:
            run(compile_command(entry, Path(entry["src"]), air), out / "logs" / f"cc-{key.replace('/', '__')}.log", f"compiling {key}")
            record.update({"mode": "original", "note": "compiled from the original source with the original command"})
        else:
            filtered, proof = K.filter_text(item["text"], keep)
            filtered_path = out / "pp" / (key.replace("/", "__") + ".filtered.metal")
            filtered_path.write_text(filtered, encoding="utf-8")
            run(compile_command(entry, filtered_path, air), out / "logs" / f"cc-{key.replace('/', '__')}.log", f"compiling filtered {key}")
            record.update({"mode": "filtered", "proof": {k: v for k, v in proof.items() if k != "keptNames"}, "keptNames": proof["keptNames"],
                           "filteredSHA256": proof["filteredSHA256"]})
        record["airBytes"] = air.stat().st_size
        record["airSHA256"] = sha256_file(air)
        if args.full_build_dir:
            full_air = next(iter(sorted(Path(args.full_build_dir).rglob(air.name))), None)
            if full_air is not None and item["retained"]:
                record["sameAsFullBuildAir"] = sha256_file(full_air) == record["airSHA256"]
        airs[air.name] = air
        per_file[key] = record
    metallib = out / "lib" / "mlx.metallib"
    command = link_command(parsed["link"], airs, metallib)
    run(command, out / "logs" / "link.log", "linking the reduced metallib")
    kept_in_filtered = sorted({n for r in per_file.values() if r["mode"] == "filtered" for n in r["keptNames"]})
    missing_in_filtered = sorted((set(keep) & set(index)) - set(kept_in_filtered))
    if missing_in_filtered:
        raise ReduceError(f"allowlist names found in the source but not kept by the filter: {missing_in_filtered[:5]}")
    manifest = {
        "schema": 1, "allowlistSHA256": sha256_file(Path(args.allowlist)), "allowlistComplete": allowlist["complete"], "allowlistNames": len(keep),
        "declarationSHA256": hashlib.sha256(declaration_bytes).hexdigest(), "fullBuildLogSHA256": sha256_file(Path(args.build_log)),
        "flagSet": parsed["flagSet"], "linkCommand": command, "files": per_file,
        "explicitInstantiationsInSource": len(index), "explicitInstantiationsKept": len(kept_in_filtered),
        "allowlistNamesNotExplicitInstantiations": unmapped,
        "allowlistNamesNotExplicitInstantiationsNote": "must live in the original base/standalone kernels; `verify` proves they are present in the reflected library",
        "metallib": {"path": str(metallib), "bytes": metallib.stat().st_size, "sha256": sha256_file(metallib)},
        "buildSeconds": round(time.time() - started, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"reduced metallib: {manifest['metallib']['bytes']} bytes sha256 {manifest['metallib']['sha256']}; "
          f"{manifest['explicitInstantiationsKept']} of {manifest['explicitInstantiationsInSource']} explicit instantiations kept, {len(unmapped)} allowlist names resolved by base kernels")
    return 0


# ---- verify ----------------------------------------------------------------

def verify_names(reduced: list, full: list, allowlist: list) -> dict:
    """The library-level proof: every allowlist name is PRESENT in the reduced library, the reduced library holds nothing the full one lacks."""
    reduced_set, full_set, wanted = set(reduced), set(full), set(allowlist)
    missing_from_reduced = sorted(wanted - reduced_set)
    unknown_to_full = sorted(wanted - full_set)
    not_in_full = sorted(reduced_set - full_set)
    return {"allowlistNames": len(wanted), "reducedFunctions": len(reduced_set), "fullFunctions": len(full_set),
            "allowlistMissingFromReduced": missing_from_reduced, "allowlistUnknownToFull": unknown_to_full, "reducedNotInFull": not_in_full,
            "untracedExtraOriginalBaseExports": sorted(reduced_set - wanted)[:400], "untracedExtraCount": len(reduced_set - wanted),
            "removedFromFull": len(full_set - reduced_set),
            "ok": not (missing_from_reduced or unknown_to_full or not_in_full)}


def cmd_verify(args) -> int:
    allowlist = A.verify(Path(args.allowlist))
    reduced, full = json.loads(Path(args.reduced_names).read_text()), json.loads(Path(args.full_names).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    if reduced.get("sha256") != manifest["metallib"]["sha256"]:
        raise ReduceError("the reflected reduced library is not the built one (file hash differs)")
    result = verify_names(reduced["names"], full["names"], allowlist["names"])
    result["reducedMetallibSHA256"] = reduced["sha256"]
    result["fullMetallibSHA256"] = full["sha256"]
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"verify: allowlist {result['allowlistNames']} names, reduced {result['reducedFunctions']} functions, full {result['fullFunctions']}, "
          f"missing from reduced {len(result['allowlistMissingFromReduced'])}, unknown to full {len(result['allowlistUnknownToFull'])} -> {'OK' if result['ok'] else 'REJECTED'}")
    return 0 if result["ok"] else 1


def repro_report(a: dict, b: dict, fixed_path: str) -> dict:
    """Two CLEAN builds from the same inputs at the same fixed path: byte-identical metallib, or REPRO UNRESOLVED (no fake pass)."""
    same = a["metallib"]["sha256"] == b["metallib"]["sha256"]
    differing = sorted(k for k in set(a["files"]) | set(b["files"]) if (a["files"].get(k) or {}).get("airSHA256") != (b["files"].get(k) or {}).get("airSHA256"))
    return {"fixedPath": fixed_path, "buildA": a["metallib"]["sha256"], "buildB": b["metallib"]["sha256"], "bytesA": a["metallib"]["bytes"], "bytesB": b["metallib"]["bytes"],
            "inputsEqual": {key: a[key] == b[key] for key in ("allowlistSHA256", "declarationSHA256", "fullBuildLogSHA256")},
            "differingAirFiles": differing, "identical": same, "verdict": "IDENTICAL" if same else "REPRO UNRESOLVED",
            "note": "same fixed path only; a build at a different path is not claimed"}


def cmd_repro(args) -> int:
    report = repro_report(json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text()), args.fixed_path)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"repro: {report['verdict']} ({report['buildA'][:12]} vs {report['buildB'][:12]}); differing AIR files {report['differingAirFiles']}")
    return 0 if report["identical"] else 1


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("scan", "reduce"):
        p = sub.add_parser(name, aliases=["build"] if name == "reduce" else [])
        p.add_argument("--build-log", required=True)
        p.add_argument("--core", required=True)
        p.add_argument("--declaration", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--min-os", help="every kernel compile must carry -mmacosx-version-min=<this>")
        p.add_argument("--nax", action="store_true", help="the build must contain MLX's NAX kernel objects")
        if name == "reduce":
            p.add_argument("--allowlist", required=True)
            p.add_argument("--full-build-dir")
            p.add_argument("--allow-incomplete", action="store_true")
    v = sub.add_parser("verify")
    for flag in ("manifest", "reduced-names", "full-names", "allowlist", "out"):
        v.add_argument(f"--{flag}", required=True)
    r = sub.add_parser("repro")
    for flag in ("a", "b", "fixed-path", "out"):
        r.add_argument(f"--{flag}", required=True)
    args = parser.parse_args(argv)
    return {"scan": cmd_scan, "build": cmd_build, "reduce": cmd_build, "verify": cmd_verify, "repro": cmd_repro}[args.command](args)


def cli(argv: list | None = None) -> int:
    try:
        return main(argv)
    except (ReduceError, K.FilterError, A.AllowlistError, OSError, KeyError, ValueError) as error:
        print(f"reduce_library: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli())
