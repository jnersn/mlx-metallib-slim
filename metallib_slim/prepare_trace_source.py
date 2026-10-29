#!/usr/bin/env python3
"""Prepares a TRACED source copy of a pinned mlx-swift checkout: the no-JIT copy of prepare_source.py plus a hook that records which kernels of the
default (precompiled) library MLX asks for. Build your app (the process that runs MLX) against it, run your workload once per GPU architecture class, and feed the
trace files to `allowlist.py build`. Fails closed.

    prepare_trace_source.py --stock-checkout <SourcePackages/checkouts/mlx-swift> --package-dir <your package> --out <dir>
        [--expect-patch <package-nojit.patch>] [--expect-trace-patch <device-names-trace.patch>] [--mlx-swift-revision R ...]

What it does, in addition to prepare_source.py --variant precompiled (whose manifest it embeds):
  * hooks Source/Cmlx/mlx/mlx/backend/metal/device.cpp of the COPY: Device::get_kernel notes the base name of every kernel it is asked for from the
    default library, before any cache lookup, and the first time a name is seen it writes one JSON line to <prefix>.<pid>.jsonl, where <prefix> is
    the absolute path in the environment variable MLX_METALLIB_SLIM_TRACE (without it, a hook costs one load and writes nothing);
  * proves the copy differs from the stock checkout in exactly Package.swift and device.cpp;
  * writes <out>/device-names-trace.patch and <out>/manifest-trace.json; with --expect-trace-patch the patch must equal that file byte for byte.
Every edit is an exact-text anchor that must occur exactly once (zero fuzz). The hook never changes a compile option, precision or source text, and
never writes to stdout or stderr. A traced build is for discovery only: never ship it.

Trace file lines: {"event":"header",...} first, {"event":"call","kind":"default_kernel","label":"<name>"} per distinct name, and
{"event":"summary","lines":N,"dropped":0,"write_failures":0,"lookups":N,"distinct":N} at exit (allowlist.py rejects a trace without it).
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


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = _load("prepare_common")
S = _load("prepare_source")
PrepareError = P.PrepareError

DEVICE = "Source/Cmlx/mlx/mlx/backend/metal/device.cpp"
TRACE_ENV = "MLX_METALLIB_SLIM_TRACE"

TRACE_BLOCK = r'''// metallib-slim kernel-name trace (traced copy only): MLX_METALLIB_SLIM_TRACE=<absolute file prefix> writes the DISTINCT base names the
// default (precompiled) library is asked for, one JSON line each, to <prefix>.<pid>.jsonl, never to a standard stream. No timing, no option, source
// or precision change. A header opens the file; a summary (lines, dropped, write_failures, lookups) closes it at exit.
namespace reduced_trace {

struct State {
  bool on = false;
  int fd = -1;
  std::mutex mtx;
  std::set<std::string> seen;
  long lines = 0;
  long dropped = 0;
  long write_failures = 0;
  long long lookups = 0;
};

State& state();

std::string escape(const std::string& s) {
  std::string out;
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += static_cast<char>(c);
    } else if (c < 0x20 || c >= 0x7f) {
      char buf[8];
      snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    } else {
      out += static_cast<char>(c);
    }
  }
  return out;
}

void write_line(State& st, const std::string& line) {
  if (st.fd < 0) {
    st.write_failures++;
    return;
  }
  auto n = ::write(st.fd, line.data(), line.size());
  if (n != static_cast<ssize_t>(line.size())) {
    st.write_failures++;
  }
}

void finish() {
  auto& st = state();
  if (!st.on) {
    return;
  }
  std::lock_guard<std::mutex> lock(st.mtx);
  std::ostringstream o;
  o << "{\"event\":\"summary\",\"pid\":" << getpid() << ",\"lines\":" << st.lines
    << ",\"dropped\":" << st.dropped << ",\"write_failures\":" << st.write_failures
    << ",\"lookups\":" << st.lookups << ",\"distinct\":" << st.seen.size() << "}\n";
  write_line(st, o.str());
  ::close(st.fd);
  st.fd = -1;
}

State& state() {
  static State* st = [] {
    auto* s = new State();
    if (const char* prefix = std::getenv("MLX_METALLIB_SLIM_TRACE"); prefix && *prefix == '/') {
      auto path = std::string(prefix) + "." + std::to_string(getpid()) + ".jsonl";
      s->fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_CLOEXEC, 0600);
      s->on = s->fd >= 0;
      if (s->on) {
        std::ostringstream o;
        o << "{\"event\":\"header\",\"pid\":" << getpid() << ",\"variant\":\"default-library-names\"}\n";
        write_line(*s, o.str());
        std::atexit(finish);
      }
    }
    return s;
  }();
  return *st;
}

void note(const std::string& name) {
  auto& st = state();
  if (!st.on) {
    return;
  }
  std::lock_guard<std::mutex> lock(st.mtx);
  st.lookups++;
  if (!st.seen.insert(name).second) {
    return;
  }
  st.lines++;
  write_line(st, "{\"event\":\"call\",\"kind\":\"default_kernel\",\"label\":\"" + escape(name) + "\"}\n");
}

} // namespace reduced_trace

'''

# (name, exact anchor, replacement). Each anchor must occur exactly once in the pinned device.cpp.
EDITS = [
    ("includes",
     "#include <cstdlib>\n#include <sstream>\n",
     "#include <fcntl.h>\n#include <unistd.h>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <mutex>\n#include <set>\n#include <sstream>\n#include <string>\n"),
    ("trace-block",
     "std::pair<MTL::Library*, NS::Error*> load_library_from_path(\n    MTL::Device* device,\n",
     TRACE_BLOCK + "std::pair<MTL::Library*, NS::Error*> load_library_from_path(\n    MTL::Device* device,\n"),
    ("default-library-lookup",
     "  const auto& kname = hash_name.empty() ? base_name : hash_name;\n",
     "  const auto& kname = hash_name.empty() ? base_name : hash_name;\n  if (mtl_lib == default_library_.get()) {\n    reduced_trace::note(base_name);\n  }\n"),
]



def apply_edits(source: str, edits: list | None = None) -> tuple:
    """Zero fuzz: every anchor must occur exactly once, otherwise nothing is produced."""
    return P.apply_edits(source, EDITS if edits is None else edits, "device.cpp")


def prepare(stock: Path, package_dir: Path, out: Path, pins: dict | None = None, expect_patch: Path | None = None,
            expect_trace_patch: Path | None = None) -> dict:
    base = S.prepare("precompiled", stock, package_dir, out, None, None, pins, expect_patch)
    copy = out / "mlx-swift"
    stock_text = (stock / DEVICE).read_text(encoding="utf-8")
    hooked, report = apply_edits(stock_text)
    target = copy / DEVICE
    target.chmod(target.stat().st_mode | 0o200)  # SwiftPM sources are read-only; the isolated copy only
    target.write_text(hooked, encoding="utf-8")
    _, stock_files = P.tree_digest(stock)
    digest, copy_files = P.tree_digest(copy)
    changed = P.file_diff(stock_files, copy_files)
    allowed = sorted(["Package.swift", DEVICE])
    if changed != allowed:
        raise PrepareError(f"the trace copy differs from the stock checkout in {changed}, allowed exactly {allowed}")
    diff = "".join(difflib.unified_diff(stock_text.splitlines(True), hooked.splitlines(True), "a/mlx/backend/metal/device.cpp", "b/mlx/backend/metal/device.cpp"))
    (out / "device-names-trace.patch").write_text(diff, encoding="utf-8")
    result = {
        "schema": 1, "sourceManifest": base, "traceEnvironment": TRACE_ENV,
        "tracePatch": P.compare_expected(expect_trace_patch, diff, "device.cpp trace patch") if expect_trace_patch else None,
        "trace": {
            "allowedChangedFiles": allowed, "changedFiles": changed, "otherFilesIdentical": True,
            "deviceCppSHA256": {"stock": stock_files[DEVICE], "hooked": copy_files[DEVICE]},
            "editsSHA256": P.edits_digest(EDITS), "edits": report,
            "patchSHA256": P.sha256_bytes(diff.encode()), "copyTreeDigestAfterHooks": digest, "stockTreeDigest": base["stockTree"]["digest"],
        },
    }
    (out / "manifest-trace.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stock-checkout", required=True, type=Path)
    parser.add_argument("--package-dir", required=True, type=Path, help="the consumer SwiftPM package that depends on mlx-swift")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expect-patch", type=Path, help="the generated Package.swift patch must equal this file byte for byte")
    parser.add_argument("--expect-trace-patch", type=Path, help="the generated device.cpp patch must equal this file byte for byte")
    P.add_pin_arguments(parser)
    args = parser.parse_args(argv)
    result = prepare(args.stock_checkout.resolve(), args.package_dir.resolve(), args.out.resolve(), P.pins_from(args), args.expect_patch, args.expect_trace_patch)
    print(json.dumps({"out": str(args.out.resolve()), "changedFiles": result["trace"]["changedFiles"], "patchSHA256": result["trace"]["patchSHA256"],
                      "tracePatch": result["tracePatch"], "traceEnvironment": TRACE_ENV}))
    return 0


def cli(argv: list | None = None) -> int:
    try:
        return main(argv)
    except (PrepareError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"prepare_trace_source: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli())
