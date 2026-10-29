# mlx-metallib-slim

`metallib-slim` cuts MLX's precompiled Metal library, `mlx.metallib`, down to
the kernels an app actually uses.

An app that runs [MLX](https://github.com/ml-explore/mlx) on Apple silicon
through [mlx-swift](https://github.com/ml-explore/mlx-swift) carries the whole
`mlx.metallib`: more than 17,000 kernels, about 180 MB, for every data type,
tile size and GPU generation MLX supports. Most apps use a few hundred of them.
This tool records which kernels your app asks for on each GPU architecture
class, then rebuilds the library with only those. Together with mlx-swift's
no-JIT runtime (kernels come from the precompiled library and are never
compiled from source on the user's Mac), it took one shipped app from about
305 MB to 40 MB; its library went from 17,319 functions (183 MB) to 303
(3 MB).

It is a set of Python 3 scripts with no dependencies, plus one small Swift
file. Every step fails closed and writes a manifest of hashes, so a reduced
library can be rebuilt byte for byte and checked before it ships.

Unofficial; not affiliated with Apple or the MLX project.

## Requirements

- macOS on Apple silicon.
- Xcode with the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`
  if `xcrun metal` is missing). `names` also needs a Metal device.
- CMake, to build the full library from MLX's sources.
- Python 3.9 or later (the system `/usr/bin/python3` works).
- An mlx-swift checkout as SwiftPM resolves it. The source edits are exact-text
  anchors written for mlx-swift 0.32.3 (MLX 1f8e74e3); another version that
  moved those lines is refused, never patched by guesswork.

## Install

```sh
git clone https://github.com/jnersn/mlx-metallib-slim.git
mlx-metallib-slim/metallib-slim --help
```

Run it from the clone, or link `metallib-slim` into a directory on your `PATH`.
For a release build, pin the clone to a commit: the reduced library's bytes
depend on the tool's code, so a build should name the revision it used.

## How it works

MLX defines most of its kernels as templates and instantiates each variant
explicitly with a `host_name`, for example
`template [[host_name("vn_Absfloat32")]] [[kernel]] decltype(...) unary_v<...>;`.
The library holds every one of them. `metallib-slim reduce` replays the
compile commands of MLX's own CMake build, file by file, with one change: each
kernel file is preprocessed with the same flags, the explicit instantiations
whose names are not on your allowlist are deleted, and the rest of the text is
compiled unchanged. Kernel bodies, helpers, headers, function constants and the
kept instantiations stay byte for byte, and an independent proof of that is
recorded per file. The objects are linked with MLX's own link command.

The allowlist comes from tracing: a copy of mlx-swift with a small hook in
`device.cpp` writes the name of every kernel MLX asks the precompiled library
for. You run your real workload with that build, once per GPU architecture
class, and the union of the traced names is the allowlist.

## Workflow

The commands below assume a SwiftPM package (`MyApp/`) that depends on
mlx-swift with a line like
`.package(url: "https://github.com/ml-explore/mlx-swift", ...),`, and a
scratch directory `build/`.

### 1. Prepare the sources

Resolve your package so SwiftPM checks out mlx-swift, then make the no-JIT
copy:

```sh
xcodebuild -resolvePackageDependencies -scheme MyApp -clonedSourcePackagesDirPath build/sources   # in MyApp/
STOCK=build/sources/checkouts/mlx-swift

metallib-slim prepare-source --stock-checkout $STOCK --package-dir MyApp --out build/src \
    --expect-patch mlx-metallib-slim/patches/mlx-swift-0.32.3/package-nojit.patch \
    --mlx-swift-revision <commit> --mlx-core-revision <commit> --mlx-c-revision <commit>
```

`build/src/mlx-swift` is a copy-on-write clone whose `Package.swift` compiles
`nojit_kernels.cpp` instead of `jit_kernels.cpp`, so MLX looks every kernel up
in `mlx.metallib` and never compiles Metal source at run time. Everything else
is proven byte-identical to the stock checkout. `build/src/package` is your
package with its mlx-swift dependency replaced by a path dependency on the copy;
build your app from there. The stock checkout is never written. The revision
options are optional; `--expect-patch` pins the reviewed one-line patch.

### 2. Build the full library

```sh
CORE=build/src/mlx-swift/Source/Cmlx/mlx; VENDOR=build/src/mlx-swift/Source/Cmlx
cmake -S $CORE -B build/cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=26.2 -DMLX_METAL_JIT=OFF -DMLX_BUILD_GGUF=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF -DMLX_BUILD_PYTHON_STUBS=OFF -DMLX_USE_CCACHE=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DFETCHCONTENT_SOURCE_DIR_METAL_CPP=$VENDOR/metal-cpp -DFETCHCONTENT_SOURCE_DIR_JSON=$VENDOR/json -DFETCHCONTENT_SOURCE_DIR_FMT=$VENDOR/fmt \
  > build/cmake-configure.log
cmake --build build/cmake --target mlx-metallib --parallel --verbose > build/cmake-build.log

metallib-slim record --build-dir build/cmake --source-core $CORE --configure-log build/cmake-configure.log \
    --build-log build/cmake-build.log --cmake-version "$(cmake --version | head -1)" --min-os 26.2 --nax --out build/full.json
```

The verbose build log is the input of the reduction: it holds every
`xcrun metal` compile command and the link command. `record` checks the build
(options, one flag set, no fast math, exactly one library) and records it.
`--nax` also requires MLX's NAX kernels, which MLX builds when the SDK and the
deployment target are macOS 26.2 or later; leave it out for an older target.

### 3. Trace the kernels

```sh
metallib-slim prepare-trace-source --stock-checkout $STOCK --package-dir MyApp --out build/trace
```

Build your app from `build/trace/package`, put the FULL library from step 2
(`build/cmake/mlx/backend/metal/kernels/mlx.metallib`) where your app loads
it from (next to the executable works), and run your whole workload once per
architecture class:

```sh
for class in natural g13g g13s g13d g15g g15s g15d g17g g17s g17d g17p g18p; do
  mkdir -p build/traces/$class
  arch=$([ $class = natural ] || echo applegpu_$class)
  MLX_METALLIB_SLIM_TRACE=$PWD/build/traces/$class/trace MLX_METAL_GPU_ARCH=$arch ./MyApp-traced --run-workload
done
```

The hook writes `<prefix>.<pid>.jsonl`: one line per distinct kernel name the
precompiled library was asked for, then a summary at exit. It changes no
compile option and writes nothing to stdout or stderr. Never ship the traced
build.

MLX chooses kernels by the GPU's architecture name, which `MLX_METAL_GPU_ARCH`
overrides. Forcing the name on one Mac makes MLX request exactly the kernels
that generation would; the kernels still run on your GPU, so this proves which
kernels are needed, not the pixels another chip would produce. These classes
cover the dispatch differences in MLX 0.32 (generation below 15, 15 to 16, 17
and later with NAX, times the g/p, s and d letters):

| Class | `MLX_METAL_GPU_ARCH` | Dispatch group |
|---|---|---|
| `natural` | unset | the Mac you run on |
| `g13g`, `g13s`, `g13d` | `applegpu_g13g`, `applegpu_g13s`, `applegpu_g13d` | generation below 15 (M1, M2) |
| `g15g`, `g15s`, `g15d` | `applegpu_g15g`, `applegpu_g15s`, `applegpu_g15d` | generation 15 and 16 (M3, M4) |
| `g17g`, `g17s`, `g17d` | `applegpu_g17g`, `applegpu_g17s`, `applegpu_g17d` | generation 17 and later (M5; NAX kernels on macOS 26.2 and later) |
| `g17p`, `g18p` | `applegpu_g17p`, `applegpu_g18p` | phone-class GPUs |

The last letter is the GPU size: `g` base and Pro, `s` Max, `d` Ultra, `p`
phone. Check the dispatch code of the MLX version you ship when you bump it.

### 4. Build the allowlist

```sh
echo '{"reduction": {"baseFiles": []}}' > declaration.json
metallib-slim allowlist build --declaration declaration.json --out allowlist.json \
    $(for c in natural g13g g13s g13d g15g g15s g15d g17g g17s g17d g17p g18p; do echo --trace $c=build/traces/$c --expect $c; done)
```

Each `--trace LABEL=PATH` takes a trace file or a directory of them; every
trace must be complete (header, summary, nothing dropped), or the build stops.
An `--expect` label without a trace is an error unless `--allow-partial`, which
marks the allowlist incomplete. `--base` adds the names of an older allowlist,
so you can extend one with new traces. The result is frozen with
`allowlist.sha256` beside it and is never rewritten; `allowlist verify` checks
it. Commit both files.

The declaration is the reduction's configuration, and the allowlist records its
hash. `reduction.baseFiles` (and `retainWholeFiles`) name kernel files, relative
to `mlx/backend/metal/kernels` and without `.metal`, that are compiled whole
from the original source. Run `metallib-slim scan` first: it preprocesses every
kernel file and fails on any the filter cannot parse; those must be listed.
With MLX 0.32.3 every file parses, so the list can be empty; keeping a few
small files whole costs little.

### 5. Reduce

```sh
metallib-slim reduce --build-log build/cmake-build.log --core $CORE --declaration declaration.json \
    --allowlist allowlist.json --out build/reduced --full-build-dir build/cmake --min-os 26.2 --nax
```

`build/reduced/lib/mlx.metallib` is the reduced library and
`build/reduced/manifest.json` records every input hash, every file's mode
(filtered or original), its proof, and the library's hash. `--out` must not
exist. `--min-os` and `--nax` apply the checks of `record` to the build log.

### 6. Verify

```sh
metallib-slim names build/reduced/lib/mlx.metallib build/reduced-names.json
metallib-slim names build/cmake/mlx/backend/metal/kernels/mlx.metallib build/full-names.json
metallib-slim verify --manifest build/reduced/manifest.json --reduced-names build/reduced-names.json \
    --full-names build/full-names.json --allowlist allowlist.json --out build/verify.json
```

`names` loads a library on the Metal device and lists its functions. `verify`
fails unless every allowlisted name is in the reduced library, every name is
known to the full one, and the reduced library holds nothing the full one
lacks.

Then run the shipping build (from `build/src/package`, no trace hook) with the
reduced library over the same workload for every architecture class. Every
job must succeed: a missing kernel makes MLX fail when it looks the kernel up.

For the binary you ship, `check-macho` compares and audits Mach-O files:
`check-macho payload-equal <a> <b>` (same code and data in two differently
signed copies), `check-macho no-coverage <binary>` (no LLVM profile or coverage
sections) and `check-macho log-coverage <xcodebuild.log>` (no coverage flags in
any compile command).

### 7. Reproduce

The reduction is deterministic: two clean builds from the same inputs give the
same bytes (in practice also at different paths, but only the same path is
claimed). To prove it for your inputs, reduce twice into fresh directories and
compare:

```sh
metallib-slim repro --a build/reduced/manifest.json --b build/reduced-2/manifest.json --fixed-path build --out build/repro.json
```

Pin the library's SHA-256 in your release checks.

## The risk: a kernel missing at run time

A kernel your app never asked for while tracing is not in the reduced library.
If the app asks for it later, MLX cannot find it and the operation fails (with
the no-JIT runtime there is no fallback to compiling it from source). Kernels
depend on the operation, the data types, the shapes (tile sizes, aligned or
not), the options and the GPU generation. To keep the set complete:

- Trace every architecture class you ship to, not only your own Mac.
- Trace a workload that covers every model, data type, size and option your
  app offers, including the edges (smallest and largest sizes, odd shapes,
  each step count and mode).
- Retrace and rebuild whenever you change MLX, mlx-swift, a model or what the
  app lets the user do.
- Run the shipping build with the reduced library over the workload for every
  class before each release.

## Commands

| Command | Does |
|---|---|
| `prepare-source` | no-JIT copy of a pinned mlx-swift checkout and a package that builds against it (`--variant jitcontrol` for an unchanged control copy) |
| `prepare-trace-source` | the same copy plus the kernel-name trace hook |
| `record` | records and checks the full `mlx-metallib` CMake build |
| `allowlist build`, `allowlist verify` | builds and freezes the allowlist from traces; verifies a frozen one |
| `scan` | preprocesses and parses every kernel file; no allowlist needed |
| `reduce` | builds the reduced library |
| `names` | lists the functions of a library (compiles `metallib_names.swift`; needs a Metal device) |
| `verify` | checks the reduced library against the allowlist and the full library |
| `repro` | compares two reduced builds |
| `check-macho` | `sections`, `payload-equal`, `no-coverage`, `log-coverage`, `coverage-proof`, `payload-report` |

`metallib-slim <command> --help` lists each command's options. The modules in
`metallib_slim/` also run on their own (`python3 -I metallib_slim/reduce_library.py reduce ...`).

`patches/mlx-swift-0.32.3/` holds the patches the two prepare commands generate
for mlx-swift 0.32.3, for review and for `--expect-patch` /
`--expect-trace-patch`.

## Tests

```sh
python3 -m unittest discover -s tests
MLX_SWIFT_CHECKOUT=build/sources/checkouts/mlx-swift python3 -m unittest discover -s tests   # also checks the edits against a real checkout
```

The tests need no GPU, no network and no Xcode; the reduction is tested with a
stand-in compiler.

## License

MIT
