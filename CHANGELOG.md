# Changelog

The project uses semantic versioning. Before 1.0.0, a minor version may change the command line, the file formats or the source edits.

## 0.1.0

First release.

- `metallib-slim prepare-source`: no-JIT copy of a pinned mlx-swift checkout (one exact-anchor `Package.swift` edit that compiles
  `nojit_kernels.cpp` instead of `jit_kernels.cpp`) and a consumer package that builds against it by path; `--expect-patch` pins the patch,
  `--mlx-swift-revision`, `--mlx-core-revision` and `--mlx-c-revision` pin the checkout.
- `metallib-slim prepare-trace-source`: the same copy plus a `device.cpp` hook that writes the distinct default-library kernel names to
  `<MLX_METALLIB_SLIM_TRACE>.<pid>.jsonl`.
- `metallib-slim allowlist build` and `allowlist verify`: the union of complete traces per label (architecture class), with expected labels,
  an optional base allowlist and a frozen `.sha256`.
- `metallib-slim record`: checks and records the full `mlx-metallib` CMake build (`--min-os`, `--nax`).
- `metallib-slim scan`, `reduce`, `verify`, `repro`: the reduction from the full build's recorded commands, with a per-file proof that only
  unselected explicit instantiations were removed, the export check against the allowlist and the full library, and the comparison of two builds.
- `metallib-slim names`: lists a library's functions on the Metal device (`metallib_names.swift`).
- `metallib-slim check-macho`: Mach-O section hashes, payload equality of differently signed binaries, and coverage checks of binaries and build logs.
- Patches for mlx-swift 0.32.3 in `patches/mlx-swift-0.32.3/`.
