# OpenCLownfish roadmap

Statuses: `open`, `doing`, `done`, `held`. Updated with every PR.

| Row | Item | Status |
|---|---|---|
| D0 | ROADMAP, CI (fmt, clippy, test, doc, deny, portfolio), CODEOWNERS, deny.toml | done (#1) |
| D1 | Survey and design note, `docs/survey-and-design.md` | done (#1); ruling B 2026-10-30 |
| P0 | Branch protection requiring the D0 checks | PM (asked 2026-10-30) |
| C1 | Crate skeleton: `openclownfish-sys` (runtime loader over `opencl-sys` types) + `openclownfish` (platform/device enumeration, `probe` example); CI guards removed | doing (PR c1-skeleton, 0.0.1) |
| C2 | Loader hardening: no-OpenCL path tested on Linux CI, ICD vendor-file enumeration, more entry points (context, queue, buffer, program, kernel) | open |
| H1 | Per-machine platform/device report (laptop done via `probe`; desktop rig and Linux open) | doing |
| M1 | macOS: GitHub macOS runner exposes an OpenCL platform but `clGetDeviceIDs` returns -30 (`CL_INVALID_VALUE`), so no devices; measured in CI run 41113987171 (2026-10-30). macOS support is NOT claimed; revisit on real Apple hardware | open (finding) |
| X1 | OpenCL C parser into Unpopped's IR (contract: `Unpopped/docs/parser-emitter-ownership-map.md`, ROADMAP 6.7) | design note `docs/parser-emitter-design.md`; after X2 |
| X2 | OpenCL C emitter from Unpopped's IR | design note written; next, after the safe layer |
| F1 | Fuel OpenCL backend, designed with the Fuel lane (Fuel repo not edited here) | held; below Rockfish, Bonefish, Vulkane |
