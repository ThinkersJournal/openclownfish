# OpenCLownfish: survey and design note

Status: draft for the PM's gate. No code. Measured 2026-10-30 on `TUF-LAPTOP` (Windows 11) unless a line says otherwise. crates.io figures are from the crates.io API on the same day.

## 1. Existing Rust OpenCL crates (measured)

| Crate | Latest | Licence | Last release | Repo last push | Role |
|---|---|---|---|---|---|
| `opencl-sys` | 0.6.5 | Apache-2.0 | 2026-10-18 | 2026-10-18 | raw FFI, one cargo feature per OpenCL version/extension; `static` is opt-in |
| `cl3` | 0.13.2 | Apache-2.0 | 2026-10-18 | 2026-10-18 | thin wrapper over `opencl-sys`; `default = ["dynamic"]` |
| `opencl3` | 0.12.4 | Apache-2.0 | 2026-10-24 | 2026-10-24 | object-style API over `cl3` |
| `ocl` | 0.19.7 | MIT OR Apache-2.0 | 2024-03-27 | 2026-02-23 | high-level API; 36 open issues |
| `ocl-core` / `cl-sys` | 0.11.5 / 0.4.3 | (not read) | 2023-10-09 | n/a | `ocl`'s lower layers, stale |
| `opencl-dynamic-sys` | 0.1.1 | (not read) | 2024-11-12 | n/a | runtime loading only, small |
| `ocl-include` | 0.6.0 | (not read) | 2023-04-18 | n/a | `#include` preprocessor for OpenCL C source |

The first three share one author (kenba) and one release cadence, and are actively maintained. Downloads (all-time): `cl3` 785k, `opencl3` 758k, `ocl` 477k.

**Not verified here:** whether `cl3`'s `dlopen2` loader fails gracefully when no ICD loader is installed (`opencl-sys` itself has no loader, see section 4); each crate's unsafe surface and soundness; extension coverage beyond the feature list above. These need a code read, not a metadata read.

**Parsers/emitters:** a crates.io search (queries "opencl c parser", "opencl kernel", "opencl") found no Rust OpenCL C parser or emitter. Control: the same searches find `ocl-include`, `opencl3`, `rlxsl` (an emitter crate for another project), so the queries do reach relevant crates. Absence of a dedicated crate is therefore likely but not proven (crates.io search is name/description based).

## 2. Hardware (measured, `clinfo` on TUF-LAPTOP, 2026-10-30)

Three ICD platforms, one device each:

| Platform | Version | Device | Type | Device OpenCL C |
|---|---|---|---|---|
| NVIDIA CUDA | OpenCL 3.0 CUDA 13.4.96 | RTX 4070 Laptop, driver 617.42 | GPU | 1.2 |
| AMD Accelerated Parallel Processing | OpenCL 2.1 AMD-APP (3652.0) | gfx1036 (Radeon 610M iGPU) | GPU | 2.0 |
| Intel(R) OpenCL (oneAPI 2026.1 runtime, registry vendor key) | OpenCL 3.0 WINDOWS | AMD Ryzen 9 7940HX, 32 CUs | CPU | 3.0 |

So the premise holds on this machine: OpenCL is a real runtime API across three vendors' stacks, and the three differ visibly (OpenCL C 1.2 / 2.0 / 3.0; NVIDIA exposes no SVM atomics or device queues, the CPU runtime exposes fine-grain system SVM). The Intel *GPU* platform is absent here because this laptop has no Intel GPU. **Not yet measured:** the desktop rig (RTX 4060, Radeon VII, Arc A770, Tesla P40) and Linux on either machine (PoCL, Mesa rusticl). Row H1.

## 3. Options: bindings and safe layer

- **A. Depend on `opencl3` (and so `cl3`, `opencl-sys`).** Fastest. Licence Apache-2.0 only (compatible with our MIT/Apache dual licence for consumers, but we cannot relicense their code). We inherit their feature-flag model, release cadence and safety decisions; the "test the limits of every vendor stack" goal needs raw access to odd paths (invalid args, extension entry points, driver errors) that a safe layer may hide. Depending on them is consistent with CireSnave's rule to use crates.io versions.
- **B. Depend on `opencl-sys` only; write our own safe layer.** The FFI layer is the large, mechanical, header-derived part and is kept current by someone else; we own the API design, error model and the stack-probing behaviour we need. Medium effort. Keeps `openclownfish-sys` as a thin re-export or empty reserved crate, which wastes one of the two reserved names.
- **C. Generate our own bindings** (`openclownfish-sys`) from the Khronos headers, loading `libOpenCL.so` / `OpenCL.dll` at runtime ourselves. Most control, including tolerant loading (missing symbols are `None`, not link failures) and per-ICD-platform dispatch, which a conformance/limits tool wants. Largest effort; we then carry the header-tracking burden, and must keep the Khronos headers' licence (Apache-2.0) and credit.

## 4. Ruling and the currency check on `opencl-sys`

Ruling (PM, 2026-10-30): **B**, own safe layer over `opencl-sys`, conditional on this lane's check that `opencl-sys` is current, tracks the OpenCL 3.0 headers, loads dynamically with no OpenCL install at build time, has an acceptable licence and is maintained; else C.

Check, `opencl-sys` 0.6.5 (crates.io tarball, read 2026-10-30):

| Condition | Result | Evidence |
|---|---|---|
| Latest version | yes, 0.6.5 is `max_version`, released 2026-10-18 | crates.io API |
| Tracks the headers | yes, "up to date with OpenCL-Headers tag v2026.05.29", adds OpenCL 3.1 headers (0.6.3), 0.6.4 incorporates later header changes | `RELEASES.md` |
| Licence | Apache-2.0 | `Cargo.toml` |
| Maintained | 3 releases in the 10 days to 2026-10-18; repo pushed 2026-10-18 | `RELEASES.md`, GitHub API |
| Dynamic loading, no OpenCL at build time | **Not provided by `opencl-sys`.** It has no loader. Its only function declarations are an `extern` block gated on `feature = "static"` carrying `#[link(name = "OpenCL")]` (`src/cl.rs:995-1000`); without `static` it ships types, constants and 230 `pub type` function-pointer aliases (`src/cl_function_types.rs`, e.g. `clGetPlatformIDs_t = Option<unsafe extern "C" fn(..)>`). Its only dependency is `libc`. The loader lives in `cl3` (`dlopen2`). | source read; a scratch crate depending on `opencl-sys` 0.6.5 without `static` builds and runs, and its binary contains no `OpenCL.dll` string (that box does have OpenCL installed, so the cfg gate is the real evidence, not the run) |

Consequence: **B holds, with a stated shape.** `openclownfish-sys` is *not* empty: it is our own runtime loader (`libOpenCL.so` / `OpenCL.dll`, Linux and Windows, missing symbols as `None`) that fills the `opencl-sys` `*_t` function-pointer types. We take types, constants and signatures from `opencl-sys` (header-tracked by someone else) and own loading and the safe API. This is the cheap half of C. Choice of loader crate (`dlopen2`, as `cl3` uses, or `libloading`) is open; neither was evaluated beyond that. The `*_t` option wrapping already models "symbol absent".

Not checked: `opencl-sys`'s MSRV (none declared in `Cargo.toml`; edition 2024 implies Rust 1.85 or later), and whether every extension we want has a `*_t` alias.

## 5. Parser and emitter (design sketch, no code)

Contract: Unpopped holds neither (`Unpopped/docs/parser-emitter-ownership-map.md` section 9; at that file's state on this checkout); the map's section 3 recorded OpenCL as having no owner, and the task assigning this lane is the owner decision. Section 5 of that doc proposes a `FrontendRegistry` where a parser crate depends on `unpopped`, never the reverse; the emitter side uses the `Backend`/`Lowering` traits. So:

- `openclownfish-clc` (name tentative): OpenCL C parse to Unpopped IR, and IR to OpenCL C emit, depending on `unpopped` from crates.io.
- Open questions for the Unpopped lane: which of `FrontendRegistry`/`Frontend` is the committed seam today (the doc calls it a proposal); how OpenCL address-space qualifiers (`__global`, `__local`, `__constant`, `__private`), vector types and the `get_global_id` family map into the IR-union (ROADMAP 6.7).
- OpenCL C versions differ per device (1.2 / 2.0 / 3.0 above); the emitter must target a declared version and feature set, and the probe from section 2 supplies it.

## 6. Fuel backend (later)

Designed with the Fuel lane; this repo does not edit Fuel. Lowest priority behind Rockfish, Bonefish and Vulkane. Prerequisite: sections 3 and 5 settled. Nothing else to say until then.

## 7. Plan

1. Land D0/D1 (this PR); PM enables branch protection (P0).
2. C1 skeleton per the ruling; H1 desktop and Linux measurements.
