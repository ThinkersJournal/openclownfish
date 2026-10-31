# OpenCL C parser and emitter: design (rows X1, X2)

Status: proposal for the Unpopped lane and the PM. No code. Measured against Unpopped
`origin/main` 5500fd1 (crate `unpopped` 0.34.12) and crates.io on 2026-10-30.

## 1. Scope

OpenCL C is owned by this project (task from the PM, 2026-10-30); the ownership map (section 3) had
recorded it as having no owner. Unpopped keeps the IR, the traits and one reference emitter, and
holds neither parser nor emitter (`docs/parser-emitter-ownership-map.md` section 9). So both
directions live in a new crate, `openclownfish-clc` (name tentative), depending on `unpopped` from
crates.io, never the reverse and never by path.

## 2. Seams that exist today (read at 5500fd1)

- **Parse side:** `unpopped::convert::Frontend` (feature `convert`): `name`, `kernel_marker`,
  `residue`, `out_name`, `in_prefix` and `parse: fn(&str) -> Option<Tree>`. It is `#[non_exhaustive]`
  with `Frontend::new(..)`, built for another crate to hand in. Its doc says the grammar crate is the
  only thing that needs a dependency, which is why `parse` is a plain function pointer. Entry points:
  `lift`, `lift_elementwise`, `lift_reduction`, `lift_scan` and their `*_with(frontend)` forms.
  The ownership map still calls a `FrontendRegistry` a proposal; `Frontend` is the committed seam.
- **Emit side:** `unpopped::Backend` (`name`, `provider`, `lower(&KernelPlan)`,
  `lower_kernel(&KernelFn)` defaulting to the typed refusal `LowerError::UnsupportedKernel`,
  dtype support per target). Rule: refuse rather than approximate.

## 3. Parser (X1)

- **Grammar (as built):** `tree-sitter-cpp`, not `tree-sitter-c`. Unpopped's shared walker reads a
  subscript's `indices` field, which only the C++ grammar has (the C grammar calls it `index`); with the C
  grammar every kernel came back `NotElementwise`. OpenCL qualifiers (`__kernel`, `__global`) parse as
  tolerated `ERROR` identifiers and do not stop the walker. No `tree-sitter-opencl` exists on crates.io.
- **Plan (as built):** `OPENCL_C: Frontend` plus `parse_opencl_c`, behind the default `parse` feature of
  `openclownfish-clc`; reuses `lift_elementwise_with`. Buffers are named `in0, in1, ..` and `out`
  (`with` a `ResourceMap` for other names). The index is a lone identifier (`size_t i = get_global_id(0)`);
  `out[get_global_id(0)]` inline is refused as not elementwise.
- **Math names (as built, Unpopped 0.35.3):** `Frontend::with_intrinsics` replaces the CUDA call tables
  (`expf`, `sqrtf`) with OpenCL's overloaded names: `exp exp2 log log2 sqrt rsqrt fabs tanh sin cos floor
  ceil trunc` and `fmax fmin pow atan2 copysign`. `fmax`/`fmin` map to the NaN-suppressing
  `FmaxIeee`/`FminIeee`. Refused (tests pin it): `native_*` and `half_*` (reduced accuracy), CUDA spellings
  (`expf`), and `abs`, which is integer-only in OpenCL C.
- **Residue (refused, never approximated):** `barrier`, `mem_fence`, `atomic_*`, `__local` buffers,
  `work_group_*` built-ins, `async_work_group_copy`, pipes, images/samplers, device-side enqueue,
  `printf`, `#pragma`. Each is a typed refusal naming the construct.
- **Index mapping:** `get_global_id(d)` maps to `Coord(d)` (section 6, answer 2). `get_local_id`,
  `get_group_id`, `get_local_size` and `get_num_groups` are residue until the IR has a work-group model.
- **Vector types** (`float4`, `int2`): refused in 0.x; the IR is scalar-element.

## 4. Emitter (X2)

- `OpenClC: Backend`, `name() = "opencl_c"`, `provider() = "openclownfish"`. It carries one flag, `fp64`,
  set with `OpenClC::for_extensions(device.extensions())`: a plan has no device, so the caller says whether
  `cl_khr_fp64` exists, and an f64 plan without it is a typed `UnsupportedDtype` (no kernel that fails to
  build on the driver).
- **Versioned target:** the emitter targets a declared OpenCL C version and extension set, taken from
  the device (`Device::opencl_c_version`, already in `probe`: laptop shows 1.2 on NVIDIA, 2.0 on AMD,
  3.0 on Intel). Spelling needing more than the target offers is `LowerError`, not a downgrade.
- **Dtypes:** `half` needs `cl_khr_fp16`, `double` needs `cl_khr_fp64`; emit the `#pragma OPENCL
  EXTENSION` only when the device lists it. Dtype support is answered per target, as the trait asks.
- **Address spaces:** buffers are `__global`; scalars are by value. No `__local` or `__constant`
  until the IR has schedules that want them.
- **Conformance:** the Unpopped suite shows the pattern (`kernel_emit_*_agrees.rs` compares emitted
  C against the reference). Ours runs emitted source through `clBuildProgram` on every device the probe
  finds, via the safe layer, and compares results with Unpopped's f64 oracle (section 7).

## 5. Order of work

1. Safe layer (context, queue, buffer, program, kernel, event) so build logs and results are
   reachable. C2 resolved the entry points; the safe types are the next roadmap rows.
2. X2 first: emitting needs only the `Backend` trait and is testable on real drivers, which serves
   the goal of finding where each vendor's compiler differs.
3. X1 second: a grammar spike with the frontend above, then lifts over a corpus.

## 6. Answers from the Unpopped lane (2026-10-30, read at their 5500fd1)

1. `Frontend` + `Frontend::new` + `lift_*_with` is the parse seam and stays public under semver.
   `FrontendRegistry` does not exist at 5500fd1 (it was a proposal); this note does not need it.
2. The intent layer is multi-dimensional: `ScalarExpr::Coord(axis)` is an output coordinate along
   `axis < rank`, so `get_global_id(d)` maps to `Coord(d)` for elementwise N-D kernels. There is no
   thread-id or work-group model (no local/group id, barriers, wave ops; IR-union U4, not landed), so
   `get_local_id`, `get_group_id` and friends stay residue. Section 3's "dim>0 is residue" is
   therefore wrong: dimensions above 0 are accepted when the kernel is elementwise.
3. `lower_kernel(&KernelFn)` is the recommended direction but is defaulted and so far proven only in
   `unpopped-cpu-c`; `lower` is not deprecated. X2 starts on `lower(&KernelPlan)`.
4. `convert` pulls `tree-sitter` + `tree-sitter-slang` until `vulkane-slang-parse` is published, then
   `tree-sitter` only. Timing is unscheduled and not to be depended on.

## 7. Measurements: emitter v1 on the laptop (Windows 11, laptop booted; rig and Linux UNVERIFIED)

`crates/openclownfish-clc/tests/device_agrees.rs` at the commit that added it: 257-element f32 kernels,
emitted by `OpenClC`, built and run on each device, compared with host f32 arithmetic rounded after every
operation ("unfused") and with Unpopped's f64 oracle. Largest distance in ULP:

| device (driver) | add sub mul neg*2.5 | a*b+c vs unfused | a*b+c vs oracle | a/b vs unfused |
|---|---|---|---|---|
| NVIDIA RTX 4070 Laptop (OpenCL 3.0 CUDA 13.4.96) | 0 | 0 | 2 | 1 |
| AMD gfx1036 (OpenCL 2.1 AMD-APP 3652.0) | 0 | 0 | 2 | 1 |
| Intel OpenCL CPU on Ryzen 9 7940HX (OpenCL 3.0) | 0 | 0 | 2 | 0 |

- Every stack honoured `#pragma OPENCL FP_CONTRACT OFF`: a*b+c matches the unfused result exactly.
- The oracle disagrees with unfused a*b+c by 2 ULP on all three. The oracle evaluates in f64 and rounds
  once, which is what a fused multiply-add gives, so it models fused arithmetic for compound bodies
  while `unpopped-cpu-c` emits unfused C. The Unpopped lane confirms this is documented behaviour (`oracle.rs` names round-once vs per-op rounding as the residual for compound bodies): single-op bodies compare BitExact, compound bodies against an unfused host reference or in tolerant mode.
- Division is within 1 ULP on the two GPU stacks and exact on the CPU stack; the test allows 3 (OpenCL's
  default permits 2.5). Correctly rounded division would need `-cl-fp32-correctly-rounded-divide-sqrt`
  where the device offers it; not used in v1.

### Unary operations (emitter 0.0.9), same method, largest ULP distance from the f64 oracle

| op | NVIDIA RTX 4070 Laptop | AMD gfx1036 | Intel CPU | accepted |
|---|---|---|---|---|
| abs, sqr | 0 | 0 | 0 | 0 |
| recip (`1.0f / x`) | 1 | 1 | 0 | 3 |
| sqrt | 1 | 1 | 0 | 3 |
| rsqrt | 1 | 1 | 1 | 4 |
| exp | 2 | 1 | 1 | 4 |
| log | 1 | 2 | 1 | 4 |

Inputs for sqrt, rsqrt and log are positive. NVIDIA and AMD `sqrt` and reciprocal are not correctly
rounded by default (1 ULP) while the Intel CPU stack is exact; the "accepted" column is OpenCL's
default allowance (3 for sqrt and division, exp and log), widened by one for rsqrt/exp/log so a
stack at the spec limit is not a flaky failure. Windows 11 laptop booted; rig and Linux UNVERIFIED.

### Max, Min, Fmax/Fmin and pow (emitter 0.0.10)

`max`/`min` (NaN-propagating) and `fmax_ieee`/`fmin_ieee` (NaN-suppressing) were run over a 9 x 9 grid of
{NaN, +inf, -inf, +0, -0, 1, -1, 2.5, -2.5} and agree with the oracle exactly on all three devices (NaN
only by being NaN; every other result by bit pattern, so the sign of a zero is checked). A first version
that made `+0` beat `-0` disagreed with the oracle, which keeps the first operand on a tie
(`max(-0, +0) = -0`); the helpers now follow it. OpenCL's own `fmax`/`fmin` leave that sign unspecified,
so the kernel carries its own helpers. IEEE maxNum/minNum leave a zero tie open, so for the two `_ieee`
ops the test accepts either zero.

`pow(x, 2.5)` over positive x: NVIDIA 2 ULP, AMD 1, Intel CPU 0 from the f64 oracle (OpenCL allows 16;
the test accepts 17). Windows 11 laptop booted; rig and Linux UNVERIFIED.

### Comparisons and Select (emitter 0.0.12)

`cmp_eq cmp_ne cmp_lt cmp_le cmp_gt cmp_ge` (each exactly 1.0 or 0.0, false on any NaN operand except `!=`)
and `select(cond, a, b)` (cond tested `!= 0`: NaN true, -0.0 false; the chosen arm moves untouched) were
run over the same 9 x 9 NaN/inf/zero grid at n=257 and n=256 and agree with the oracle bit for bit on all
three laptop devices (Windows 11 booted; rig and Linux UNVERIFIED). No vendor needed a workaround: all
three compilers follow C's NaN rules for the relational operators without `-cl-finite-math-only`.

### f64 (emitter 0.0.13)

`tests/device_agrees_f64.rs` runs 20 cases (arithmetic, unary, exp/log/pow, Max/Min/Fmax, comparisons,
Select) at n=257 and n=256 on every device that lists `cl_khr_fp64` (all three here), against the f64
oracle. Largest ULP distance:

| op | NVIDIA | AMD | Intel CPU | accepted |
|---|---|---|---|---|
| add sub mul, a*b+c, neg, abs, sqr, div, recip, sqrt | 0 | 0 | 0 | 0 (div, recip, sqrt: 3) |
| exp, rsqrt | 1 | 1 | 1 | 4 |
| log | 0 | 1 | 0 | 4 |
| pow(x, 2.5) | 1 | 1 | 1 | 17 |

Unlike f32 (where NVIDIA and AMD division, reciprocal and sqrt were 1 ULP off), f64 division, reciprocal and
sqrt are exact on all three stacks, and `a*b+c` matches the oracle bit for bit because the oracle's f64
arithmetic is per-operation like the device's with FP_CONTRACT OFF. The NaN/inf/zero grid is bit-exact for
max, min, fmax, cmp_lt, cmp_ne and select. A device without `cl_khr_fp64` (none on this laptop; the Arc
A770 is a candidate on the rig, UNVERIFIED) gets a typed refusal from the backend, tested without a device.
Windows 11 laptop booted; rig and Linux UNVERIFIED.
