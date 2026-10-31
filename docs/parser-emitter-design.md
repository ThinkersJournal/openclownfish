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

- **Grammar:** OpenCL C 1.2 to 3.0 is C99 plus address-space qualifiers, vector types, built-ins and
  `__kernel`. crates.io has no `tree-sitter-opencl` (API: "does not exist"; control: `tree-sitter-c`
  0.24.2, `tree-sitter-cpp` 0.23.4 and `tree-sitter-cuda` 0.21.2 resolve through the same query). Use
  `tree-sitter-c`: qualifiers parse as unknown identifiers or attribute-like tokens, so the frontend
  must recognise them itself. Verify on real kernels before committing to this; the fallback is a
  small hand-written recogniser for the kernel subset, as `lift.rs` was.
- **Plan:** `pub const OPENCL_C: Frontend = Frontend::new("opencl-c", "__kernel", RESIDUE, ..)`
  plus `parse_opencl_c`. Reuse `lift*_with`, so elementwise, reduction and scan come for free.
- **Residue (refused, never approximated):** `barrier`, `mem_fence`, `atomic_*`, `__local` buffers,
  `work_group_*` built-ins, `async_work_group_copy`, pipes, images/samplers, device-side enqueue,
  `printf`, `#pragma`. Each is a typed refusal naming the construct.
- **Index mapping:** `get_global_id(d)` maps to `Coord(d)` (section 6, answer 2). `get_local_id`,
  `get_group_id`, `get_local_size` and `get_num_groups` are residue until the IR has a work-group model.
- **Vector types** (`float4`, `int2`): refused in 0.x; the IR is scalar-element.

## 4. Emitter (X2)

- `OpenClC: Backend`, `name() = "opencl-c"`, `provider() = "openclownfish"`.
- **Versioned target:** the emitter targets a declared OpenCL C version and extension set, taken from
  the device (`Device::opencl_c_version`, already in `probe`: laptop shows 1.2 on NVIDIA, 2.0 on AMD,
  3.0 on Intel). Spelling needing more than the target offers is `LowerError`, not a downgrade.
- **Dtypes:** `half` needs `cl_khr_fp16`, `double` needs `cl_khr_fp64`; emit the `#pragma OPENCL
  EXTENSION` only when the device lists it. Dtype support is answered per target, as the trait asks.
- **Address spaces:** buffers are `__global`; scalars are by value. No `__local` or `__constant`
  until the IR has schedules that want them.
- **Conformance:** the Unpopped suite shows the pattern (`kernel_emit_*_agrees.rs` compares emitted
  C against the reference). Ours runs emitted source through `clBuildProgram` and compares results
  with `unpopped-cpu-c`, on every platform the probe finds, via the safe layer from C3 onward.

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
