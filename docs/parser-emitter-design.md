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
- **Index mapping:** `get_global_id(0)` is the elementwise index. Other dimensions, `get_local_id`,
  `get_group_id` and `get_global_size` are residue until the IR has a place for them.
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

## 6. Questions for the Unpopped lane

1. Is `convert::Frontend` the committed parse seam, with `FrontendRegistry` only a convenience?
2. Does the IR carry multi-dimensional index (`get_global_id(1)`), or only a flat index?
3. Is `lower_kernel(&KernelFn)` the preferred emit path over `lower(&KernelPlan)` for new backends?
4. Is the `convert` feature expected to stay opt-in, and does it pull `tree-sitter-slang` for us?
   (It does today: the feature lists both grammars.)
