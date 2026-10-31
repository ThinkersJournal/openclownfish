# Changelog

## 0.0.24

- Emitter: `OpenClC::lower_variants` offers a `blockscan` variant of every scan the serial emitter serves: one work-group per row, a local-memory Hillis-Steele scan per chunk of `OCF_BLOCK` elements (default 64, set with `-D OCF_BLOCK=n`), a carry across chunks for longer rows. Float Sum/Prod are `DeterministicallyDivergent`; Max/Min and integer scans are `BitIdentical`. 260 cases at block sizes 8 and 32 agree with the oracle on all 3 laptop devices (X2).

## 0.0.23

- Test: no tracked text file may contain a raw NUL byte (scans the git index; population control; seen red on an injected NUL).

## 0.0.22

- Emitter: strided offsets and the coordinate unravel come from unpopped::cfamily (strided_offset_expr, unravel_decomp) instead of local copies; emitted kernels are unchanged and every device differential still passes.
- Deps: unpopped and unpopped-vocab 0.38.1.

## 0.0.21

- Loader: `icd` module lists the drivers the system registers (Windows: Khronos registry key and display-adapter `OpenCLDriverName`, read through the `winreg` crate, no program is run; Unix: `.icd` files) and whether each loads; the probe prints them before the platform list (C2).
- Deps: unpopped and unpopped-vocab 0.38.0.

## 0.0.20

- Parser: naive OpenCL C accumulator-loop reductions (sum, prod, max, min, with a pre-map) and scans (cumsum, cummax) lift through the OPENCL_C frontend; the emitted kernel matches the source kernel run as written, bit-exact on every device (X1).

## 0.0.19

- Emitter: serial per-row scans (Sum, Prod, Max, Min; f32, f64, i32, i64; forward or reverse; inclusive or exclusive; per-element pre-map) in Baracuda's argument convention; 132 cases bit-exact on every device (X2).

## 0.0.18

- Deps: unpopped and unpopped-vocab 0.37.0.
- Tests: f64 reductions have a device differential, bit-exact on every cl_khr_fp64 device (36 more cases per device) (X2).

## 0.0.17

- Emitter: serial per-output reductions (Sum, Mean, Prod, Max, Min; f32, i32, i64) over any axis set, with or without keepdim, in Baracuda's argument convention; 100 cases bit-exact (Mean within 1 ULP) on every device (X2).

## 0.0.16

- Docs: HANDOFF rewritten for the next session.

## 0.0.15

- Emitter: i32/i64 wrapping + - *, bitwise and shift operators, bit-exact on every device (X2).

## 0.0.14

- Emitter: strided schedule (transposed, broadcast, rank-3, Coord) in Baracuda's argument convention, bit-exact on every device (X2).

## 0.0.13

- Emitter: f64 behind cl_khr_fp64 (OpenClC::for_extensions); OpenClC is now a struct, use OpenClC::new() (X2).

## 0.0.12

- Emitter: six comparisons and Select, bit-exact over the NaN/inf/zero grid on every device (X2).

## 0.0.11

- Device limits and extensions (compute units, work-group size, memory, fp64/fp16); probe prints host OS and limits; docs/device-reports.md (H1).

## 0.0.10

- Emitter: Max, Min, FmaxIeee, FminIeee, Pow; parser lifts OpenCL math names via Unpopped 0.35.3 Frontend tables (X1/X2).

## 0.0.9

- Emitter: abs, sqr, recip, sqrt, rsqrt, exp, log with device differential tests (X2).

## 0.0.8

- X1 v1: OPENCL_C frontend lifts elementwise OpenCL C kernels to the IR (feature parse); emitter accepts Vectorized plans.

## 0.0.7

- Added openclownfish-clc: v1 OpenCL C emitter (f32 contiguous elementwise) with a device-vs-oracle differential test (X2).

## 0.0.6

- Safe layer part 2: Program (build log surfaced), Kernel, unsafe launch, Event with profiling (C4).

## 0.0.5

- Safe layer part 1: Context, Queue, Buffer with bounds-checked blocking read/write (C3). Design note records the Unpopped lane's answers.

## 0.0.4

- Docs: OpenCL C parser/emitter design note (X1/X2); ROADMAP rows updated.

## 0.0.3

- Loader resolves 46 OpenCL entry points; Runtime::missing_entry_points() and probe report what a driver lacks (C2).

## 0.0.2

- Docs: ROADMAP row M1 (macOS), no macOS support claimed.
