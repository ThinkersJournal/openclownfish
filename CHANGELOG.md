# Changelog

## 0.0.29

- Emitter: reductions over several inputs are served (dot product, weighted Max/Min, a dot with a fused post), by the serial kernel and the `blockreduce` variant. Kernel arguments: the input pointers, `out`, `shape{d}`, then `s{k}_{d}` for every input `k` in turn, `so_{j}`, `n_out`; each input is read through its own strides, so a broadcast input (stride 0) needs only its smaller buffer. A single-input reduction takes the same arguments as before. The cases (f32/f64/i32/i64, two shapes, four axis sets, dense and broadcast second input, serial and block sizes 8 and 32) agree bit for bit with the oracle on all 3 laptop devices (X2).

## 0.0.28

- Emitter: reductions with a fused post-expression are served, by the serial kernel and the `blockreduce` variant alike. The post reads the finalized fold (after `Mean`'s division) as `Reduced(0)`, may use constants, and is spelled in the element type (`norm2 = sqrt(sum(x*x))`, scale-and-offset, integer `red*red`). An input in a post is refused (Unpopped's plan gate rejects it first). 144 cases per device (serial and block sizes 8 and 32, f32/f64/i32/i64, two shapes, four axis sets) agree with the oracle on all 3 laptop devices; posts that round (`sqrt`, `Mean`) are compared within 4 ulp, the rest bit for bit (X2).

## 0.0.27

- Emitter: `OpenClC::lower_variants` also offers a `blockreduce` variant of every reduction the serial emitter serves: one work-group per output element (global = `n_out * B`, local = `B`, `-D OCF_BLOCK=n`, default 64). Each work item folds a contiguous chunk of the row-major reduced extent, then a neighbour-pair tree in `__local` memory combines the chunks in order, so Max/Min keep the serial fold's tie and NaN rule. Float Sum/Mean/Prod are `DeterministicallyDivergent`; Max/Min and integer reductions are `BitIdentical`. An empty reduced extent returns the serial seed. 170 oracle cases per device at block sizes 8 and 32, a NaN/signed-zero Max/Min bit-comparison against the serial kernel, and the empty-extent case agree on all 3 laptop devices (X2).
- Internal: the reduction emitter's parameter list and addressing are shared between the serial and block kernels; the serial kernel's output is unchanged.

## 0.0.26

- Deps: unpopped and unpopped-vocab 0.38.4. Workspace tests, including every device differential on the 3 laptop devices, pass unchanged.

## 0.0.25

- Deps: unpopped and unpopped-vocab 0.38.3.
- Test: integer `<<` and `>>` with negative and over-wide counts (-bits..2*bits) are bit-exact against the ruled total semantics (count unsigned, modulo the width) on every device; no emitter change was needed because OpenCL C masks the count. Integer division remains refused.

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
