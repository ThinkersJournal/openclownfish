# Changelog

## 0.0.34

- Emitter: f16 elementwise kernels, contiguous and strided (X2 PR3), the same way as bf16: stored as its 16-bit pattern in a `ushort`, computed in `float`, through an integer decode/encode in the kernel. It needs no `cl_khr_fp16`, so it runs on NVIDIA too, which has no such extension; native `half` arithmetic was not used because it would round after every operation where the oracle rounds once at the store. A store rounds to nearest even; an f16 NaN created by an operation is the canonical quiet NaN with its sign (the oracle's rule); `Neg`, `Abs`, `Max`, `Min` and `Select` move bits as for bf16. A constant must be exactly representable in f16.
- Tests: `device_agrees_bf16.rs` became `device_agrees_half.rs` and covers both formats: 252 elementwise case runs and the strided cases agree with the oracle on all 3 laptop devices (NVIDIA included), and every one of the 65536 patterns of each format goes through decode, an exact scaling (x1, x0.5, x2) and encode, which exercises subnormal rounding and overflow. A planted tie-to-even bug in the f16 encoder fails it.
- f16 reductions and scans are declined by name, as for bf16.

## 0.0.33

- Emitter: bf16 elementwise kernels, contiguous and strided (X2 PR2). bf16 is stored as its 16-bit pattern in a `ushort` and computed in `float`: a load is a shift, a store rounds to nearest even and keeps a NaN quiet with its payload's top bits. As in Unpopped's oracle, `Neg`, `Abs`, `Max`, `Min` and `Select` on stored values move bits (a NaN payload, a signalling NaN and the sign of a zero survive); every other operation computes in float and rounds once at the store. `+ - * /`, the comparisons, `Select`, `Max`/`Min`, `Neg`/`Abs`, `sqr recip sqrt rsqrt exp log` are served; a constant must be exactly representable in bf16. 126 elementwise case runs plus the strided cases agree with the oracle on all 3 laptop devices: bit moves, comparisons and `+ - *` byte for byte (two NaNs agree where an operation creates one), `/ sqrt exp log` within one bf16 step.
- bf16 reductions and scans are declined by name (elementwise only so far); f16 is the next PR.

## 0.0.32

- Emitter: i8, u8, i16 and u16 elementwise kernels and scans (serial and block-scan variants). `+ - *` and `<<` compute in `uint` and convert back, so they wrap at the element width (a 16-bit product overflows `int`); shifts promote to `int` and take the count modulo 32, Unpopped's rule for every type narrower than 32 bits. 192 elementwise case runs plus the extended scan cases agree bit for bit with the oracle on all 3 laptop devices; the oracle also agrees with the ruled shift formula (X2).
- Narrow-integer reductions are declined with a named reason: Unpopped's oracle scopes integer reductions to i32 and i64 and panics on a narrower dtype, so they would have no referee. bf16 and f16 are not started.
- Internal: the emitter's integer limits (the Max and Min identities) come from the type, not a per-dtype match.

## 0.0.31

- Deps: unpopped and unpopped-vocab 0.40.0 (Breaking upstream: `Stmt::AtomicRmw` in the kernel IR). The emitter lowers `KernelPlan` and never consumes the kernel-layer `Stmt` IR, so no arm is needed and no emitted kernel changes; a kernel-IR emitter (atomic add/max/min on integers) is not started. Every device differential passes unchanged on all 3 laptop devices.

## 0.0.30

- Deps: unpopped and unpopped-vocab 0.39.1 (0.39 is Breaking upstream: windows with no in-bounds tap are declined, and the oracle takes integer shift counts modulo the width). The emitter serves only elementwise, reduction and scan accesses, so it has no windowed op that the tap rule could reach; the integer shift differential (negative and over-wide counts) and every other device differential pass unchanged on all 3 laptop devices.

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
