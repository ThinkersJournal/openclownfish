# Changelog

## 0.0.47

- Dependencies: `unpopped` and `unpopped-vocab` 0.45.12 to 0.45.17 (the oracle now covers the integer Max/Min cases and the new U33 rule). No emitter change; the whole device suite is re-run against the new oracle.

## 0.0.46

- Emitter: the block-reduce variant now serves `And`, `Or`, `Xor` (integer) and `MaxNum`, `MinNum`. `And` starts the tree from all ones and an empty extent still gives 0, as the serial fold does. A float `MaxNum` or `MinNum` starts from a NaN identity so a row of only NaN stays NaN, and reports `DeterministicallyDivergent` because such a row can return another NaN payload than the serial fold's last NaN; every other new case is `BitIdentical`. `ArgMax` and `ArgMin` still have no block variant.
- Tests: `device_agrees_reduce_ops_block.rs` runs 840 cases per device against the oracle (eight integer dtypes, f32, f64, bf16, f16, five axis shapes, block sizes 4, 16 and 64). A mutant giving `And` an identity of 0 is caught.

## 0.0.45

- Emitter: the block-scan and block-reduce variants (`lower_variants`) now serve bf16 and f16. The work-group tree folds in `float` over the decoded values, the result is rounded to the format once at the store, and the arguments are the stored `ushort` patterns. `Sum`, `Mean` and `Prod` report `DeterministicallyDivergent` (the tree reassociates), `Max` and `Min` `BitIdentical`. A fused post-expression over a half is still declined, for the variant as for the serial kernel, and the variants still keep to `Sum`, `Mean`, `Prod`, `Max` and `Min`.
- Tests: `device_agrees_half_block.rs` runs 630 cases on 3 devices against the oracle (both formats, five reduction ops over four axis shapes, scans forward, reverse, inclusive and exclusive over rows longer than the block, block sizes 4, 16 and 64). A mutant that drops the re-encode at the block-scan store is caught.

## 0.0.44

- Emitter: serial `ArgMax` and `ArgMin` over every dtype served (floats, halves and all eight integers), giving an `I64` index in the row-major order of the reduced axes. This is the first kernel whose output dtype differs from the compute dtype: `lower` now admits that only for an arg reduction with `out_dtype = Some(I64)`, which the plan gate requires. NaN orders as the greatest value, ties keep the lower index, and a row of only NaN gives 0 (KISS-OPS-6.11-0007, the oracle's `arg_fold`). The block-reduce variant still declines them.
- Tests: `device_agrees_arg_reduce.rs` runs 198 cases on 3 devices (11 dtypes, both ops, three axis shapes) where the kernel, the oracle and a host fold written from the rule all agree, on a corpus with ties, NaNs, an all-negative row, equal zeros and an all-NaN row. Letting ties go to the last index and letting `ArgMin` take a NaN each fail it.

## 0.0.43

- Emitter: serial `And`, `Or` and `Xor` reductions over every integer dtype (seeded by the first element, as the oracle does; a float bit fold is declined by name), and `MaxNum` and `MinNum` over floats, halves and integers (a NaN element is skipped and a NaN accumulator replaced, so a row of only NaN stays NaN, where `Max` and `Min` keep a NaN once seen). The block-reduce variant stays on Sum, Mean, Prod, Max and Min. `ArgMax` and `ArgMin` are still declined by name.
- Tests: `device_agrees_reduce_ops.rs` runs 840 reduction cases on 3 devices against the oracle (eight integer dtypes, f32, f64, bf16, f16, five axis shapes), with a control that the data separates the `Max` and `MaxNum` NaN rules. Disabling the NaN skip and turning the `And` fold into an `Or` each fail it.

## 0.0.42

- Emitter: bf16 and f16 serial reductions (`Sum`, `Prod`, `Max`, `Min`, `Mean`) and scans (forward, reverse, inclusive, exclusive). The fold runs in `float` over the decoded values and each stored result is rounded once to the format, to nearest even, which is what the oracle's `f64` fold plus one encode gives on exactly representable data. A fused post-expression over a half is declined by name (only the plain fold is checked), and the block-scan and block-reduce variants decline halves.
- Tests: `device_agrees_half_reduce.rs` runs 234 reduction and scan cases on 3 devices against the oracle (`Mean` within one step of the format, the rest bit for bit). Doubling the stored reduction and the stored scan value are each caught. The old by-name refusal of half reductions is replaced by a check that they lower and that a fused post is declined.

## 0.0.41

- Emitter: u32 and u64 kernels (elementwise, serial reductions and scans). Wrapping `+ - *` and `<<` go through the same unsigned type, `>>` is logical, a shift count is masked by OpenCL C, and integer `Div` is guarded by the zero-divisor rule only (no `MIN / -1` case); integer constants at the unsigned types check `0..2^bits-1` (u8 and u16 constants used the 64-bit signed range before).
- Tests: `device_agrees_unsigned.rs` runs 108 elementwise cases (the oracle, a host formula and the device all agree) and 48 reduction and scan cases on 3 devices. A zero-divisor mutant and an integer `Prod` mutant each fail it. The old by-name refusal of unsigned division is replaced by a refusal of an unserved dtype (`F32Strict`).

## 0.0.40

- Deps: unpopped and unpopped-vocab 0.45.12 (from 0.43.2), per the PM's repin notice. Unpopped now declines the four patent-flagged statements (`AsyncWait`, `CopyAsync`, `AsyncCommit`, `Collective`) by name at its own plan gate (board 434 C); this emitter lowers none of them (ROADMAP PS1).

## 0.0.39

- Docs: ROADMAP row PS1 records the patent sweep for async scheduling (PM notice, board 431 hit 22): this crate lowers no `AsyncWait`, `CopyAsync` or `AsyncCommit`, and the parser refuses OpenCL's `async_work_group_copy` and `wait_group_events` by name. The same row covers `Stmt::Collective` (barrier aggregation, board 431 hit 23): no lowering here, and the parser refuses `work_group_` and `sub_group_`. No code change.

## 0.0.38

- Emitter: i8, u8, i16 and u16 `Sum`, `Prod`, `Max` and `Min` reductions now lower (ROADMAP E19). Unpopped's oracle covers them from 0.40.16, so the by-name decline is gone; integer `Mean` stays declined by name.
- Tests: `device_agrees_narrow_reduce.rs` runs 240 narrow reductions on 3 devices bit for bit against the oracle (wrapping included). Turning the integer `Sum`'s `+` into `-` makes it fail.

## 0.0.37

- Tests: ROADMAP E18 idioms proved through this crate's `OPENCL_C` front end. A `Mean` post, exclusive, reverse and reverse-exclusive scans lift, emit and agree bit for bit with the source kernel on every device (11 cases per device); the N-D `get_global_id` elementwise form and the 2-D row loop lift, and near-miss spellings are refused.
- Flipping `exclusive` or `reverse` in the serial scan emitter makes the test fail.

## 0.0.36

- Tests: the bf16 and f16 device differential (`device_agrees_half.rs`) now covers `FmaxIeee`, `FminIeee` and `Pow`, which were emitted but unchecked since 0.0.33 and 0.0.34. 288 case runs agree on all 3 laptop devices (NVIDIA RTX 4070 Laptop, AMD gfx1036, Intel CPU; Windows booted), 36 of them new. `Pow` is held to one step of the format, as the other float-computed operations are. No emitter change.
- The first run found one disagreement, and it is not a kernel defect: for `FmaxIeee(-0, +0)` the device returned `-0` and the oracle `+0`. IEEE 754 maxNum/minNum, OpenCL `fmax` and Rust's `f64::max` leave the sign of a zero tie unspecified, and the f32 differential already treats it as open. The half test now does the same, but only when both inputs are zero (`Check::ZeroTie`); every other result is byte for byte, except that two NaNs agree.
- Removing the NaN handling from `ocl_fmax` (so a NaN operand propagates) makes the test fail on a NaN input.

## 0.0.35

- Deps: unpopped and unpopped-vocab 0.43.2 (from 0.40.0). `ReduceOp` gained `And`, `Or`, `Xor`, `MaxNum`, `MinNum`, `ArgMax` and `ArgMin`; the emitter declines each by name in `reduction_facts` (an unreachable arm backs the match), so no emitted kernel changes.
- Emitter: integer `Div`, and the integer `Rem` composition over it, by the total rule of Unpopped `docs/int-div-shift-semantics.md` (ROADMAP E16): a zero divisor gives the dividend and `MIN / -1` gives `MIN`, through an `ocl_idiv` helper that evaluates each operand once (OpenCL C 6.5.1 leaves both unspecified). Integer comparisons (1 or 0 in the element type), `Select` and `ConstInt` are spelled too, so the `Rem` composition plans. `OpenClC` now returns `true` from `Backend::total_integer_ops`, which opens Unpopped's plan gate for it at I32 and I64. U32 and U64 stay refused as unsupported dtypes (the emitter has no unsigned type), and 8- and 16-bit `Div` is not admitted by the gate.
- Tests: `device_agrees_int_div.rs` runs `Div`, `Rem`, comparison with `Select`, and `ConstInt` (including `MIN`) at I32 and I64 on all 3 laptop devices (NVIDIA RTX 4070 Laptop, AMD gfx1036, Intel CPU; Windows booted): 60 case runs, each against a host function written from the document and against Unpopped's plan oracle. The inputs hold every pairing of `MIN`, `MAX`, `-1`, `0`, `1` and large values. Removing the guard from `ocl_idiv` makes the test fail on the first device.
- Not covered: the Intel CPU runtime issue in `docs/device-reports.md` is unchanged; rig results are UNVERIFIED (the PM has no read of the rig's boot state).

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
