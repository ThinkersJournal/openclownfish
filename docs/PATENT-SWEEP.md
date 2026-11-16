# Patent sweep: openclownfish 0.1 @ main after #55

Date: 2026-11-16; Run by: the openclownfish lane, Claude Sonnet 5.5; Method: AI-assisted read of records already made; **NOT legal advice; NOT a freedom-to-operate opinion.**
Version lines covered: **0.1** (0.1.3). Process: `C:\Projects\PATENT-SWEEP-PROCESS.md`.

**This record is partial, and says so.** It is not a fresh sweep of every idea. The only searches ever run for this repository are the ones the PM and this lane made for the async/collective and atomics families (board 431 and 441, 2026-11-13 and 2026-11-14), listed below with their verdicts. Every other idea is listed as **none searched, named unread**: no query was run and no claim was read for it. Nothing here clears the project.

## Ideas swept

Plain claim-like sentences, read from the source at this commit:

1. **A safe wrapper over an OpenCL runtime** (context, queue, program, kernel, buffer, event handles with ownership and error mapping). Searched: no. Verdict: none searched, named unread.
2. **Emitting OpenCL C from a kernel plan** (elementwise, reduction, scan, ArgMax/ArgMin, gather and layout ops) for the Unpopped backend interface, with a per-device differential test against an interpreter. Searched: no. none searched, named unread.
3. **Block-scan and block-reduce variants** that combine through local memory and `barrier(CLK_LOCAL_MEM_FENCE)` with the reduction written out in the kernel. Searched: yes, as part of the Collective read of board 431 hit 23 (see hit 3 below).
4. **A parser from OpenCL C to a plan**, refusing named constructs it does not model (`RESIDUE`). Searched: no. none searched, named unread.
5. **Lowering a kernel-IR (`Stmt`) function to an OpenCL C kernel**: integer and float arithmetic, `Let`/`Assign`/`Load`/`Store`, `If`/`While`/`For`/`Break`/`Continue`, block barrier, `Shared` arrays. Searched: no for the control-flow and arithmetic parts. none searched, named unread.
6. **Integer `AtomicRmw` and `AtomicCas` spelled as OpenCL C 2.0 `atomic_*_explicit` calls**, opt-in. Searched: yes (hits 4 to 7 below).
7. **Async copy and wait lowering** (`AsyncWait`, `CopyAsync`, `AsyncCommit`). Not implemented here; the emitter has no path for it and the parser refuses the OpenCL names. Searched: yes by the PM (board 431 hit 22; hits 1 and 2).
8. **Evaluating an F32 kernel statement at double to match an oracle that rounds once per statement** (with `cl_khr_fp64`), declining what cannot match. Searched: no. none searched, named unread.
9. **Opt-in native half precision** behind `cl_khr_fp16`. Searched: no. none searched, named unread.

## Queries (source, query, date)

Source: Google Patents; claims read by the PM (verbatim) for the hits marked so. The queries are as recorded in `ROADMAP.md` row PS1; the async/collective queries by the PM are on board 431 and are not reproduced here (not held in this repository).

| # | Idea | Query | Date |
|---|---|---|---|
| Q1 | 6 | atomic read-modify-write by a single instruction | 2026-11-14 |
| Q2 | 6 | compare-and-swap with scope and an IR | 2026-11-14 |
| Q3 | 7 | board 431 (PM): async copy with a wait on a thread's own instructions | 2026-11-13 |
| Q4 | 3 | board 431 (PM): barrier aggregation instruction | 2026-11-13 |

Control for Q1 and Q2: the same searches returned US8624916B2, US7627723B1, US11972261B2 and US9513976B2, so they find what they should.

## Hits

| # | patent | assignee | claim read | overlap | our difference | verdict |
|---|---|---|---|---|---|---|
| 1 | US12118382B2 | NVIDIA | claim 1 (prevent first instructions of a thread while its second instructions run) | OpenCL `async_work_group_copy` with `wait_group_events` | this crate lowers no async statement; parser refuses the names | clear (not implemented here; re-read if a lowering is added) |
| 2 | US11294713B2 | NVIDIA | claim 1, same family | same | same | clear, as hit 1 |
| 3 | US8539204B2 | NVIDIA | claim 1 and 13 (barrier aggregation instruction; hardware barrier execution unit) | `work_group_reduce_*`, `work_group_scan_*`, `sub_group_*` | no `Stmt::Collective` lowering; parser refuses those prefixes; block variants use plain `barrier` and local memory | clear as read by the PM (ruled A, board 431 hit 23) |
| 4 | US7627723B1 | NVIDIA | independent claims 1, 10 and 20 (instruction unit, several execution units, operation routed by address range); read verbatim by the PM | atomic operations are in its description, not its claims | this emitter spells Khronos atomic builtins as text; it adds no unit and no address routing | clear as read; ruled A (board 441, CireSnave 2026-11-14), risk note in `kernel.rs` rustdoc |
| 5 | US9513976B2 | Intel | CAS-based lock using a software parameter emulating a full/empty bit; read verbatim by the PM | CAS | no lock built on an emulated full/empty bit | clear as read; same ruling |
| 6 | US8624916B2 | (raster-operations blend unit) | read verbatim | none relevant | hardware unit | clear as read |
| 7 | US11972261B2 | (hardware CAS module) | paraphrase only | none relevant | hardware module | clear as read; a paraphrase, not a full claim read |

## Needs review (to the PM)

None from what was read. The ideas marked "none searched, named unread" are not clear: they are unread.
Re-read triggers recorded in `ROADMAP.md` PS1: a lowering that adds an execution unit or address routing, a lock built on an emulated full/empty bit, or any lowering of `AsyncWait`, `CopyAsync`, `AsyncCommit` or `Collective`.

## Limitations

AI-assisted; two of the seven hits were read by the PM and the rest taken from their records; ideas 1, 2, 4, 5 (control flow and arithmetic), 8 and 9 were not searched. Google Patents status was not re-checked at this commit. This is not a freedom-to-operate opinion.

### 0.1.3 (0.1 line) - 2026-11-16

Version line 0.1 (0.1.0 repin, 0.1.1 integer `For`, 0.1.2 F32/F64 kernels, 0.1.3 pin move). 0.1.1 and 0.1.2 each recorded "no new search" in the CHANGELOG: plain loops and float arithmetic add no atomics, execution units, address routing or full/empty-bit lock, and float atomics stay declined. That is a reasoned read, not a search, and is recorded as such here.
