# Patent sweep: openclownfish 0.1 @ main after #55

Date: 2026-11-16; Run by: the openclownfish lane, Claude Sonnet 5.5; Method: AI-assisted read of records already made; **NOT legal advice; NOT a freedom-to-operate opinion.**
Version lines covered: **0.1** (0.1.3; the 2026-11-16 addition below covers ideas 1 and 10 at the commit that adds it). Process: `C:\Projects\PATENT-SWEEP-PROCESS.md`.

**This record is partial, and says so.** It is not a fresh sweep of every idea. The only searches ever run for this repository are the ones the PM and this lane made for the async/collective and atomics families (board 431 and 441, 2026-11-13 and 2026-11-14), listed below with their verdicts. Every other idea is listed as **none searched, named unread**: no query was run and no claim was read for it. Nothing here clears the project. **Addition 2026-11-16 (PM [TASK] openclownfish-publish-prep, part B):** ideas 1 and 10, the two the published crates `openclownfish` and `openclownfish-sys` rest on, were searched; see hits 8 to 14. A null is not clearance.

## Ideas swept

Plain claim-like sentences, read from the source at this commit:

1. **A safe wrapper over an OpenCL runtime** (context, queue, program, kernel, buffer, event handles with ownership and error mapping). Searched: yes, 2026-11-16 (Q5, Q6; hits 8 to 10). Verdict: no hit reads on it; see the hits table and the limits of the search.
2. **Emitting OpenCL C from a kernel plan** (elementwise, reduction, scan, ArgMax/ArgMin, gather and layout ops) for the Unpopped backend interface, with a per-device differential test against an interpreter. Searched: no. none searched, named unread.
3. **Block-scan and block-reduce variants** that combine through local memory and `barrier(CLK_LOCAL_MEM_FENCE)` with the reduction written out in the kernel. Searched: yes, as part of the Collective read of board 431 hit 23 (see hit 3 below).
4. **A parser from OpenCL C to a plan**, refusing named constructs it does not model (`RESIDUE`). Searched: no. none searched, named unread.
5. **Lowering a kernel-IR (`Stmt`) function to an OpenCL C kernel**: integer and float arithmetic, `Let`/`Assign`/`Load`/`Store`, `If`/`While`/`For`/`Break`/`Continue`, block barrier, `Shared` arrays. Searched: no for the control-flow and arithmetic parts. none searched, named unread.
6. **Integer `AtomicRmw` and `AtomicCas` spelled as OpenCL C 2.0 `atomic_*_explicit` calls**, opt-in. Searched: yes (hits 4 to 7 below).
7. **Async copy and wait lowering** (`AsyncWait`, `CopyAsync`, `AsyncCommit`). Not implemented here; the emitter has no path for it and the parser refuses the OpenCL names. Searched: yes by the PM (board 431 hit 22; hits 1 and 2).
8. **Evaluating an F32 kernel statement at double to match an oracle that rounds once per statement** (with `cl_khr_fp64`), declining what cannot match. Searched: no. none searched, named unread.
9. **Opt-in native half precision** behind `cl_khr_fp16`. Searched: no. none searched, named unread.
10. **Reading which OpenCL drivers the system registers** (the Windows registry Khronos `Vendors` key in both views, the `OpenCLDriverName` value under each display adapter class key, `*.icd` files on Linux), **testing whether each registered library opens, and loading the OpenCL library at run time with every entry point resolved as an `Option`** so a missing symbol fails only the call that needs it (`openclownfish-sys`, `icd.rs` and `lib.rs`). Searched: yes, 2026-11-16 (Q7 to Q10; hits 11 to 14). Verdict: no hit reads on it; see the hits table and the limits of the search.

## Queries (source, query, date)

Source: Google Patents; claims read by the PM (verbatim) for the hits marked so. The queries are as recorded in `ROADMAP.md` row PS1; the async/collective queries by the PM are on board 431 and are not reproduced here (not held in this repository).

| # | Idea | Query | Date |
|---|---|---|---|
| Q1 | 6 | atomic read-modify-write by a single instruction | 2026-11-14 |
| Q2 | 6 | compare-and-swap with scope and an IR | 2026-11-14 |
| Q3 | 7 | board 431 (PM): async copy with a wait on a thread's own instructions | 2026-11-13 |
| Q4 | 3 | board 431 (PM): barrier aggregation instruction | 2026-11-13 |
| Q5 | 1 | patent safe wrapper language binding GPU compute runtime handle ownership automatic release error codes mapped exceptions OpenCL | 2026-11-16 |
| Q6 | 1 | (same idea, second look) read the two WebCL-family hits Q5 returned, US9348676B2 and US9448823B2, plus the safe-handle hit US7610322B2 | 2026-11-16 |
| Q7 | 10 | patent installable client driver loader enumerating registered vendor drivers registry key dynamically loading library compute API | 2026-11-16 |
| Q8 | 10 | patent runtime loading shared library resolving function entry points missing symbol null fallback dlopen API dispatch table | 2026-11-16 |
| Q9 | 10 | OpenCL installable client driver ICD loader platform dispatch vendor library patent | 2026-11-16 |
| Q10 | 10 | patent determining graphics driver registered in system configuration but fails to load, enumerating display adapter driver names from registry, diagnostic report of compute devices | 2026-11-16 |

Source for Q5 to Q10: a web search restricted to `patents.google.com` (the Google Patents query page needs a browser), one query each; candidates were then read in full with `C:\Projects\tools\patent-claims.py` (independent claims first). Control for Q5 to Q10: the same search method, run on the atomics idea with the query "patent instruction unit execution units atomic operation routed by address range shared memory", returned US7627723B1 first, a hit already in this record (hit 4), so the method finds a known patent. A sound control proves the method works, not that these queries asked the right question.

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
| 8 | US9348676B2 | Google Technology Holdings (Active per Google, est. expiry 2033) | independent claims 1, 17, 20 read in full: a machine processor that acquires a **download script** whose commands bind to functions of an OpenCL function library and a graphics library, creates parallel contexts for respective OpenCL devices and runs the script portions in them (dependent claim 5 names a web browser) | OpenCL contexts and buffers | no download script, no browser or web server, no graphics-library binding, no script run per context; this crate is a library a program calls | clear on the independent claims as read |
| 9 | US9448823B2 | Google Technology Holdings (Expired - Fee Related per Google) | independent claims 1 and 20 read: a client requests an **extension** from a web server, the extension naming bindings to functions of a parallel-processing library | same family as hit 8 | no extension repository, no web server | clear as read |
| 10 | US7610322B2 | Microsoft (Expired per Google, est. 2025-03-19) | independent claims 1, 7, 14, 20 read: a **safe handle administrator** and a handle administrator, a reference counter, and status flags for expedited release | handle ownership and release | this crate wraps each OpenCL object in a Rust value that retains and releases it on drop, with no administrator pair, no shared counter flags, no expedited-release request | clear as read; expired anyway |
| 11 | US7689985B2 | Microsoft (Expired per Google, est. 2027-03-17) | independent claims 1, 16, 20, 21 read: a **version library** of operating-system framework components, a client driver and a registration architecture that binds the two by revision | registry-based driver binding | we read the registry and `*.icd` files to list drivers and open each library; no version library, no binding by revision | clear as read |
| 12 | US7793279B1 | VMware (Expired per Google) | independent claims 1 and 16 read: swapping a device's **driver while the OS stays running**, using the OS map of device identifiers to drivers | OS driver map | we replace no driver and write no registry | clear as read |
| 13 | US5916308A, EP1386230B1 | IBM (both expired per Google) | claim 1 of each read: DLL linking by compiler, prelinker and runtime library without system linker or loader; a **relay file** that substitutes functions for symbols at start-up | run-time symbol resolution | we call the OS loader (`libloading`: `dlopen`/`LoadLibrary`, `dlsym`/`GetProcAddress`) and store each result as an `Option`; no prelinker, no relay file, no symbol substitution | clear as read |
| 14 | US7080247B2 | EVGA (Expired per Google, est. 2021) | independent claims 1, 5, 9 read: **identify the motherboard chipset, remove all installed display driver files, install new ones** | display driver registry handling | we read registrations and install, remove and modify nothing | clear as read |
| 15 | US20030005176A1 | Intel (Expired - Fee Related per Google) | claim 1 read: compare the memory location of two kinds of **bus configuration information** for a device and select one | a search hit for a diagnostic of a driver that failed to load; the claim does not read that way | we read no bus configuration | clear as read |

## Needs review (to the PM)

None from what was read, including hits 8 to 15 (all clear on the independent claims as read; only US9348676B2 is listed Active by Google, and its claims need a download script and OpenCL parallel contexts, which this crate has neither of). The ideas marked "none searched, named unread" are not clear: they are unread.
Re-read triggers recorded in `ROADMAP.md` PS1: a lowering that adds an execution unit or address routing, a lock built on an emulated full/empty bit, or any lowering of `AsyncWait`, `CopyAsync`, `AsyncCommit` or `Collective`.

## Limitations

AI-assisted; two of the seven hits were read by the PM and the rest taken from their records; ideas 2, 4, 5 (control flow and arithmetic), 8 and 9 were not searched. For ideas 1 and 10 (added 2026-11-16): the search was one web query per phrasing (Q5 to Q10), not the Google Patents query interface, so its coverage is thin and a null is not clearance; patents using other words for a loader, a handle wrapper or a registry reader would not be found; non-US filings were not systematically searched; the Khronos ICD loader and the OpenCL specification are published prior art (from general knowledge, not read here) and no patent claiming the ICD mechanism turned up; claim text from the shared tool is machine-converted and no claim was checked against the official record; expiry and status are Google Patents' estimates; hit 8's status is the only one listed Active. Nobody has read a patent attorney's view of any of this. Google Patents status was not re-checked at this commit. This is not a freedom-to-operate opinion.

### 0.1.3 (0.1 line) - 2026-11-16

Version line 0.1 (0.1.0 repin, 0.1.1 integer `For`, 0.1.2 F32/F64 kernels, 0.1.3 pin move). 0.1.1 and 0.1.2 each recorded "no new search" in the CHANGELOG: plain loops and float arithmetic add no atomics, execution units, address routing or full/empty-bit lock, and float atomics stay declined. That is a reasoned read, not a search, and is recorded as such here.

### 0.1.6 - 2026-11-16

Patent search of ideas 1 (safe wrapper) and 10 (ICD and registry reading, run-time loading), ten queries total with the control above; hits 8 to 15 read at their independent claims, all clear as read. No code change.
