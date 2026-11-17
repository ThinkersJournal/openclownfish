# openclownfish-clc

OpenCL C emitter and parser for Unpopped. **Not published** (`publish = false`); it follows Unpopped's releases. Nothing here needs an OpenCL install to build; the device tests need a device and run where one exists.

## Declined by name

The kernel-IR emitter (`lower_kernel`) refuses these with a typed `LowerError::UnsupportedKernel` that names the statement. It never lowers them to a hardware operation.

- **Mbarrier statements: `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, `CopyBulk` (on an mbarrier), `Tcgen05Commit` (it arrives on an mbarrier).** NVIDIA US11803380B2. OpenCL C is vendor-neutral: an emitter that is not NVIDIA-specific could infringe by lowering these to hardware operations, so this one lowers none (CireSnave 2026-11-16: "we consider ourselves not infringing"). `Tcgen05Commit` is cited as an extension of US11803380B2 per the PM, not read separately. Not a clearance, not legal advice. The match is over the real `Stmt` variants (Unpopped 0.56.1) and its tests build real statements.
- **Async and collective statements: `AsyncWait`, `CopyAsync`, `AsyncCommit`, `Collective`, `CollectiveTuple`.** NVIDIA US12118382B2, US11294713B2 and US8539204B2. Each is a typed `UnsupportedKernel` that names the statement and cites the patents and board 434 C (CireSnave 2026-11-13), by a `match` over the real `Stmt` variants (`patent_declined`), tested bare and nested in `If`, `For` and `While`. `CollectiveTuple` is declined the same way but cannot be reached through `lower_kernel` in a test (its combine function needs a module, which the checker refuses alone), so it has a unit test only.
- **Floating-point atomics, 16-bit atomics and `F32` `Div`**: declined for correctness or risk reasons recorded in `ROADMAP.md` (X2, PS1) and `docs/device-reports.md`.
- **`WgmmaWait`** (a wait on a count of pending matrix-multiply operations): NVIDIA US12204897B2 and its continuation US12710961B2. Same reason and same rule as the mbarrier statements: this emitter is not NVIDIA-specific and lowers none of these to a hardware operation; **builders of emitters for other manufacturers' hardware should not lower them either, because of the potential for infringement** (CireSnave 2026-11-16: "Add a note and specifically notify our non-Nvidia-emitter builders about the potential for infringement and not to").

`docs/PATENT-SWEEP.md` records what was searched and what was not. A refusal here is a design choice, not a clearance.
