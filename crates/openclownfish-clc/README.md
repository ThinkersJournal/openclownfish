# openclownfish-clc

OpenCL C emitter and parser for Unpopped. **Not published** (`publish = false`); it follows Unpopped's releases. Nothing here needs an OpenCL install to build; the device tests need a device and run where one exists.

## Declined by name

The kernel-IR emitter (`lower_kernel`) refuses these with a typed `LowerError::UnsupportedKernel` that names the statement. It never lowers them to a hardware operation.

- **Mbarrier statements: `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, `CopyBulk` (on an mbarrier), `Tcgen05Commit` (it arrives on an mbarrier).** NVIDIA US11803380B2. OpenCL C is vendor-neutral: an emitter that is not NVIDIA-specific could infringe by lowering these to hardware operations, so this one lowers none (CireSnave 2026-11-16: "we consider ourselves not infringing"). Not a clearance, not legal advice. Unpopped 0.47.0 has no such statements yet; a later one is refused by name from its `Debug` variant name. **This matcher is temporary**: a `Debug` string is brittle (a rename or a derive change silently stops it), so it is to be replaced by an exhaustive `match` arm and tests on real statements at the repin to the first Unpopped version that contains them.
- **Async and collective statements: `AsyncWait`, `CopyAsync`, `AsyncCommit`, `Collective`, `CollectiveTuple`.** NVIDIA US12118382B2, US11294713B2 and US8539204B2.
- **Floating-point atomics, 16-bit atomics and `F32` `Div`**: declined for correctness or risk reasons recorded in `ROADMAP.md` (X2, PS1) and `docs/device-reports.md`.
- **`WgmmaWait`** (a wait on a count of pending matrix-multiply operations): NVIDIA US12204897B2 and its continuation US12710961B2. Same reason and same rule as the mbarrier statements: this emitter is not NVIDIA-specific and lowers none of these to a hardware operation; **builders of emitters for other manufacturers' hardware should not lower them either, because of the potential for infringement** (CireSnave 2026-11-16: "Add a note and specifically notify our non-Nvidia-emitter builders about the potential for infringement and not to").

`docs/PATENT-SWEEP.md` records what was searched and what was not. A refusal here is a design choice, not a clearance.
