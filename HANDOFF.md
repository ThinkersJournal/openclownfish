# OpenCLownfish lane HANDOFF (rewritten 2026-10-30)

You own **OpenCLownfish**: safe Rust OpenCL interface, OpenCL C parser+emitter for Unpopped, later a Fuel backend. Crates reserved on crates.io at 0.0.0 (`openclownfish`, `openclownfish-sys`); the PM allocates every version at gate time. Repo ThinkersJournal/openclownfish. Read `ROADMAP.md`, `docs/survey-and-design.md` (ruling B: own safe layer over `opencl-sys` types + our own runtime loader) and `docs/parser-emitter-design.md`.

CireSnave goal (2026-10-30, verbatim): *"so we can really test the limits of every manufacturer's software stack."* Do NOT claim macOS support (row M1).

## State at 2026-10-30
- main 5dd286d = 0.0.4: loader with 46 entry points (C2), M1 docs, parser/emitter design note (X1/X2). The Unpopped lane answered the 4 design questions (design note section 6).
- Open: PR for branch `c3-safe-layer` (worktree `C:\Projects\openclownfish-wt-c3`): Context/Queue/Buffer (C3), 0.0.5 proposed. After merge: remove worktree, delete branch via API (CLAUDE.md section 3).

## Next (unblocked, start without asking)
1. **C4** safe layer part 2: Program (per-device build log surfaced), Kernel, arg setting, NDRange launch, Event profiling. Needs only the C2 entry points.
2. **X2** emitter on `lower(&KernelPlan)` (design note section 4), once C4 can build and run emitted source.
3. H1 measurements on the desktop rig (RTX 4060, Radeon VII, Arc A770, Tesla P40) and Linux (PoCL, rusticl) via `cargo run --example probe`: needs the PM/CireSnave (board 329). Laptop done.

## Rules that bite here
`C:\Projects\CLAUDE.md` applies. Worktree per branch, never branch in the primary. Never merge, publish or allocate versions; `[READY]` to the PM (cwd `C:\Projects`; peer IDs rotate, use `list_peers`). A PR without a version line is refused at the gate: bump workspace Cargo.toml, Cargo.lock, CHANGELOG.md and verify each applied. Pin sonnet on any subagent. No secrets or private paths in committed files. Use Write for source files with odd quoting.
