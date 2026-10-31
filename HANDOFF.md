# OpenCLownfish lane HANDOFF (rewritten 2026-10-30)

You own **OpenCLownfish**: safe Rust OpenCL interface, OpenCL C parser+emitter for Unpopped, later a Fuel backend. Crates reserved on crates.io at 0.0.0 (`openclownfish`, `openclownfish-sys`); the PM allocates every version at gate time. Repo ThinkersJournal/openclownfish. Read `ROADMAP.md`, `docs/survey-and-design.md` (ruling B: own safe layer over `opencl-sys` types + our own runtime loader) and `docs/parser-emitter-design.md`.

CireSnave goal (2026-10-30, verbatim): *"so we can really test the limits of every manufacturer's software stack."* Do NOT claim macOS support (row M1).

## State at 2026-10-30
- main d7df6af = 0.0.5: loader (C2), design note, Context/Queue/Buffer (C3).
- Open: PR for branch `c4-program-kernel` (worktree `C:\Projects\openclownfish-wt-c4`): Program/Kernel/launch/Event (C4), 0.0.6. After merge: remove worktree, delete branch via API (CLAUDE.md section 3).

## Next (unblocked, start without asking)
1. **X2** emitter on `lower(&KernelPlan)` (design note section 4), once C4 can build and run emitted source.
2. H1 measurements on the desktop rig (RTX 4060, Radeon VII, Arc A770, Tesla P40) and Linux (PoCL, rusticl) via `cargo run --example probe`: needs the PM/CireSnave (board 329). Laptop done.

## Rules that bite here
`C:\Projects\CLAUDE.md` applies. Worktree per branch, never branch in the primary. Never merge, publish or allocate versions; `[READY]` to the PM (cwd `C:\Projects`; peer IDs rotate, use `list_peers`). A PR without a version line is refused at the gate: bump workspace Cargo.toml, Cargo.lock, CHANGELOG.md and verify each applied. Pin sonnet on any subagent. No secrets or private paths in committed files. Use Write for source files with odd quoting.
