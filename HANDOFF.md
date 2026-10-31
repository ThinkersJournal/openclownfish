# OpenCLownfish lane HANDOFF (rewritten 2026-10-30)

You own **OpenCLownfish**: safe Rust OpenCL interface, OpenCL C parser+emitter for Unpopped, later a Fuel backend. Crates reserved on crates.io at 0.0.0 (`openclownfish`, `openclownfish-sys`); the PM allocates every version at gate time. Repo ThinkersJournal/openclownfish. Read `ROADMAP.md`, `docs/survey-and-design.md` (ruling B: own safe layer over `opencl-sys` types + our own runtime loader) and `docs/parser-emitter-design.md`.

CireSnave goal (2026-10-30, verbatim): *"so we can really test the limits of every manufacturer's software stack."* Do NOT claim macOS support (row M1).

## State at 2026-10-30
- main d7df6af = 0.0.5; PR #7 (C4 Program/Kernel/launch, 0.0.6) at the PM gate.
- Branch `x2-emitter` (worktree `C:\Projects\openclownfish-wt-x2`) is stacked on `c4-program-kernel`: v1 emitter crate `openclownfish-clc` + device differential test, 0.0.7 proposed. NOT pushed until #7 merges; then `git rebase --onto origin/main <old c4 tip 2235775>`, push, [READY]. After merge: remove worktree, delete branch via API.
- Findings to send the Unpopped lane (design note section 7): the f64 oracle models fused a*b+c, `unpopped-cpu-c` emits unfused C; they differ by 2 ULP.
- PM ruling: the desktop rig dual-boots (Windows OR Linux). H1 results are labelled with OS and booted state; the other OS stays UNVERIFIED until it boots. Metal/Triton/Mojo parked after all other backends.
- Unpopped 0.35 (tree-sitter 0.27, breaking) is coming; X1 builds against it. Emitter uses 0.34.13 from crates.io.

## Next (unblocked, start without asking)
1. Broaden X2: more unary/binary ops (Abs, Sqrt, Max, Min, Select), f64/ints with `cl_khr_fp64` gating, Strided schedule; each with a device differential.
2. X1 parser once Unpopped 0.35 is published.
3. H1 on the rig per boot (needs the PM/CireSnave, board 329). Laptop done.

## Rules that bite here
`C:\Projects\CLAUDE.md` applies. Worktree per branch, never branch in the primary. Never merge, publish or allocate versions; `[READY]` to the PM (cwd `C:\Projects`; peer IDs rotate, use `list_peers`). A PR without a version line is refused at the gate: bump workspace Cargo.toml, Cargo.lock, CHANGELOG.md and verify each applied. Pin sonnet on any subagent. No secrets or private paths in committed files. Use Write for source files with odd quoting.
