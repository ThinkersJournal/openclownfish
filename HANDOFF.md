# OpenCLownfish lane HANDOFF (rewritten 2026-10-30)

You own **OpenCLownfish**: safe Rust OpenCL interface, OpenCL C parser+emitter for Unpopped, later a Fuel backend. Crates reserved on crates.io at 0.0.0 (`openclownfish`, `openclownfish-sys`); the PM allocates every version at gate time. Repo ThinkersJournal/openclownfish. Read `ROADMAP.md`, `docs/survey-and-design.md` (ruling B: own safe layer over `opencl-sys` types + our own runtime loader) and `docs/parser-emitter-design.md`.

CireSnave goal (2026-10-30, verbatim): *"so we can really test the limits of every manufacturer's software stack."* Do NOT claim macOS support (row M1).

## State at 2026-10-31
- main 6a68cb5 = 0.0.7: loader, safe layer C3/C4, v1 emitter `openclownfish-clc` (X2).
- Branch `x1-parser` (worktree `C:\Projects\openclownfish-wt-x1`): X1 v1 parser (`OPENCL_C` Frontend over tree-sitter-cpp, feature `parse`), emitter now also accepts Vectorized plans (element-counted), 0.0.8 proposed. After merge: remove worktree, delete branch via API.
- Asked the Unpopped lane for a per-Frontend math-name table; until then only fabs/abs/fmax/fmin lift.
- PM ruling: the desktop rig dual-boots (Windows OR Linux). H1 results are labelled with OS and booted state; the other OS stays UNVERIFIED until it boots. Metal/Triton/Mojo parked after all other backends.

## Next (unblocked, start without asking)
1. Broaden X2: more unary/binary ops (Abs, Sqrt, Max, Min, Select), f64/ints with `cl_khr_fp64` gating, Strided schedule; each with a device differential.
2. X1 beyond elementwise (reduction, scan) once the name table lands; N-D `get_global_id(d)` to `Coord(d)`.
3. H1 on the rig per boot (needs the PM/CireSnave, board 329). Laptop done.

## Rules that bite here
`C:\Projects\CLAUDE.md` applies. Worktree per branch, never branch in the primary. Never merge, publish or allocate versions; `[READY]` to the PM (cwd `C:\Projects`; peer IDs rotate, use `list_peers`). A PR without a version line is refused at the gate: bump workspace Cargo.toml, Cargo.lock, CHANGELOG.md and verify each applied. Pin sonnet on any subagent. No secrets or private paths in committed files. Use Write for source files with odd quoting.
