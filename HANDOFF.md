# OpenCLownfish lane HANDOFF (rewritten 2026-10-30)

You own **OpenCLownfish**: safe Rust OpenCL interface, OpenCL C parser+emitter for Unpopped, later a Fuel backend. Crates reserved on crates.io at 0.0.0 (`openclownfish`, `openclownfish-sys`); the PM allocates every version at gate time. Repo ThinkersJournal/openclownfish. Read `ROADMAP.md`, `docs/survey-and-design.md` (ruling B: own safe layer over `opencl-sys` types + our own runtime loader) and `docs/parser-emitter-design.md`.

CireSnave goal (2026-10-30, verbatim): *"so we can really test the limits of every manufacturer's software stack."* Do NOT claim macOS support (row M1).

## State at 2026-10-31
- main 4c70fb5 = 0.0.13: loader, safe layer C3/C4, emitter (arith, unary, Max/Min/Fmax/Fmin/Pow, comparisons, Select, f64 behind `OpenClC::for_extensions`), parser X1 v1 (`openclownfish-clc`, unpopped 0.35.3), device limits + `docs/device-reports.md`.
- Branch `x2-strided` (worktree `C:\Projects\openclownfish-wt-strided`): strided schedule (Baracuda's argument convention, re-spelled from baracuda-cuda-emit 0.24.0), `tests/device_agrees_strided.rs`; 0.0.14 proposed. After merge: remove worktree, delete branch via API.
- PM ruling: the desktop rig dual-boots (Windows OR Linux). H1 results are labelled with OS and booted state; the other OS stays UNVERIFIED until it boots. Metal/Triton/Mojo parked after all other backends.

## Next (unblocked, start without asking)
1. X2: integer dtypes (Div/Const have int hazards: see unpopped `assert_no_int_div_or_const`), reductions/scans after the elementwise surface; each with a device differential.
2. X1 beyond elementwise (reduction, scan); N-D `get_global_id(d)` to `Coord(d)`.
3. C2 rest: ICD vendor-file enumeration. H1 on the rig per boot (needs the PM/CireSnave, board 329): the Arc A770 may lack fp64, which exercises the refusal path on a real device.

## Rules that bite here
`C:\Projects\CLAUDE.md` applies. Worktree per branch, never branch in the primary. Never merge, publish or allocate versions; `[READY]` to the PM (cwd `C:\Projects`; peer IDs rotate, use `list_peers`). A PR without a version line is refused at the gate: bump workspace Cargo.toml, Cargo.lock, CHANGELOG.md and verify each applied. Pin sonnet on any subagent. No secrets or private paths in committed files. Use Write for source files with odd quoting.
