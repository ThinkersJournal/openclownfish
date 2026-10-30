# OpenCLownfish lane HANDOFF (new lane, created 2026-10-30 by the PM)

You own **OpenCLownfish**: a comprehensive Rust interface to OpenCL, with the OpenCL C parser and emitter for Unpopped and, eventually, a Fuel backend. Crates reserved on crates.io at 0.0.0: `openclownfish`, `openclownfish-sys` (owner ciresnave). Repo: ThinkersJournal/openclownfish (this checkout, `C:\Projects\openclownfish`).

CireSnave, 2026-10-30, verbatim: *"We need an OpenCL crate and lane to create the parser and emitter for OpenCL and eventually create the backend for Fuel for OpenCL."* *"Rather than Clownfish, what about OpenCLownfish?"* (name taken). His goal: *"so we can really test the limits of every manufacturer's software stack."* OpenCL is a real runtime API with NVIDIA, AMD and Intel drivers and CPU implementations (e.g. PoCL); it is one more route to test each vendor's stack. **Verify that on this hardware; do not assume.**

## Scope
- Survey existing Rust OpenCL crates (opencl3, cl3, ocl and others): licence, activity, coverage, safety; recommend depend / generate our own bindings / own safe layer (options first). Load libOpenCL / OpenCL.dll at runtime, Linux AND Windows.
- Parser + emitter for OpenCL C on Unpopped's contract (read `C:\Projects\Unpopped\docs\parser-emitter-ownership-map.md`, ROADMAP 6.7). Unpopped holds neither.
- Later: the OpenCL backend for Fuel (`C:\Projects\fuel`); design with the Fuel lane; do not edit its repo. Priority is below Rockfish, Bonefish and Vulkane: start with survey + design + the crate skeleton.
- Hardware: the laptop has an NVIDIA RTX 4070 and an AMD GPU; the desktop rig (dual boot) has RTX 4060, Radeon VII, Arc A770, Tesla P40 (a second P40 and a MI50 on order). Report which OpenCL platforms each machine exposes, measured.

(see COMMON rules below)
## Rules that bite here
Read `C:\Projects\CLAUDE.md` (loaded) and `C:\Projects\AGENT-ONBOARDING.md` once. Work in a worktree (`C:\Projects\<repo>-wt-<topic>`), never branch in the primary checkout. Version bump on every pushed change (pre-1.0 breaking = the SECOND number); the PM allocates the number at gate time. crates.io versions only, no path or git dependencies; all dependencies on their latest versions. Trace and verify every third-party file against its origin and keep its licence and credit (CireSnave: "I am not a plageurist"). Every claim verified on the forge or by measurement; every absence needs a positive control. Compact `[READY]`/`[STATUS]` messages, short. Peers change ID on every restart: `list_peers`, match on cwd. Restart at task boundaries (HANDOFF rewrite, `lane-restart assert-idle` last, then `--self`). This is an open-source project from day one: no secrets, no private paths in committed files.

## First deliverables (all lanes of this family)
0. `ROADMAP.md` with rows and statuses; a CI workflow (fmt, clippy, tests, cargo-deny, and the portfolio checks workflow the other repos use: copy the shape from `C:\Projects\capant\.github`) in your first PR; then `[ASK]` the PM to enable branch protection requiring those checks (CI and protection everywhere).
1. A survey and design note (options first, recommendation, no code) before any crate code.
