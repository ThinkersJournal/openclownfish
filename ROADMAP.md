# OpenCLownfish roadmap

Statuses: `open`, `doing`, `done`, `held`. Updated with every PR.

| Row | Item | Status |
|---|---|---|
| D0 | ROADMAP, CI (fmt, clippy, test, doc, deny, portfolio), CODEOWNERS, deny.toml | doing (this PR) |
| D1 | Survey and design note, `docs/survey-and-design.md` (options first, no code) | doing (this PR); recommendation awaits PM/CireSnave ruling |
| P0 | Branch protection requiring the D0 checks | open: `[ASK]` to the PM after D0 merges |
| C1 | Crate skeleton: `openclownfish-sys` + `openclownfish` workspace; remove the no-Cargo.toml guards from `ci.yml` | held on D1 ruling |
| C2 | Runtime loading of libOpenCL / OpenCL.dll, Linux and Windows | held on C1 |
| H1 | Per-machine platform/device report (laptop done, see survey section 2; desktop rig open) | doing |
| X1 | OpenCL C parser into Unpopped's IR (contract: `Unpopped/docs/parser-emitter-ownership-map.md`, ROADMAP 6.7) | held on D1, C1 |
| X2 | OpenCL C emitter from Unpopped's IR | held on X1 design |
| F1 | Fuel OpenCL backend, designed with the Fuel lane (Fuel repo not edited here) | held; below Rockfish, Bonefish, Vulkane |
