# openclownfish

OpenCL for Rust, loaded at run time, built to test the limits of every manufacturer's software stack. Two crates are published; one is not.

| Crate | What it is |
| --- | --- |
| `openclownfish` | Safe Rust interface: platforms, devices, contexts, queues, buffers, programs, kernels, events. |
| `openclownfish-sys` | The run-time loader for `OpenCL.dll` / `libOpenCL.so` and the ICD registration reader. |
| `openclownfish-clc` | OpenCL C emitter and parser for Unpopped. **Not published** (`publish = false`); it follows Unpopped's releases. |

Nothing needs an OpenCL install to build. `cargo run --example probe` prints every platform, device and registered driver on the machine.

## Tested, and not

- **Windows 11, one laptop, 3 devices: tested.** NVIDIA GeForce RTX 4070 Laptop GPU (OpenCL 3.0, OpenCL C 1.2), AMD gfx1036 (OpenCL C 2.0), Intel OpenCL runtime on an AMD Ryzen 9 7940HX CPU (OpenCL C 3.0). `docs/device-reports.md` has the probe output.
- **macOS: no devices.** GitHub's macOS runner reports none (error -30). Not supported.
- **Linux: unverified.** Built and unit-tested on GitHub's Linux runner (no device); never run against PoCL, Mesa rusticl or a vendor driver.
- **The desktop rig** (RTX 4060, Radeon VII, Arc A770, Tesla P40): UNVERIFIED.
- Supported targets: Windows and Linux. GitHub's hosted runners have no GPU, so CI does not prove a kernel runs.

See `ROADMAP.md`, `docs/survey-and-design.md` and `docs/PATENT-SWEEP.md` (a partial record: it says what was searched and what was not).

Minimum Rust version of the published crates: 1.88. Licensed under MIT or Apache-2.0, at your option.
