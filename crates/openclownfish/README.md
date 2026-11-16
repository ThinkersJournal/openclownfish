# openclownfish

A safe Rust interface to OpenCL. The OpenCL library (`OpenCL.dll` or `libOpenCL.so`) is found and loaded at run time, so nothing here needs an OpenCL install to build, and a machine without a driver gets an `Err`, not a link failure.

## What exists

- `Runtime::load`, then `platforms()` and `devices()`: every platform and device the loader exposes, with name, vendor, version, OpenCL C version, kind, extensions, compute units, work-group size and memory limits.
- `Context`, `Queue` (plain and profiling), `Buffer` with bounds-checked `write_buffer` and `read_buffer`.
- `Program::build` from OpenCL C source: a failed build is an `Error::Build` carrying the driver's log verbatim.
- `Kernel` with typed and byte arguments, `Queue::launch` (an `unsafe fn`: OpenCL C is unchecked), `Event::wait` and `Event::duration_ns`.
- `icd`: the drivers the system registers (the Windows registry, or `*.icd` files on Linux), and whether each one loads. This can report a driver that is registered but fails to load, which the OpenCL loader hides.
- An entry point a library lacks is reported by `Runtime::missing_entry_points` and fails only the call that needs it.

```rust,no_run
use openclownfish::{Error, Runtime};

fn main() -> Result<(), Error> {
    let rt = Runtime::load()?;
    for platform in rt.platforms()? {
        println!("{} ({})", platform.name()?, platform.version()?);
        for device in platform.devices()? {
            println!("  {} | {:?} | {}", device.name()?, device.kind()?, device.opencl_c_version()?);
        }
    }
    Ok(())
}
```

`cargo run --example probe` prints the same, plus the registered drivers.

## Where it has been tested, and where it has not

Plainly:

- **Tested on devices: Windows 11 on one laptop, 3 devices.** NVIDIA GeForce RTX 4070 Laptop GPU (NVIDIA CUDA, OpenCL 3.0, OpenCL C 1.2), AMD gfx1036 (AMD Accelerated Parallel Processing, OpenCL C 2.0), and the Intel OpenCL runtime on the AMD Ryzen 9 7940HX CPU (OpenCL C 3.0). Buffer round trips, a vector add and the device-limit checks run on all three.
- **macOS: no devices.** The GitHub macOS runner reports no OpenCL device (error -30), so only the no-device paths run there. macOS support is not claimed.
- **Linux: unverified.** The Linux code paths (`libOpenCL.so`, `*.icd` files) compile and their no-device tests run on GitHub's Linux runner, but no device has run on Linux: PoCL and Mesa rusticl are untested.
- **Supported targets are Windows and Linux only.** The loader also names macOS's `OpenCL.framework` path, untested.
- A second machine with an RTX 4060, Radeon VII, Arc A770 and Tesla P40 has not been run: UNVERIFIED.
- GitHub's hosted runners have no GPU, so CI proves that the crate builds, that the unit tests pass, and that "no device" is handled. It does not prove a kernel runs.

Minimum Rust version: 1.88 (set by the `libloading` dependency). Early: the surface is small and will change in 0.x releases.

## Licence

MIT or Apache-2.0, at your option. The OpenCL types and function signatures come from the [`opencl-sys`](https://crates.io/crates/opencl-sys) crate (Apache-2.0).
