# openclownfish-sys

A runtime loader for the OpenCL ICD library. It finds `OpenCL.dll` or `libOpenCL.so` when your program runs and resolves each entry point as an `Option`, so a symbol the library lacks is `None`, never a load failure. Nothing here needs an OpenCL install to build. The safe API on top of it is the [`openclownfish`](https://crates.io/crates/openclownfish) crate; use this one only if you want the raw, unsafe function pointers.

## What exists

- `OpenCl::load()` and `OpenCl::missing_entry_points()`: open the system library (trying the usual names for the OS) and resolve 41 OpenCL entry points (the 1.x set plus a few 2.0 ones).
- `types`: a re-export of [`opencl-sys`](https://crates.io/crates/opencl-sys) 0.6 (Apache-2.0), built without its `static` feature, which supplies the types, constants and function-pointer signatures.
- `icd`: the drivers the system registers, read without calling OpenCL. On Windows, the Khronos registry key `HKLM\SOFTWARE\Khronos\OpenCL\Vendors` (both registry views) and the `OpenCLDriverName` value GPU drivers write under each display adapter; on Linux, `*.icd` files in `$OCL_ICD_VENDORS` or `/etc/OpenCL/vendors`. `icd::check_load` tries to open a registered library. This reports what the configuration says, not what the loader will do.

## Where it has been tested, and where it has not

- **Windows 11, one laptop: tested**, with the loader and `icd` run against 3 real devices through the `openclownfish` crate (NVIDIA GeForce RTX 4070 Laptop GPU, AMD gfx1036, Intel OpenCL runtime on an AMD Ryzen 9 7940HX CPU).
- **macOS: no devices.** The GitHub macOS runner reports none (error -30). macOS is not supported: the loader names Apple's `OpenCL.framework` path, but no run has reached a device, so that path is untested.
- **Linux: unverified.** The `*.icd` code is unit-tested and built on GitHub's Linux runner, which has no OpenCL device; it has not been run against PoCL, Mesa rusticl or a vendor driver.
- **Supported targets are Windows and Linux only.**
- A second machine (RTX 4060, Radeon VII, Arc A770, Tesla P40) has not been run: UNVERIFIED.

Minimum Rust version: 1.88 (set by the `libloading` dependency). Early: expect changes in 0.x releases.

## Licence

MIT or Apache-2.0, at your option. Types come from `opencl-sys` (Apache-2.0); loading uses `libloading` (ISC) and, on Windows, `winreg` (MIT).
