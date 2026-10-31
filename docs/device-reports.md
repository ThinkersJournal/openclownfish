# Device reports

Output of `cargo run --example probe`, one section per machine **and per boot**. The desktop rig
dual-boots, so a result is only valid for the OS it was taken on; the other OS stays UNVERIFIED until it
boots (PM ruling, board 329, 2026-10-31).

## Laptop, Windows 11 (booted), openclownfish 0.0.11, 2026-10-31

```
host: windows x86_64
library: OpenCL.dll
missing entry points: []
platform: NVIDIA CUDA | NVIDIA Corporation | OpenCL 3.0 CUDA 13.4.96
  device: NVIDIA GeForce RTX 4070 Laptop GPU | Gpu | OpenCL 3.0 CUDA | C OpenCL C 1.2
    units 36 | work-group 1024 | global 8187 MiB | max alloc 2046 MiB | local 48 KiB | fp64 yes | fp16 no | 26 extensions
platform: AMD Accelerated Parallel Processing | Advanced Micro Devices, Inc. | OpenCL 2.1 AMD-APP (3652.0)
  device: gfx1036 | Gpu | OpenCL 2.0 AMD-APP (3652.0) | C OpenCL C 2.0
    units 1 | work-group 256 | global 36769 MiB | max alloc 30818 MiB | local 64 KiB | fp64 yes | fp16 yes | 25 extensions
platform: Intel(R) OpenCL | Intel(R) Corporation | OpenCL 3.0 WINDOWS
  device: AMD Ryzen 9 7940HX with Radeon Graphics | Cpu | OpenCL 3.0 (Build 0) | C OpenCL C 3.0
    units 32 | work-group 8192 | global 64727 MiB | max alloc 32363 MiB | local 256 KiB | fp64 yes | fp16 yes | 46 extensions
```

Notes: the NVIDIA stack reports OpenCL C 1.2 on an OpenCL 3.0 platform (3.0 allows that), so a
kernel needing 2.0 features cannot be assumed there. `cl_khr_fp16` is absent on the NVIDIA device,
which gates any half-precision emission.

## Desktop rig (RTX 4060, Radeon VII, Arc A770, Tesla P40)

- Windows: UNVERIFIED (not booted for this project yet).
- Linux (PoCL, Mesa rusticl): UNVERIFIED (not booted for this project yet).
