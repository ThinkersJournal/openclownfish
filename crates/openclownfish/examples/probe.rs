//! Print every OpenCL platform and device this machine exposes.
use openclownfish::{Error, Runtime};

fn main() -> Result<(), Error> {
    let rt = Runtime::load()?;
    // Results are labelled with the OS they were taken on: the rig dual-boots.
    println!("host: {} {}", std::env::consts::OS, std::env::consts::ARCH);
    println!("library: {}", rt.library());
    println!("missing entry points: {:?}", rt.missing_entry_points());
    for p in rt.platforms()? {
        println!(
            "platform: {} | {} | {}",
            p.name()?,
            p.vendor()?,
            p.version()?
        );
        for d in p.devices()? {
            println!(
                "  device: {} | {:?} | {} | C {}",
                d.name()?,
                d.kind()?,
                d.version()?,
                d.opencl_c_version()?
            );
            let exts = d.extensions()?;
            let has = |name: &str| {
                if exts.iter().any(|e| e == name) {
                    "yes"
                } else {
                    "no"
                }
            };
            println!(
                "    units {} | work-group {} | global {} MiB | max alloc {} MiB | local {} KiB | fp64 {} | fp16 {} | {} extensions",
                d.compute_units()?,
                d.max_work_group_size()?,
                d.global_mem_bytes()? >> 20,
                d.max_alloc_bytes()? >> 20,
                d.local_mem_bytes()? >> 10,
                has("cl_khr_fp64"),
                has("cl_khr_fp16"),
                exts.len()
            );
        }
    }
    Ok(())
}
