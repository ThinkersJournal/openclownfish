//! Print every OpenCL platform and device this machine exposes.
use openclownfish::{Error, Runtime, icd};

fn main() -> Result<(), Error> {
    // Results are labelled with the OS they were taken on: the rig dual-boots.
    println!("host: {} {}", std::env::consts::OS, std::env::consts::ARCH);
    // The drivers the system registers, before the loader hides any that fail.
    for e in icd::registered_icds() {
        let state = match icd::check_load(&e) {
            Ok(()) => "loads".to_owned(),
            Err(why) => format!("FAILS TO LOAD: {why}"),
        };
        println!(
            "icd: {} | {:?} | {} | {state}",
            e.library,
            e.source,
            if e.enabled { "enabled" } else { "disabled" }
        );
    }
    let rt = Runtime::load()?;
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
