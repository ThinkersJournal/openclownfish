//! Print every OpenCL platform and device this machine exposes.
use openclownfish::{Error, Runtime};

fn main() -> Result<(), Error> {
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
        }
    }
    Ok(())
}
