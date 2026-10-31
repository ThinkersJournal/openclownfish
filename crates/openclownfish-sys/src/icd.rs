//! Which OpenCL drivers the system says are installed, read without calling
//! OpenCL.
//!
//! The ICD loader (`OpenCL.dll` / `libOpenCL.so`) decides which vendor drivers
//! to load from system configuration. Reading that configuration here lets a
//! probe report a driver that is registered but fails to load, which the
//! loader itself hides: it just returns fewer platforms.
//!
//! * **Linux and other Unix**: `*.icd` files, each holding one library name, in
//!   `$OCL_ICD_VENDORS` (a directory, or one file) or `/etc/OpenCL/vendors`.
//! * **Windows**: the Khronos registry key (`HKLM\SOFTWARE\Khronos\OpenCL\Vendors`,
//!   in both registry views; a `DWORD` of 0 means enabled) and the
//!   `OpenCLDriverName` value that GPU drivers write under each display adapter
//!   class key. Most GPU drivers register only the second way, so reading the
//!   first alone misses them. The registry is read through the Windows registry
//!   API (the `winreg` crate); no program is run.
//!
//! This is what the configuration says, not what the loader will do: a driver
//! listed here may still be rejected by the loader (for example when its
//! hardware is absent), and a platform can be exposed by a mechanism not read
//! here (the loader's own environment overrides, `OPENCL_VENDOR_PATH`).

use libloading::Library;

/// Where a registration was found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IcdSource {
    /// The Khronos registry key's `Vendors` list (Windows).
    KhronosRegistry,
    /// A display adapter's `OpenCLDriverName` value (Windows).
    DisplayDriverKey,
    /// An `.icd` file; the field is its path.
    VendorFile(String),
}

/// One registered driver library.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IcdEntry {
    /// Where it was registered.
    pub source: IcdSource,
    /// The library name or path as registered.
    pub library: String,
    /// `false` when the registration marks it disabled (a Khronos key value
    /// other than 0).
    pub enabled: bool,
}

/// A Khronos registry value: the value name is the library path and the
/// `DWORD` is 0 when the driver is enabled.
#[must_use]
pub fn khronos_entry(name: &str, value: u32) -> Option<IcdEntry> {
    (!name.is_empty()).then(|| IcdEntry {
        source: IcdSource::KhronosRegistry,
        library: name.to_owned(),
        enabled: value == 0,
    })
}

/// A display adapter's `OpenCLDriverName` value.
#[must_use]
pub fn display_entry(library: &str) -> Option<IcdEntry> {
    (!library.is_empty()).then(|| IcdEntry {
        source: IcdSource::DisplayDriverKey,
        library: library.to_owned(),
        enabled: true,
    })
}

/// Parse an `.icd` file: the first non-blank, non-comment line is the library.
#[must_use]
pub fn parse_icd_file(path: &str, contents: &str) -> Option<IcdEntry> {
    let library = contents
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))?;
    Some(IcdEntry {
        source: IcdSource::VendorFile(path.to_owned()),
        library: library.to_owned(),
        enabled: true,
    })
}

/// The registry reads. An absent key is an empty list, not an error: most
/// machines have no Khronos key in the 32-bit view.
#[cfg(windows)]
fn windows_icds() -> Vec<IcdEntry> {
    use winreg::RegKey;
    use winreg::enums::{HKEY_LOCAL_MACHINE, KEY_READ, KEY_WOW64_32KEY, KEY_WOW64_64KEY};

    const KHRONOS: &str = r"SOFTWARE\Khronos\OpenCL\Vendors";
    const DISPLAY_CLASS: &str =
        r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}";

    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
    let mut v = Vec::new();
    for view in [KEY_WOW64_64KEY, KEY_WOW64_32KEY] {
        let Ok(key) = hklm.open_subkey_with_flags(KHRONOS, KEY_READ | view) else {
            continue;
        };
        for (name, _) in key.enum_values().filter_map(Result::ok) {
            // A value that is not a DWORD is not a registration.
            if let Ok(n) = key.get_value::<u32, _>(&name) {
                v.extend(khronos_entry(&name, n));
            }
        }
    }
    if let Ok(class) = hklm.open_subkey_with_flags(DISPLAY_CLASS, KEY_READ) {
        for adapter in class.enum_keys().filter_map(Result::ok) {
            let Ok(adapter) = class.open_subkey_with_flags(&adapter, KEY_READ) else {
                continue;
            };
            if let Ok(lib) = adapter.get_value::<String, _>("OpenCLDriverName") {
                v.extend(display_entry(&lib));
            }
        }
    }
    v
}

/// Every driver the system configuration registers, in a stable order, with
/// duplicates (the same library from two sources) kept.
#[must_use]
pub fn registered_icds() -> Vec<IcdEntry> {
    #[cfg(windows)]
    {
        windows_icds()
    }
    #[cfg(not(windows))]
    {
        let dir = std::env::var("OCL_ICD_VENDORS").unwrap_or_else(|_| "/etc/OpenCL/vendors".into());
        let path = std::path::Path::new(&dir);
        let mut files: Vec<std::path::PathBuf> = if path.is_dir() {
            std::fs::read_dir(path)
                .map(|rd| {
                    rd.filter_map(Result::ok)
                        .map(|e| e.path())
                        .filter(|p| p.extension().is_some_and(|x| x == "icd"))
                        .collect()
                })
                .unwrap_or_default()
        } else {
            vec![path.to_path_buf()]
        };
        files.sort();
        files
            .iter()
            .filter_map(|f| {
                let text = std::fs::read_to_string(f).ok()?;
                parse_icd_file(&f.to_string_lossy(), &text)
            })
            .collect()
    }
}

/// Try to open the registered library, as the ICD loader would. `Err` carries
/// the system's reason, which is the diagnostic a missing platform otherwise
/// lacks.
///
/// # Errors
/// The library could not be loaded.
pub fn check_load(entry: &IcdEntry) -> Result<(), String> {
    // SAFETY: loading a shared library runs its initialisers; that is the
    // documented cost of using OpenCL at all, and this is a driver the system
    // registered for exactly that.
    unsafe { Library::new(&entry.library) }
        .map(drop)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn khronos_values_with_spaces_and_a_disabled_entry() {
        let on = khronos_entry(r"C:\Program Files (x86)\Intel\bin\intelocl64.dll", 0).unwrap();
        assert_eq!(on.source, IcdSource::KhronosRegistry);
        assert!(on.enabled);
        assert!(!khronos_entry(r"C:\x\off.dll", 1).unwrap().enabled);
        // Control: an empty name is not a registration.
        assert!(khronos_entry("", 0).is_none());
    }

    #[test]
    fn display_driver_values() {
        let e = display_entry(r"C:\DriverStore\amdocl64.dll").unwrap();
        assert_eq!(e.source, IcdSource::DisplayDriverKey);
        assert_eq!(e.library, r"C:\DriverStore\amdocl64.dll");
        assert!(display_entry("").is_none());
    }

    #[test]
    fn icd_file_skips_comments_and_blanks() {
        let e = parse_icd_file(
            "/etc/OpenCL/vendors/x.icd",
            "# vendor\n\nlibnvidia-opencl.so.1\n",
        )
        .unwrap();
        assert_eq!(e.library, "libnvidia-opencl.so.1");
        assert!(parse_icd_file("e.icd", "\n# only a comment\n").is_none());
    }

    #[test]
    fn a_missing_library_reports_why() {
        // Negative control for `check_load`: the probe relies on Err meaning
        // "registered but will not load".
        let e = IcdEntry {
            source: IcdSource::VendorFile("none.icd".into()),
            library: "openclownfish-no-such-driver.dll".into(),
            enabled: true,
        };
        assert!(!check_load(&e).unwrap_err().is_empty());
    }

    /// Whatever this machine registers, listing it must not panic, and every
    /// entry names a library from a source this platform reads.
    #[test]
    fn enumerating_this_machine_returns_well_formed_entries() {
        for e in registered_icds() {
            assert!(!e.library.is_empty(), "{e:?}");
            #[cfg(windows)]
            assert!(
                matches!(
                    e.source,
                    IcdSource::KhronosRegistry | IcdSource::DisplayDriverKey
                ),
                "{e:?}"
            );
            #[cfg(not(windows))]
            assert!(matches!(e.source, IcdSource::VendorFile(_)), "{e:?}");
        }
    }
}
