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
//! * **Windows**: the Khronos registry key (`HKLM\SOFTWARE\Khronos\OpenCL\Vendors`
//!   and its 32-bit view; a `DWORD` of 0 means enabled) and the `OpenCLDriverName`
//!   value that GPU drivers write under each display adapter class key. Most
//!   GPU drivers register only the second way, so reading the first alone misses
//!   them. The registry is read by running `reg.exe`, which is part of Windows,
//!   so no dependency is added.
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

/// Parse the output of `reg query HKLM\SOFTWARE\Khronos\OpenCL\Vendors`: lines
/// of `<library path>    REG_DWORD    0x0`. The path may contain spaces.
#[must_use]
pub fn parse_khronos_reg(text: &str) -> Vec<IcdEntry> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim_end_matches('\r');
            let (head, value) = line.rsplit_once("REG_DWORD")?;
            let library = head.trim().to_owned();
            let value = value.trim();
            let n = value
                .strip_prefix("0x")
                .map_or_else(|| value.parse(), |h| u32::from_str_radix(h, 16))
                .ok()?;
            (!library.is_empty()).then_some(IcdEntry {
                source: IcdSource::KhronosRegistry,
                library,
                enabled: n == 0,
            })
        })
        .collect()
}

/// Parse the output of `reg query <display class key> /s /v OpenCLDriverName`:
/// lines of `OpenCLDriverName    REG_SZ    <library path>`.
#[must_use]
pub fn parse_display_reg(text: &str) -> Vec<IcdEntry> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim_end_matches('\r').trim();
            let rest = line.strip_prefix("OpenCLDriverName")?;
            let (_, library) = rest.split_once("REG_SZ")?;
            let library = library.trim().to_owned();
            (!library.is_empty()).then_some(IcdEntry {
                source: IcdSource::DisplayDriverKey,
                library,
                enabled: true,
            })
        })
        .collect()
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

#[cfg(windows)]
fn reg_query(args: &[&str]) -> String {
    // An absent key makes `reg` exit non-zero with nothing on stdout: no entries.
    std::process::Command::new("reg")
        .arg("query")
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

/// Every driver the system configuration registers, in a stable order, with
/// duplicates (the same library from two sources) kept.
#[must_use]
pub fn registered_icds() -> Vec<IcdEntry> {
    #[cfg(windows)]
    {
        let mut v = Vec::new();
        for key in [
            r"HKLM\SOFTWARE\Khronos\OpenCL\Vendors",
            r"HKLM\SOFTWARE\WOW6432Node\Khronos\OpenCL\Vendors",
        ] {
            v.extend(parse_khronos_reg(&reg_query(&[key])));
        }
        v.extend(parse_display_reg(&reg_query(&[
            r"HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}",
            "/s",
            "/v",
            "OpenCLDriverName",
        ])));
        v
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
    fn khronos_key_with_spaces_and_a_disabled_entry() {
        let text = "\r\nHKEY_LOCAL_MACHINE\\SOFTWARE\\Khronos\\OpenCL\\Vendors\r\n    C:\\Program Files (x86)\\Intel\\bin\\intelocl64.dll    REG_DWORD    0x0\r\n    C:\\x\\off.dll    REG_DWORD    0x1\r\n\r\n";
        let v = parse_khronos_reg(text);
        assert_eq!(v.len(), 2, "{v:?}");
        assert_eq!(
            v[0].library,
            r"C:\Program Files (x86)\Intel\bin\intelocl64.dll"
        );
        assert!(v[0].enabled);
        assert!(!v[1].enabled);
        // Control: the key header line is not an entry.
        assert!(v.iter().all(|e| !e.library.starts_with("HKEY")));
    }

    #[test]
    fn display_driver_values() {
        let text = "HKEY_LOCAL_MACHINE\\...\\0000\r\n    OpenCLDriverName    REG_SZ    C:\\DriverStore\\amdocl64.dll\r\n\r\nHKEY_LOCAL_MACHINE\\...\\0001\r\n    OpenCLDriverName    REG_SZ    C:\\DriverStore\\nvopencl64.dll\r\n\r\nEnd of search: 2 match(es) found.\r\n";
        let v = parse_display_reg(text);
        assert_eq!(
            v.iter().map(|e| e.library.as_str()).collect::<Vec<_>>(),
            [
                r"C:\DriverStore\amdocl64.dll",
                r"C:\DriverStore\nvopencl64.dll"
            ]
        );
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

    #[test]
    fn enumerating_this_machine_never_fails() {
        // Whatever this machine registers, listing it must not panic.
        let _ = registered_icds();
    }
}
