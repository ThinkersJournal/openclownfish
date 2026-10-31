//! Safe Rust interface to OpenCL, loaded at run time.
//!
//! Start with [`Runtime::load`], then [`Runtime::platforms`] and
//! [`Platform::devices`]. Nothing here needs an OpenCL install to build.

mod context;
mod program;

pub use context::{Buffer, Context, Queue};
pub use openclownfish_sys::icd;
pub use program::{Event, Kernel, Program};

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use openclownfish_sys::OpenCl;
use openclownfish_sys::types::{
    CL_DEVICE_EXTENSIONS, CL_DEVICE_GLOBAL_MEM_SIZE, CL_DEVICE_LOCAL_MEM_SIZE,
    CL_DEVICE_MAX_COMPUTE_UNITS, CL_DEVICE_MAX_MEM_ALLOC_SIZE, CL_DEVICE_MAX_WORK_GROUP_SIZE,
    CL_DEVICE_NAME, CL_DEVICE_NOT_FOUND, CL_DEVICE_OPENCL_C_VERSION, CL_DEVICE_TYPE,
    CL_DEVICE_TYPE_ALL, CL_DEVICE_VENDOR, CL_DEVICE_VERSION, CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
    CL_PLATFORM_VERSION, CL_SUCCESS, cl_device_id, cl_int, cl_platform_id, cl_uint,
};

/// `CL_PLATFORM_NOT_FOUND_KHR`: the ICD loader is present but has no ICDs.
const PLATFORM_NOT_FOUND: cl_int = -1001;

/// Errors from this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The OpenCL library could not be loaded.
    #[error(transparent)]
    Load(#[from] openclownfish_sys::LoadError),
    /// The loaded library lacks an entry point.
    #[error("the OpenCL library does not export {0}")]
    MissingSymbol(&'static str),
    /// An OpenCL call returned a non-success status.
    #[error("{call} failed with OpenCL status {code}")]
    Cl {
        /// Name of the call.
        call: &'static str,
        /// Raw `cl_int` status.
        code: cl_int,
    },
    /// `clBuildProgram` failed; `log` is the driver's build log for the device.
    #[error(
        "clBuildProgram failed with OpenCL status {code}:
{log}"
    )]
    Build {
        /// Raw `cl_int` status.
        code: cl_int,
        /// Build log, verbatim.
        log: String,
    },
    /// A numeric property came back with a width other than 4 or 8 bytes.
    #[error("{call} returned a {size}-byte number, expected 4 or 8")]
    UnexpectedSize {
        /// Name of the call.
        call: &'static str,
        /// Bytes the driver returned.
        size: usize,
    },
    /// A string argument contains an interior NUL byte.
    #[error("{0} contains a NUL byte")]
    InvalidString(&'static str),
    /// A buffer range does not fit the buffer; no driver call was made.
    #[error("range {offset}..{offset}+{len} does not fit a buffer of {size} bytes")]
    OutOfBounds {
        /// Requested byte offset.
        offset: usize,
        /// Requested byte length.
        len: usize,
        /// Buffer size in bytes.
        size: usize,
    },
}

/// A loaded OpenCL runtime.
#[derive(Clone)]
pub struct Runtime(Arc<OpenCl>);

/// An OpenCL platform (one vendor's driver stack).
pub struct Platform {
    rt: Arc<OpenCl>,
    id: cl_platform_id,
}

/// Kind of device, from `CL_DEVICE_TYPE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// CPU.
    Cpu,
    /// GPU.
    Gpu,
    /// Accelerator.
    Accelerator,
    /// Any other bit pattern, raw.
    Other(u64),
}

/// An OpenCL device.
pub struct Device {
    rt: Arc<OpenCl>,
    id: cl_device_id,
}

fn check(call: &'static str, code: cl_int) -> Result<(), Error> {
    if code == CL_SUCCESS {
        Ok(())
    } else {
        Err(Error::Cl { call, code })
    }
}

fn text(raw: &[u8]) -> String {
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end]).into_owned()
}

impl Runtime {
    /// Load the system's OpenCL library.
    pub fn load() -> Result<Self, Error> {
        Ok(Self(Arc::new(OpenCl::load()?)))
    }

    /// Entry points the opened library does not export.
    #[must_use]
    pub fn missing_entry_points(&self) -> Vec<&'static str> {
        self.0.missing_entry_points()
    }

    /// Name of the OpenCL library that was opened.
    pub fn library(&self) -> &str {
        &self.0.library
    }

    /// All platforms the ICD loader exposes. An empty list is not an error.
    pub fn platforms(&self) -> Result<Vec<Platform>, Error> {
        let f = self
            .0
            .get_platform_ids
            .ok_or(Error::MissingSymbol("clGetPlatformIDs"))?;
        let mut n: cl_uint = 0;
        // SAFETY: valid out-pointer, zero entries requested.
        let code = unsafe { f(0, ptr::null_mut(), &mut n) };
        if code == PLATFORM_NOT_FOUND {
            return Ok(Vec::new());
        }
        check("clGetPlatformIDs", code)?;
        let mut ids: Vec<cl_platform_id> = vec![ptr::null_mut(); n as usize];
        // SAFETY: `ids` has room for `n` entries.
        check("clGetPlatformIDs", unsafe {
            f(n, ids.as_mut_ptr(), ptr::null_mut())
        })?;
        Ok(ids
            .into_iter()
            .map(|id| Platform {
                rt: self.0.clone(),
                id,
            })
            .collect())
    }
}

impl Platform {
    fn info(&self, param: cl_uint) -> Result<String, Error> {
        let f = self
            .rt
            .get_platform_info
            .ok_or(Error::MissingSymbol("clGetPlatformInfo"))?;
        let mut size = 0usize;
        // SAFETY: size query with null buffer.
        check("clGetPlatformInfo", unsafe {
            f(self.id, param, 0, ptr::null_mut(), &mut size)
        })?;
        let mut buf = vec![0u8; size];
        // SAFETY: `buf` is `size` bytes.
        check("clGetPlatformInfo", unsafe {
            f(
                self.id,
                param,
                size,
                buf.as_mut_ptr().cast::<c_void>(),
                ptr::null_mut(),
            )
        })?;
        Ok(text(&buf))
    }

    /// `CL_PLATFORM_NAME`.
    pub fn name(&self) -> Result<String, Error> {
        self.info(CL_PLATFORM_NAME)
    }

    /// `CL_PLATFORM_VENDOR`.
    pub fn vendor(&self) -> Result<String, Error> {
        self.info(CL_PLATFORM_VENDOR)
    }

    /// `CL_PLATFORM_VERSION`.
    pub fn version(&self) -> Result<String, Error> {
        self.info(CL_PLATFORM_VERSION)
    }

    /// All devices on this platform. An empty list is not an error.
    pub fn devices(&self) -> Result<Vec<Device>, Error> {
        let f = self
            .rt
            .get_device_ids
            .ok_or(Error::MissingSymbol("clGetDeviceIDs"))?;
        let mut n: cl_uint = 0;
        // SAFETY: count query with null buffer.
        let code = unsafe { f(self.id, CL_DEVICE_TYPE_ALL, 0, ptr::null_mut(), &mut n) };
        if code == CL_DEVICE_NOT_FOUND {
            return Ok(Vec::new());
        }
        check("clGetDeviceIDs", code)?;
        let mut ids: Vec<cl_device_id> = vec![ptr::null_mut(); n as usize];
        // SAFETY: `ids` has room for `n` entries.
        check("clGetDeviceIDs", unsafe {
            f(
                self.id,
                CL_DEVICE_TYPE_ALL,
                n,
                ids.as_mut_ptr(),
                ptr::null_mut(),
            )
        })?;
        Ok(ids
            .into_iter()
            .map(|id| Device {
                rt: self.rt.clone(),
                id,
            })
            .collect())
    }
}

impl Device {
    fn raw(&self, param: cl_uint) -> Result<Vec<u8>, Error> {
        let f = self
            .rt
            .get_device_info
            .ok_or(Error::MissingSymbol("clGetDeviceInfo"))?;
        let mut size = 0usize;
        // SAFETY: size query with null buffer.
        check("clGetDeviceInfo", unsafe {
            f(self.id, param, 0, ptr::null_mut(), &mut size)
        })?;
        let mut buf = vec![0u8; size];
        // SAFETY: `buf` is `size` bytes.
        check("clGetDeviceInfo", unsafe {
            f(
                self.id,
                param,
                size,
                buf.as_mut_ptr().cast::<c_void>(),
                ptr::null_mut(),
            )
        })?;
        Ok(buf)
    }

    fn info(&self, param: cl_uint) -> Result<String, Error> {
        Ok(text(&self.raw(param)?))
    }

    /// `CL_DEVICE_NAME`.
    pub fn name(&self) -> Result<String, Error> {
        self.info(CL_DEVICE_NAME)
    }

    /// `CL_DEVICE_VENDOR`.
    pub fn vendor(&self) -> Result<String, Error> {
        self.info(CL_DEVICE_VENDOR)
    }

    /// `CL_DEVICE_VERSION`.
    pub fn version(&self) -> Result<String, Error> {
        self.info(CL_DEVICE_VERSION)
    }

    /// `CL_DEVICE_OPENCL_C_VERSION`.
    pub fn opencl_c_version(&self) -> Result<String, Error> {
        self.info(CL_DEVICE_OPENCL_C_VERSION)
    }

    /// `CL_DEVICE_TYPE`.
    pub fn kind(&self) -> Result<DeviceKind, Error> {
        let raw = self.raw(CL_DEVICE_TYPE)?;
        let mut bytes = [0u8; 8];
        let n = raw.len().min(8);
        bytes[..n].copy_from_slice(&raw[..n]);
        Ok(match u64::from_le_bytes(bytes) {
            2 => DeviceKind::Cpu,
            4 => DeviceKind::Gpu,
            8 => DeviceKind::Accelerator,
            other => DeviceKind::Other(other),
        })
    }

    /// A numeric device property, whose width (4 or 8 bytes) the driver sets.
    fn number(&self, param: cl_uint) -> Result<u64, Error> {
        let raw = self.raw(param)?;
        Ok(match raw.len() {
            4 => u64::from(u32::from_ne_bytes(raw[..4].try_into().expect("4 bytes"))),
            8 => u64::from_ne_bytes(raw[..8].try_into().expect("8 bytes")),
            size => {
                return Err(Error::UnexpectedSize {
                    call: "clGetDeviceInfo",
                    size,
                });
            }
        })
    }

    /// `CL_DEVICE_EXTENSIONS`, one name per entry.
    pub fn extensions(&self) -> Result<Vec<String>, Error> {
        Ok(self
            .info(CL_DEVICE_EXTENSIONS)?
            .split_whitespace()
            .map(str::to_owned)
            .collect())
    }

    /// Whether the device lists extension `name` (for example `cl_khr_fp64`).
    pub fn has_extension(&self, name: &str) -> Result<bool, Error> {
        Ok(self.extensions()?.iter().any(|e| e == name))
    }

    /// `CL_DEVICE_MAX_COMPUTE_UNITS`.
    pub fn compute_units(&self) -> Result<u32, Error> {
        Ok(u32::try_from(self.number(CL_DEVICE_MAX_COMPUTE_UNITS)?).unwrap_or(u32::MAX))
    }

    /// `CL_DEVICE_MAX_WORK_GROUP_SIZE`, the most work-items per work-group.
    pub fn max_work_group_size(&self) -> Result<u64, Error> {
        self.number(CL_DEVICE_MAX_WORK_GROUP_SIZE)
    }

    /// `CL_DEVICE_GLOBAL_MEM_SIZE` in bytes.
    pub fn global_mem_bytes(&self) -> Result<u64, Error> {
        self.number(CL_DEVICE_GLOBAL_MEM_SIZE)
    }

    /// `CL_DEVICE_MAX_MEM_ALLOC_SIZE` in bytes: the largest single buffer.
    pub fn max_alloc_bytes(&self) -> Result<u64, Error> {
        self.number(CL_DEVICE_MAX_MEM_ALLOC_SIZE)
    }

    /// `CL_DEVICE_LOCAL_MEM_SIZE` in bytes.
    pub fn local_mem_bytes(&self) -> Result<u64, Error> {
        self.number(CL_DEVICE_LOCAL_MEM_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every device that answers reports limits that make sense together. A
    /// property the driver refuses is reported by the probe, not hidden here.
    #[test]
    fn device_limits_are_consistent() {
        let Ok(rt) = Runtime::load() else { return };
        let Ok(platforms) = rt.platforms() else {
            return;
        };
        let mut checked = 0;
        for p in platforms {
            let Ok(devices) = p.devices() else { continue };
            for d in devices {
                let (Ok(units), Ok(wg), Ok(global), Ok(alloc), Ok(exts)) = (
                    d.compute_units(),
                    d.max_work_group_size(),
                    d.global_mem_bytes(),
                    d.max_alloc_bytes(),
                    d.extensions(),
                ) else {
                    continue;
                };
                assert!(units > 0 && wg > 0, "{:?}", d.name());
                assert!(
                    alloc <= global,
                    "{:?}: alloc {alloc} > global {global}",
                    d.name()
                );
                assert!(exts.iter().all(|e| !e.contains(char::is_whitespace)));
                assert_eq!(
                    d.has_extension("cl_khr_fp64").unwrap(),
                    exts.iter().any(|e| e == "cl_khr_fp64")
                );
                checked += 1;
            }
        }
        eprintln!("device limits checked on {checked} device(s)");
    }

    #[test]
    fn text_stops_at_nul() {
        assert_eq!(text(b"abc\0def"), "abc");
        assert_eq!(text(b"abc"), "abc");
        assert_eq!(text(b""), "");
    }

    /// Write then read back through every device that works; a stack that
    /// cannot make a context is skipped, not failed (this is a measurement
    /// tool: the probe reports such stacks).
    #[test]
    fn buffer_round_trip_and_bounds() {
        let Ok(rt) = Runtime::load() else { return };
        let Ok(platforms) = rt.platforms() else {
            return;
        };
        let mut ran = 0;
        for p in platforms {
            let Ok(devices) = p.devices() else { continue };
            for d in devices {
                let Ok(ctx) = Context::new(&d) else { continue };
                let Ok(queue) = ctx.create_queue() else {
                    continue;
                };
                let Ok(buf) = ctx.create_buffer(16) else {
                    continue;
                };
                let data: Vec<u8> = (0..16).collect();
                queue.write_buffer(&buf, 0, &data).unwrap();
                let mut out = [0u8; 16];
                queue.read_buffer(&buf, 0, &mut out).unwrap();
                queue.finish().unwrap();
                assert_eq!(&out[..], &data[..], "device {:?}", d.name());
                assert!(matches!(
                    queue.write_buffer(&buf, 8, &data),
                    Err(Error::OutOfBounds { .. })
                ));
                ran += 1;
            }
        }
        eprintln!("buffer round trip ran on {ran} device(s)");
    }

    /// Passes with or without an OpenCL install, and when a vendor stack
    /// errors on enumeration (the macOS CI runner returns status -30 from
    /// `clGetDeviceIDs`): driver errors are printed, not asserted away.
    #[test]
    fn load_and_enumerate_never_panics() {
        let rt = match Runtime::load() {
            Ok(rt) => rt,
            Err(Error::Load(_)) => return,
            Err(e) => panic!("unexpected error: {e}"),
        };
        let platforms = match rt.platforms() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("platform enumeration: {e}");
                return;
            }
        };
        for p in platforms {
            match p.devices() {
                Ok(devices) => {
                    for d in devices {
                        if let Err(e) = d.name() {
                            eprintln!("device name: {e}");
                        }
                    }
                }
                Err(e) => eprintln!("device enumeration: {e}"),
            }
        }
    }
}
