//! Safe Rust interface to OpenCL, loaded at run time.
//!
//! Start with [`Runtime::load`], then [`Runtime::platforms`] and
//! [`Platform::devices`]. Nothing here needs an OpenCL install to build.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use openclownfish_sys::OpenCl;
use openclownfish_sys::types::{
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

    /// Name of the library that was opened.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_stops_at_nul() {
        assert_eq!(text(b"abc\0def"), "abc");
        assert_eq!(text(b"abc"), "abc");
        assert_eq!(text(b""), "");
    }

    /// Passes with or without an OpenCL install: both outcomes are handled.
    #[test]
    fn load_and_enumerate_never_panics() {
        match Runtime::load() {
            Ok(rt) => {
                for p in rt.platforms().expect("platform enumeration") {
                    for d in p.devices().expect("device enumeration") {
                        d.name().expect("device name");
                    }
                }
            }
            Err(Error::Load(_)) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
}
