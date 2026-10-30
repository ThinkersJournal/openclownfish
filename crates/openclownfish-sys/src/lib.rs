//! Runtime loader for the OpenCL ICD library.
//!
//! Types, constants and function-pointer signatures come from the
//! [`opencl-sys`](https://crates.io/crates/opencl-sys) crate (built without its
//! `static` feature, so nothing links against OpenCL). This crate finds
//! `libOpenCL.so` / `OpenCL.dll` at run time and resolves entry points, each as
//! an `Option`: a symbol the library lacks is `None`, never a load failure.

pub use opencl_sys as types;

use libloading::Library;
use opencl_sys::cl_function_types::{
    clGetDeviceIDs_t, clGetDeviceInfo_t, clGetPlatformIDs_t, clGetPlatformInfo_t,
};

/// Library names tried in order.
#[cfg(windows)]
const CANDIDATES: &[&str] = &["OpenCL.dll"];
#[cfg(target_os = "macos")]
const CANDIDATES: &[&str] = &["/System/Library/Frameworks/OpenCL.framework/OpenCL"];
#[cfg(all(unix, not(target_os = "macos")))]
const CANDIDATES: &[&str] = &["libOpenCL.so.1", "libOpenCL.so"];

/// Failure to load the OpenCL library.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// No candidate library could be opened.
    #[error("no OpenCL library found (tried {tried:?}); last error: {last}")]
    NotFound {
        /// Names tried.
        tried: Vec<String>,
        /// Error from the last attempt.
        last: String,
    },
}

/// A loaded OpenCL library and its resolved entry points.
///
/// The function pointers stay valid as long as this value lives; keep it
/// alive for as long as any object created through it.
pub struct OpenCl {
    _lib: Library,
    /// Name of the library that was opened.
    pub library: String,
    /// `clGetPlatformIDs`
    pub get_platform_ids: clGetPlatformIDs_t,
    /// `clGetPlatformInfo`
    pub get_platform_info: clGetPlatformInfo_t,
    /// `clGetDeviceIDs`
    pub get_device_ids: clGetDeviceIDs_t,
    /// `clGetDeviceInfo`
    pub get_device_info: clGetDeviceInfo_t,
}

/// Resolve `$name` as the `Option<fn>` alias `$alias`; `None` when absent.
///
/// `Option<fn>` has the layout of the bare pointer, so the loader's address
/// is read as the alias directly. Must be called inside `unsafe`, with the
/// alias being the header signature for `$name`.
macro_rules! sym {
    ($lib:expr, $name:literal, $alias:ty) => {
        $lib.get::<$alias>(concat!($name, "\0").as_bytes())
            .ok()
            .and_then(|s| *s)
    };
}

impl OpenCl {
    /// Load the system's OpenCL ICD library.
    pub fn load() -> Result<Self, LoadError> {
        let mut last = String::new();
        for name in CANDIDATES {
            // SAFETY: loading a shared library runs its initialisers; that is
            // the documented cost of using OpenCL at all.
            match unsafe { Library::new(*name) } {
                Ok(lib) => return Ok(Self::from_library(lib, (*name).to_owned())),
                Err(e) => last = e.to_string(),
            }
        }
        Err(LoadError::NotFound {
            tried: CANDIDATES.iter().map(|s| (*s).to_owned()).collect(),
            last,
        })
    }

    fn from_library(lib: Library, library: String) -> Self {
        // SAFETY: each alias is the signature `opencl-sys` derives from the
        // Khronos header for that symbol.
        unsafe {
            Self {
                get_platform_ids: sym!(lib, "clGetPlatformIDs", clGetPlatformIDs_t),
                get_platform_info: sym!(lib, "clGetPlatformInfo", clGetPlatformInfo_t),
                get_device_ids: sym!(lib, "clGetDeviceIDs", clGetDeviceIDs_t),
                get_device_info: sym!(lib, "clGetDeviceInfo", clGetDeviceInfo_t),
                library,
                _lib: lib,
            }
        }
    }
}
