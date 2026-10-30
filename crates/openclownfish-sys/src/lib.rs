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
    clBuildProgram_t, clCreateBuffer_t, clCreateCommandQueue_t,
    clCreateCommandQueueWithProperties_t, clCreateContext_t, clCreateKernel_t,
    clCreateProgramWithBinary_t, clCreateProgramWithSource_t, clEnqueueCopyBuffer_t,
    clEnqueueFillBuffer_t, clEnqueueNDRangeKernel_t, clEnqueueReadBuffer_t, clEnqueueWriteBuffer_t,
    clFinish_t, clFlush_t, clGetContextInfo_t, clGetDeviceIDs_t, clGetDeviceInfo_t,
    clGetEventInfo_t, clGetEventProfilingInfo_t, clGetKernelInfo_t, clGetKernelWorkGroupInfo_t,
    clGetMemObjectInfo_t, clGetPlatformIDs_t, clGetPlatformInfo_t, clGetProgramBuildInfo_t,
    clGetProgramInfo_t, clReleaseCommandQueue_t, clReleaseContext_t, clReleaseEvent_t,
    clReleaseKernel_t, clReleaseMemObject_t, clReleaseProgram_t, clRetainCommandQueue_t,
    clRetainContext_t, clRetainEvent_t, clRetainKernel_t, clRetainMemObject_t, clRetainProgram_t,
    clSetKernelArg_t, clWaitForEvents_t,
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
    /// `clCreateContext`
    pub create_context: clCreateContext_t,
    /// `clRetainContext`
    pub retain_context: clRetainContext_t,
    /// `clReleaseContext`
    pub release_context: clReleaseContext_t,
    /// `clGetContextInfo`
    pub get_context_info: clGetContextInfo_t,
    /// `clCreateCommandQueueWithProperties`
    pub create_command_queue_with_properties: clCreateCommandQueueWithProperties_t,
    /// `clCreateCommandQueue`
    pub create_command_queue: clCreateCommandQueue_t,
    /// `clRetainCommandQueue`
    pub retain_command_queue: clRetainCommandQueue_t,
    /// `clReleaseCommandQueue`
    pub release_command_queue: clReleaseCommandQueue_t,
    /// `clFlush`
    pub flush: clFlush_t,
    /// `clFinish`
    pub finish: clFinish_t,
    /// `clCreateBuffer`
    pub create_buffer: clCreateBuffer_t,
    /// `clRetainMemObject`
    pub retain_mem_object: clRetainMemObject_t,
    /// `clReleaseMemObject`
    pub release_mem_object: clReleaseMemObject_t,
    /// `clGetMemObjectInfo`
    pub get_mem_object_info: clGetMemObjectInfo_t,
    /// `clEnqueueReadBuffer`
    pub enqueue_read_buffer: clEnqueueReadBuffer_t,
    /// `clEnqueueWriteBuffer`
    pub enqueue_write_buffer: clEnqueueWriteBuffer_t,
    /// `clEnqueueCopyBuffer`
    pub enqueue_copy_buffer: clEnqueueCopyBuffer_t,
    /// `clEnqueueFillBuffer`
    pub enqueue_fill_buffer: clEnqueueFillBuffer_t,
    /// `clEnqueueNDRangeKernel`
    pub enqueue_nd_range_kernel: clEnqueueNDRangeKernel_t,
    /// `clCreateProgramWithSource`
    pub create_program_with_source: clCreateProgramWithSource_t,
    /// `clCreateProgramWithBinary`
    pub create_program_with_binary: clCreateProgramWithBinary_t,
    /// `clBuildProgram`
    pub build_program: clBuildProgram_t,
    /// `clRetainProgram`
    pub retain_program: clRetainProgram_t,
    /// `clReleaseProgram`
    pub release_program: clReleaseProgram_t,
    /// `clGetProgramInfo`
    pub get_program_info: clGetProgramInfo_t,
    /// `clGetProgramBuildInfo`
    pub get_program_build_info: clGetProgramBuildInfo_t,
    /// `clCreateKernel`
    pub create_kernel: clCreateKernel_t,
    /// `clRetainKernel`
    pub retain_kernel: clRetainKernel_t,
    /// `clReleaseKernel`
    pub release_kernel: clReleaseKernel_t,
    /// `clSetKernelArg`
    pub set_kernel_arg: clSetKernelArg_t,
    /// `clGetKernelInfo`
    pub get_kernel_info: clGetKernelInfo_t,
    /// `clGetKernelWorkGroupInfo`
    pub get_kernel_work_group_info: clGetKernelWorkGroupInfo_t,
    /// `clWaitForEvents`
    pub wait_for_events: clWaitForEvents_t,
    /// `clGetEventInfo`
    pub get_event_info: clGetEventInfo_t,
    /// `clGetEventProfilingInfo`
    pub get_event_profiling_info: clGetEventProfilingInfo_t,
    /// `clRetainEvent`
    pub retain_event: clRetainEvent_t,
    /// `clReleaseEvent`
    pub release_event: clReleaseEvent_t,
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

    /// Names of entry points this library does not export (for example the
    /// OpenCL 2.0 `clCreateCommandQueueWithProperties` on a 1.2 stack).
    #[must_use]
    pub fn missing_entry_points(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if self.get_platform_ids.is_none() {
            missing.push("clGetPlatformIDs");
        }
        if self.get_platform_info.is_none() {
            missing.push("clGetPlatformInfo");
        }
        if self.get_device_ids.is_none() {
            missing.push("clGetDeviceIDs");
        }
        if self.get_device_info.is_none() {
            missing.push("clGetDeviceInfo");
        }
        if self.create_context.is_none() {
            missing.push("clCreateContext");
        }
        if self.retain_context.is_none() {
            missing.push("clRetainContext");
        }
        if self.release_context.is_none() {
            missing.push("clReleaseContext");
        }
        if self.get_context_info.is_none() {
            missing.push("clGetContextInfo");
        }
        if self.create_command_queue_with_properties.is_none() {
            missing.push("clCreateCommandQueueWithProperties");
        }
        if self.create_command_queue.is_none() {
            missing.push("clCreateCommandQueue");
        }
        if self.retain_command_queue.is_none() {
            missing.push("clRetainCommandQueue");
        }
        if self.release_command_queue.is_none() {
            missing.push("clReleaseCommandQueue");
        }
        if self.flush.is_none() {
            missing.push("clFlush");
        }
        if self.finish.is_none() {
            missing.push("clFinish");
        }
        if self.create_buffer.is_none() {
            missing.push("clCreateBuffer");
        }
        if self.retain_mem_object.is_none() {
            missing.push("clRetainMemObject");
        }
        if self.release_mem_object.is_none() {
            missing.push("clReleaseMemObject");
        }
        if self.get_mem_object_info.is_none() {
            missing.push("clGetMemObjectInfo");
        }
        if self.enqueue_read_buffer.is_none() {
            missing.push("clEnqueueReadBuffer");
        }
        if self.enqueue_write_buffer.is_none() {
            missing.push("clEnqueueWriteBuffer");
        }
        if self.enqueue_copy_buffer.is_none() {
            missing.push("clEnqueueCopyBuffer");
        }
        if self.enqueue_fill_buffer.is_none() {
            missing.push("clEnqueueFillBuffer");
        }
        if self.enqueue_nd_range_kernel.is_none() {
            missing.push("clEnqueueNDRangeKernel");
        }
        if self.create_program_with_source.is_none() {
            missing.push("clCreateProgramWithSource");
        }
        if self.create_program_with_binary.is_none() {
            missing.push("clCreateProgramWithBinary");
        }
        if self.build_program.is_none() {
            missing.push("clBuildProgram");
        }
        if self.retain_program.is_none() {
            missing.push("clRetainProgram");
        }
        if self.release_program.is_none() {
            missing.push("clReleaseProgram");
        }
        if self.get_program_info.is_none() {
            missing.push("clGetProgramInfo");
        }
        if self.get_program_build_info.is_none() {
            missing.push("clGetProgramBuildInfo");
        }
        if self.create_kernel.is_none() {
            missing.push("clCreateKernel");
        }
        if self.retain_kernel.is_none() {
            missing.push("clRetainKernel");
        }
        if self.release_kernel.is_none() {
            missing.push("clReleaseKernel");
        }
        if self.set_kernel_arg.is_none() {
            missing.push("clSetKernelArg");
        }
        if self.get_kernel_info.is_none() {
            missing.push("clGetKernelInfo");
        }
        if self.get_kernel_work_group_info.is_none() {
            missing.push("clGetKernelWorkGroupInfo");
        }
        if self.wait_for_events.is_none() {
            missing.push("clWaitForEvents");
        }
        if self.get_event_info.is_none() {
            missing.push("clGetEventInfo");
        }
        if self.get_event_profiling_info.is_none() {
            missing.push("clGetEventProfilingInfo");
        }
        if self.retain_event.is_none() {
            missing.push("clRetainEvent");
        }
        if self.release_event.is_none() {
            missing.push("clReleaseEvent");
        }
        missing
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
                create_context: sym!(lib, "clCreateContext", clCreateContext_t),
                retain_context: sym!(lib, "clRetainContext", clRetainContext_t),
                release_context: sym!(lib, "clReleaseContext", clReleaseContext_t),
                get_context_info: sym!(lib, "clGetContextInfo", clGetContextInfo_t),
                create_command_queue_with_properties: sym!(
                    lib,
                    "clCreateCommandQueueWithProperties",
                    clCreateCommandQueueWithProperties_t
                ),
                create_command_queue: sym!(lib, "clCreateCommandQueue", clCreateCommandQueue_t),
                retain_command_queue: sym!(lib, "clRetainCommandQueue", clRetainCommandQueue_t),
                release_command_queue: sym!(lib, "clReleaseCommandQueue", clReleaseCommandQueue_t),
                flush: sym!(lib, "clFlush", clFlush_t),
                finish: sym!(lib, "clFinish", clFinish_t),
                create_buffer: sym!(lib, "clCreateBuffer", clCreateBuffer_t),
                retain_mem_object: sym!(lib, "clRetainMemObject", clRetainMemObject_t),
                release_mem_object: sym!(lib, "clReleaseMemObject", clReleaseMemObject_t),
                get_mem_object_info: sym!(lib, "clGetMemObjectInfo", clGetMemObjectInfo_t),
                enqueue_read_buffer: sym!(lib, "clEnqueueReadBuffer", clEnqueueReadBuffer_t),
                enqueue_write_buffer: sym!(lib, "clEnqueueWriteBuffer", clEnqueueWriteBuffer_t),
                enqueue_copy_buffer: sym!(lib, "clEnqueueCopyBuffer", clEnqueueCopyBuffer_t),
                enqueue_fill_buffer: sym!(lib, "clEnqueueFillBuffer", clEnqueueFillBuffer_t),
                enqueue_nd_range_kernel: sym!(
                    lib,
                    "clEnqueueNDRangeKernel",
                    clEnqueueNDRangeKernel_t
                ),
                create_program_with_source: sym!(
                    lib,
                    "clCreateProgramWithSource",
                    clCreateProgramWithSource_t
                ),
                create_program_with_binary: sym!(
                    lib,
                    "clCreateProgramWithBinary",
                    clCreateProgramWithBinary_t
                ),
                build_program: sym!(lib, "clBuildProgram", clBuildProgram_t),
                retain_program: sym!(lib, "clRetainProgram", clRetainProgram_t),
                release_program: sym!(lib, "clReleaseProgram", clReleaseProgram_t),
                get_program_info: sym!(lib, "clGetProgramInfo", clGetProgramInfo_t),
                get_program_build_info: sym!(lib, "clGetProgramBuildInfo", clGetProgramBuildInfo_t),
                create_kernel: sym!(lib, "clCreateKernel", clCreateKernel_t),
                retain_kernel: sym!(lib, "clRetainKernel", clRetainKernel_t),
                release_kernel: sym!(lib, "clReleaseKernel", clReleaseKernel_t),
                set_kernel_arg: sym!(lib, "clSetKernelArg", clSetKernelArg_t),
                get_kernel_info: sym!(lib, "clGetKernelInfo", clGetKernelInfo_t),
                get_kernel_work_group_info: sym!(
                    lib,
                    "clGetKernelWorkGroupInfo",
                    clGetKernelWorkGroupInfo_t
                ),
                wait_for_events: sym!(lib, "clWaitForEvents", clWaitForEvents_t),
                get_event_info: sym!(lib, "clGetEventInfo", clGetEventInfo_t),
                get_event_profiling_info: sym!(
                    lib,
                    "clGetEventProfilingInfo",
                    clGetEventProfilingInfo_t
                ),
                retain_event: sym!(lib, "clRetainEvent", clRetainEvent_t),
                release_event: sym!(lib, "clReleaseEvent", clReleaseEvent_t),
                library,
                _lib: lib,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With a library present, the always-required enumeration entry points
    /// are never reported missing; without one, loading fails cleanly.
    #[test]
    fn enumeration_symbols_resolve_when_loaded() {
        if let Ok(cl) = OpenCl::load() {
            let missing = cl.missing_entry_points();
            assert!(!missing.contains(&"clGetPlatformIDs"), "{missing:?}");
            assert!(!missing.contains(&"clGetDeviceIDs"), "{missing:?}");
        }
    }
}
