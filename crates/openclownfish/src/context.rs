//! Contexts, command queues and buffers.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use openclownfish_sys::OpenCl;
use openclownfish_sys::types::{
    CL_MEM_READ_WRITE, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_PROPERTIES, CL_TRUE, cl_command_queue,
    cl_context, cl_int, cl_mem, cl_queue_properties,
};

use crate::{Device, Error, check};

/// Turn a creation call's `errcode_ret` and returned handle into a result.
pub(crate) fn created<T>(
    call: &'static str,
    handle: *mut T,
    code: cl_int,
) -> Result<*mut T, Error> {
    check(call, code)?;
    if handle.is_null() {
        // A driver that returns NULL with CL_SUCCESS broke its contract; say so.
        return Err(Error::Cl { call, code: 0 });
    }
    Ok(handle)
}

/// An OpenCL context over one device.
pub struct Context {
    pub(crate) rt: Arc<OpenCl>,
    pub(crate) raw: cl_context,
    pub(crate) device: Device,
}

impl Context {
    /// Create a context for `device` (`clCreateContext`).
    pub fn new(device: &Device) -> Result<Self, Error> {
        let f = device
            .rt
            .create_context
            .ok_or(Error::MissingSymbol("clCreateContext"))?;
        let mut code: cl_int = 0;
        // SAFETY: one valid device id, no properties, no callback, valid out-pointer.
        let raw = unsafe { f(ptr::null(), 1, &device.id, None, ptr::null_mut(), &mut code) };
        let raw = created("clCreateContext", raw, code)?;
        Ok(Self {
            rt: device.rt.clone(),
            raw,
            device: Device {
                rt: device.rt.clone(),
                id: device.id,
            },
        })
    }

    /// Create a buffer of `len` bytes (`CL_MEM_READ_WRITE`, uninitialised).
    pub fn create_buffer(&self, len: usize) -> Result<Buffer, Error> {
        let f = self
            .rt
            .create_buffer
            .ok_or(Error::MissingSymbol("clCreateBuffer"))?;
        let mut code: cl_int = 0;
        // SAFETY: valid context, no host pointer, valid out-pointer.
        let raw = unsafe { f(self.raw, CL_MEM_READ_WRITE, len, ptr::null_mut(), &mut code) };
        let raw = created("clCreateBuffer", raw, code)?;
        Ok(Buffer {
            rt: self.rt.clone(),
            raw,
            len,
        })
    }

    /// Create an in-order command queue on the context's device.
    ///
    /// Uses `clCreateCommandQueueWithProperties` when the library has it
    /// (OpenCL 2.0+), else the deprecated `clCreateCommandQueue` so 1.x stacks
    /// still work.
    pub fn create_queue(&self) -> Result<Queue, Error> {
        self.queue(false)
    }

    /// Like [`Context::create_queue`] with `CL_QUEUE_PROFILING_ENABLE`, so
    /// [`Event::duration_ns`] works for work enqueued on it.
    pub fn create_profiling_queue(&self) -> Result<Queue, Error> {
        self.queue(true)
    }

    fn queue(&self, profiling: bool) -> Result<Queue, Error> {
        let mut code: cl_int = 0;
        let raw = if let Some(f) = self.rt.create_command_queue_with_properties {
            let props: [cl_queue_properties; 3] = [
                cl_queue_properties::from(CL_QUEUE_PROPERTIES),
                if profiling {
                    CL_QUEUE_PROFILING_ENABLE
                } else {
                    0
                },
                0,
            ];
            // SAFETY: valid context and device, zero-terminated property list, valid out-pointer.
            unsafe { f(self.raw, self.device.id, props.as_ptr(), &mut code) }
        } else if let Some(f) = self.rt.create_command_queue {
            let flags = if profiling {
                CL_QUEUE_PROFILING_ENABLE
            } else {
                0
            };
            // SAFETY: valid context and device, property bits, valid out-pointer.
            unsafe { f(self.raw, self.device.id, flags, &mut code) }
        } else {
            return Err(Error::MissingSymbol("clCreateCommandQueueWithProperties"));
        };
        let raw = created("clCreateCommandQueue", raw, code)?;
        Ok(Queue {
            rt: self.rt.clone(),
            raw,
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Some(f) = self.rt.release_context {
            // SAFETY: `raw` is a context we created and have not released.
            unsafe { f(self.raw) };
        }
    }
}

/// A device memory buffer. Releasing it is safe while queued work still uses
/// it: OpenCL keeps the object alive until that work completes.
pub struct Buffer {
    rt: Arc<OpenCl>,
    pub(crate) raw: cl_mem,
    len: usize,
}

impl Buffer {
    /// Size in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is zero bytes long.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(f) = self.rt.release_mem_object {
            // SAFETY: `raw` is a buffer we created and have not released.
            unsafe { f(self.raw) };
        }
    }
}

/// An in-order command queue.
pub struct Queue {
    pub(crate) rt: Arc<OpenCl>,
    pub(crate) raw: cl_command_queue,
}

/// `[offset, offset + len)` must lie inside a buffer of `size` bytes.
fn in_bounds(size: usize, offset: usize, len: usize) -> bool {
    offset.checked_add(len).is_some_and(|end| end <= size)
}

impl Queue {
    /// Copy `data` into `buffer` at byte `offset`; returns when the copy is done.
    ///
    /// Fails with [`Error::OutOfBounds`] before calling the driver if the range
    /// does not fit the buffer.
    pub fn write_buffer(&self, buffer: &Buffer, offset: usize, data: &[u8]) -> Result<(), Error> {
        if !in_bounds(buffer.len, offset, data.len()) {
            return Err(Error::OutOfBounds {
                offset,
                len: data.len(),
                size: buffer.len,
            });
        }
        let f = self
            .rt
            .enqueue_write_buffer
            .ok_or(Error::MissingSymbol("clEnqueueWriteBuffer"))?;
        // SAFETY: blocking write of `data.len()` bytes from a live slice into a
        // range checked above; no wait list, no event.
        check("clEnqueueWriteBuffer", unsafe {
            f(
                self.raw,
                buffer.raw,
                CL_TRUE,
                offset,
                data.len(),
                data.as_ptr().cast::<c_void>(),
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        })
    }

    /// Copy `out.len()` bytes from `buffer` at byte `offset` into `out`;
    /// returns when the copy is done.
    pub fn read_buffer(&self, buffer: &Buffer, offset: usize, out: &mut [u8]) -> Result<(), Error> {
        if !in_bounds(buffer.len, offset, out.len()) {
            return Err(Error::OutOfBounds {
                offset,
                len: out.len(),
                size: buffer.len,
            });
        }
        let f = self
            .rt
            .enqueue_read_buffer
            .ok_or(Error::MissingSymbol("clEnqueueReadBuffer"))?;
        // SAFETY: blocking read of `out.len()` bytes into a live mutable slice
        // from a range checked above; no wait list, no event.
        check("clEnqueueReadBuffer", unsafe {
            f(
                self.raw,
                buffer.raw,
                CL_TRUE,
                offset,
                out.len(),
                out.as_mut_ptr().cast::<c_void>(),
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        })
    }

    /// Block until every queued command has finished (`clFinish`).
    pub fn finish(&self) -> Result<(), Error> {
        let f = self.rt.finish.ok_or(Error::MissingSymbol("clFinish"))?;
        // SAFETY: valid queue.
        check("clFinish", unsafe { f(self.raw) })
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        if let Some(f) = self.rt.release_command_queue {
            // SAFETY: `raw` is a queue we created and have not released.
            unsafe { f(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::in_bounds;

    #[test]
    fn bounds() {
        assert!(in_bounds(8, 0, 8));
        assert!(in_bounds(8, 8, 0));
        assert!(!in_bounds(8, 4, 5));
        assert!(!in_bounds(8, usize::MAX, 2), "overflow must not wrap");
    }
}
