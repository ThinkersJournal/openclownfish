//! Programs, kernels, launches and events.

use std::ffi::{CString, c_void};
use std::ptr;
use std::sync::Arc;

use openclownfish_sys::OpenCl;
use openclownfish_sys::types::{
    CL_PROFILING_COMMAND_END, CL_PROFILING_COMMAND_START, CL_PROGRAM_BUILD_LOG, CL_SUCCESS,
    cl_event, cl_int, cl_kernel, cl_program, cl_uint, cl_ulong,
};

use crate::context::created;
use crate::{Buffer, Context, Error, Queue, check, text};

/// An OpenCL program built for its context's device.
pub struct Program {
    rt: Arc<OpenCl>,
    raw: cl_program,
    device: crate::Device,
}

impl Program {
    /// Compile `source` with `options` (`clCreateProgramWithSource` +
    /// `clBuildProgram`) for the context's device.
    ///
    /// A failed build is [`Error::Build`] carrying the driver's log verbatim.
    pub fn build(ctx: &Context, source: &str, options: &str) -> Result<Self, Error> {
        let create = ctx
            .rt
            .create_program_with_source
            .ok_or(Error::MissingSymbol("clCreateProgramWithSource"))?;
        let build = ctx
            .rt
            .build_program
            .ok_or(Error::MissingSymbol("clBuildProgram"))?;
        let opts = CString::new(options).map_err(|_| Error::InvalidString("options"))?;
        let mut strings = [source.as_ptr().cast::<std::ffi::c_char>()];
        let lengths = [source.len()];
        let mut code: cl_int = 0;
        // SAFETY: one string with an explicit length (no NUL needed), valid context.
        let raw = unsafe {
            create(
                ctx.raw,
                1,
                strings.as_mut_ptr(),
                lengths.as_ptr(),
                &mut code,
            )
        };
        let raw = created("clCreateProgramWithSource", raw, code)?;
        let program = Self {
            rt: ctx.rt.clone(),
            raw,
            device: crate::Device {
                rt: ctx.rt.clone(),
                id: ctx.device.id,
            },
        };
        // SAFETY: valid program and device list of one, NUL-terminated options, no callback.
        let code = unsafe {
            build(
                program.raw,
                1,
                &program.device.id,
                opts.as_ptr(),
                None,
                ptr::null_mut(),
            )
        };
        if code != CL_SUCCESS {
            return Err(Error::Build {
                code,
                log: program
                    .build_log()
                    .unwrap_or_else(|e| format!("<no log: {e}>")),
            });
        }
        Ok(program)
    }

    /// The driver's build log for this program (may hold warnings on success).
    pub fn build_log(&self) -> Result<String, Error> {
        let f = self
            .rt
            .get_program_build_info
            .ok_or(Error::MissingSymbol("clGetProgramBuildInfo"))?;
        let mut size = 0usize;
        // SAFETY: size query with a null buffer.
        check("clGetProgramBuildInfo", unsafe {
            f(
                self.raw,
                self.device.id,
                CL_PROGRAM_BUILD_LOG,
                0,
                ptr::null_mut(),
                &mut size,
            )
        })?;
        let mut buf = vec![0u8; size];
        // SAFETY: `buf` is `size` bytes.
        check("clGetProgramBuildInfo", unsafe {
            f(
                self.raw,
                self.device.id,
                CL_PROGRAM_BUILD_LOG,
                size,
                buf.as_mut_ptr().cast::<c_void>(),
                ptr::null_mut(),
            )
        })?;
        Ok(text(&buf))
    }

    /// Create the kernel called `name` (`clCreateKernel`).
    pub fn kernel(&self, name: &str) -> Result<Kernel, Error> {
        let f = self
            .rt
            .create_kernel
            .ok_or(Error::MissingSymbol("clCreateKernel"))?;
        let cname = CString::new(name).map_err(|_| Error::InvalidString("kernel name"))?;
        let mut code: cl_int = 0;
        // SAFETY: valid built program, NUL-terminated name, valid out-pointer.
        let raw = unsafe { f(self.raw, cname.as_ptr(), &mut code) };
        let raw = created("clCreateKernel", raw, code)?;
        Ok(Kernel {
            rt: self.rt.clone(),
            raw,
        })
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        if let Some(f) = self.rt.release_program {
            // SAFETY: `raw` is a program we created and have not released.
            unsafe { f(self.raw) };
        }
    }
}

/// A kernel with its arguments.
pub struct Kernel {
    rt: Arc<OpenCl>,
    raw: cl_kernel,
}

macro_rules! scalar_args {
    ($($name:ident: $t:ty),*) => {$(
        /// Set argument `index` to this scalar.
        pub fn $name(&mut self, index: u32, value: $t) -> Result<(), Error> {
            self.set_arg_bytes(index, &value.to_ne_bytes())
        }
    )*};
}

impl Kernel {
    /// Set argument `index` to `buffer` (`clSetKernelArg`). Keep the buffer
    /// alive until work using it has finished.
    pub fn set_arg_buffer(&mut self, index: u32, buffer: &Buffer) -> Result<(), Error> {
        let f = self
            .rt
            .set_kernel_arg
            .ok_or(Error::MissingSymbol("clSetKernelArg"))?;
        // SAFETY: the value is the address of a live `cl_mem`, sized as one.
        check("clSetKernelArg", unsafe {
            f(
                self.raw,
                index,
                size_of_val(&buffer.raw),
                ptr::from_ref(&buffer.raw).cast::<c_void>(),
            )
        })
    }

    /// Set argument `index` to raw bytes; the driver rejects a wrong size.
    pub fn set_arg_bytes(&mut self, index: u32, bytes: &[u8]) -> Result<(), Error> {
        let f = self
            .rt
            .set_kernel_arg
            .ok_or(Error::MissingSymbol("clSetKernelArg"))?;
        // SAFETY: `bytes` is a live slice of the stated length.
        check("clSetKernelArg", unsafe {
            f(
                self.raw,
                index,
                bytes.len(),
                bytes.as_ptr().cast::<c_void>(),
            )
        })
    }

    scalar_args!(set_arg_i32: i32, set_arg_u32: u32, set_arg_i64: i64, set_arg_u64: u64, set_arg_f32: f32, set_arg_f64: f64);
}

impl Drop for Kernel {
    fn drop(&mut self) {
        if let Some(f) = self.rt.release_kernel {
            // SAFETY: `raw` is a kernel we created and have not released.
            unsafe { f(self.raw) };
        }
    }
}

/// A queued command's completion handle.
pub struct Event {
    rt: Arc<OpenCl>,
    raw: cl_event,
}

impl Event {
    /// Block until the command completes (`clWaitForEvents`).
    pub fn wait(&self) -> Result<(), Error> {
        let f = self
            .rt
            .wait_for_events
            .ok_or(Error::MissingSymbol("clWaitForEvents"))?;
        // SAFETY: one valid event.
        check("clWaitForEvents", unsafe { f(1, &self.raw) })
    }

    fn profiling(&self, param: cl_uint) -> Result<cl_ulong, Error> {
        let f = self
            .rt
            .get_event_profiling_info
            .ok_or(Error::MissingSymbol("clGetEventProfilingInfo"))?;
        let mut v: cl_ulong = 0;
        // SAFETY: `v` is exactly the 8 bytes requested.
        check("clGetEventProfilingInfo", unsafe {
            f(
                self.raw,
                param,
                size_of::<cl_ulong>(),
                ptr::from_mut(&mut v).cast::<c_void>(),
                ptr::null_mut(),
            )
        })?;
        Ok(v)
    }

    /// Device-side run time in nanoseconds (`END - START`). Waits first.
    /// Needs a queue from [`Context::create_profiling_queue`]; otherwise the
    /// driver's `CL_PROFILING_INFO_NOT_AVAILABLE` surfaces as [`Error::Cl`].
    pub fn duration_ns(&self) -> Result<u64, Error> {
        self.wait()?;
        let start = self.profiling(CL_PROFILING_COMMAND_START)?;
        let end = self.profiling(CL_PROFILING_COMMAND_END)?;
        Ok(end.saturating_sub(start))
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Some(f) = self.rt.release_event {
            // SAFETY: `raw` is an event the driver gave us and we have not released.
            unsafe { f(self.raw) };
        }
    }
}

impl Queue {
    /// Enqueue `kernel` over `global` work-items (`clEnqueueNDRangeKernel`),
    /// optionally with a `local` work-group size of the same rank. Returns an
    /// [`Event`]; use it or [`Queue::finish`] before reading results.
    ///
    /// # Safety
    ///
    /// OpenCL C is unchecked: the kernel must not read or write outside the
    /// buffers bound to it, and every buffer bound to it must stay alive until
    /// the work completes. This crate cannot verify either.
    pub unsafe fn launch(
        &self,
        kernel: &Kernel,
        global: &[usize],
        local: Option<&[usize]>,
    ) -> Result<Event, Error> {
        let f = self
            .rt
            .enqueue_nd_range_kernel
            .ok_or(Error::MissingSymbol("clEnqueueNDRangeKernel"))?;
        if global.is_empty() || local.is_some_and(|l| l.len() != global.len()) {
            return Err(Error::Cl {
                call: "clEnqueueNDRangeKernel",
                code: -53, // CL_INVALID_WORK_DIMENSION, caught before the call
            });
        }
        let mut raw: cl_event = ptr::null_mut();
        // SAFETY: valid queue and kernel; `global`/`local` have `work_dim` entries
        // (checked above); the caller upholds the contract in `# Safety`.
        check("clEnqueueNDRangeKernel", unsafe {
            f(
                self.raw,
                kernel.raw,
                global.len() as cl_uint,
                ptr::null(),
                global.as_ptr(),
                local.map_or(ptr::null(), <[usize]>::as_ptr),
                0,
                ptr::null(),
                &mut raw,
            )
        })?;
        Ok(Event {
            rt: self.rt.clone(),
            raw,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{Context, Error, Program, Runtime};

    const ADD: &str = "__kernel void add(__global const float* a, __global const float* b, \
                       __global float* c) { size_t i = get_global_id(0); c[i] = a[i] + b[i]; }";

    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_ne_bytes()).collect()
    }

    /// Build, launch and read back on every device that can; a stack that
    /// cannot is skipped and counted, since this is a measurement tool.
    #[test]
    fn vector_add_on_every_device() {
        let Ok(rt) = Runtime::load() else { return };
        let Ok(platforms) = rt.platforms() else {
            return;
        };
        let mut ran = 0;
        for p in platforms {
            let Ok(devices) = p.devices() else { continue };
            for d in devices {
                let Ok(ctx) = Context::new(&d) else { continue };
                let Ok(queue) = ctx.create_profiling_queue() else {
                    continue;
                };
                let Ok(prog) = Program::build(&ctx, ADD, "") else {
                    continue;
                };
                let mut k = prog.kernel("add").unwrap();
                let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
                let b: Vec<f32> = (0..64).map(|i| 2.0 * i as f32).collect();
                let (ba, bb, bc) = (
                    ctx.create_buffer(256).unwrap(),
                    ctx.create_buffer(256).unwrap(),
                    ctx.create_buffer(256).unwrap(),
                );
                queue.write_buffer(&ba, 0, &f32s(&a)).unwrap();
                queue.write_buffer(&bb, 0, &f32s(&b)).unwrap();
                k.set_arg_buffer(0, &ba).unwrap();
                k.set_arg_buffer(1, &bb).unwrap();
                k.set_arg_buffer(2, &bc).unwrap();
                // SAFETY: the kernel touches indices 0..64 of three 64-float buffers.
                let ev = unsafe { queue.launch(&k, &[64], None) }.unwrap();
                ev.wait().unwrap();
                let mut out = vec![0u8; 256];
                queue.read_buffer(&bc, 0, &mut out).unwrap();
                let got: Vec<f32> = out
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_ne_bytes(*c))
                    .collect();
                let want: Vec<f32> = (0..64).map(|i| 3.0 * i as f32).collect();
                assert_eq!(got, want, "device {:?}", d.name());
                // Profiling is best effort per stack; the call must not panic.
                let _ = ev.duration_ns();
                ran += 1;
            }
        }
        eprintln!("vector add ran on {ran} device(s)");
    }

    #[test]
    fn bad_source_surfaces_the_build_log() {
        let Ok(rt) = Runtime::load() else { return };
        let Ok(platforms) = rt.platforms() else {
            return;
        };
        for p in platforms {
            let Ok(devices) = p.devices() else { continue };
            for d in devices {
                let Ok(ctx) = Context::new(&d) else { continue };
                match Program::build(&ctx, "__kernel void k( {", "") {
                    Err(Error::Build { code, log }) => {
                        eprintln!("{:?}: status {code}, log {} bytes", d.name(), log.len());
                    }
                    Err(_) => {}
                    Ok(_) => panic!("garbage compiled on {:?}", d.name()),
                }
            }
        }
    }
}
