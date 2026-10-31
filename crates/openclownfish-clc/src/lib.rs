//! OpenCL C emitter for [Unpopped](https://crates.io/crates/unpopped).
//!
//! [`OpenClC`] implements `unpopped::backend::Backend`. It is narrow on
//! purpose: contiguous single-output `f32` (and, when the device has
//! `cl_khr_fp64`, `f64`) elementwise plans whose body uses inputs, finite
//! exactly-representable constants, `+ - * /` and the unary operations negate,
//! abs, square, reciprocal, sqrt, rsqrt, exp and log, `Max`/`Min`
//! (NaN-propagating) and `FmaxIeee`/`FminIeee` (NaN-suppressing), `Pow`, the six
//! comparisons (1.0 or 0.0) and `Select`. Everything else is a typed refusal,
//! never an approximation.
//!
//! Whether `f64` may be emitted is the caller's to say, from the target device's
//! extension list: [`OpenClC::for_extensions`]. A plan carries no device, so the
//! backend does.

#[cfg(feature = "parse")]
mod parse;
#[cfg(feature = "parse")]
pub use parse::{OPENCL_C, parse_opencl_c};

use unpopped::backend::{Backend, DtypeSupport, GeneratedKernel, LowerError};
use unpopped::ir::{BinaryOp, ScalarExpr, UnaryOp};
use unpopped::plan::{KernelPlan, Schedule};
use unpopped_vocab::{ElementKind, TargetId};

/// The OpenCL C backend.
#[derive(Copy, Clone, Debug, Default)]
pub struct OpenClC {
    fp64: bool,
}

impl OpenClC {
    /// A backend that emits `f32` only.
    #[must_use]
    pub const fn new() -> Self {
        Self { fp64: false }
    }

    /// Allow (or forbid) `f64` plans. Only set this for a device that lists
    /// `cl_khr_fp64`; a kernel using `double` fails to build elsewhere.
    #[must_use]
    pub const fn with_fp64(self, allowed: bool) -> Self {
        Self { fp64: allowed }
    }

    /// A backend for a device with these extensions (for example from
    /// `openclownfish::Device::extensions`): `f64` is allowed iff `cl_khr_fp64`
    /// is among them.
    #[must_use]
    pub fn for_extensions<'a>(extensions: impl IntoIterator<Item = &'a str>) -> Self {
        Self::new().with_fp64(extensions.into_iter().any(|e| e == "cl_khr_fp64"))
    }
}

/// The C spelling of the kernel's element type.
#[derive(Copy, Clone)]
struct Ty {
    /// `float` or `double`.
    c: &'static str,
    /// Literal suffix: `f` for float, none for double.
    suffix: &'static str,
    /// Used in the kernel name.
    tag: &'static str,
}

const F32: Ty = Ty {
    c: "float",
    suffix: "f",
    tag: "f32",
};
const F64: Ty = Ty {
    c: "double",
    suffix: "",
    tag: "f64",
};

fn refuse(detail: impl Into<String>) -> LowerError {
    LowerError::UnsupportedOp {
        detail: format!("opencl_c v1: {}", detail.into()),
    }
}

/// Spell `e` as a fully parenthesised OpenCL C expression of type `t`.
fn expr(e: &ScalarExpr, n_inputs: u8, t: Ty) -> Result<String, LowerError> {
    let sfx = t.suffix;
    Ok(match e {
        ScalarExpr::Input(i) if *i < n_inputs => format!("in{i}[i]"),
        ScalarExpr::Input(i) => return Err(refuse(format!("input {i} out of range"))),
        ScalarExpr::Const(c) => {
            if t.c == "float" {
                #[allow(clippy::cast_possible_truncation)]
                let f = *c as f32;
                if !f.is_finite() || f64::from(f) != *c {
                    return Err(refuse(format!(
                        "constant {c:?} is not exactly representable as a finite f32"
                    )));
                }
                // `{:?}` always carries a '.' or exponent, so the literal is a float.
                format!("{f:?}f")
            } else if c.is_finite() {
                format!("{c:?}")
            } else {
                return Err(refuse(format!("constant {c:?} is not finite")));
            }
        }
        ScalarExpr::Add(a, b) => bin(a, "+", b, n_inputs, t)?,
        ScalarExpr::Sub(a, b) => bin(a, "-", b, n_inputs, t)?,
        ScalarExpr::Mul(a, b) => bin(a, "*", b, n_inputs, t)?,
        ScalarExpr::Div(a, b) => bin(a, "/", b, n_inputs, t)?,
        ScalarExpr::Unary(op, a) => {
            let x = expr(a, n_inputs, t)?;
            match op {
                UnaryOp::Neg => format!("(-{x})"),
                UnaryOp::Abs => format!("fabs({x})"),
                // A helper rather than `(x * x)`, which would repeat `x` and
                // double the source at every nesting level.
                UnaryOp::Sqr => format!("ocl_sqr({x})"),
                UnaryOp::Recip => format!("(1.0{sfx} / {x})"),
                UnaryOp::Sqrt => format!("sqrt({x})"),
                UnaryOp::Rsqrt => format!("rsqrt({x})"),
                UnaryOp::Exp => format!("exp({x})"),
                UnaryOp::Log => format!("log({x})"),
                other => return Err(refuse(format!("unary op not supported: {other:?}"))),
            }
        }
        // A comparison is 1.0 or 0.0 with C's NaN rules: false on any NaN
        // operand, except `!=`, which is true.
        ScalarExpr::Binary(
            op @ (BinaryOp::CmpEq
            | BinaryOp::CmpNe
            | BinaryOp::CmpLt
            | BinaryOp::CmpLe
            | BinaryOp::CmpGt
            | BinaryOp::CmpGe),
            a,
            b,
        ) => {
            let sym = match op {
                BinaryOp::CmpEq => "==",
                BinaryOp::CmpNe => "!=",
                BinaryOp::CmpLt => "<",
                BinaryOp::CmpLe => "<=",
                BinaryOp::CmpGt => ">",
                _ => ">=",
            };
            format!(
                "(({} {sym} {}) ? 1.0{sfx} : 0.0{sfx})",
                expr(a, n_inputs, t)?,
                expr(b, n_inputs, t)?
            )
        }
        // The condition is tested `!= 0`, so NaN is true and -0.0 is false. The
        // chosen arm is moved, not computed on.
        ScalarExpr::Select(c, a, b) => format!(
            "(({} != 0.0{sfx}) ? {} : {})",
            expr(c, n_inputs, t)?,
            expr(a, n_inputs, t)?,
            expr(b, n_inputs, t)?
        ),
        ScalarExpr::Binary(op, a, b) => {
            let name = match op {
                BinaryOp::Max => "ocl_max",
                BinaryOp::Min => "ocl_min",
                BinaryOp::FmaxIeee => "ocl_fmax",
                BinaryOp::FminIeee => "ocl_fmin",
                // OpenCL's `pow` follows C99 Annex F for the special cases.
                BinaryOp::Pow => "pow",
                other => return Err(refuse(format!("binary op not supported: {other:?}"))),
            };
            format!(
                "{name}({}, {})",
                expr(a, n_inputs, t)?,
                expr(b, n_inputs, t)?
            )
        }
        other => return Err(refuse(format!("expression node not supported: {other:?}"))),
    })
}

/// Helpers the kernel carries, emitted only when its body calls them. `float`
/// in the text stands for the kernel's element type.
///
/// `Max`/`Min` propagate NaN and `FmaxIeee`/`FminIeee` suppress it, and OpenCL's
/// own `fmax`/`fmin` leave the sign of a zero result unspecified for `(-0, +0)`.
/// These helpers follow Unpopped's normative reference instead: NaN is moved
/// through verbatim, and on a tie the first operand wins (`a >= b ? a : b`), so
/// `max(-0, +0)` is `-0` on every stack.
const HELPERS: &[(&str, &str)] = &[
    (
        "ocl_sqr(",
        "inline float ocl_sqr(float x) { return x * x; }\n",
    ),
    (
        "ocl_max(",
        "inline float ocl_max(float a, float b) {\n    if (isnan(a)) { return a; }\n    if (isnan(b)) { return b; }\n    return a >= b ? a : b;\n}\n",
    ),
    (
        "ocl_min(",
        "inline float ocl_min(float a, float b) {\n    if (isnan(a)) { return a; }\n    if (isnan(b)) { return b; }\n    return a <= b ? a : b;\n}\n",
    ),
    (
        "ocl_fmax(",
        "inline float ocl_fmax(float a, float b) {\n    if (isnan(a)) { return b; }\n    if (isnan(b)) { return a; }\n    return a >= b ? a : b;\n}\n",
    ),
    (
        "ocl_fmin(",
        "inline float ocl_fmin(float a, float b) {\n    if (isnan(a)) { return b; }\n    if (isnan(b)) { return a; }\n    return a <= b ? a : b;\n}\n",
    ),
];

fn bin(a: &ScalarExpr, op: &str, b: &ScalarExpr, n: u8, t: Ty) -> Result<String, LowerError> {
    Ok(format!("({} {op} {})", expr(a, n, t)?, expr(b, n, t)?))
}

impl Backend for OpenClC {
    fn name(&self) -> &str {
        "opencl_c"
    }

    fn provider(&self) -> &str {
        "openclownfish"
    }

    fn supports_dtype(&self, dtype: ElementKind, _target: TargetId) -> DtypeSupport {
        match dtype {
            ElementKind::F32 => DtypeSupport::Supported,
            ElementKind::F64 if self.fp64 => DtypeSupport::Supported,
            _ => DtypeSupport::Unsupported,
        }
    }

    fn lower(&self, plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        if plan.out_dtype != plan.dtype {
            return Err(LowerError::UnsupportedDtype {
                dtype: plan.out_dtype,
                detail: "opencl_c v1: output dtype must equal the compute dtype".to_owned(),
            });
        }
        let t = match plan.dtype {
            ElementKind::F32 => F32,
            ElementKind::F64 if self.fp64 => F64,
            ElementKind::F64 => {
                return Err(LowerError::UnsupportedDtype {
                    dtype: plan.dtype,
                    detail: "opencl_c v1: f64 needs cl_khr_fp64; build the backend with \
                             OpenClC::for_extensions from the target device"
                        .to_owned(),
                });
            }
            other => {
                return Err(LowerError::UnsupportedDtype {
                    dtype: other,
                    detail: "opencl_c v1: f32 and (with cl_khr_fp64) f64 only".to_owned(),
                });
            }
        };
        if plan.n_outputs != 1 {
            return Err(LowerError::UnsupportedPlanShape {
                detail: format!(
                    "opencl_c v1: single output only; op '{}' has {}",
                    plan.op_name, plan.n_outputs
                ),
            });
        }
        // `Vectorized` is a width hint for backends that vectorise; this kernel
        // is element-counted (`effective_count_width` stays 1), which is correct
        // for any contiguous elementwise plan.
        if !matches!(
            plan.schedule,
            Schedule::Scalar | Schedule::Vectorized { .. }
        ) {
            return Err(LowerError::UnsupportedSchedule {
                detail: format!(
                    "opencl_c v1: contiguous scalar or vectorized schedule only, got {:?}",
                    plan.schedule
                ),
            });
        }
        let body = expr(plan.body, plan.n_inputs, t)?;
        let name = format!("openclownfish_{}_{}", plan.op_name, t.tag);
        let mut s = String::new();
        s.push_str("// Generated by openclownfish-clc (opencl_c backend) - do not edit.\n");
        s.push_str(&format!(
            "// op: {} | cell: {}\n",
            plan.op_name,
            plan.key.to_token()
        ));
        if t.c == "double" {
            s.push_str("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
        }
        // Without this a compiler may fuse a*b+c into one rounding, so results
        // would differ from the unfused reference and between vendors.
        s.push_str("#pragma OPENCL FP_CONTRACT OFF\n\n");
        let mut helpers = false;
        for (call, text) in HELPERS {
            if body.contains(call) {
                s.push_str(&text.replace("float", t.c));
                helpers = true;
            }
        }
        if helpers {
            s.push('\n');
        }
        s.push_str(&format!("__kernel void {name}(\n"));
        for i in 0..plan.n_inputs {
            s.push_str(&format!("    __global const {}* in{i},\n", t.c));
        }
        s.push_str(&format!("    __global {}* out,\n    long n)\n{{\n", t.c));
        s.push_str("    long i = (long)get_global_id(0);\n");
        s.push_str(&format!("    if (i < n) {{ out[i] = {body}; }}\n}}\n"));
        Ok(GeneratedKernel::new(name, s))
    }
}
