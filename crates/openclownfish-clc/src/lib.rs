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
//! `i32` and `i64` plans are served for the subset the IR allows on integers:
//! wrapping `+ - *` and the bitwise and shift operators.
//!
//! Whether `f64` may be emitted is the caller's to say, from the target device's
//! extension list: [`OpenClC::for_extensions`]. A plan carries no device, so the
//! backend does.

#[cfg(feature = "parse")]
mod parse;
#[cfg(feature = "parse")]
pub use parse::{OPENCL_C, parse_opencl_c};

use unpopped::backend::{Backend, DtypeSupport, GeneratedKernel, LowerError};
use unpopped::ir::{BinaryOp, ReadIndex, ScalarExpr, UnaryOp, View, WriteIndex};
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
    /// An integer type: arithmetic wraps and the operator set is different.
    int: bool,
    /// The unsigned type of the same width, through which integer `+ - *` and
    /// `<<` are computed so overflow wraps instead of being undefined.
    uns: &'static str,
}

const F32: Ty = Ty {
    c: "float",
    suffix: "f",
    tag: "f32",
    int: false,
    uns: "",
};
const F64: Ty = Ty {
    c: "double",
    suffix: "",
    tag: "f64",
    int: false,
    uns: "",
};
const I32: Ty = Ty {
    c: "int",
    suffix: "",
    tag: "i32",
    int: true,
    uns: "uint",
};
const I64: Ty = Ty {
    c: "long",
    suffix: "L",
    tag: "i64",
    int: true,
    uns: "ulong",
};

fn refuse(detail: impl Into<String>) -> LowerError {
    LowerError::UnsupportedOp {
        detail: format!("opencl_c v1: {}", detail.into()),
    }
}

/// What an expression is spelled against.
#[derive(Copy, Clone)]
struct Cx<'a> {
    n_inputs: u8,
    t: Ty,
    /// `Some` for the strided schedule, where leaves read through offsets.
    strided: Option<&'a StridedCx>,
}

/// The strided schedule's compile-time facts, from the structure key.
struct StridedCx {
    rank: usize,
    /// Per input: every iteration axis is broadcast, so its load is hoisted.
    fully_bcast: Vec<bool>,
}

/// How input `i` is read: by linear index, through its strided offset, or from
/// the value hoisted out of the loop when every axis is broadcast.
fn input_leaf(i: u8, cx: Cx<'_>) -> String {
    match cx.strided {
        None => format!("in{i}[i]"),
        Some(st) if st.fully_bcast[usize::from(i)] => format!("h{i}"),
        Some(_) => format!("in{i}[o{i}]"),
    }
}

/// The integer subset the IR admits (`ir.rs` op x dtype table): `+ - *` that
/// wrap, and bitwise and shift operators. Division, unary and comparison
/// operators are not legal on integers there, and `Select` and integer
/// constants (`ConstInt`) have no integer lowering in v1 (the plan gate says
/// so), so they are refused here too.
///
/// Signed overflow is undefined in C, so `+ - *` and `<<` are computed in the
/// unsigned type of the same width and converted back, which wraps on every
/// compiler. Shift amounts outside `0..bits` are the caller's to avoid, as in
/// the IR contract.
fn int_expr(e: &ScalarExpr, cx: Cx<'_>) -> Result<String, LowerError> {
    let (t, n_inputs) = (cx.t, cx.n_inputs);
    let wrap = |a: &ScalarExpr, op: &str, b: &ScalarExpr| -> Result<String, LowerError> {
        Ok(format!(
            "(({})(({}){} {op} ({}){}))",
            t.c,
            t.uns,
            int_expr(a, cx)?,
            t.uns,
            int_expr(b, cx)?
        ))
    };
    let plain = |a: &ScalarExpr, op: &str, b: &ScalarExpr| -> Result<String, LowerError> {
        Ok(format!("({} {op} {})", int_expr(a, cx)?, int_expr(b, cx)?))
    };
    Ok(match e {
        ScalarExpr::Input(i) if *i < n_inputs => input_leaf(*i, cx),
        ScalarExpr::Input(i) => return Err(refuse(format!("input {i} out of range"))),
        ScalarExpr::Add(a, b) => wrap(a, "+", b)?,
        ScalarExpr::Sub(a, b) => wrap(a, "-", b)?,
        ScalarExpr::Mul(a, b) => wrap(a, "*", b)?,
        ScalarExpr::Binary(BinaryOp::BitAnd, a, b) => plain(a, "&", b)?,
        ScalarExpr::Binary(BinaryOp::BitOr, a, b) => plain(a, "|", b)?,
        ScalarExpr::Binary(BinaryOp::BitXor, a, b) => plain(a, "^", b)?,
        ScalarExpr::Binary(BinaryOp::Shl, a, b) => wrap(a, "<<", b)?,
        // Arithmetic on signed types, as the IR contract states.
        ScalarExpr::Binary(BinaryOp::Shr, a, b) => plain(a, ">>", b)?,
        other => {
            return Err(refuse(format!(
                "expression node not supported on {}: {other:?}",
                t.c
            )));
        }
    })
}

/// Spell `e` as a fully parenthesised OpenCL C expression of type `t`.
fn expr(e: &ScalarExpr, cx: Cx<'_>) -> Result<String, LowerError> {
    if cx.t.int {
        return int_expr(e, cx);
    }
    let (t, n_inputs) = (cx.t, cx.n_inputs);
    let sfx = t.suffix;
    Ok(match e {
        ScalarExpr::Input(i) if *i < n_inputs => input_leaf(*i, cx),
        // The unravelled output coordinate along axis `d`, as the element type.
        ScalarExpr::Coord(d) if cx.strided.is_some_and(|st| usize::from(*d) < st.rank) => {
            format!("(({}) c{d})", t.c)
        }
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
        ScalarExpr::Add(a, b) => bin(a, "+", b, cx)?,
        ScalarExpr::Sub(a, b) => bin(a, "-", b, cx)?,
        ScalarExpr::Mul(a, b) => bin(a, "*", b, cx)?,
        ScalarExpr::Div(a, b) => bin(a, "/", b, cx)?,
        ScalarExpr::Unary(op, a) => {
            let x = expr(a, cx)?;
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
                expr(a, cx)?,
                expr(b, cx)?
            )
        }
        // The condition is tested `!= 0`, so NaN is true and -0.0 is false. The
        // chosen arm is moved, not computed on.
        ScalarExpr::Select(c, a, b) => format!(
            "(({} != 0.0{sfx}) ? {} : {})",
            expr(c, cx)?,
            expr(a, cx)?,
            expr(b, cx)?
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
            format!("{name}({}, {})", expr(a, cx)?, expr(b, cx)?)
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

fn bin(a: &ScalarExpr, op: &str, b: &ScalarExpr, cx: Cx<'_>) -> Result<String, LowerError> {
    Ok(format!("({} {op} {})", expr(a, cx)?, expr(b, cx)?))
}

/// What the strided schedule needs from the plan, or why this emitter cannot
/// serve it. Gathers, scatters, runtime base offsets and permuting views change
/// the kernel's argument list and addressing; they are refused, not guessed.
fn strided_facts(plan: &KernelPlan<'_>) -> Result<StridedCx, LowerError> {
    let shape_err = |what: &str| LowerError::UnsupportedPlanShape {
        detail: format!(
            "opencl_c v1: strided schedule without {what}; op '{}'",
            plan.op_name
        ),
    };
    if plan
        .read_index
        .iter()
        .any(|r| !matches!(r, ReadIndex::Direct))
    {
        return Err(shape_err("gathers"));
    }
    if !matches!(plan.write_index, WriteIndex::Direct) {
        return Err(shape_err("scatters"));
    }
    if plan.base_offsets.iter().any(|b| !b.is_zero()) || !plan.out_base_offset.is_zero() {
        return Err(shape_err("runtime base offsets"));
    }
    if plan.views.iter().any(|v| {
        !matches!(
            v,
            View::Identity | View::Broadcast { .. } | View::Reshape { .. }
        )
    }) {
        return Err(shape_err("permuting or layout views"));
    }
    let rank = usize::from(plan.key.rank);
    let fully_bcast = (0..usize::from(plan.n_inputs))
        .map(|k| rank > 0 && (0..rank).all(|d| plan.key.operands[k].bcast.is_set(d as u8)))
        .collect();
    Ok(StridedCx { rank, fully_bcast })
}

/// The element offset of operand `k` (`s{k}_{d}` strides; broadcast axes have
/// stride 0 and drop out) or of the output (`so_{d}`).
fn offset_expr(plan: &KernelPlan<'_>, operand: usize, stride: &str, rank: usize) -> String {
    let terms: Vec<String> = (0..rank)
        .filter(|d| !plan.key.operands[operand].bcast.is_set(*d as u8))
        .map(|d| format!("c{d}*{stride}_{d}"))
        .collect();
    if terms.is_empty() {
        "0".to_owned()
    } else {
        terms.join(" + ")
    }
}

/// The rest of a strided kernel after `out`: the runtime shape, every input's
/// strides, the output's strides and the element count, in the argument order
/// Baracuda's CUDA strided kernels use (`shape{d}`, `s{i}_{d}`, `so_{d}`, `n`),
/// then a one-work-item-per-element body that unravels the linear index
/// row-major (last axis fastest) and reads each operand through its strides.
fn emit_strided_body(
    s: &mut String,
    plan: &KernelPlan<'_>,
    st: &StridedCx,
    ctype: &str,
    body: &str,
) {
    let rank = st.rank;
    for d in 0..rank {
        s.push_str(&format!("    long shape{d},\n"));
    }
    for i in 0..plan.n_inputs {
        for d in 0..rank {
            s.push_str(&format!("    long s{i}_{d},\n"));
        }
    }
    for d in 0..rank {
        s.push_str(&format!("    long so_{d},\n"));
    }
    s.push_str("    long n)\n{\n");
    s.push_str("    long i = (long)get_global_id(0);\n    if (i >= n) { return; }\n");
    for (k, hoist) in st.fully_bcast.iter().enumerate() {
        if *hoist {
            s.push_str(&format!("    {} h{k} = in{k}[0];\n", ctype));
        }
    }
    s.push_str("    long lin = i;\n");
    for d in (0..rank).rev() {
        s.push_str(&format!(
            "    long c{d} = lin % shape{d}; lin /= shape{d};\n"
        ));
    }
    for (k, hoist) in st.fully_bcast.iter().enumerate() {
        if !*hoist {
            let off = offset_expr(plan, k, &format!("s{k}"), rank);
            s.push_str(&format!("    long o{k} = {off};\n"));
        }
    }
    let oo = offset_expr(plan, usize::from(plan.n_inputs), "so", rank);
    s.push_str(&format!("    long oo = {oo};\n"));
    s.push_str(&format!("    out[oo] = {body};\n}}\n"));
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
            ElementKind::I32 | ElementKind::I64 => DtypeSupport::Supported,
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
            ElementKind::I32 => I32,
            ElementKind::I64 => I64,
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
                    detail: "opencl_c v1: f32, i32, i64 and (with cl_khr_fp64) f64 only".to_owned(),
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
        let strided = match plan.schedule {
            Schedule::Scalar | Schedule::Vectorized { .. } => None,
            Schedule::Strided => Some(strided_facts(plan)?),
            other => {
                return Err(LowerError::UnsupportedSchedule {
                    detail: format!(
                        "opencl_c v1: scalar, vectorized or strided schedule only, got {other:?}"
                    ),
                });
            }
        };
        let cx = Cx {
            n_inputs: plan.n_inputs,
            t,
            strided: strided.as_ref(),
        };
        let body = expr(plan.body, cx)?;
        let kind = if strided.is_some() {
            format!("_strided_r{}", plan.key.rank)
        } else {
            String::new()
        };
        let name = format!("openclownfish_{}_{}{kind}", plan.op_name, t.tag);
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
        s.push_str(&format!("    __global {}* out,\n", t.c));
        match &strided {
            None => {
                s.push_str("    long n)\n{\n");
                s.push_str("    long i = (long)get_global_id(0);\n");
                s.push_str(&format!("    if (i < n) {{ out[i] = {body}; }}\n}}\n"));
            }
            Some(st) => emit_strided_body(&mut s, plan, st, t.c, &body),
        }
        Ok(GeneratedKernel::new(name, s))
    }
}
