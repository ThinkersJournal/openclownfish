//! OpenCL C emitter for [Unpopped](https://crates.io/crates/unpopped).
//!
//! [`OpenClC`] implements `unpopped::backend::Backend`. It is narrow on
//! purpose: contiguous single-output `f32` (and, when the device has
//! `cl_khr_fp64`, `f64`) elementwise plans whose body uses inputs, finite
//! exactly-representable constants, `+ - * /` and the unary operations negate,
//! abs, square, reciprocal, sqrt, rsqrt, exp and log, `Max`/`Min`
//! (NaN-propagating) and `FmaxIeee`/`FminIeee` (NaN-suppressing), `Pow`, the six
//! comparisons (1.0 or 0.0) and `Select`. Everything else is a typed refusal,
//! never an approximation. Reductions (`Schedule::Reduction`) are served as a
//! serial per-output fold over any axis set, one input, identity post-expression,
//! and scans (`Schedule::Scan`) as a serial per-row fold of the innermost axis;
//! `lower_variants` also offers a work-group-per-row `blockscan` variant of each scan.
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

use unpopped::backend::{
    Backend, DtypeSupport, GeneratedKernel, LowerError, Variant, VariantFidelity,
};
use unpopped::cfamily::{strided_offset_expr, unravel_decomp};
use unpopped::ir::{Access, BinaryOp, ReadIndex, ReduceOp, ScalarExpr, UnaryOp, View, WriteIndex};
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
/// compiler. A shift count is taken unsigned and modulo the bit width: OpenCL C
/// masks it (6.5.10), which is the kernel-layer rule of unpopped 0.38.3
/// (`docs/int-div-shift-semantics.md`), so out-of-range and negative counts need
/// no guard. Integer division stays refused until the plan gate admits it (E16);
/// OpenCL C leaves `b == 0` and `MIN / -1` unspecified, so it will need a guard.
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
/// stride 0 and drop out) or of the output (`so_{d}`), spelled by Unpopped's
/// shared C-family helper.
fn offset_expr(plan: &KernelPlan<'_>, operand: usize, stride: &str, rank: usize) -> String {
    strided_offset_expr(plan.key.operands[operand].bcast, "c", stride, rank, None)
}

/// The rest of a strided kernel after `out`: the runtime shape, every input's
/// strides, the output's strides and the element count, in the argument order
/// Baracuda's CUDA strided kernels use (`shape{d}`, `s{i}_{d}`, `so_{d}`, `n`),
/// then a one-work-item-per-element body that unravels the linear index
/// row-major (last axis fastest) and reads each operand through its strides.
/// What a reduction kernel folds: the op and which input axes it removes.
struct ReduceCx {
    op: ReduceOp,
    keepdim: bool,
    /// Ascending input axes folded away; the rest are kept.
    reduced: Vec<usize>,
    kept: Vec<usize>,
}

/// The reduction plan's facts, or why this emitter cannot serve it. Scope (v1):
/// one input, identity post-expression, `Sum`/`Mean`/`Prod`/`Max`/`Min`, no
/// integer `Mean` (it is a float-output op, unrepresentable in one dtype).
fn reduction_facts(plan: &KernelPlan<'_>, t: Ty) -> Result<ReduceCx, LowerError> {
    let Access::Reduction {
        op,
        axes,
        keepdim,
        post,
        ..
    } = plan.access
    else {
        return Err(refuse("reduction schedule without a reduction access"));
    };
    if plan.n_inputs != 1 {
        return Err(refuse("reductions take a single input"));
    }
    if !matches!(post, ScalarExpr::Reduced(0)) {
        return Err(refuse("fused reduction post-expressions"));
    }
    if t.int && matches!(op, ReduceOp::Mean) {
        return Err(refuse("integer Mean"));
    }
    let rank = usize::from(plan.key.rank);
    if rank == 0 {
        return Err(refuse("rank-0 reductions"));
    }
    let reduced: Vec<usize> = if axes.is_empty() {
        vec![rank - 1]
    } else {
        (0..rank).filter(|&d| axes.is_set(d as u8)).collect()
    };
    let kept = (0..rank).filter(|d| !reduced.contains(d)).collect();
    Ok(ReduceCx {
        op: *op,
        keepdim: *keepdim,
        reduced,
        kept,
    })
}

/// The kernel parameters after `out`, through the opening brace: `shape{d}` per
/// input axis, `s0_{d}` input strides, `so_{j}` output strides (per input axis
/// with `keepdim`, else per kept axis), then `n_out`, in Baracuda's order.
fn emit_reduction_params(s: &mut String, rank: usize, rc: &ReduceCx) {
    let n_so = if rc.keepdim { rank } else { rc.kept.len() };
    for d in 0..rank {
        s.push_str(&format!("    long shape{d},\n"));
    }
    for d in 0..rank {
        s.push_str(&format!("    long s0_{d},\n"));
    }
    for d in 0..n_so {
        s.push_str(&format!("    long so_{d},\n"));
    }
    s.push_str(
        "    long n_out)
{\n",
    );
}

/// An index expression: the terms added, or `0` when there are none.
fn sum(terms: Vec<String>) -> String {
    if terms.is_empty() {
        "0".to_owned()
    } else {
        terms.join(" + ")
    }
}

/// Declares `base` (the input offset of output element `o`'s first folded
/// element) and `oo` (its output offset); `o` must be in scope.
fn emit_reduction_addressing(s: &mut String, rc: &ReduceCx) {
    s.push_str("    long lin = o;\n");
    s.push_str(&unravel_decomp(
        rc.kept.iter().rev().copied(),
        "    ",
        "lin",
        "long",
        |a| format!("ck{a}"),
        |a| format!("shape{a}"),
        false,
    ));
    let base = sum(rc.kept.iter().map(|a| format!("ck{a}*s0_{a}")).collect());
    let oo = sum(rc
        .kept
        .iter()
        .enumerate()
        .map(|(j, a)| {
            if rc.keepdim {
                format!("ck{a}*so_{a}")
            } else {
                format!("ck{a}*so_{j}")
            }
        })
        .collect());
    s.push_str(&format!(
        "    long base = {base};
    long oo = {oo};\n"
    ));
}

/// The rest of a reduction kernel after `out`. One work item per output element
/// folds the reduced axes serially, in row-major order over them, in the element
/// type.
fn emit_reduction_body(s: &mut String, rank: usize, rc: &ReduceCx, t: Ty, body: &str) {
    emit_reduction_params(s, rank, rc);
    s.push_str(
        "    long o = (long)get_global_id(0);
    if (o >= n_out) { return; }\n",
    );
    emit_reduction_addressing(s, rc);
    let ty = t.c;
    let u = t.uns;
    let (init, step) = match rc.op {
        ReduceOp::Sum | ReduceOp::Mean => (
            format!("({ty})0"),
            if t.int {
                format!("acc = ({ty})(({u})acc + ({u})({body}));")
            } else {
                format!("acc += {body};")
            },
        ),
        ReduceOp::Prod => (
            format!("({ty})1"),
            if t.int {
                format!("acc = ({ty})(({u})acc * ({u})({body}));")
            } else {
                format!("acc *= {body};")
            },
        ),
        ReduceOp::Max | ReduceOp::Min => {
            let max = matches!(rc.op, ReduceOp::Max);
            let cmp = if max { ">" } else { "<" };
            // An empty reduced extent leaves the monoid identity (dtype extreme
            // for integers, 0 for floats), as Baracuda's emitter does.
            let seed = match (t.tag, max) {
                ("i32", true) => "(-2147483647 - 1)".to_owned(),
                ("i32", false) => "2147483647".to_owned(),
                ("i64", true) => "(-9223372036854775807L - 1L)".to_owned(),
                ("i64", false) => "9223372036854775807L".to_owned(),
                _ => format!("({ty})0"),
            };
            (
                seed,
                format!(
                    "{{ {ty} e = {body}; acc = has ? ((e != e || e {cmp} acc) ? e : acc) : e; has = 1; }}"
                ),
            )
        }
    };
    s.push_str(&format!("    {ty} acc = {init};\n"));
    if matches!(rc.op, ReduceOp::Max | ReduceOp::Min) {
        s.push_str("    int has = 0;\n");
    }
    for &r in &rc.reduced {
        s.push_str(&format!(
            "    for (long cr{r} = 0; cr{r} < shape{r}; ++cr{r}) {{\n"
        ));
    }
    let idx = sum(std::iter::once("base".to_owned())
        .chain(rc.reduced.iter().map(|r| format!("cr{r}*s0_{r}")))
        .collect());
    s.push_str(&format!(
        "    long i = {idx};
    {step}\n"
    ));
    for _ in &rc.reduced {
        s.push_str("    }\n");
    }
    let fin = if matches!(rc.op, ReduceOp::Mean) {
        let div = rc
            .reduced
            .iter()
            .map(|r| format!("shape{r}"))
            .collect::<Vec<_>>()
            .join(" * ");
        format!("acc / ({ty})({div})")
    } else {
        "acc".to_owned()
    };
    s.push_str(&format!(
        "    out[oo] = {fin};
}}\n"
    ));
}

/// What a scan kernel computes along the innermost axis.
struct ScanCx {
    op: ReduceOp,
    reverse: bool,
    exclusive: bool,
}

/// The scan plan's facts, and its per-element pre-map, or why this emitter
/// cannot serve it. Scope (v1): the serial per-row fold (not the block-scan
/// variant), the innermost axis, one input, identity epilogue, no `Mean`.
fn scan_facts<'p>(
    plan: &KernelPlan<'p>,
    block: bool,
) -> Result<(ScanCx, &'p ScalarExpr), LowerError> {
    let Schedule::Scan {
        op,
        axis,
        reverse,
        exclusive,
        ..
    } = plan.schedule
    else {
        return Err(refuse("scan schedule expected"));
    };
    let Access::Scan { pre, post, .. } = plan.access else {
        return Err(refuse("scan schedule without a scan access"));
    };
    if block {
        return Err(refuse("the block-scan variant"));
    }
    if matches!(op, ReduceOp::Mean) {
        return Err(refuse("Mean is not a scan monoid"));
    }
    if plan.n_inputs != 1 {
        return Err(refuse("scans take a single input"));
    }
    if usize::from(axis) + 1 != usize::from(plan.key.rank) {
        return Err(refuse("scans over a non-innermost axis"));
    }
    if !matches!(post, ScalarExpr::Reduced(0)) {
        return Err(refuse("fused scan epilogues"));
    }
    Ok((
        ScanCx {
            op,
            reverse,
            exclusive,
        },
        pre,
    ))
}

/// The scan monoid's identity, spelled for the element type.
fn scan_ident(op: ReduceOp, t: Ty) -> String {
    let ty = t.c;
    match op {
        ReduceOp::Sum | ReduceOp::Mean => format!("({ty})0"),
        ReduceOp::Prod => format!("({ty})1"),
        _ => match (t.tag, matches!(op, ReduceOp::Max)) {
            ("i32", true) => "(-2147483647 - 1)".to_owned(),
            ("i32", false) => "2147483647".to_owned(),
            ("i64", true) => "(-9223372036854775807L - 1L)".to_owned(),
            ("i64", false) => "9223372036854775807L".to_owned(),
            (_, true) => format!("(({ty})-INFINITY)"),
            (_, false) => format!("(({ty})INFINITY)"),
        },
    }
}

/// The block-scan variant's kernel: one work-group per row, `OCF_BLOCK` work
/// items (a power of two, default 64, set with `-D OCF_BLOCK=n` at build time).
/// A Hillis-Steele inclusive scan in `__local` memory per chunk of `OCF_BLOCK`
/// elements; a row longer than that threads a carry through its chunks.
///
/// `comb(a, b)` has `a` earlier in scan order. Max/Min take the later operand
/// when it is NaN or strictly better and otherwise the earlier one, so ties
/// keep the earlier element and the last NaN wins - the serial fold's result,
/// and associative, so the tree agrees with it bit for bit. The combiner picks
/// an operand rather than computing one, so signed zeros and NaN payloads move.
fn emit_blockscan_body(s: &mut String, sc: &ScanCx, t: Ty, pre: &str) {
    let ty = t.c;
    let ident = scan_ident(sc.op, t);
    s.push_str(
        "    long n_out,
    long k)
{
    __local ",
    );
    s.push_str(&format!("{ty} buf[OCF_BLOCK];\n"));
    s.push_str(
        "    long row = (long)get_group_id(0);
    if (row >= n_out) { return; }
    long lid = (long)get_local_id(0);
    long base = row * k;\n",
    );
    s.push_str(&format!("    {ty} carry = {ident};\n"));
    s.push_str("    for (long c0 = 0; c0 < k; c0 += OCF_BLOCK) {\n");
    s.push_str("        long p = c0 + lid;\n");
    s.push_str(&format!("        {ty} v = {ident};\n"));
    let j = if sc.reverse { "k - 1 - p" } else { "p" };
    s.push_str(&format!(
        "        long i = base + ({j});
        if (p < k) {{ v = {pre}; }}
        buf[lid] = v;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (long off = 1; off < OCF_BLOCK; off <<= 1) {{
            {ty} left = buf[lid >= off ? lid - off : lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            {ty} mine = buf[lid];
            {ty} next = lid >= off ? ocf_comb(left, mine) : mine;
            buf[lid] = next;
            barrier(CLK_LOCAL_MEM_FENCE);
        }}\n"
    ));
    let val = if sc.exclusive {
        "(lid > 0 ? ocf_comb(carry, buf[lid - 1]) : carry)"
    } else {
        "ocf_comb(carry, buf[lid])"
    };
    s.push_str(&format!(
        "        if (p < k) {{ out[i] = {val}; }}
        {ty} total = buf[OCF_BLOCK - 1];
        carry = ocf_comb(carry, total);
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
}}\n"
    ));
}

/// Text that precedes the block-scan kernel: the block-size macro and combiner.
/// The block-size macro and the `ocf_comb` combiner for a monoid (`Mean` folds as `Sum`).
fn block_prelude(op: ReduceOp, t: Ty) -> String {
    let (ty, u) = (t.c, t.uns);
    let comb = match op {
        ReduceOp::Sum | ReduceOp::Mean | ReduceOp::Prod => {
            let c = if matches!(op, ReduceOp::Sum | ReduceOp::Mean) {
                "+"
            } else {
                "*"
            };
            if t.int {
                format!("{ty} ocf_comb({ty} a, {ty} b) {{ return ({ty})(({u})a {c} ({u})b); }}")
            } else {
                format!("{ty} ocf_comb({ty} a, {ty} b) {{ return a {c} b; }}")
            }
        }
        _ => {
            let c = if matches!(op, ReduceOp::Max) {
                ">"
            } else {
                "<"
            };
            format!("{ty} ocf_comb({ty} a, {ty} b) {{ return (b != b || b {c} a) ? b : a; }}")
        }
    };
    format!("#ifndef OCF_BLOCK\n#define OCF_BLOCK 64\n#endif\n\n{comb}\n\n")
}

/// The block-reduce variant's kernel after `out`: one work-group per output
/// element, `OCF_BLOCK` work items. Item `lid` folds a contiguous chunk of the
/// row-major reduced extent serially; a neighbour-pair tree in `__local` memory
/// then combines the chunks in order, so `ocf_comb(a, b)` always has `a` earlier.
/// That keeps Max/Min (ties keep the earlier operand, the last NaN wins) the
/// serial fold's bits, signed zeros and NaN payloads included.
fn emit_blockreduce_body(s: &mut String, rank: usize, rc: &ReduceCx, t: Ty, body: &str) {
    let ty = t.c;
    let ident = scan_ident(rc.op, t);
    emit_reduction_params(s, rank, rc);
    s.push_str(&format!("    __local {ty} buf[OCF_BLOCK];\n"));
    s.push_str(
        "    long o = (long)get_group_id(0);
    if (o >= n_out) { return; }
    long lid = (long)get_local_id(0);\n",
    );
    emit_reduction_addressing(s, rc);
    let total = rc
        .reduced
        .iter()
        .map(|r| format!("shape{r}"))
        .collect::<Vec<_>>()
        .join(" * ");
    s.push_str(&format!(
        "    long total = {total};
    long chunk = (total + OCF_BLOCK - 1) / OCF_BLOCK;
    long lo = lid * chunk;
    long hi = lo + chunk;
    if (hi > total) {{ hi = total; }}
    {ty} acc = {ident};
    for (long rr = lo; rr < hi; ++rr) {{
        long rl = rr;\n"
    ));
    s.push_str(&unravel_decomp(
        rc.reduced.iter().rev().copied(),
        "        ",
        "rl",
        "long",
        |a| format!("cr{a}"),
        |a| format!("shape{a}"),
        false,
    ));
    let idx = std::iter::once("base".to_owned())
        .chain(rc.reduced.iter().map(|r| format!("cr{r}*s0_{r}")))
        .collect::<Vec<_>>()
        .join(" + ");
    s.push_str(&format!(
        "        long i = {idx};
        acc = ocf_comb(acc, {body});
    }}
    buf[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (long st = 1; st < OCF_BLOCK; st <<= 1) {{
        if (lid % (2 * st) == 0 && lid + st < OCF_BLOCK) {{
            buf[lid] = ocf_comb(buf[lid], buf[lid + st]);
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}\n"
    ));
    let fin = match rc.op {
        ReduceOp::Mean => format!("buf[0] / ({ty})(total)"),
        // An empty reduced extent leaves 0 for floats (the serial seed), not the
        // infinity the tree starts from; integer seeds are the identity itself.
        ReduceOp::Max | ReduceOp::Min if !t.int => format!("(total == 0 ? ({ty})0 : buf[0])"),
        _ => "buf[0]".to_owned(),
    };
    s.push_str(&format!(
        "    if (lid == 0) {{ out[oo] = {fin}; }}
}}\n"
    ));
}

/// The rest of a scan kernel after `out`: `n_out` rows of `k` elements, in
/// Baracuda's argument order. One work item owns a row and folds it serially in
/// scan order (descending when `reverse`), in the element type.
fn emit_scan_body(s: &mut String, sc: &ScanCx, t: Ty, pre: &str) {
    let (ty, u) = (t.c, t.uns);
    let is_max = matches!(sc.op, ReduceOp::Max);
    let ident = scan_ident(sc.op, t);
    s.push_str(
        "    long n_out,
    long k)
{\n",
    );
    s.push_str(
        "    long row = (long)get_global_id(0);
    if (row >= n_out) { return; }\n",
    );
    s.push_str("    long base = row * k;\n");
    s.push_str(&format!("    {ty} acc = {ident};\n"));
    let fold = match sc.op {
        ReduceOp::Sum | ReduceOp::Prod => {
            let c = if matches!(sc.op, ReduceOp::Sum) {
                "+"
            } else {
                "*"
            };
            if t.int {
                format!("acc = ({ty})(({u})acc {c} ({u})v);")
            } else {
                format!("acc = acc {c} v;")
            }
        }
        _ => {
            let c = if is_max { ">" } else { "<" };
            s.push_str("    int have = 0;\n");
            format!("if (!have || v != v || v {c} acc) {{ acc = v; have = 1; }}")
        }
    };
    let first = if matches!(sc.op, ReduceOp::Max | ReduceOp::Min) {
        format!("have ? acc : {ident}")
    } else {
        "acc".to_owned()
    };
    s.push_str("    for (long jj = 0; jj < k; ++jj) {\n");
    let j = if sc.reverse { "k - 1 - jj" } else { "jj" };
    s.push_str(&format!("        long i = base + ({j});\n"));
    s.push_str(&format!("        {ty} v = {pre};\n"));
    if sc.exclusive {
        s.push_str(&format!(
            "        out[i] = {first};
        {fold}\n"
        ));
    } else {
        s.push_str(&format!(
            "        {fold}
        out[i] = acc;\n"
        ));
    }
    s.push_str(
        "    }
}\n",
    );
}

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
    s.push_str(&unravel_decomp(
        (0..rank).rev(),
        "    ",
        "lin",
        "long",
        |d| format!("c{d}"),
        |d| format!("shape{d}"),
        false,
    ));
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

    fn lower_variants(&self, plan: &KernelPlan<'_>) -> Result<Vec<Variant>, LowerError> {
        let is_scan = match plan.schedule {
            Schedule::Scan { block: false, .. } => true,
            Schedule::Reduction { .. } => false,
            _ => return Ok(Vec::new()),
        };
        // Any decline of the serial base is a decline here too (never worse).
        let Ok(base) = self.lower(plan) else {
            return Ok(Vec::new());
        };
        let t = match plan.dtype {
            ElementKind::F32 => F32,
            ElementKind::I32 => I32,
            ElementKind::I64 => I64,
            ElementKind::F64 => F64,
            _ => return Ok(Vec::new()),
        };
        let cx = Cx {
            n_inputs: plan.n_inputs,
            t,
            strided: None,
        };
        let (variant, op, body, sc, rc);
        if is_scan {
            let Ok((s, pre)) = scan_facts(plan, false) else {
                return Ok(Vec::new());
            };
            variant = "blockscan";
            op = s.op;
            body = expr(pre, cx)?;
            sc = Some(s);
            rc = None;
        } else {
            let Ok(r) = reduction_facts(plan, t) else {
                return Ok(Vec::new());
            };
            variant = "blockreduce";
            op = r.op;
            body = expr(plan.body, cx)?;
            sc = None;
            rc = Some(r);
        }
        let name = format!("{}_{variant}", base.name);
        let mut s = String::new();
        s.push_str("// Generated by openclownfish-clc (opencl_c backend) - do not edit.\n");
        s.push_str(&format!(
            "// op: {} | cell: {} | variant: {variant}\n",
            plan.op_name,
            plan.key.to_token()
        ));
        if t.c == "double" {
            s.push_str("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
        }
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
        s.push_str(&block_prelude(op, t));
        s.push_str(&format!("__kernel void {name}(\n"));
        s.push_str(&format!("    __global const {}* in0,\n", t.c));
        s.push_str(&format!("    __global {}* out,\n", t.c));
        let usage = if let (Some(sc), None) = (&sc, &rc) {
            emit_blockscan_body(&mut s, sc, t, &body);
            "one work-group per row: global = n_out * B, local = B, B a power of two \
             no larger than the device's work-group limit (default 64; build with \
             -D OCF_BLOCK=B to change it). Arguments as the serial scan: in0, out, \
             n_out, k. A row longer than B threads a carry through ceil(k/B) chunks. \
             Local memory: B elements of the compute type."
        } else {
            let rc = rc.as_ref().expect("reduction facts");
            emit_blockreduce_body(&mut s, usize::from(plan.key.rank), rc, t, &body);
            "one work-group per output element: global = n_out * B, local = B, B a power \
             of two no larger than the device's work-group limit (default 64; build with \
             -D OCF_BLOCK=B to change it). Arguments as the serial reduction: in0, out, \
             shape{d}, s0_{d}, so_{j}, n_out. Each work item folds a contiguous chunk of \
             the row-major reduced extent, then a neighbour-pair tree combines the \
             chunks in order. Local memory: B elements of the compute type."
        };
        // Float Sum/Mean/Prod are reassociated by the tree; everything else
        // picks an operand or wraps exactly, so it is the serial fold's bits.
        let reassociates = !t.int && matches!(op, ReduceOp::Sum | ReduceOp::Mean | ReduceOp::Prod);
        let fidelity = if reassociates {
            VariantFidelity::DeterministicallyDivergent
        } else {
            VariantFidelity::BitIdentical
        };
        Ok(vec![Variant::new(
            variant,
            vec![GeneratedKernel::new(name, s)],
            fidelity,
            usage.to_owned(),
        )])
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
        let mut reduction = None;
        let mut scan = None;
        let strided = match plan.schedule {
            Schedule::Scalar | Schedule::Vectorized { .. } => None,
            Schedule::Scan { block, .. } => {
                scan = Some(scan_facts(plan, block)?);
                None
            }
            Schedule::Reduction { .. } => {
                reduction = Some(reduction_facts(plan, t)?);
                None
            }
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
        // A scan folds its per-element pre-map; the epilogue is the identity.
        let body = expr(scan.as_ref().map_or(plan.body, |(_, pre)| pre), cx)?;
        let kind = if let Some((sc, _)) = &scan {
            format!(
                "_scan_{}{}{}",
                format!("{:?}", sc.op).to_lowercase(),
                if sc.reverse { "_rev" } else { "" },
                if sc.exclusive { "_excl" } else { "" }
            )
        } else if let Some(rc) = &reduction {
            let ax: String = rc.reduced.iter().map(|r| r.to_string()).collect();
            format!(
                "_reduce_{}_ax{ax}{}",
                format!("{:?}", rc.op).to_lowercase(),
                if rc.keepdim { "_kd" } else { "" }
            )
        } else if strided.is_some() {
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
            None if scan.is_some() => {
                let (sc, _) = scan.as_ref().unwrap();
                emit_scan_body(&mut s, sc, t, &body);
            }
            None if reduction.is_some() => {
                let rc = reduction.as_ref().unwrap();
                emit_reduction_body(&mut s, usize::from(plan.key.rank), rc, t, &body);
            }
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
