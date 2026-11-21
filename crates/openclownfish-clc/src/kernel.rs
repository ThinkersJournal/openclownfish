// SPDX-License-Identifier: MIT OR Apache-2.0
//! A kernel-layer [`KernelFn`] spelled as an OpenCL C kernel: the statement-IR path
//! (`Backend::lower_kernel`), a deliberately small subset.
//!
//! What is spelled: integer and float (`F32`, `F64`) kernels with a launch shape, rank-1
//! buffers, scalar arguments, `Shared` arrays of static extent, `Let`/`Assign`/`Load`/`Store`,
//! `If`/`While`/`For`/`Break`/`Continue`/`Return`, a block `Barrier`, and the two atomic statements
//! `AtomicRmw` and `AtomicCas`. Everything else is declined by name with
//! [`LowerError::UnsupportedKernel`]; nothing is spelled by default.
//!
//! # Floats
//!
//! `F32` and `F64` (`F64` only with [`OpenClC::with_fp64`], and the build needs `cl_khr_fp64`)
//! elements in buffers, scalars, locals and `Shared` arrays; `Const`, `Add`, `Sub`, `Mul`,
//! `Div`, `Neg`, `Fma` and the six comparisons. `#pragma OPENCL FP_CONTRACT OFF` is written so
//! each operation rounds once. An `F32` expression follows the oracle: it is evaluated at
//! `f64` and rounded once per statement. With `with_fp64` the emitter does exactly that (the
//! leaves are widened to `double`); without it an `F32` statement may hold at most one rounding
//! operation (where the two agree) and anything more, `Fma` included, is declined by name.
//! `F32` `Div` is declined in both modes: OpenCL C bounds float division at 2.5 ULP and NVIDIA's
//! is 1 ULP off the oracle, even when the source divides at `double` (the compiler narrows it).
//! Also spelled: float `Max`, `Min`, `FmaxIeee`, `FminIeee` (helpers that follow Unpopped's NaN and
//! signed-zero rules), `Abs`, `Sqr` and `Select`; the exact float `Floor`, `Ceil`, `Trunc`,
//! `Round` (ties to even), `Relu`, `Sign` and `Step` (bit for bit); and the float `Sqrt`, `Exp`,
//! `Log`, `Tanh`, `Sin`, `Cos`, `Exp2`, `Expm1`, `Log2`, `Log10`, `Log1p`, `Sinh`, `Cosh`, `Tan`,
//! `Asin`, `Acos`, `Atan`, `Asinh`, `Acosh`, `Atanh` and `Cbrt` through the device's built-ins
//! and the float binary `Copysign`, `RemTrunc` (`fmod`, exact), `Pow`, `Atan2` (built-ins, ULP
//! limits) and an F64 floored `Rem`; a float `Nextafter` and an F32 `Rem` are declined by name
//! (NOT bit-exact: OpenCL allows several ULP, the device test states a tolerance per op).
//! An integer-to-float `Cast` (`Wrap`; `NearestEven`, `TowardZero`, `TowardNegInf`, `TowardPosInf`, to `F32`/`F64`)
//! is spelled with `convert_float_*`/`convert_double_*`, as is an `F64`-to-`F32` cast and an exact `F32`-to-`F64`
//! one; `NearestAway` (but for `F32` to `F64`) is declined.
//! A float-to-integer `Cast` (`Clamp`, any rounding mode, NaN to 0; to `I32`, `U32`, `I64`,
//! `U64`) is spelled, bit for bit. Declined by name: float atomics, a float cast that is not `Clamp`, `Recip`, `Rsqrt`, `Sigmoid`, `Silu` (a
//! float division), `Erf`, `Erfc`, `Gelu`, `Lgamma`, and `F16`/`Bf16`. `Div` relies on
//! the device's division, which OpenCL C bounds at 2.5 ULP rather than demanding correct
//! rounding; the device test checks it bit for bit on every device here.
//!
//! # Counted loops
//!
//! `For` follows the oracle: the three bounds are read once, the loop runs while the counter is
//! below `end` (above it for a negative step), and it stops rather than wrap when the next
//! counter value would leave the type. The advance sits in the C `for` header so `Continue`
//! still takes it, and it is computed in the unsigned type so a signed counter never overflows
//! in C. The counter and bounds are one of the four integer types. A zero step is a caller
//! precondition (the oracle traps; the emitted loop would not end). 0.47.0's `Stmt::For` has no
//! unroll hint, so none is spelled.
//!
//! # Atomics
//!
//! They are the OpenCL C 2.0 (Khronos) `atomic_fetch_*_explicit`, `atomic_exchange_explicit` and
//! `atomic_compare_exchange_strong_explicit`, on `int`, `uint`, `long` and `ulong`. The caller
//! opts in with [`OpenClC::with_c20_atomics`] and must build the program for OpenCL C 2.0
//! (`-cl-std=CL2.0`) on a device that reports it. Scope `Block` is `memory_scope_work_group` and
//! `Device` is `memory_scope_device`; every other scope, the `ClusterPeer`, `Peer` and `Multicast`
//! places, floats and 16-bit elements are declined.
//!
//! # Patent risk note
//!
//! Published on CireSnave's ruling of 2026-11-14 ("441: A.", board item 441), not a clearance.
//! Two patents were read against this emitter and found outside as read: US7627723B1 (NVIDIA,
//! atomic memory operators in a parallel processor; its independent claims need execution units
//! with address-range routing of the instruction) and US9513976B2 (Intel; a compare-and-swap
//! lock using a software parameter that emulates a full/empty bit). This module only spells
//! Khronos-defined OpenCL C 2.0 atomic builtins as text; it has neither element. A project that
//! builds a lowering with execution units, address routing or an emulated full/empty bit owns
//! that risk. See the PS1 row of `ROADMAP.md`.
//!
//! # Declined by name: NVIDIA-patent statements
//!
//! `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, `CopyBulk` (on an mbarrier) and `Tcgen05Commit` (it arrives on an
//! mbarrier) (NVIDIA US11803380B2), and `WgmmaWait` (NVIDIA US12204897B2, continuation
//! US12710961B2: a wait on a count of pending matrix-multiply operations), are
//! refused with a typed [`LowerError::UnsupportedKernel`] that names the statement and NVIDIA
//! US11803380B2. OpenCL C is vendor-neutral: if a non-NVIDIA-specific emitter lowered these to
//! hardware operations it could infringe, so this one lowers none of them (CireSnave
//! 2026-11-16: "we consider ourselves not infringing"). Not a clearance; not legal advice.
//!
//! # The kernel
//!
//! `__kernel void NAME(args)` in `KernelFn::params` order: a buffer is `__global T* bI` (`const`
//! for a `Read` buffer), a scalar is `T pI`. The buffer is addressed `bI[coord]`: rank 1, no view
//! strides, no base. The interpreter traps an out-of-bounds access and OpenCL C cannot, so a
//! kernel must keep its accesses inside its buffers (a precondition, not a check).

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fmt::Write as _;

use unpopped::backend::{GeneratedKernel, LowerError};
use unpopped::exec::{Extent, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, RoundMode, Saturation, ScalarExpr, UnaryOp};
use unpopped::kernel::{
    AtomicOp, BarrierKind, Block, KernelFn, MemoryOrder, ParamKind, Place, Stmt, check_kernel,
};
use unpopped_vocab::ElementKind;

use crate::OpenClC;

fn refuse(detail: impl Into<String>) -> LowerError {
    LowerError::UnsupportedKernel {
        detail: format!("opencl_c: {}", detail.into()),
    }
}

/// The C type, the atomic type and the unsigned C type of the same width for an integer element.
fn int_c(e: ElementKind) -> Option<(&'static str, &'static str, &'static str)> {
    match e {
        ElementKind::I32 => Some(("int", "atomic_int", "uint")),
        ElementKind::U32 => Some(("uint", "atomic_uint", "uint")),
        ElementKind::I64 => Some(("long", "atomic_long", "ulong")),
        ElementKind::U64 => Some(("ulong", "atomic_ulong", "ulong")),
        _ => None,
    }
}

fn bits(e: ElementKind) -> u32 {
    match e {
        ElementKind::I32 | ElementKind::U32 => 32,
        _ => 64,
    }
}

fn is_float(e: ElementKind) -> bool {
    matches!(e, ElementKind::F32 | ElementKind::F64)
}

fn ctype(e: ElementKind, what: &str) -> Result<&'static str, LowerError> {
    match e {
        ElementKind::F32 => Ok("float"),
        ElementKind::F64 => Ok("double"),
        _ => int_c(e).map(|c| c.0).ok_or_else(|| {
            refuse(format!(
                "{what} has element {e:?}; only I32, U32, I64, U64, F32 and F64 are spelled"
            ))
        }),
    }
}

fn order(o: MemoryOrder) -> &'static str {
    match o {
        MemoryOrder::Relaxed => "memory_order_relaxed",
        MemoryOrder::Acquire => "memory_order_acquire",
        MemoryOrder::Release => "memory_order_release",
        MemoryOrder::AcqRel => "memory_order_acq_rel",
        _ => "memory_order_seq_cst",
    }
}

/// The failure order of a compare-exchange: it may not release.
fn fail_order(o: MemoryOrder) -> &'static str {
    match o {
        MemoryOrder::Relaxed | MemoryOrder::Release => "memory_order_relaxed",
        MemoryOrder::Acquire | MemoryOrder::AcqRel => "memory_order_acquire",
        _ => "memory_order_seq_cst",
    }
}

fn scope(s: Scope) -> Result<&'static str, LowerError> {
    match s {
        Scope::Block => Ok("memory_scope_work_group"),
        Scope::Device => Ok("memory_scope_device"),
        other => Err(refuse(format!(
            "an atomic at scope {other:?}; only Block and Device are spelled"
        ))),
    }
}

struct Em<'a> {
    k: &'a KernelFn,
    atomics: bool,
    /// `F32` expressions are evaluated in `double` and rounded once per statement, as the oracle
    /// does (needs `cl_khr_fp64`; set by `OpenClC::with_fp64`).
    wide: bool,
    out: String,
    depth: usize,
    /// Helper functions the body calls, by name (see [`HELPERS`]); written once before the kernel.
    helpers: RefCell<BTreeSet<(&'static str, bool)>>,
}

impl Em<'_> {
    fn line(&mut self, t: &str) {
        for _ in 0..self.depth {
            self.out.push_str("    ");
        }
        self.out.push_str(t);
        self.out.push('\n');
    }

    fn local_elem(&self, id: u32) -> Result<ElementKind, LowerError> {
        self.k
            .locals
            .get(id as usize)
            .map(|d| d.elem)
            .ok_or_else(|| refuse(format!("local {id} is not declared")))
    }

    /// The element type of `e`, for the forms this emitter spells.
    fn ty(&self, e: &ScalarExpr) -> Result<ElementKind, LowerError> {
        match e {
            ScalarExpr::Local(i) => self.local_elem(*i),
            ScalarExpr::ConstInt(_, k) => Ok(*k),
            ScalarExpr::Const(_) if is_float(self.k.compute) => Ok(self.k.compute),
            ScalarExpr::Exec(_) => Ok(ElementKind::U32),
            ScalarExpr::Div(a, b) => {
                let (x, y) = (self.ty(a)?, self.ty(b)?);
                if x == y && is_float(x) {
                    Ok(x)
                } else {
                    Err(refuse("a Div other than of two floats of one type"))
                }
            }
            ScalarExpr::Unary(UnaryOp::Neg, x) => {
                let x = self.ty(x)?;
                if is_float(x) {
                    Ok(x)
                } else {
                    Err(refuse("a Neg of a non-float"))
                }
            }
            ScalarExpr::Unary(UnaryOp::Abs | UnaryOp::Sqr, x) => {
                let x = self.ty(x)?;
                if is_float(x) {
                    Ok(x)
                } else {
                    Err(refuse("an Abs or Sqr of a non-float"))
                }
            }
            ScalarExpr::Unary(op, x) if unary_kind(op).is_some() => {
                let x = self.ty(x)?;
                if is_float(x) {
                    Ok(x)
                } else {
                    Err(refuse("a float unary op of a non-float"))
                }
            }
            ScalarExpr::Select(c, a, b) => {
                self.ty(c)?;
                let (x, y) = (self.ty(a)?, self.ty(b)?);
                if x == y {
                    Ok(x)
                } else {
                    Err(refuse(format!("a Select of {x:?} and {y:?}")))
                }
            }
            ScalarExpr::Fma(a, b, c) => {
                let (x, y, z) = (self.ty(a)?, self.ty(b)?, self.ty(c)?);
                if x == y && y == z && is_float(x) {
                    Ok(x)
                } else {
                    Err(refuse("an Fma other than of three floats of one type"))
                }
            }
            ScalarExpr::Cast(x, spec) => {
                if is_float(spec.to) {
                    // Integer to float: one conversion in the mode's rounding (`NearestAway`
                    // has no OpenCL conversion). Float to float is not spelled.
                    let from = self.ty(x)?;
                    if spec.sat != Saturation::Wrap {
                        return Err(refuse("a cast to a float that is not Wrap"));
                    }
                    // F32 to F64 is exact (any mode); F64 to F32 and integer to float round.
                    if spec.round == RoundMode::NearestAway
                        && !(from == ElementKind::F32 && spec.to == ElementKind::F64)
                    {
                        return Err(refuse("a float cast rounding ties away"));
                    }
                    return Ok(spec.to);
                }
                if is_float(self.ty(x)?) {
                    // The oracle's float-to-integer cast rounds by the mode, clamps and maps
                    // NaN to 0; only `Clamp` is defined for it.
                    if spec.sat != Saturation::Clamp {
                        return Err(refuse("a cast from a float that is not Clamp"));
                    }
                    return int_c(spec.to)
                        .map(|_| spec.to)
                        .ok_or_else(|| refuse("a cast to this type"));
                }
                if spec.sat != Saturation::Wrap {
                    return Err(refuse("a saturating cast"));
                }
                int_c(spec.to)
                    .map(|_| spec.to)
                    .ok_or_else(|| refuse("a cast to this type"))
            }
            ScalarExpr::Add(a, b) | ScalarExpr::Sub(a, b) | ScalarExpr::Mul(a, b) => {
                let (x, y) = (self.ty(a)?, self.ty(b)?);
                if x == y {
                    Ok(x)
                } else {
                    Err(refuse(format!("operands of {x:?} and {y:?}")))
                }
            }
            ScalarExpr::Binary(_, a, b) => {
                let (x, y) = (self.ty(a)?, self.ty(b)?);
                if x != y {
                    return Err(refuse(format!("operands of {x:?} and {y:?}")));
                }
                Ok(x)
            }
            other => Err(refuse(format!("the expression form {other:?}"))),
        }
    }

    /// How many rounding operations an `F32` expression performs (`Fma` counts as two: the
    /// oracle's `f64` fused multiply-add rounded to `f32` is not the device's `f32` one).
    fn f32_roundings(&self, e: &ScalarExpr) -> usize {
        let is32 = |x: &ScalarExpr| self.ty(x).ok() == Some(ElementKind::F32);
        match e {
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b) => {
                usize::from(is32(e)) + self.f32_roundings(a) + self.f32_roundings(b)
            }
            ScalarExpr::Fma(a, b, c) => {
                2 * usize::from(is32(e))
                    + self.f32_roundings(a)
                    + self.f32_roundings(b)
                    + self.f32_roundings(c)
            }
            // A square rounds once, like a multiply; `Abs` and `Neg` only change a sign bit.
            ScalarExpr::Unary(UnaryOp::Sqr, x) => usize::from(is32(e)) + self.f32_roundings(x),
            // A device built-in rounds once (within its own ULP bound, not the oracle's); the
            // exact ones (floor, relu, ...) do not round.
            ScalarExpr::Unary(op, x) if matches!(unary_kind(op), Some(UnaryKind::Rounded(_))) => {
                usize::from(is32(e)) + self.f32_roundings(x)
            }
            // An integer converted to `F32` rounds once; a float-to-integer cast does not.
            ScalarExpr::Cast(x, _) => usize::from(is32(e)) + self.f32_roundings(x),
            ScalarExpr::Unary(_, x) => self.f32_roundings(x),
            // The chosen arm is moved, not computed on; the condition is counted too, because
            // its own rounding decides which arm is chosen.
            ScalarExpr::Select(c, a, b) => {
                self.f32_roundings(c) + self.f32_roundings(a) + self.f32_roundings(b)
            }
            // `Pow` and `Atan2` are device built-ins that round once; `Copysign`, `RemTrunc`
            // (`fmod`) and the comparisons are exact.
            ScalarExpr::Binary(op, a, b) => {
                usize::from(is32(e) && matches!(op, BinaryOp::Pow | BinaryOp::Atan2))
                    + self.f32_roundings(a)
                    + self.f32_roundings(b)
            }
            _ => 0,
        }
    }

    fn expr(&self, e: &ScalarExpr) -> Result<String, LowerError> {
        if !self.wide && self.f32_roundings(e) > 1 {
            return Err(refuse(
                "an F32 expression with more than one rounding operation needs with_fp64 \
                 (the oracle evaluates it at f64 and rounds once per statement)",
            ));
        }
        use unpopped::exec::ExecId;
        Ok(match e {
            ScalarExpr::Local(i) if self.wide && self.local_elem(*i)? == ElementKind::F32 => {
                format!("((double)l{i})")
            }
            ScalarExpr::Local(i) => format!("l{i}"),
            ScalarExpr::ConstInt(v, k) => {
                let (c, _, _) =
                    int_c(*k).ok_or_else(|| refuse("an integer constant of this type"))?;
                match k {
                    ElementKind::I32 => format!("(({c}){v})"),
                    ElementKind::U32 => format!("(({c}){}u)", *v as u32),
                    ElementKind::I64 => format!("(({c}){v}l)"),
                    _ => format!("(({c}){}ul)", *v as u64),
                }
            }
            ScalarExpr::Const(v) => {
                if !v.is_finite() {
                    return Err(refuse("a non-finite constant"));
                }
                let c = ctype(self.k.compute, "a constant")?;
                // The shortest decimal that reads back to the same f64, then one conversion
                // to the compute type (round to nearest even), as the oracle does.
                if self.k.compute == ElementKind::F32 {
                    // The oracle keeps the f64 value inside an expression it evaluates at f64.
                    if self.wide {
                        format!("((double){v:?})")
                    } else if f64::from(*v as f32) == *v {
                        format!("((float){v:?})")
                    } else {
                        return Err(refuse(
                            "an F32 constant that is not exactly an f32 needs with_fp64                              (the oracle uses its f64 value)",
                        ));
                    }
                } else {
                    format!("(({c}){v:?})")
                }
            }
            ScalarExpr::Div(a, b) => {
                if self.ty(e)? == ElementKind::F32 {
                    // OpenCL C float division is within 2.5 ULP, not correctly rounded, and a
                    // compiler may narrow a double division of widened floats back to it:
                    // NVIDIA's is 1 ULP off the oracle here, with or without with_fp64.
                    return Err(refuse(
                        "an F32 Div (the device's float division is not correctly rounded)",
                    ));
                }
                format!("(({}) / ({}))", self.expr(a)?, self.expr(b)?)
            }
            ScalarExpr::Unary(UnaryOp::Neg, x) => {
                self.ty(e)?;
                format!("(-({}))", self.expr(x)?)
            }
            ScalarExpr::Unary(UnaryOp::Abs, x) => {
                self.ty(e)?;
                format!("fabs({})", self.expr(x)?)
            }
            ScalarExpr::Unary(UnaryOp::Sqr, x) => {
                let t = self.ty(e)?;
                format!("{}({})", self.helper("ocl_sqr", t), self.expr(x)?)
            }
            // The OpenCL C built-in of the evaluation type (`double` when wide). The `Rounded`
            // ones are not bit-exact (the spec allows several ULP, so the device test states a
            // tolerance per op); the `Exact` ones and the helpers are bit for bit.
            ScalarExpr::Unary(op, x) if unary_kind(op).is_some() => {
                let t = self.ty(e)?;
                let inner = self.expr(x)?;
                match unary_kind(op) {
                    Some(UnaryKind::Rounded(f) | UnaryKind::Exact(f)) => format!("{f}({inner})"),
                    Some(UnaryKind::Helper(h)) => format!("{}({inner})", self.helper(h, t)),
                    None => unreachable!("guarded"),
                }
            }
            // The condition is tested `!= 0`: NaN is true and -0.0 is false, as in the oracle.
            // The chosen arm is moved, not computed on.
            ScalarExpr::Select(c, a, b) => {
                self.ty(e)?;
                format!(
                    "((({}) != 0) ? ({}) : ({}))",
                    self.expr(c)?,
                    self.expr(a)?,
                    self.expr(b)?
                )
            }
            ScalarExpr::Fma(a, b, c) => {
                self.ty(e)?;
                format!(
                    "fma(({}), ({}), ({}))",
                    self.expr(a)?,
                    self.expr(b)?,
                    self.expr(c)?
                )
            }
            ScalarExpr::Exec(id) => match id {
                ExecId::ThreadIdx(a) => format!("((uint)get_local_id({a}))"),
                ExecId::BlockIdx(a) => format!("((uint)get_group_id({a}))"),
                ExecId::BlockDim(a) => format!("((uint)get_local_size({a}))"),
                ExecId::GridDim(a) => format!("((uint)get_num_groups({a}))"),
                other => return Err(refuse(format!("the id {}", other.name()))),
            },
            // Signed overflow is undefined in C: wrap through the unsigned type of the same width.
            ScalarExpr::Add(a, b) | ScalarExpr::Sub(a, b) | ScalarExpr::Mul(a, b) => {
                let t = self.ty(e)?;
                let op = match e {
                    ScalarExpr::Add(..) => '+',
                    ScalarExpr::Sub(..) => '-',
                    _ => '*',
                };
                if is_float(t) {
                    return Ok(format!("(({}) {op} ({}))", self.expr(a)?, self.expr(b)?));
                }
                let (c, _, u) = int_c(t).ok_or_else(|| refuse("an arithmetic type"))?;
                format!(
                    "(({c})(({u})({}) {op} ({u})({})))",
                    self.expr(a)?,
                    self.expr(b)?
                )
            }
            ScalarExpr::Cast(x, spec) => {
                let from = self.ty(x)?;
                if is_float(spec.to) {
                    let inner = self.expr(x)?;
                    if from == ElementKind::F32 && spec.to == ElementKind::F64 {
                        return Ok(format!("convert_double({inner})"));
                    }
                    let conv = match (spec.to, spec.round) {
                        (ElementKind::F32, RoundMode::NearestEven) => "convert_float_rte",
                        (ElementKind::F32, RoundMode::TowardZero) => "convert_float_rtz",
                        (ElementKind::F32, RoundMode::TowardNegInf) => "convert_float_rtn",
                        (ElementKind::F32, RoundMode::TowardPosInf) => "convert_float_rtp",
                        (_, RoundMode::NearestEven) => "convert_double_rte",
                        (_, RoundMode::TowardZero) => "convert_double_rtz",
                        (_, RoundMode::TowardNegInf) => "convert_double_rtn",
                        (_, RoundMode::TowardPosInf) => "convert_double_rtp",
                        _ => {
                            return Err(refuse(
                                "a cast rounding mode this emitter has not been taught",
                            ));
                        }
                    };
                    return Ok(format!("{conv}({inner})"));
                }
                let (c, _, u) = int_c(spec.to).ok_or_else(|| refuse("a cast to this type"))?;
                let inner = self.expr(x)?;
                if is_float(from) {
                    // Round with the built-in of the mode (exact), then clamp and map NaN to 0
                    // in a helper, so no out-of-range float-to-integer conversion is executed.
                    let round = match spec.round {
                        RoundMode::NearestEven => "rint",
                        RoundMode::TowardZero => "trunc",
                        RoundMode::TowardNegInf => "floor",
                        RoundMode::TowardPosInf => "ceil",
                        RoundMode::NearestAway => "round",
                        _ => {
                            return Err(refuse(
                                "a cast rounding mode this emitter has not been taught",
                            ));
                        }
                    };
                    let h = match spec.to {
                        ElementKind::I32 => "ocl_f2i_i32",
                        ElementKind::U32 => "ocl_f2i_u32",
                        ElementKind::I64 => "ocl_f2i_i64",
                        _ => "ocl_f2i_u64",
                    };
                    return Ok(format!("{}({round}({inner}))", self.helper(h, from)));
                }
                if bits(spec.to) > bits(from) {
                    format!("(({c})({inner}))")
                } else {
                    // Narrowing or same width: modular, through the unsigned type.
                    format!("(({c})(({u})({inner})))")
                }
            }
            ScalarExpr::Binary(op, a, b) => {
                let t = self.ty(e)?;
                let c = ctype(t, "a comparison")?;
                let (x, y) = (self.expr(a)?, self.expr(b)?);
                if is_float(t) {
                    // Unpopped's reference rules, not OpenCL's `fmax`/`fmin` (which leave the
                    // sign of a zero result unspecified): `Max`/`Min` propagate NaN,
                    // `FmaxIeee`/`FminIeee` suppress it, a tie takes the first operand.
                    let name = match op {
                        BinaryOp::Max => Some("ocl_max"),
                        BinaryOp::Min => Some("ocl_min"),
                        BinaryOp::FmaxIeee => Some("ocl_fmax"),
                        BinaryOp::FminIeee => Some("ocl_fmin"),
                        _ => None,
                    };
                    if let Some(name) = name {
                        return Ok(format!("{}({x}, {y})", self.helper(name, t)));
                    }
                    // Device built-ins. `Pow` and `Atan2` are within OpenCL's ULP bounds, not
                    // the oracle's bits; `Copysign` and `fmod` (`RemTrunc`) are exact.
                    let builtin = match op {
                        BinaryOp::Pow => Some("pow"),
                        BinaryOp::Atan2 => Some("atan2"),
                        BinaryOp::Copysign => Some("copysign"),
                        BinaryOp::RemTrunc => Some("fmod"),
                        _ => None,
                    };
                    if let Some(f) = builtin {
                        return Ok(format!("{f}(({x}), ({y}))"));
                    }
                    if matches!(op, BinaryOp::Rem) {
                        // The floored remainder `a - floor(a / b) * b` needs a division, and a
                        // float division is declined (see `Div`); only F64 is spelled.
                        if t == ElementKind::F32 {
                            return Err(refuse(
                                "an F32 Rem (it divides, and the device's float division is not correctly rounded)",
                            ));
                        }
                        return Ok(format!("{}({x}, {y})", self.helper("ocl_rem", t)));
                    }
                    if matches!(op, BinaryOp::Nextafter) {
                        return Err(refuse(
                            "a float Nextafter (it steps the compute type's own lattice, which an F32 expression evaluated at double would not)",
                        ));
                    }
                }
                let cmp = |o: &str| format!("(({x}) {o} ({y}) ? ({c})1 : ({c})0)");
                match op {
                    BinaryOp::CmpEq => cmp("=="),
                    BinaryOp::CmpNe => cmp("!="),
                    BinaryOp::CmpLt => cmp("<"),
                    BinaryOp::CmpLe => cmp("<="),
                    BinaryOp::CmpGt => cmp(">"),
                    BinaryOp::CmpGe => cmp(">="),
                    BinaryOp::Max => format!("max(({x}), ({y}))"),
                    BinaryOp::Min => format!("min(({x}), ({y}))"),
                    other => return Err(refuse(format!("the binary op {other:?}"))),
                }
            }
            other => return Err(refuse(format!("the expression form {other:?}"))),
        })
    }

    /// The name of the helper `base` for an expression of type `t`, and a request to emit it. An
    /// `F32` expression is a `double` one when evaluated wide.
    fn helper(&self, base: &'static str, t: ElementKind) -> String {
        let double = t == ElementKind::F64 || self.wide;
        self.helpers.borrow_mut().insert((base, double));
        format!("{base}_{}", if double { 'd' } else { 'f' })
    }

    /// The element of `place` as a C lvalue, and its element type and address space.
    fn place(&self, p: &Place) -> Result<(String, ElementKind, &'static str), LowerError> {
        match p {
            Place::Logical { buffer, coords } => {
                let param = self
                    .k
                    .params
                    .get(*buffer as usize)
                    .ok_or_else(|| refuse("a buffer index outside the parameters"))?;
                let ParamKind::Buffer { elem, .. } = &param.kind else {
                    return Err(refuse("a place on a scalar parameter"));
                };
                let [c] = coords.as_slice() else {
                    return Err(refuse("a buffer access with other than one coordinate"));
                };
                Ok((
                    format!("b{buffer}[(long)({})]", self.expr(c)?),
                    *elem,
                    "__global",
                ))
            }
            Place::Alloc { alloc, index } => {
                let a = self
                    .k
                    .allocs
                    .get(*alloc as usize)
                    .ok_or_else(|| refuse("an allocation index outside the allocations"))?;
                Ok((
                    format!("a{alloc}[(long)({})]", self.expr(index)?),
                    a.elem,
                    "__local",
                ))
            }
            other => Err(refuse(format!(
                "the place {}: only buffers and Shared arrays are spelled",
                place_name(other)
            ))),
        }
    }

    fn block(&mut self, b: &Block) -> Result<(), LowerError> {
        for s in &b.0 {
            self.stmt(s)?;
        }
        Ok(())
    }

    fn nested(&mut self, head: &str, b: &Block) -> Result<(), LowerError> {
        self.line(head);
        self.depth += 1;
        self.block(b)?;
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    /// A counted loop, spelled to match the oracle: the three bounds are read once, the loop
    /// runs while the counter is below `end` (above it for a negative step), and it stops
    /// rather than wrap when the next counter value would leave the type. The advance is in the
    /// `for` header so `continue` still takes it, and it is done in the unsigned type so a
    /// signed counter never overflows in C. A zero step is a caller precondition (the oracle
    /// traps; here it would not end).
    fn for_loop(
        &mut self,
        local: u32,
        bounds: [&ScalarExpr; 3],
        body: &Block,
    ) -> Result<(), LowerError> {
        let e = self.local_elem(local)?;
        let (c, _, u) = int_c(e).ok_or_else(|| refuse("a For counter that is not an integer"))?;
        for b in bounds {
            if self.ty(b)? != e {
                return Err(refuse("For bounds of another type than the counter"));
            }
        }
        let signed = matches!(e, ElementKind::I32 | ElementKind::I64);
        let (max, min) = match e {
            ElementKind::I32 => ("INT_MAX", "INT_MIN"),
            ElementKind::I64 => ("LONG_MAX", "LONG_MIN"),
            ElementKind::U32 => ("UINT_MAX", "0"),
            _ => ("ULONG_MAX", "0"),
        };
        let (s0, e0, s1) = (
            self.expr(bounds[0])?,
            self.expr(bounds[1])?,
            self.expr(bounds[2])?,
        );
        let (l, en, st, dn) = (
            format!("l{local}"),
            format!("fe{local}"),
            format!("fs{local}"),
            format!("fd{local}"),
        );
        self.line("{");
        self.depth += 1;
        self.line(&format!("{c} {en} = {e0};"));
        self.line(&format!("{c} {st} = {s1};"));
        self.line(&format!("int {dn} = 0;"));
        let (cond, stop) = if signed {
            (
                format!("({st} > 0 ? {l} < {en} : {l} > {en})"),
                format!("({st} > 0 ? {l} > {max} - {st} : {l} < {min} - {st})"),
            )
        } else {
            (format!("{l} < {en}"), format!("{l} > {max} - {st}"))
        };
        let head = format!(
            "for ({l} = {s0}; {dn} == 0 && {cond}; {dn} = {stop}, {l} = ({c})(({u}){l} + ({u}){st})) {{"
        );
        self.nested(&head, body)?;
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::Let { local, init } | Stmt::Assign { local, value: init } => {
                let c = ctype(self.local_elem(*local)?, "a local")?;
                let v = self.expr(init)?;
                self.line(&format!("l{local} = ({c})({v});"));
            }
            Stmt::Load { local, place } => {
                let c = ctype(self.local_elem(*local)?, "a local")?;
                let (lv, ..) = self.place(place)?;
                self.line(&format!("l{local} = ({c})({lv});"));
            }
            Stmt::Store { place, value } => {
                let (lv, e, _) = self.place(place)?;
                let c = ctype(e, "a stored element")?;
                let v = self.expr(value)?;
                self.line(&format!("{lv} = ({c})({v});"));
            }
            Stmt::If { cond, then, els } => {
                let c = self.expr(cond)?;
                self.line(&format!("if (({c}) != 0) {{"));
                self.depth += 1;
                self.block(then)?;
                self.depth -= 1;
                if !els.0.is_empty() {
                    self.line("} else {");
                    self.depth += 1;
                    self.block(els)?;
                    self.depth -= 1;
                }
                self.line("}");
            }
            Stmt::While { cond, body } => {
                let c = self.expr(cond)?;
                self.nested(&format!("while (({c}) != 0) {{"), body)?;
            }
            Stmt::For {
                local,
                start,
                end,
                step,
                body,
                // The unroll hint is non-semantic (every value gives the same result); not lowered.
                ..
            } => self.for_loop(*local, [start, end, step], body)?,
            Stmt::Break => self.line("break;"),
            Stmt::Continue => self.line("continue;"),
            Stmt::Return => self.line("return;"),
            Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Full,
            } => self.line("barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);"),
            Stmt::Barrier { scope, .. } => {
                return Err(refuse(format!(
                    "a barrier at scope {scope:?}; only Block is spelled"
                )));
            }
            Stmt::AtomicRmw {
                op,
                place,
                value,
                old,
                order: mo,
                scope: sc,
            } => {
                self.need_atomics("AtomicRmw")?;
                let (lv, e, space) = self.place(place)?;
                let (c, at, _) = int_c(e).ok_or_else(|| {
                    refuse(format!("an atomic on {e:?}; only I32, U32, I64 and U64"))
                })?;
                let f = match op {
                    AtomicOp::Add => "atomic_fetch_add_explicit",
                    AtomicOp::Max => "atomic_fetch_max_explicit",
                    AtomicOp::Min => "atomic_fetch_min_explicit",
                    AtomicOp::And => "atomic_fetch_and_explicit",
                    AtomicOp::Or => "atomic_fetch_or_explicit",
                    AtomicOp::Xor => "atomic_fetch_xor_explicit",
                    AtomicOp::Exchange => "atomic_exchange_explicit",
                    _ => return Err(refuse("an atomic op this emitter has not been taught")),
                };
                let call = format!(
                    "{f}((volatile {space} {at}*)&{lv}, ({c})({}), {}, {})",
                    self.expr(value)?,
                    order(*mo),
                    scope(*sc)?
                );
                match old {
                    Some(l) => {
                        let lc = ctype(self.local_elem(*l)?, "the old-value local")?;
                        self.line(&format!("l{l} = ({lc})({call});"));
                    }
                    None => self.line(&format!("{call};")),
                }
            }
            Stmt::AtomicCas {
                place,
                expected,
                desired,
                old,
                order: mo,
                scope: sc,
            } => {
                self.need_atomics("AtomicCas")?;
                let (lv, e, space) = self.place(place)?;
                let (c, at, _) = int_c(e).ok_or_else(|| {
                    refuse(format!("an atomic on {e:?}; only I32, U32, I64 and U64"))
                })?;
                self.line("{");
                self.depth += 1;
                self.line(&format!("{c} cas = ({c})({});", self.expr(expected)?));
                self.line(&format!(
                    "atomic_compare_exchange_strong_explicit((volatile {space} {at}*)&{lv}, &cas, ({c})({}), {}, {}, {});",
                    self.expr(desired)?,
                    order(*mo),
                    fail_order(*mo),
                    scope(*sc)?
                ));
                if let Some(l) = old {
                    let lc = ctype(self.local_elem(*l)?, "the old-value local")?;
                    self.line(&format!("l{l} = ({lc})(cas);"));
                }
                self.depth -= 1;
                self.line("}");
            }
            other => {
                if let Some((name, patent)) = patent_declined(other) {
                    return Err(refuse(format!(
                        "the statement {name}: declined by name. NVIDIA {patent}: OpenCL C is \
                         vendor-neutral and this emitter lowers none of these statements to a hardware operation"
                    )));
                }
                return Err(refuse(format!("the statement {}", stmt_name(other))));
            }
        }
        Ok(())
    }

    fn need_atomics(&self, what: &str) -> Result<(), LowerError> {
        if self.atomics {
            Ok(())
        } else {
            Err(refuse(format!(
                "{what} without OpenClC::with_c20_atomics(true): atomics need an OpenCL C 2.0 device"
            )))
        }
    }
}

fn place_name(p: &Place) -> &'static str {
    match p {
        Place::Logical { .. } => "Logical",
        Place::Alloc { .. } => "Alloc",
        Place::ClusterPeer { .. } => "ClusterPeer",
        Place::Peer { .. } => "Peer",
        Place::Multicast { .. } => "Multicast",
        _ => "an unknown place",
    }
}

/// The NVIDIA-patent statements this emitter declines by name, each with its patent (CireSnave
/// 2026-11-16: "we consider ourselves not infringing" by never lowering them to hardware
/// operations from a vendor-neutral emitter, and by telling non-NVIDIA emitter builders not to).
/// `US11803380B2` covers mbarrier operations (`Tcgen05Commit` arrives on an mbarrier and is cited
/// as an extension of it per the PM, not read separately);
/// `US12204897B2` (continuation `US12710961B2`) covers a wait on a count of pending
/// matrix-multiply operations (`WgmmaWait`). An arm over the real `Stmt` variants (Unpopped 0.56.2).
fn patent_declined(s: &Stmt) -> Option<(&'static str, &'static str)> {
    const MBAR: &str = "US11803380B2";
    const ASYNC: &str = "US12118382B2 and US11294713B2 (board 434 C)";
    const COLLECTIVE: &str = "US8539204B2 (board 434 C)";
    match s {
        Stmt::MbarInit { .. } => Some(("MbarInit", MBAR)),
        Stmt::MbarArrive { .. } => Some(("MbarArrive", MBAR)),
        Stmt::MbarCompleteTx { .. } => Some(("MbarCompleteTx", MBAR)),
        Stmt::MbarWait { .. } => Some(("MbarWait", MBAR)),
        Stmt::CopyBulk { .. } => Some(("CopyBulk", MBAR)),
        Stmt::Tcgen05Commit { .. } => Some(("Tcgen05Commit", MBAR)),
        Stmt::WgmmaWait { .. } => Some(("WgmmaWait", "US12204897B2 (continuation US12710961B2)")),
        // Board 434 C (CireSnave 2026-11-13): async scheduling and collectives are declined by name.
        Stmt::CopyAsync { .. } => Some(("CopyAsync", ASYNC)),
        Stmt::AsyncCommit => Some(("AsyncCommit", ASYNC)),
        Stmt::AsyncWait { .. } => Some(("AsyncWait", ASYNC)),
        Stmt::Collective { .. } => Some(("Collective", COLLECTIVE)),
        Stmt::CollectiveTuple { .. } => Some(("CollectiveTuple", COLLECTIVE)),
        _ => None,
    }
}

fn stmt_name(s: &Stmt) -> &'static str {
    match s {
        Stmt::MaskedLoad { .. } => "MaskedLoad",
        Stmt::MaskedStore { .. } => "MaskedStore",
        Stmt::MulticastLoadReduce { .. } => "MulticastLoadReduce",
        Stmt::Fence { .. } => "Fence",
        Stmt::Mma { .. } => "Mma",
        Stmt::Call { .. } => "Call",
        Stmt::CallTuple { .. } => "CallTuple",
        Stmt::ExternCall { .. } => "ExternCall",
        _ => "a statement this emitter has not been taught",
    }
}

/// How a float unary op is spelled: an OpenCL built-in that rounds once (within its own ULP
/// bound), one that is exact on every input, or a helper function.
#[derive(Clone, Copy)]
enum UnaryKind {
    Rounded(&'static str),
    Exact(&'static str),
    Helper(&'static str),
}

/// The spelling of a float unary op, `None` for the ones declined (`Recip`, `Rsqrt`, `Sigmoid`,
/// `Silu`, `Erf`, `Erfc`, `Gelu`, `Lgamma`: a float division, or an erf/gamma whose oracle
/// accuracy is unchecked).
fn unary_kind(op: &UnaryOp) -> Option<UnaryKind> {
    use UnaryKind::{Exact, Helper, Rounded};
    Some(match op {
        UnaryOp::Sqrt => Rounded("sqrt"),
        UnaryOp::Exp => Rounded("exp"),
        UnaryOp::Log => Rounded("log"),
        UnaryOp::Tanh => Rounded("tanh"),
        UnaryOp::Sin => Rounded("sin"),
        UnaryOp::Cos => Rounded("cos"),
        UnaryOp::Exp2 => Rounded("exp2"),
        UnaryOp::Expm1 => Rounded("expm1"),
        UnaryOp::Log2 => Rounded("log2"),
        UnaryOp::Log10 => Rounded("log10"),
        UnaryOp::Log1p => Rounded("log1p"),
        UnaryOp::Sinh => Rounded("sinh"),
        UnaryOp::Cosh => Rounded("cosh"),
        UnaryOp::Tan => Rounded("tan"),
        UnaryOp::Asin => Rounded("asin"),
        UnaryOp::Acos => Rounded("acos"),
        UnaryOp::Atan => Rounded("atan"),
        UnaryOp::Asinh => Rounded("asinh"),
        UnaryOp::Acosh => Rounded("acosh"),
        UnaryOp::Atanh => Rounded("atanh"),
        UnaryOp::Cbrt => Rounded("cbrt"),
        UnaryOp::Floor => Exact("floor"),
        UnaryOp::Ceil => Exact("ceil"),
        UnaryOp::Trunc => Exact("trunc"),
        // Round to nearest, ties to even (the default rounding mode), as the oracle does.
        UnaryOp::Round => Exact("rint"),
        UnaryOp::Relu => Helper("ocl_relu"),
        UnaryOp::Sign => Helper("ocl_sign"),
        UnaryOp::Step => Helper("ocl_step"),
        _ => return None,
    })
}

/// Helper functions a kernel carries when its body calls them; `REAL` is `float` or `double` and
/// `_S` is `_f` or `_d`. They follow Unpopped's normative reference, not OpenCL's `fmax`/`fmin`,
/// which leave the sign of a zero result unspecified for `(-0, +0)`: `Max`/`Min` move a NaN
/// through, `FmaxIeee`/`FminIeee` suppress it, and on a tie the first operand wins.
const HELPERS: &[(&str, &str)] = &[
    (
        "ocl_f2i_i32",
        "inline int ocl_f2i_i32_S(REAL r) {
    if (isnan(r)) { return 0; }
    if (r >= (REAL)2147483648.0) { return INT_MAX; }
    if (r < (REAL)-2147483648.0) { return INT_MIN; }
    return (int)r;
}
",
    ),
    (
        "ocl_f2i_u32",
        "inline uint ocl_f2i_u32_S(REAL r) {
    if (isnan(r)) { return 0u; }
    if (r >= (REAL)4294967296.0) { return UINT_MAX; }
    if (r < (REAL)0.0) { return 0u; }
    return (uint)r;
}
",
    ),
    (
        "ocl_f2i_i64",
        "inline long ocl_f2i_i64_S(REAL r) {
    if (isnan(r)) { return 0l; }
    if (r >= (REAL)9223372036854775808.0) { return LONG_MAX; }
    if (r < (REAL)-9223372036854775808.0) { return LONG_MIN; }
    return (long)r;
}
",
    ),
    (
        "ocl_f2i_u64",
        "inline ulong ocl_f2i_u64_S(REAL r) {
    if (isnan(r)) { return 0ul; }
    if (r >= (REAL)18446744073709551616.0) { return ULONG_MAX; }
    if (r < (REAL)0.0) { return 0ul; }
    return (ulong)r;
}
",
    ),
    (
        "ocl_rem",
        "inline REAL ocl_rem_S(REAL a, REAL b) { return a - floor(a / b) * b; }
",
    ),
    (
        "ocl_relu",
        "inline REAL ocl_relu_S(REAL x) { return x < 0 ? (REAL)0 : x; }
",
    ),
    (
        "ocl_sign",
        "inline REAL ocl_sign_S(REAL x) { return x > 0 ? (REAL)1 : (x < 0 ? (REAL)-1 : (REAL)0); }
",
    ),
    (
        "ocl_step",
        "inline REAL ocl_step_S(REAL x) { return x > 0 ? (REAL)1 : (REAL)0; }
",
    ),
    (
        "ocl_sqr",
        "inline REAL ocl_sqr_S(REAL x) { return x * x; }
",
    ),
    (
        "ocl_max",
        "inline REAL ocl_max_S(REAL a, REAL b) {
    if (isnan(a)) { return a; }
    if (isnan(b)) { return b; }
    return a >= b ? a : b;
}
",
    ),
    (
        "ocl_min",
        "inline REAL ocl_min_S(REAL a, REAL b) {
    if (isnan(a)) { return a; }
    if (isnan(b)) { return b; }
    return a <= b ? a : b;
}
",
    ),
    (
        "ocl_fmax",
        "inline REAL ocl_fmax_S(REAL a, REAL b) {
    if (isnan(a)) { return b; }
    if (isnan(b)) { return a; }
    return a >= b ? a : b;
}
",
    ),
    (
        "ocl_fmin",
        "inline REAL ocl_fmin_S(REAL a, REAL b) {
    if (isnan(a)) { return b; }
    if (isnan(b)) { return a; }
    return a <= b ? a : b;
}
",
    ),
];

/// The source of the helpers `used` names, in a stable order.
fn helper_source(used: &BTreeSet<(&'static str, bool)>) -> String {
    let mut s = String::new();
    for (base, double) in used {
        let (_, template) = HELPERS
            .iter()
            .find(|(n, _)| n == base)
            .expect("a helper that is requested exists");
        let (real, suffix) = if *double {
            ("double", "_d")
        } else {
            ("float", "_f")
        };
        s.push_str(
            &template
                .replace("REAL", real)
                .replace("_S(", &format!("{suffix}(")),
        );
    }
    s
}

fn is_ident(s: &str) -> bool {
    let mut c = s.chars();
    c.next()
        .is_some_and(|f| f.is_ascii_alphabetic() || f == '_')
        && c.all(|x| x.is_ascii_alphanumeric() || x == '_')
}

pub(crate) fn lower_kernel(oc: &OpenClC, k: &KernelFn) -> Result<GeneratedKernel, LowerError> {
    check_kernel(k).map_err(|e| refuse(format!("kernel `{}` is ill-formed: {e:?}", k.name)))?;
    if !is_ident(&k.name) {
        return Err(refuse(format!(
            "the kernel name `{}` is not an identifier",
            k.name
        )));
    }
    let Some(l) = k.launch.as_ref() else {
        return Err(refuse("a kernel without a launch shape"));
    };
    if l.subgroup.is_some() || l.cluster.is_some() || l.devices != 1 {
        return Err(refuse(
            "a launch with a subgroup, a cluster or several devices",
        ));
    }
    ctype(k.compute, "the compute dtype")?;
    let mut used = vec![k.compute];
    used.extend(k.locals.iter().map(|d| d.elem));
    used.extend(k.allocs.iter().map(|a| a.elem));
    for p in &k.params {
        match &p.kind {
            ParamKind::Buffer { elem, .. } => used.push(*elem),
            ParamKind::Scalar { ty, .. } => used.push(*ty),
            _ => {}
        }
    }
    if used.contains(&ElementKind::F64) && !oc.fp64 {
        return Err(refuse(
            "an F64 element needs a device with cl_khr_fp64 (OpenClC::with_fp64)",
        ));
    }

    let mut head = String::new();
    let mut scalars = Vec::new();
    for (i, p) in k.params.iter().enumerate() {
        if !head.is_empty() {
            head.push_str(", ");
        }
        match &p.kind {
            ParamKind::Buffer { elem, role, rank } => {
                if *rank != Some(1) {
                    return Err(refuse(format!("buffer `{}` is not rank 1", p.name)));
                }
                let c = ctype(*elem, &format!("buffer `{}`", p.name))?;
                let q = if matches!(role, unpopped::kernel::BufferRole::Read) {
                    "const "
                } else {
                    ""
                };
                let _ = write!(head, "__global {q}{c}* b{i}");
            }
            ParamKind::Scalar { ty, local } => {
                let c = ctype(*ty, &format!("scalar `{}`", p.name))?;
                let _ = write!(head, "{c} p{i}");
                scalars.push((*local, i));
            }
            _ => return Err(refuse("a parameter kind this emitter has not been taught")),
        }
    }
    for (i, d) in k.locals.iter().enumerate() {
        if d.lanes != 1 {
            return Err(refuse(format!("local {i} is packed")));
        }
        ctype(d.elem, &format!("local {i}"))?;
    }

    let mut em = Em {
        k,
        atomics: oc.c20_atomics(),
        wide: oc.fp64,
        out: String::new(),
        depth: 1,
        helpers: RefCell::new(BTreeSet::new()),
    };
    for (i, a) in k.allocs.iter().enumerate() {
        let c = ctype(a.elem, &format!("array `{}`", a.name))?;
        if a.space != MemorySpace::Shared {
            return Err(refuse(format!("array `{}` is not Shared", a.name)));
        }
        let Extent::Static(n) = a.extent else {
            return Err(refuse(format!("array `{}` has a runtime extent", a.name)));
        };
        em.line(&format!("__local {c} a{i}[{n}];"));
    }
    for (i, d) in k.locals.iter().enumerate() {
        let c = ctype(d.elem, "a local")?;
        match scalars.iter().find(|(l, _)| *l as usize == i) {
            Some((_, p)) => em.line(&format!("{c} l{i} = p{p};")),
            None => em.line(&format!("{c} l{i} = ({c})0;")),
        }
    }
    em.block(&k.body)?;

    let mut s = String::new();
    let _ = writeln!(
        s,
        "// Built from a kernel-layer KernelFn; atomics need OpenCL C 2.0 (-cl-std=CL2.0)."
    );
    if used.contains(&ElementKind::F64) {
        s.push_str("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
    }
    if oc.fp64 && !used.contains(&ElementKind::F64) && used.contains(&ElementKind::F32) {
        // `F32` expressions are evaluated in `double` here, which needs the extension too.
        s.push_str("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
    }
    if used.iter().any(|e| is_float(*e)) {
        // One rounding per operation, as the oracle has: no fused multiply-add behind our back.
        s.push_str("#pragma OPENCL FP_CONTRACT OFF\n");
    }
    s.push_str(&helper_source(&em.helpers.borrow()));
    let _ = writeln!(s, "__kernel void {}({head}) {{", k.name);
    s.push_str(&em.out);
    s.push_str("}\n");
    Ok(GeneratedKernel::new(k.name.clone(), s))
}

#[cfg(test)]
mod tests {
    use super::patent_declined;
    use unpopped::exec::Scope;
    use unpopped::ir::ScalarExpr;
    use unpopped::kernel::{CollectiveKind, Place, Stmt, VoteKind};
    use unpopped_vocab::ElementKind;

    fn place() -> Place {
        Place::Alloc {
            alloc: 0,
            index: ScalarExpr::ConstInt(0, ElementKind::U32),
        }
    }
    fn collective() -> Stmt {
        Stmt::Collective {
            kind: CollectiveKind::Vote(VoteKind::Any),
            value: n(),
            result: 0,
            scope: Scope::Block,
        }
    }
    fn n() -> ScalarExpr {
        ScalarExpr::ConstInt(1, ElementKind::U32)
    }

    #[test]
    fn each_nvidia_patent_statement_is_declined_by_name() {
        let declined = [
            (
                "MbarInit",
                Stmt::MbarInit {
                    bar: place(),
                    count: n(),
                },
            ),
            (
                "MbarArrive",
                Stmt::MbarArrive {
                    bar: place(),
                    tx: None,
                },
            ),
            (
                "MbarCompleteTx",
                Stmt::MbarCompleteTx {
                    bar: place(),
                    bytes: n(),
                },
            ),
            (
                "MbarWait",
                Stmt::MbarWait {
                    bar: place(),
                    parity: n(),
                },
            ),
            (
                "CopyBulk",
                Stmt::CopyBulk {
                    dst: place(),
                    src: place(),
                    elems: 4,
                    bar: place(),
                },
            ),
            ("Tcgen05Commit", Stmt::Tcgen05Commit { bar: place() }),
            ("WgmmaWait", Stmt::WgmmaWait { pending: 0 }),
            ("AsyncCommit", Stmt::AsyncCommit),
            ("AsyncWait", Stmt::AsyncWait { pending: 0 }),
            (
                "CopyAsync",
                Stmt::CopyAsync {
                    dst: place(),
                    src: place(),
                    elems: 4,
                    valid: None,
                    cache: Default::default(),
                    l2_prefetch: None,
                },
            ),
            ("Collective", collective()),
            (
                "CollectiveTuple",
                Stmt::CollectiveTuple {
                    kind: CollectiveKind::Vote(VoteKind::All),
                    values: vec![n()],
                    results: vec![0],
                    scope: Scope::Block,
                },
            ),
        ];
        assert_eq!(declined.len(), 12);
        for (name, s) in &declined {
            assert_eq!(patent_declined(s).map(|(n, _)| n), Some(*name));
        }
    }

    #[test]
    fn each_name_cites_its_own_patent() {
        let p = |s: &Stmt| patent_declined(s).map(|(_, p)| p).unwrap();
        assert_eq!(
            p(&Stmt::MbarWait {
                bar: place(),
                parity: n()
            }),
            "US11803380B2"
        );
        assert_eq!(p(&Stmt::Tcgen05Commit { bar: place() }), "US11803380B2");
        let wg = p(&Stmt::WgmmaWait { pending: 0 });
        assert!(
            wg.contains("US12204897B2") && wg.contains("US12710961B2"),
            "{wg}"
        );
        let a = p(&Stmt::AsyncWait { pending: 0 });
        assert!(
            a.contains("US12118382B2") && a.contains("US11294713B2") && a.contains("434 C"),
            "{a}"
        );
        assert_eq!(p(&Stmt::AsyncCommit), a);
        let c = p(&collective());
        assert!(c.contains("US8539204B2") && c.contains("434 C"), "{c}");
    }

    #[test]
    fn other_statements_are_not_swept_up() {
        // Control: the match is not "always Some".
        for s in [Stmt::Break, Stmt::WgmmaCommit, Stmt::Return] {
            assert_eq!(patent_declined(&s), None, "{s:?}");
        }
    }
}
