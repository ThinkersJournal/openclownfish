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
//! Declined by name: float atomics, `Max` and `Min` on floats, a
//! cast from a float, the transcendental unary ops, `Select`, and `F16`/`Bf16`. `Div` relies on
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

use std::fmt::Write as _;

use unpopped::backend::{GeneratedKernel, LowerError};
use unpopped::exec::{Extent, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, Saturation, ScalarExpr, UnaryOp};
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
            ScalarExpr::Fma(a, b, c) => {
                let (x, y, z) = (self.ty(a)?, self.ty(b)?, self.ty(c)?);
                if x == y && y == z && is_float(x) {
                    Ok(x)
                } else {
                    Err(refuse("an Fma other than of three floats of one type"))
                }
            }
            ScalarExpr::Cast(x, spec) => {
                if is_float(self.ty(x)?) {
                    return Err(refuse("a cast from a float"));
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
            ScalarExpr::Unary(_, x) | ScalarExpr::Cast(x, _) => self.f32_roundings(x),
            ScalarExpr::Binary(_, a, b) => self.f32_roundings(a) + self.f32_roundings(b),
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
                let (c, _, u) = int_c(spec.to).ok_or_else(|| refuse("a cast to this type"))?;
                let inner = self.expr(x)?;
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
                if is_float(t) && matches!(op, BinaryOp::Max | BinaryOp::Min) {
                    return Err(refuse(
                        "a float Max or Min (NaN and signed-zero rules are not matched)",
                    ));
                }
                let (x, y) = (self.expr(a)?, self.expr(b)?);
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
                if let Some((name, patent)) = patent_declined(&format!("{other:?}")) {
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
/// matrix-multiply operations (`WgmmaWait`).
///
/// **TEMPORARY.** Matching a `Debug` string is brittle: a variant rename or a `Debug` change
/// silently stops it. Replace it with an exhaustive `match` arm over the real `Stmt` variants,
/// with tests that build real statements, at the repin to the first Unpopped version that
/// contains them (owed; see `HANDOFF.md`, PM 2026-11-16).
const NVIDIA_PATENT_DECLINED: &[(&str, &str)] = &[
    ("MbarInit", "US11803380B2"),
    ("MbarArrive", "US11803380B2"),
    ("MbarCompleteTx", "US11803380B2"),
    ("MbarWait", "US11803380B2"),
    ("CopyBulk", "US11803380B2"),
    ("Tcgen05Commit", "US11803380B2"),
    ("WgmmaWait", "US12204897B2 (continuation US12710961B2)"),
];

/// The declined statement a `Debug` rendering names, with its patent, if any. It reads the
/// leading identifier of `format!("{stmt:?}")`, which a derived `Debug` makes the variant name,
/// so a statement Unpopped adds later is caught by name without this crate naming the variant
/// (Unpopped 0.47.0 has none of them yet).
fn patent_declined(debug: &str) -> Option<(&'static str, &'static str)> {
    let head = debug
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .next()
        .unwrap_or("");
    NVIDIA_PATENT_DECLINED
        .iter()
        .copied()
        .find(|(n, _)| *n == head)
}

fn stmt_name(s: &Stmt) -> &'static str {
    match s {
        Stmt::MaskedLoad { .. } => "MaskedLoad",
        Stmt::MaskedStore { .. } => "MaskedStore",
        Stmt::MulticastLoadReduce { .. } => "MulticastLoadReduce",
        Stmt::Fence { .. } => "Fence",
        Stmt::Collective { .. } => "Collective",
        Stmt::CollectiveTuple { .. } => "CollectiveTuple",
        Stmt::Mma { .. } => "Mma",
        Stmt::CopyAsync { .. } => "CopyAsync",
        Stmt::AsyncCommit => "AsyncCommit",
        Stmt::AsyncWait { .. } => "AsyncWait",
        Stmt::Call { .. } => "Call",
        Stmt::CallTuple { .. } => "CallTuple",
        Stmt::ExternCall { .. } => "ExternCall",
        _ => "a statement this emitter has not been taught",
    }
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
    let _ = writeln!(s, "__kernel void {}({head}) {{", k.name);
    s.push_str(&em.out);
    s.push_str("}\n");
    Ok(GeneratedKernel::new(k.name.clone(), s))
}

#[cfg(test)]
mod tests {
    use super::{NVIDIA_PATENT_DECLINED, patent_declined};

    #[test]
    fn each_nvidia_patent_statement_is_declined_by_name() {
        // A derived `Debug` prints the variant name first, whatever its fields.
        for (name, patent) in NVIDIA_PATENT_DECLINED {
            assert_eq!(patent_declined(name), Some((*name, *patent)), "unit form");
            assert_eq!(
                patent_declined(&format!("{name} {{ barrier: 0, count: 1 }}")),
                Some((*name, *patent)),
                "struct form"
            );
            assert_eq!(
                patent_declined(&format!("{name}(7)")),
                Some((*name, *patent)),
                "tuple form"
            );
        }
        assert_eq!(NVIDIA_PATENT_DECLINED.len(), 7);
    }

    #[test]
    fn each_name_cites_its_own_patent() {
        let patent = |s: &str| patent_declined(s).map(|(_, p)| p);
        assert_eq!(patent("MbarWait { b: 0 }"), Some("US11803380B2"));
        assert_eq!(patent("Tcgen05Commit { b: 0 }"), Some("US11803380B2"));
        let wg = patent("WgmmaWait { n: 0 }").unwrap();
        assert!(
            wg.contains("US12204897B2") && wg.contains("US12710961B2"),
            "{wg}"
        );
    }

    #[test]
    fn other_statements_are_not_swept_up() {
        // Control: the matcher is not "always Some". Similar names pass.
        for other in [
            "CopyAsync { dst: 0 }",
            "AsyncWait { n: 0 }",
            "WgmmaWaitX { n: 0 }",
            "MbarInitX { a: 1 }",
            "CopyBulkExtra",
            "Let { local: 0 }",
            "",
        ] {
            assert_eq!(patent_declined(other), None, "{other}");
        }
    }
}
