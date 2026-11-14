//! A kernel-layer [`KernelFn`] spelled as an OpenCL C kernel: the statement-IR path
//! (`Backend::lower_kernel`), a deliberately small subset.
//!
//! What is spelled: integer kernels (`I32`, `U32`, `I64`, `U64`) with a launch shape, rank-1
//! buffers, scalar arguments, `Shared` arrays of static extent, `Let`/`Assign`/`Load`/`Store`,
//! `If`/`While`/`Break`/`Continue`/`Return`, a block `Barrier`, and the two atomic statements
//! `AtomicRmw` and `AtomicCas`. Everything else is declined by name with
//! [`LowerError::UnsupportedKernel`]; nothing is spelled by default.
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
//! # The kernel
//!
//! `__kernel void NAME(args)` in `KernelFn::params` order: a buffer is `__global T* bI` (`const`
//! for a `Read` buffer), a scalar is `T pI`. The buffer is addressed `bI[coord]`: rank 1, no view
//! strides, no base. The interpreter traps an out-of-bounds access and OpenCL C cannot, so a
//! kernel must keep its accesses inside its buffers (a precondition, not a check).

use std::fmt::Write as _;

use unpopped::backend::{GeneratedKernel, LowerError};
use unpopped::exec::{Extent, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, Saturation, ScalarExpr};
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

fn ctype(e: ElementKind, what: &str) -> Result<&'static str, LowerError> {
    int_c(e).map(|c| c.0).ok_or_else(|| {
        refuse(format!(
            "{what} has element {e:?}; only I32, U32, I64 and U64 are spelled"
        ))
    })
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
            ScalarExpr::Exec(_) => Ok(ElementKind::U32),
            ScalarExpr::Cast(x, spec) => {
                self.ty(x)?;
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

    fn expr(&self, e: &ScalarExpr) -> Result<String, LowerError> {
        use unpopped::exec::ExecId;
        Ok(match e {
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
                let (c, _, u) = int_c(t).ok_or_else(|| refuse("an arithmetic type"))?;
                let op = match e {
                    ScalarExpr::Add(..) => '+',
                    ScalarExpr::Sub(..) => '-',
                    _ => '*',
                };
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
                let (c, _, _) = int_c(t).ok_or_else(|| refuse("a comparison type"))?;
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
            other => return Err(refuse(format!("the statement {}", stmt_name(other)))),
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

fn stmt_name(s: &Stmt) -> &'static str {
    match s {
        Stmt::MaskedLoad { .. } => "MaskedLoad",
        Stmt::MaskedStore { .. } => "MaskedStore",
        Stmt::For { .. } => "For",
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
    let _ = writeln!(s, "__kernel void {}({head}) {{", k.name);
    s.push_str(&em.out);
    s.push_str("}\n");
    Ok(GeneratedKernel::new(k.name.clone(), s))
}
