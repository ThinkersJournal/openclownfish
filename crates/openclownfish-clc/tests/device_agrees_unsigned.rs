//! u32 and u64 kernels on every device: elementwise, serial reductions and serial
//! scans, byte for byte against Unpopped's oracle (the oracle covers both from
//! 0.40.19).
//!
//! The elementwise cases are also checked against a host formula that writes the rule
//! out: `+ - *` and `<<` wrap, `>>` is logical, a shift count is taken modulo the width
//! (OpenCL C 6.5.10), and a zero divisor gives the dividend. So the oracle and the
//! device are each held to the rule, not only to each other.

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::ir::{BinaryOp, Expr, OpDef, ReduceOp, input, reduced};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::build_plan_for;
use unpopped::{build_plan, generate};
use unpopped_vocab::{ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, structure_key};

const KINDS: [ElementKind; 2] = [ElementKind::U32, ElementKind::U64];

fn bits(k: ElementKind) -> u32 {
    if k == ElementKind::U32 { 32 } else { 64 }
}

fn width(k: ElementKind) -> usize {
    (bits(k) / 8) as usize
}

fn mask(k: ElementKind) -> u64 {
    if k == ElementKind::U32 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    }
}

/// Where wrapping, sign-like top bits and zero go wrong.
fn pool(k: ElementKind) -> [u64; 9] {
    let top = mask(k);
    [top, 1, 0, 2, 7, top / 2 + 1, top - 1, 3, 100_000]
}

fn bytes(k: ElementKind, v: &[u64]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| x.to_le_bytes()[..width(k)].to_vec())
        .collect()
}

fn buffer(k: ElementKind, shape: &[i64], v: &[u64]) -> TypedBuffer {
    if k == ElementKind::U32 {
        let x: Vec<u32> = v.iter().map(|&a| a as u32).collect();
        TypedBuffer::from_u32(shape, &x)
    } else {
        TypedBuffer::from_u64(shape, v)
    }
}

fn dense(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

struct Case {
    name: &'static str,
    shift: bool,
    body: Expr,
}

fn cases() -> Vec<Case> {
    let c = |name, shift, body| Case { name, shift, body };
    vec![
        c("add", false, input(0) + input(1)),
        c("sub", false, input(0) - input(1)),
        c("mul", false, input(0) * input(1)),
        c("and", false, input(0).binary(BinaryOp::BitAnd, input(1))),
        c("or", false, input(0).binary(BinaryOp::BitOr, input(1))),
        c("xor", false, input(0).binary(BinaryOp::BitXor, input(1))),
        c("shl", true, input(0).binary(BinaryOp::Shl, input(1))),
        c("shr", true, input(0).binary(BinaryOp::Shr, input(1))),
        c("div", false, input(0) / input(1)),
    ]
}

fn ruled(k: ElementKind, name: &str, x: u64, y: u64) -> u64 {
    let m = mask(k);
    let n = bits(k);
    let r = match name {
        "add" => x.wrapping_add(y),
        "sub" => x.wrapping_sub(y),
        "mul" => x.wrapping_mul(y),
        "and" => x & y,
        "or" => x | y,
        "xor" => x ^ y,
        "shl" => x.wrapping_shl((y % u64::from(n)) as u32),
        "shr" => (x & m) >> (y % u64::from(n)),
        // A zero divisor gives the dividend.
        _ => x.checked_div(y).unwrap_or(x),
    };
    r & m
}

#[test]
fn unsigned_elementwise_agrees_with_the_oracle_and_the_rule_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let backend = OpenClC::new();
    let (mut ran, mut checks) = (0, 0);
    for p in platforms {
        let Ok(devices) = p.devices() else { continue };
        for d in devices {
            let label = format!(
                "{} / {}",
                p.name().unwrap_or_default(),
                d.name().unwrap_or_default()
            );
            let Ok(ctx) = Context::new(&d) else { continue };
            let queue = ctx.create_queue().unwrap();
            for kind in KINDS {
                let w = width(kind);
                let pl = pool(kind);
                for n_el in [257usize, 256] {
                    for c in cases() {
                        let desc = OperandDesc::new(1, &[n_el as i64], &[1], kind, 256);
                        let operands = [desc, desc, desc];
                        let key =
                            structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89);
                        let op = OpDef::elementwise(c.name, 2, &[kind], c.body.clone());
                        let kernel = generate(&op, &key, &backend);
                        let prog = Program::build(&ctx, &kernel.source, "").unwrap_or_else(|e| {
                            panic!("{label}: {kind:?} {}: {e}\n{}", c.name, kernel.source)
                        });
                        let mut k = prog.kernel(&kernel.name).unwrap();
                        let a: Vec<u64> = (0..n_el).map(|i| pl[i % 9]).collect();
                        let b: Vec<u64> = (0..n_el)
                            .map(|i| {
                                if c.shift {
                                    (i % 150) as u64
                                } else {
                                    pl[(i / 9) % 9]
                                }
                            })
                            .collect();
                        let bufs: Vec<_> = [&a, &b]
                            .iter()
                            .map(|v| {
                                let buf = ctx.create_buffer(n_el * w).unwrap();
                                queue.write_buffer(&buf, 0, &bytes(kind, v)).unwrap();
                                buf
                            })
                            .collect();
                        let out = ctx.create_buffer(n_el * w).unwrap();
                        queue
                            .write_buffer(&out, 0, &vec![0xA5u8; n_el * w])
                            .unwrap();
                        k.set_arg_buffer(0, &bufs[0]).unwrap();
                        k.set_arg_buffer(1, &bufs[1]).unwrap();
                        k.set_arg_buffer(2, &out).unwrap();
                        k.set_arg_i64(3, n_el as i64).unwrap();
                        // SAFETY: the kernel guards `i < n`; every buffer holds n_el elements.
                        unsafe { queue.launch(&k, &[n_el], None) }
                            .unwrap()
                            .wait()
                            .unwrap();
                        let mut got = vec![0u8; n_el * w];
                        queue.read_buffer(&out, 0, &mut got).unwrap();

                        let plan = build_plan_for(&op, &key, &backend);
                        let shape = [n_el as i64];
                        let tb = [buffer(kind, &shape, &a), buffer(kind, &shape, &b)];
                        let oracle = evaluate(&plan, &operands, &tb, &[]);
                        let oracle = &oracle[0].raw_bytes()[..n_el * w];
                        let want: Vec<u64> =
                            (0..n_el).map(|i| ruled(kind, c.name, a[i], b[i])).collect();
                        let want = bytes(kind, &want);
                        assert_eq!(
                            oracle,
                            &want[..],
                            "{label}: oracle vs the rule, {kind:?} {} n={n_el}",
                            c.name
                        );
                        assert_eq!(
                            got, want,
                            "{label}: {kind:?} {} n={n_el}\n{}",
                            c.name, kernel.source
                        );
                        checks += 1;
                    }
                }
            }
            ran += 1;
        }
    }
    eprintln!("unsigned elementwise: {checks} case runs bit-exact on {ran} device(s)");
}

const SHAPE: [i64; 2] = [4, 7];

fn data(k: ElementKind) -> TypedBuffer {
    let n: usize = SHAPE.iter().product::<i64>() as usize;
    let pl = pool(k);
    let v: Vec<u64> = (0..n).map(|i| pl[i % 9]).collect();
    buffer(k, &SHAPE, &v)
}

#[test]
fn unsigned_reductions_and_scans_agree_with_the_oracle_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let backend = OpenClC::new();
    let (mut ran, mut checks) = (0, 0);
    for p in platforms {
        let Ok(devices) = p.devices() else { continue };
        for d in devices {
            let label = format!(
                "{} / {}",
                p.name().unwrap_or_default(),
                d.name().unwrap_or_default()
            );
            let Ok(ctx) = Context::new(&d) else { continue };
            let queue = ctx.create_queue().unwrap();
            for kind in KINDS {
                let w = width(kind);
                let inp = data(kind);
                let inb = ctx.create_buffer(inp.bytes.len()).unwrap();
                queue.write_buffer(&inb, 0, &inp.bytes).unwrap();
                let n_el: i64 = SHAPE.iter().product();
                let row = SHAPE[1];
                let n_out = SHAPE[0];

                // Reductions over the last axis.
                for rop in [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Max, ReduceOp::Min] {
                    let a = OperandDesc::new(2, &SHAPE, &dense(&SHAPE), kind, 256);
                    let o = OperandDesc::new(1, &[n_out], &[1], kind, 256);
                    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
                    let mut m = AxisMask::EMPTY;
                    m.set(1);
                    let opdef = OpDef::reduction_axes("r", 1, &[kind], input(0), rop, m, false);
                    let plan = build_plan(&opdef, &key);
                    let kernel = generate(&opdef, &key, &backend);
                    let tag = format!("{kind:?} reduce {rop:?}");
                    let prog = Program::build(&ctx, &kernel.source, "")
                        .unwrap_or_else(|e| panic!("{label}: {tag}: {e}\n{}", kernel.source));
                    let mut k = prog.kernel(&kernel.name).unwrap();
                    let nb = n_out as usize * w;
                    let out = ctx.create_buffer(nb).unwrap();
                    queue.write_buffer(&out, 0, &vec![0xA5u8; nb]).unwrap();
                    k.set_arg_buffer(0, &inb).unwrap();
                    k.set_arg_buffer(1, &out).unwrap();
                    for (i, x) in [SHAPE[0], SHAPE[1], row, 1, 1, n_out].iter().enumerate() {
                        k.set_arg_i64(2 + i as u32, *x).unwrap();
                    }
                    // SAFETY: one work item per output element, guarded by `o < n_out`.
                    unsafe { queue.launch(&k, &[n_out as usize], None) }
                        .unwrap()
                        .wait()
                        .unwrap();
                    let mut got = vec![0u8; nb];
                    queue.read_buffer(&out, 0, &mut got).unwrap();
                    let want = evaluate(&plan, &[a, o], std::slice::from_ref(&inp), &[])
                        .remove(0)
                        .bytes;
                    assert_eq!(got, &want[..nb], "{label}: {tag}\n{}", kernel.source);
                    checks += 1;
                }

                // Scans along the last axis: forward and reverse, inclusive and exclusive.
                for (rop, reverse, exclusive) in [
                    (ReduceOp::Sum, false, false),
                    (ReduceOp::Sum, true, true),
                    (ReduceOp::Max, false, true),
                    (ReduceOp::Min, true, false),
                ] {
                    let a = OperandDesc::new(2, &SHAPE, &dense(&SHAPE), kind, 256);
                    let key = structure_key(OpCategory::Scan, &[a, a], ArchSku::Sm89);
                    let opdef = OpDef::scan(
                        "s",
                        1,
                        &[kind],
                        rop,
                        1,
                        reverse,
                        exclusive,
                        input(0),
                        reduced(0),
                    );
                    let plan = build_plan(&opdef, &key);
                    let kernel = generate(&opdef, &key, &backend);
                    let tag = format!("{kind:?} scan {rop:?} rev={reverse} excl={exclusive}");
                    let prog = Program::build(&ctx, &kernel.source, "")
                        .unwrap_or_else(|e| panic!("{label}: {tag}: {e}\n{}", kernel.source));
                    let mut k = prog.kernel(&kernel.name).unwrap();
                    let nb = n_el as usize * w;
                    let out = ctx.create_buffer(nb).unwrap();
                    queue.write_buffer(&out, 0, &vec![0xA5u8; nb]).unwrap();
                    k.set_arg_buffer(0, &inb).unwrap();
                    k.set_arg_buffer(1, &out).unwrap();
                    k.set_arg_i64(2, n_out).unwrap();
                    k.set_arg_i64(3, row).unwrap();
                    // SAFETY: one work item per row, guarded by `row < n_out`.
                    unsafe { queue.launch(&k, &[n_out as usize], None) }
                        .unwrap()
                        .wait()
                        .unwrap();
                    let mut got = vec![0u8; nb];
                    queue.read_buffer(&out, 0, &mut got).unwrap();
                    let want = evaluate(&plan, &[a, a], std::slice::from_ref(&inp), &[])
                        .remove(0)
                        .bytes;
                    assert_eq!(got, &want[..nb], "{label}: {tag}\n{}", kernel.source);
                    checks += 1;
                }
            }
            ran += 1;
        }
    }
    eprintln!("unsigned reductions and scans: {checks} case runs bit-exact on {ran} device(s)");
}
