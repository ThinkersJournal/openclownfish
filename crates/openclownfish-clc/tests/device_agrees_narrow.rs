//! i8, u8, i16 and u16 elementwise kernels on every device against Unpopped's
//! oracle, byte for byte.
//!
//! Each narrow type computes in the promoted `int` (or `uint`) and converts back,
//! which wraps at the element width: the pool holds both extremes, -1 or the top
//! half, and values whose sum or product overflows. A 16-bit product overflows
//! `int`, so a kernel that multiplied in the promoted signed type would show up
//! here. Shifts promote to `int` and take the count modulo 32, the rule Unpopped
//! gives every type narrower than 32 bits; the counts run from -40 (signed types)
//! to 39, and from 0 to 79 for unsigned ones.

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::ir::{BinaryOp, Expr, OpDef, ReduceOp, input};
use unpopped::oracle::evaluate;
use unpopped::{build_plan, generate};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

mod common;

struct Case {
    name: &'static str,
    /// Input 1 is a shift count instead of from the pool.
    shift: bool,
    body: Expr,
}

fn cases() -> Vec<Case> {
    let bin = |name, shift, body| Case { name, shift, body };
    vec![
        bin("add", false, input(0) + input(1)),
        bin("sub", false, input(0) - input(1)),
        bin("mul", false, input(0) * input(1)),
        bin("and", false, input(0).binary(BinaryOp::BitAnd, input(1))),
        bin("or", false, input(0).binary(BinaryOp::BitOr, input(1))),
        bin("xor", false, input(0).binary(BinaryOp::BitXor, input(1))),
        bin("shl", true, input(0).binary(BinaryOp::Shl, input(1))),
        bin("shr", true, input(0).binary(BinaryOp::Shr, input(1))),
    ]
}

fn signed(d: ElementKind) -> bool {
    matches!(d, ElementKind::I8 | ElementKind::I16)
}

/// Truncate `v` to `d`'s width and sign, as a value.
fn wrap(d: ElementKind, v: i64) -> i64 {
    match d {
        ElementKind::I8 => i64::from(v as i8),
        ElementKind::U8 => i64::from(v as u8),
        ElementKind::I16 => i64::from(v as i16),
        _ => i64::from(v as u16),
    }
}

/// The ruled result of a shift: promote to `int`, take the count as unsigned
/// bits modulo 32, truncate back.
fn shifted(d: ElementKind, left: bool, a: i64, count: i64) -> i64 {
    let c = (count as i32) as u32;
    let r = if left {
        ((a as i32 as u32).wrapping_shl(c)) as i32
    } else {
        (a as i32).wrapping_shr(c)
    };
    wrap(d, i64::from(r))
}

fn ruled(kind: ElementKind, name: &str, x: i64, y: i64) -> i64 {
    match name {
        "add" => wrap(kind, x.wrapping_add(y)),
        "sub" => wrap(kind, x.wrapping_sub(y)),
        "mul" => wrap(kind, x.wrapping_mul(y)),
        "and" => wrap(kind, x & y),
        "or" => wrap(kind, x | y),
        "xor" => wrap(kind, x ^ y),
        "shl" => shifted(kind, true, x, y),
        _ => shifted(kind, false, x, y),
    }
}

#[test]
fn narrow_integer_kernels_agree_with_the_oracle_on_every_device() {
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
            for kind in common::NARROW {
                let w = common::width(kind);
                let pool = common::pool(kind);
                for n_el in [257usize, 256] {
                    for c in cases() {
                        let desc = OperandDesc::new(1, &[n_el as i64], &[1], kind, 256);
                        let operands = [desc, desc, desc];
                        let key =
                            structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89);
                        let op = OpDef::elementwise(c.name, 2, &[kind], c.body.clone());
                        let kernel = generate(&op, &key, &backend);
                        let prog = match Program::build(&ctx, &kernel.source, "") {
                            Ok(p) => p,
                            Err(e) => panic!(
                                "{label}: {kind:?} {} failed to build:\n{e}\n{}",
                                c.name, kernel.source
                            ),
                        };
                        let mut k = prog.kernel(&kernel.name).unwrap();
                        let a: Vec<i64> = (0..n_el).map(|i| pool[i % 9]).collect();
                        let b: Vec<i64> = (0..n_el)
                            .map(|i| {
                                if c.shift {
                                    let n = (i % 80) as i64;
                                    if signed(kind) { n - 40 } else { n }
                                } else {
                                    pool[(i / 9) % 9]
                                }
                            })
                            .collect();
                        let bufs: Vec<_> = [&a, &b]
                            .iter()
                            .map(|v| {
                                let buf = ctx.create_buffer(n_el * w).unwrap();
                                queue
                                    .write_buffer(&buf, 0, &common::narrow_bytes(kind, v))
                                    .unwrap();
                                buf
                            })
                            .collect();
                        let out = ctx.create_buffer(n_el * w).unwrap();
                        // A sentinel no result is likely to equal, so a kernel
                        // that never wrote is caught.
                        queue
                            .write_buffer(&out, 0, &vec![0xA5u8; n_el * w])
                            .unwrap();
                        k.set_arg_buffer(0, &bufs[0]).unwrap();
                        k.set_arg_buffer(1, &bufs[1]).unwrap();
                        k.set_arg_buffer(2, &out).unwrap();
                        k.set_arg_i64(3, n_el as i64).unwrap();
                        // SAFETY: the kernel guards `i < n`; every buffer holds n_el elements.
                        let ev = unsafe { queue.launch(&k, &[n_el], None) }.unwrap();
                        ev.wait().unwrap();
                        let mut got = vec![0u8; n_el * w];
                        queue.read_buffer(&out, 0, &mut got).unwrap();

                        let plan = build_plan(&op, &key);
                        let shape = [n_el as i64];
                        let tb = [
                            common::narrow_from(kind, &shape, &a),
                            common::narrow_from(kind, &shape, &b),
                        ];
                        let oracle = evaluate(&plan, &operands, &tb, &[]);
                        let oracle = &oracle[0].raw_bytes()[..n_el * w];
                        let want: Vec<i64> =
                            (0..n_el).map(|i| ruled(kind, c.name, a[i], b[i])).collect();
                        let want = common::narrow_bytes(kind, &want);
                        assert_eq!(
                            oracle,
                            &want[..],
                            "{label}: oracle vs the ruled formula, {kind:?} {} n={n_el}",
                            c.name
                        );
                        for i in 0..n_el {
                            assert_eq!(
                                &got[i * w..(i + 1) * w],
                                &want[i * w..(i + 1) * w],
                                "{label}: {kind:?} {} n={n_el} i={i} a={} b={}\n{}",
                                c.name,
                                a[i],
                                b[i],
                                kernel.source
                            );
                        }
                        checks += 1;
                    }
                }
            }
            ran += 1;
        }
    }
    eprintln!("narrow integer kernels: {checks} case runs bit-exact on {ran} device(s)");
}

#[test]
fn narrow_reductions_are_declined_by_name() {
    let backend = OpenClC::new();
    for kind in common::NARROW {
        // Control: the same shape at i32 lowers, so the decline below is about
        // the dtype and not the plan.
        let a = OperandDesc::new(2, &[4, 6], &[6, 1], kind, 256);
        let o = OperandDesc::new(1, &[4], &[1], kind, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let op = OpDef::reduction("r", 1, &[kind], input(0), ReduceOp::Sum);
        let plan = build_plan(&op, &key);
        let err = backend
            .lower(&plan)
            .expect_err("a narrow reduction lowered");
        assert!(
            format!("{err:?}").contains("scopes integer reductions to i32 and i64"),
            "{kind:?}: {err:?}"
        );
        assert!(backend.lower_variants(&plan).unwrap().is_empty());
    }
    let a = OperandDesc::new(2, &[4, 6], &[6, 1], ElementKind::I32, 256);
    let o = OperandDesc::new(1, &[4], &[1], ElementKind::I32, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    let op = OpDef::reduction("r", 1, &[ElementKind::I32], input(0), ReduceOp::Sum);
    assert!(backend.lower(&build_plan(&op, &key)).is_ok());
}
