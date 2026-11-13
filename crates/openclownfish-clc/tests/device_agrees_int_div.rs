//! Integer `Div`, `Rem`, comparison, `Select` and `ConstInt` on every device (E16).
//!
//! `docs/int-div-shift-semantics.md` in Unpopped is the single source: a zero divisor gives the
//! dividend, a signed `MIN / -1` gives `MIN`, anything else truncates toward zero, and an integer
//! remainder is `Select(b == 0, 0, a - (a / b) * b)` over that division. OpenCL C leaves the first
//! two unspecified (6.5.1), so the kernel carries a guard and these inputs hit it: every pairing of
//! `MIN`, `MAX`, `-1`, `0`, `1` and large values appears in the first 100 elements.
//!
//! Each result is checked against two references that do not share a method: a host function
//! written from the document, and Unpopped's plan oracle (`build_plan_for` with this backend).

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::ir::{BinaryOp, Expr, OpDef, input, konst_int};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::build_plan_for;
use unpopped::try_generate;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const POOL32: [i32; 10] = [
    i32::MAX,
    i32::MIN,
    -1,
    0,
    1,
    2,
    12345,
    -98765,
    0x7fff_0000,
    0x1234_5678,
];
const POOL64: [i64; 10] = [
    i64::MAX,
    i64::MIN,
    -1,
    0,
    1,
    2,
    1 << 40,
    -(1 << 41) - 7,
    0x7fff_ffff_0000_0000,
    0x1234_5678_9abc_def0,
];

fn div64(a: i64, b: i64) -> i64 {
    if b == 0 || (a == i64::MIN && b == -1) {
        a
    } else {
        a / b
    }
}

/// `Select(b == 0, 0, a - (a / b) * b)`, wrapping, over the total division.
fn rem64(a: i64, b: i64) -> i64 {
    if b == 0 {
        0
    } else {
        a.wrapping_sub(div64(a, b).wrapping_mul(b))
    }
}

struct Case {
    name: &'static str,
    /// The host reference, on the values widened to `i64`; the result is narrowed to the width.
    host: fn(i64, i64) -> i64,
    body: fn(ElementKind) -> Expr,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "idiv",
            host: div64,
            body: |_| input(0) / input(1),
        },
        Case {
            name: "irem",
            host: rem64,
            body: |k| {
                let zero = konst_int(0, k);
                let back = (input(0) / input(1)) * input(1);
                input(1)
                    .binary(BinaryOp::CmpEq, konst_int(0, k))
                    .select(zero, input(0) - back)
            },
        },
        Case {
            name: "icmp_select",
            host: |a, b| if a < b { a } else { b },
            body: |_| {
                input(0)
                    .binary(BinaryOp::CmpLt, input(1))
                    .select(input(0), input(1))
            },
        },
        Case {
            name: "iconst",
            host: |a, _| a.wrapping_add(7),
            body: |k| input(0) + konst_int(7, k),
        },
        Case {
            name: "iconst_min",
            host: |a, _| a,
            // `MIN` as a constant: `Select(a == MIN, MIN, a)` is `a`, so the host is the identity.
            body: |k| {
                let min = match k {
                    ElementKind::I32 => i64::from(i32::MIN),
                    _ => i64::MIN,
                };
                input(0)
                    .binary(BinaryOp::CmpEq, konst_int(min, k))
                    .select(konst_int(min, k), input(0))
            },
        },
    ]
}

fn pool(kind: ElementKind, n: usize, which: usize) -> Vec<i64> {
    (0..n)
        .map(|i| {
            let k = if which == 0 { i % 10 } else { (i / 10) % 10 };
            if kind == ElementKind::I32 {
                i64::from(POOL32[k])
            } else {
                POOL64[k]
            }
        })
        .collect()
}

fn buffer(kind: ElementKind, v: &[i64]) -> TypedBuffer {
    if kind == ElementKind::I32 {
        let w: Vec<i32> = v.iter().map(|x| *x as i32).collect();
        TypedBuffer::from_i32(&[v.len() as i64], &w)
    } else {
        TypedBuffer::from_i64(&[v.len() as i64], v)
    }
}

fn raw(kind: ElementKind, v: &[i64]) -> Vec<u8> {
    if kind == ElementKind::I32 {
        v.iter().flat_map(|x| (*x as i32).to_ne_bytes()).collect()
    } else {
        v.iter().flat_map(|x| x.to_ne_bytes()).collect()
    }
}

/// The host result at the element width, as bytes.
fn host_bytes(kind: ElementKind, c: &Case, a: &[i64], b: &[i64]) -> Vec<u8> {
    let r: Vec<i64> = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            if kind == ElementKind::I32 {
                // The 32-bit rule is the 64-bit rule on the sign-extended value, narrowed.
                let (x, y) = (*x as i32, *y as i32);
                let v = match c.name {
                    "idiv" => {
                        if y == 0 || (x == i32::MIN && y == -1) {
                            x
                        } else {
                            x / y
                        }
                    }
                    "irem" => {
                        if y == 0 {
                            0
                        } else {
                            let q = if x == i32::MIN && y == -1 { x } else { x / y };
                            x.wrapping_sub(q.wrapping_mul(y))
                        }
                    }
                    "icmp_select" => x.min(y),
                    "iconst" => x.wrapping_add(7),
                    _ => x,
                };
                i64::from(v)
            } else if c.name == "iconst_min" {
                *x
            } else {
                (c.host)(*x, *y)
            }
        })
        .collect();
    raw(kind, &r)
}

#[test]
fn integer_div_and_friends_agree_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let backend = OpenClC::new();
    assert!(backend.total_integer_ops(), "the backend must opt in");
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
            for kind in [ElementKind::I32, ElementKind::I64] {
                let width = if kind == ElementKind::I32 { 4 } else { 8 };
                for n_el in [257usize, 256] {
                    for c in cases() {
                        let desc = OperandDesc::new(1, &[n_el as i64], &[1], kind, 256);
                        let operands = vec![desc; 3];
                        let key =
                            structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89);
                        let op = OpDef::elementwise(c.name, 2, &[kind], (c.body)(kind));
                        let kernel = match try_generate(&op, &key, &backend) {
                            Ok(k) => k,
                            Err(e) => panic!("{label}: {kind:?} {} refused: {e}", c.name),
                        };
                        let prog = match Program::build(&ctx, &kernel.source, "") {
                            Ok(p) => p,
                            Err(e) => panic!(
                                "{label}: {kind:?} {} failed to build:\n{e}\n{}",
                                c.name, kernel.source
                            ),
                        };
                        let mut k = prog.kernel(&kernel.name).unwrap();
                        let inputs: Vec<Vec<i64>> = (0..2).map(|w| pool(kind, n_el, w)).collect();
                        let bufs: Vec<_> = inputs
                            .iter()
                            .map(|v| {
                                let b = ctx.create_buffer(n_el * width).unwrap();
                                queue.write_buffer(&b, 0, &raw(kind, v)).unwrap();
                                b
                            })
                            .collect();
                        let out = ctx.create_buffer(n_el * width).unwrap();
                        // A sentinel, so a kernel that never wrote is caught.
                        queue
                            .write_buffer(&out, 0, &vec![0xA5u8; n_el * width])
                            .unwrap();
                        for (i, b) in bufs.iter().enumerate() {
                            k.set_arg_buffer(i as u32, b).unwrap();
                        }
                        k.set_arg_buffer(2, &out).unwrap();
                        k.set_arg_i64(3, n_el as i64).unwrap();
                        // SAFETY: the kernel guards `i < n`; every buffer holds n_el elements.
                        let ev = unsafe { queue.launch(&k, &[n_el], None) }.unwrap();
                        ev.wait().unwrap();
                        let mut got = vec![0u8; n_el * width];
                        queue.read_buffer(&out, 0, &mut got).unwrap();

                        let want_host = host_bytes(kind, &c, &inputs[0], &inputs[1]);
                        // The oracle is the second, independent reference.
                        let plan = build_plan_for(&op, &key, &backend);
                        let tb: Vec<TypedBuffer> = inputs.iter().map(|v| buffer(kind, v)).collect();
                        let want_oracle =
                            evaluate(&plan, &operands, &tb, &[])[0].raw_bytes().to_vec();
                        for i in 0..n_el {
                            let r = i * width..(i + 1) * width;
                            let ctxt = || {
                                format!(
                                    "{label}: {kind:?} {} n={n_el} i={i} inputs {:?}",
                                    c.name,
                                    inputs.iter().map(|v| v[i]).collect::<Vec<_>>()
                                )
                            };
                            assert_eq!(&got[r.clone()], &want_host[r.clone()], "host: {}", ctxt());
                            assert_eq!(&got[r.clone()], &want_oracle[r], "oracle: {}", ctxt());
                        }
                        checks += 1;
                    }
                }
            }
            ran += 1;
        }
    }
    eprintln!("integer div and friends: {checks} case runs exact on {ran} device(s)");
}

/// The boundary values the guard exists for, spelled out so a vacuous input pool cannot hide them.
#[test]
fn the_pools_contain_the_boundary_pairs() {
    let (a, b) = (
        pool(ElementKind::I32, 100, 0),
        pool(ElementKind::I32, 100, 1),
    );
    let has = |x: i64, y: i64| a.iter().zip(&b).any(|(p, q)| *p == x && *q == y);
    assert!(has(i64::from(i32::MIN), -1), "MIN / -1");
    assert!(has(12345, 0), "x / 0");
    let (a, b) = (
        pool(ElementKind::I64, 100, 0),
        pool(ElementKind::I64, 100, 1),
    );
    assert!(a.iter().zip(&b).any(|(p, q)| *p == i64::MIN && *q == -1));
}

/// A dtype this backend does not serve is refused, not approximated. u32 and u64 are served
/// now (`device_agrees_unsigned.rs` checks them against the oracle), so the control is the
/// strict-f32 dtype, which has no spelling here.
#[test]
fn an_unserved_dtype_is_refused_by_name() {
    let backend = OpenClC::new();
    let desc = OperandDesc::new(1, &[8], &[1], ElementKind::F32Strict, 256);
    let key = structure_key(
        OpCategory::BinaryElementwise,
        &[desc, desc, desc],
        ArchSku::Sm89,
    );
    let op = OpDef::elementwise("sadd", 2, &[ElementKind::F32Strict], input(0) + input(1));
    let r = std::panic::catch_unwind(|| try_generate(&op, &key, &backend));
    assert!(
        matches!(r, Err(_) | Ok(Err(_))),
        "an F32Strict kernel was produced"
    );
}
