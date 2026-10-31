//! i32 and i64 kernels on every device against Unpopped's oracle, byte for byte.
//!
//! The inputs are the integers where wrapping, sign and shifts go wrong: the
//! extremes, -1, 0, 1 and large values whose products overflow. A signed
//! overflow is undefined in C, so a compiler that exploited it would show up
//! here as a result that differs from two's-complement wrapping.

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::ir::{BinaryOp, Expr, OpDef, input, konst};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::{build_plan, generate};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// A value pool per width; every pairing appears in the first 100 elements.
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

struct Case {
    name: &'static str,
    n_inputs: usize,
    /// Input 1 is a shift amount: in `0..bits` instead of from the pool.
    shift: bool,
    body: Expr,
}

fn cases() -> Vec<Case> {
    let bin = |name, shift, body| Case {
        name,
        n_inputs: 2,
        shift,
        body,
    };
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

fn pool(kind: ElementKind, n: usize, which: usize, shift: bool) -> Vec<i64> {
    let bits = if kind == ElementKind::I32 { 32 } else { 64 };
    (0..n)
        .map(|i| {
            if shift && which == 1 {
                return (i % bits) as i64;
            }
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

#[test]
fn integer_kernels_agree_with_the_oracle_on_every_device() {
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
            for kind in [ElementKind::I32, ElementKind::I64] {
                let width = if kind == ElementKind::I32 { 4 } else { 8 };
                for n_el in [257usize, 256] {
                    for c in cases() {
                        let desc = OperandDesc::new(1, &[n_el as i64], &[1], kind, 256);
                        let mut operands = vec![desc; c.n_inputs];
                        operands.push(desc);
                        let key =
                            structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89);
                        let op =
                            OpDef::elementwise(c.name, c.n_inputs as u8, &[kind], c.body.clone());
                        let kernel = generate(&op, &key, &backend);
                        let prog = match Program::build(&ctx, &kernel.source, "") {
                            Ok(p) => p,
                            Err(e) => panic!(
                                "{label}: {kind:?} {} failed to build:\n{e}\n{}",
                                c.name, kernel.source
                            ),
                        };
                        let mut k = prog.kernel(&kernel.name).unwrap();
                        let inputs: Vec<Vec<i64>> = (0..c.n_inputs)
                            .map(|w| pool(kind, n_el, w, c.shift))
                            .collect();
                        let bufs: Vec<_> = inputs
                            .iter()
                            .map(|v| {
                                let b = ctx.create_buffer(n_el * width).unwrap();
                                queue.write_buffer(&b, 0, &raw(kind, v)).unwrap();
                                b
                            })
                            .collect();
                        let out = ctx.create_buffer(n_el * width).unwrap();
                        // A sentinel no result is likely to equal, so a kernel
                        // that never wrote is caught.
                        queue
                            .write_buffer(&out, 0, &vec![0xA5u8; n_el * width])
                            .unwrap();
                        for (i, b) in bufs.iter().enumerate() {
                            k.set_arg_buffer(i as u32, b).unwrap();
                        }
                        k.set_arg_buffer(c.n_inputs as u32, &out).unwrap();
                        k.set_arg_i64(c.n_inputs as u32 + 1, n_el as i64).unwrap();
                        // SAFETY: the kernel guards `i < n`; every buffer holds n_el elements.
                        let ev = unsafe { queue.launch(&k, &[n_el], None) }.unwrap();
                        ev.wait().unwrap();
                        let mut got = vec![0u8; n_el * width];
                        queue.read_buffer(&out, 0, &mut got).unwrap();

                        let plan = build_plan(&op, &key);
                        let tb: Vec<TypedBuffer> = inputs.iter().map(|v| buffer(kind, v)).collect();
                        let want = evaluate(&plan, &operands, &tb, &[]);
                        let want = want[0].raw_bytes();
                        for i in 0..n_el {
                            let (g, w) = (
                                &got[i * width..(i + 1) * width],
                                &want[i * width..(i + 1) * width],
                            );
                            assert_eq!(
                                g,
                                w,
                                "{label}: {kind:?} {} n={n_el} i={i} inputs {:?}",
                                c.name,
                                inputs.iter().map(|v| v[i]).collect::<Vec<_>>()
                            );
                        }
                        checks += 1;
                    }
                }
            }
            ran += 1;
        }
    }
    eprintln!("integer kernels: {checks} case runs bit-exact on {ran} device(s)");
}

#[test]
fn integer_plans_outside_the_ir_table_are_refused() {
    let desc = OperandDesc::new(1, &[8], &[1], ElementKind::I32, 256);
    let backend = OpenClC::new();
    for (name, body) in [
        ("max", input(0).max(input(1))),
        ("cmp", input(0).binary(BinaryOp::CmpLt, input(1))),
    ] {
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[desc, desc, desc],
            ArchSku::Sm89,
        );
        let op = OpDef::elementwise(name, 2, &[ElementKind::I32], body);
        // The plan gate may already reject these; either way the backend must
        // not hand back a kernel.
        let lowered = std::panic::catch_unwind(|| {
            let plan = build_plan(&op, &key);
            backend.lower(&plan)
        });
        assert!(
            matches!(lowered, Err(_) | Ok(Err(_))),
            "{name}: an integer kernel was produced"
        );
    }
    // A float constant has no meaning at an integer dtype.
    let key = structure_key(OpCategory::UnaryElementwise, &[desc, desc], ArchSku::Sm89);
    let op = OpDef::elementwise("c", 1, &[ElementKind::I32], input(0) + konst(1.0));
    let lowered = std::panic::catch_unwind(|| backend.lower(&build_plan(&op, &key)));
    assert!(matches!(lowered, Err(_) | Ok(Err(_))));
}
