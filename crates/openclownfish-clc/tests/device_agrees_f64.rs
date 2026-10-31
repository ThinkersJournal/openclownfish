//! The f64 emitter on every device that lists `cl_khr_fp64`, against Unpopped's
//! f64 oracle. Devices without the extension are reported and counted, and the
//! emitter's refusal for them is checked: f64 must be a typed refusal, never a
//! kernel that fails to build on the driver.
//!
//! In f64 the oracle's per-operation rounding is the device's, so unlike f32
//! `a*b+c` must match it bit for bit when the compiler honours FP_CONTRACT OFF.

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::ir::{BinaryOp, OpDef, UnaryOp, input, konst};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::{build_plan, generate};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const F: [ElementKind; 1] = [ElementKind::F64];

struct Case {
    name: &'static str,
    op: OpDef,
    cat: OpCategory,
    n_inputs: usize,
    max_ulps: u64,
    positive: bool,
    special: bool,
}

fn case(name: &'static str, n_inputs: usize, max_ulps: u64, body: unpopped::ir::Expr) -> Case {
    Case {
        name,
        op: OpDef::elementwise(name, n_inputs as u8, &F, body),
        cat: if n_inputs == 1 {
            OpCategory::UnaryElementwise
        } else {
            OpCategory::BinaryElementwise
        },
        n_inputs,
        max_ulps,
        positive: false,
        special: false,
    }
}

fn cases() -> Vec<Case> {
    let mut v = vec![
        case("add", 2, 0, input(0) + input(1)),
        case("sub", 2, 0, input(0) - input(1)),
        case("mul", 2, 0, input(0) * input(1)),
        case("muladd", 3, 0, input(0) * input(1) + input(2)),
        case("div", 2, 3, input(0) / input(1)),
        case("neg_scale", 1, 0, input(0).unary(UnaryOp::Neg) * konst(2.5)),
        case("abs", 1, 0, input(0).unary(UnaryOp::Abs)),
        case("sqr", 1, 0, input(0).unary(UnaryOp::Sqr)),
        case("recip", 1, 3, input(0).unary(UnaryOp::Recip)),
        case("exp", 1, 4, input(0).unary(UnaryOp::Exp)),
    ];
    for (name, body, ulps) in [
        ("sqrt", input(0).unary(UnaryOp::Sqrt), 3),
        ("rsqrt", input(0).unary(UnaryOp::Rsqrt), 4),
        ("log", input(0).unary(UnaryOp::Log), 4),
        ("pow", input(0).binary(BinaryOp::Pow, konst(2.5)), 17),
    ] {
        let mut c = case(name, 1, ulps, body);
        c.positive = true;
        v.push(c);
    }
    for (name, body) in [
        ("max", input(0).max(input(1))),
        ("min", input(0).min(input(1))),
        ("fmax_ieee", input(0).binary(BinaryOp::FmaxIeee, input(1))),
        ("cmp_lt", input(0).binary(BinaryOp::CmpLt, input(1))),
        ("cmp_ne", input(0).binary(BinaryOp::CmpNe, input(1))),
        ("select", input(0).select(input(1), konst(-3.0))),
    ] {
        let mut c = case(name, 2, 0, body);
        c.special = true;
        v.push(c);
    }
    v
}

const SPECIAL: [f64; 9] = [
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.5,
    -2.5,
];

fn data(n: usize, which: usize, c: &Case) -> Vec<f64> {
    (0..n)
        .map(|i| {
            if c.special {
                let k = if which == 0 { i % 9 } else { (i / 9) % 9 };
                return SPECIAL[k];
            }
            let x = (i as f64 + 1.0) * 0.173 + which as f64 * 0.37;
            let x = if i % 7 == 3 { -x * 1.0001 } else { x * 1.3 };
            if c.positive { x.abs() + 0.01 } else { x }
        })
        .collect()
}

fn bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn doubles(b: &[u8]) -> Vec<f64> {
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_ne_bytes(*c))
        .collect()
}

/// Distance in representable f64 values.
fn ulps(a: f64, b: f64) -> u64 {
    fn key(x: f64) -> i128 {
        let i = i128::from(x.to_bits() as i64);
        if i < 0 { i128::from(i64::MIN) - i } else { i }
    }
    key(a).abs_diff(key(b)) as u64
}

#[test]
fn f64_kernels_agree_with_the_oracle_on_every_fp64_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let (mut ran, mut no_fp64) = (0, 0);
    for p in platforms {
        let Ok(devices) = p.devices() else { continue };
        for d in devices {
            let label = format!(
                "{} / {}",
                p.name().unwrap_or_default(),
                d.name().unwrap_or_default()
            );
            let Ok(exts) = d.extensions() else { continue };
            let backend = OpenClC::for_extensions(exts.iter().map(String::as_str));
            if !exts.iter().any(|e| e == "cl_khr_fp64") {
                // The refusal is the behaviour under test for this device.
                let c = &cases()[0];
                let desc = OperandDesc::new(1, &[8], &[1], ElementKind::F64, 256);
                let key = structure_key(c.cat, &[desc, desc, desc], ArchSku::Sm89);
                let err = backend.lower(&build_plan(&c.op, &key)).unwrap_err();
                assert!(err.to_string().contains("cl_khr_fp64"), "{err}");
                eprintln!("{label}: no cl_khr_fp64, f64 refused");
                no_fp64 += 1;
                continue;
            }
            let Ok(ctx) = Context::new(&d) else { continue };
            let queue = ctx.create_queue().unwrap();
            for n_el in [257usize, 256] {
                for c in cases() {
                    let desc = OperandDesc::new(1, &[n_el as i64], &[1], ElementKind::F64, 256);
                    let mut operands = vec![desc; c.n_inputs];
                    operands.push(desc);
                    let key = structure_key(c.cat, &operands, ArchSku::Sm89);
                    let kernel = generate(&c.op, &key, &backend);
                    let prog = match Program::build(&ctx, &kernel.source, "") {
                        Ok(p) => p,
                        Err(e) => panic!(
                            "{label}: {} failed to build:\n{e}\n{}",
                            c.name, kernel.source
                        ),
                    };
                    let mut k = prog.kernel(&kernel.name).unwrap();
                    let inputs: Vec<Vec<f64>> =
                        (0..c.n_inputs).map(|w| data(n_el, w, &c)).collect();
                    let bufs: Vec<_> = inputs
                        .iter()
                        .map(|v| {
                            let b = ctx.create_buffer(n_el * 8).unwrap();
                            queue.write_buffer(&b, 0, &bytes(v)).unwrap();
                            b
                        })
                        .collect();
                    let out = ctx.create_buffer(n_el * 8).unwrap();
                    queue
                        .write_buffer(&out, 0, &bytes(&vec![f64::NAN; n_el]))
                        .unwrap();
                    for (i, b) in bufs.iter().enumerate() {
                        k.set_arg_buffer(i as u32, b).unwrap();
                    }
                    k.set_arg_buffer(c.n_inputs as u32, &out).unwrap();
                    k.set_arg_i64(c.n_inputs as u32 + 1, n_el as i64).unwrap();
                    // SAFETY: the kernel guards `i < n`; every buffer holds n_el doubles.
                    let ev = unsafe { queue.launch(&k, &[n_el], None) }.unwrap();
                    ev.wait().unwrap();
                    let mut raw = vec![0u8; n_el * 8];
                    queue.read_buffer(&out, 0, &mut raw).unwrap();
                    let got = doubles(&raw);

                    let plan = build_plan(&c.op, &key);
                    let tb: Vec<TypedBuffer> = inputs
                        .iter()
                        .map(|v| TypedBuffer::from_f64(&[n_el as i64], v))
                        .collect();
                    let want = evaluate(&plan, &operands, &tb, &[])
                        .into_iter()
                        .next()
                        .unwrap()
                        .to_f64_vec();

                    let mut worst = 0;
                    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                        let ctxmsg = || {
                            format!(
                                "{label}: {} n={n_el} i={i} inputs {:?} -> {g:?}, oracle {w:?}",
                                c.name,
                                inputs.iter().map(|v| v[i]).collect::<Vec<_>>()
                            )
                        };
                        if w.is_nan() {
                            assert!(g.is_nan(), "{}", ctxmsg());
                        } else if c.special {
                            let zero_tie_open = c.name.ends_with("_ieee") && *w == 0.0;
                            let ok = if zero_tie_open {
                                *g == 0.0
                            } else {
                                g.to_bits() == w.to_bits()
                            };
                            assert!(ok, "{}", ctxmsg());
                        } else {
                            assert!(!g.is_nan(), "NEVER-WROTE or NaN: {}", ctxmsg());
                            worst = worst.max(ulps(*g, *w));
                        }
                    }
                    assert!(
                        worst <= c.max_ulps,
                        "{label}: {} n={n_el} off by {worst} ulp (limit {})",
                        c.name,
                        c.max_ulps
                    );
                    if n_el == 257 {
                        eprintln!(
                            "{label}: f64 {} worst {worst} ulp (limit {})",
                            c.name, c.max_ulps
                        );
                    }
                }
            }
            ran += 1;
        }
    }
    eprintln!("f64 kernels ran on {ran} device(s); {no_fp64} without cl_khr_fp64");
}

#[test]
fn f64_is_refused_unless_the_backend_was_told_the_device_has_fp64() {
    let desc = OperandDesc::new(1, &[8], &[1], ElementKind::F64, 256);
    let key = structure_key(
        OpCategory::BinaryElementwise,
        &[desc, desc, desc],
        ArchSku::Sm89,
    );
    let op = OpDef::elementwise("add", 2, &F, input(0) + input(1));
    let plan = build_plan(&op, &key);
    for backend in [
        OpenClC::new(),
        OpenClC::for_extensions(["cl_khr_fp16", "cl_khr_byte_addressable_store"]),
    ] {
        let err = backend.lower(&plan).unwrap_err();
        assert!(err.to_string().contains("cl_khr_fp64"), "{err}");
    }
    let ok = OpenClC::for_extensions(["cl_khr_fp64"])
        .lower(&plan)
        .unwrap();
    assert!(ok.source.contains("double"), "{}", ok.source);
    assert!(ok.source.contains("cl_khr_fp64"), "{}", ok.source);
}
