//! bf16 and f16 elementwise kernels on every device against Unpopped's oracle.
//!
//! Both are stored as a 16-bit pattern and computed in `float`; neither needs
//! `cl_khr_fp16`, so this runs on every device. The oracle treats `Neg`, `Abs`,
//! `Max`, `Min` and `Select` on stored values as bit moves (a NaN payload, a
//! signalling NaN, the sign of a zero all survive) and every other operation as a
//! float computation rounded once, to nearest even, at the store. So:
//!
//! - bit moves and the exact operations (`+ - *`, squares, comparisons) are
//!   compared byte for byte, except that the NaN an operation *creates* has a
//!   sign and payload the hardware picks, so two NaNs agree;
//! - operations the device rounds in its own way (`/`, `sqrt`, `exp`, `log`, ...)
//!   may differ from the oracle's f64 detour by one step of the format;
//! - the pool holds both infinities, both zeros, quiet and signalling NaNs with
//!   payloads, the largest finite value, and the edges of the subnormal range;
//! - a second test sends every one of the 65536 patterns through the codec
//!   (decode, scale by an exact power of two, encode), which exercises subnormal
//!   rounding and overflow to infinity.

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::ir::{BinaryOp, Expr, OpDef, ReduceOp, UnaryOp, input, konst};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::{build_plan, generate};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const FORMATS: [ElementKind; 2] = [ElementKind::Bf16, ElementKind::F16];

/// 18 patterns: 1, 2, -2, 0.5, -1, 3, +0, -0, +inf, -inf, a quiet NaN with a
/// payload, a signalling NaN, its negation, the largest finite, the smallest
/// subnormal, 100, the smallest normal and the largest subnormal.
fn pool(kind: ElementKind) -> [u16; 18] {
    match kind {
        ElementKind::Bf16 => [
            0x3f80, 0x4000, 0xc000, 0x3f00, 0xbf80, 0x4040, 0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc1,
            0x7f81, 0xff81, 0x7f7f, 0x0001, 0x42c8, 0x0080, 0x007f,
        ],
        _ => [
            0x3c00, 0x4000, 0xc000, 0x3800, 0xbc00, 0x4200, 0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e01,
            0x7c01, 0xfc01, 0x7bff, 0x0001, 0x5640, 0x0400, 0x03ff,
        ],
    }
}

fn is_nan(kind: ElementKind, h: u16) -> bool {
    h & 0x7fff
        > if kind == ElementKind::Bf16 {
            0x7f80
        } else {
            0x7c00
        }
}

#[derive(Clone, Copy)]
enum Check {
    /// Byte for byte.
    Bits,
    /// Byte for byte, except that two NaNs agree.
    ArithNan,
    /// Within one step of the format (and two NaNs agree).
    Ulp1,
}

fn agrees(kind: ElementKind, check: Check, got: u16, want: u16) -> bool {
    if got == want {
        return true;
    }
    match check {
        Check::Bits => false,
        Check::ArithNan => is_nan(kind, got) && is_nan(kind, want),
        Check::Ulp1 => {
            (is_nan(kind, got) && is_nan(kind, want))
                || (got >> 15 == want >> 15 && (got & 0x7fff).abs_diff(want & 0x7fff) <= 1)
        }
    }
}

struct Case {
    name: &'static str,
    n_inputs: usize,
    body: Expr,
    check: Check,
}

fn cases() -> Vec<Case> {
    let two = |name, body, check| Case {
        name,
        n_inputs: 2,
        body,
        check,
    };
    let one = |name, body, check| Case {
        name,
        n_inputs: 1,
        body,
        check,
    };
    let bin = |op| input(0).binary(op, input(1));
    vec![
        two("add", input(0) + input(1), Check::ArithNan),
        two("sub", input(0) - input(1), Check::ArithNan),
        two("mul", input(0) * input(1), Check::ArithNan),
        two("div", input(0) / input(1), Check::Ulp1),
        two("max", bin(BinaryOp::Max), Check::Bits),
        two("min", bin(BinaryOp::Min), Check::Bits),
        two("cmplt", bin(BinaryOp::CmpLt), Check::Bits),
        two("cmpeq", bin(BinaryOp::CmpEq), Check::Bits),
        two("cmpne", bin(BinaryOp::CmpNe), Check::Bits),
        two(
            "select",
            input(0)
                .binary(BinaryOp::CmpGt, input(1))
                .select(input(0), input(1)),
            Check::Bits,
        ),
        // A composed tree: the sum is rounded to float, not to the format, before
        // the multiply, so the oracle (f64, one rounding) may differ by a step.
        two("sum_times", (input(0) + input(1)) * input(1), Check::Ulp1),
        // Max over a computed arm: the arm is rounded to the format, then moved.
        two(
            "max_sum",
            input(0).max(input(0) + input(1)),
            Check::ArithNan,
        ),
        two("mul_const", input(0) * konst(2.0), Check::ArithNan),
        one("neg", input(0).unary(UnaryOp::Neg), Check::Bits),
        one("abs", input(0).unary(UnaryOp::Abs), Check::Bits),
        one("sqr", input(0).unary(UnaryOp::Sqr), Check::ArithNan),
        one("recip", input(0).unary(UnaryOp::Recip), Check::Ulp1),
        one("sqrt", input(0).unary(UnaryOp::Sqrt), Check::Ulp1),
        one("rsqrt", input(0).unary(UnaryOp::Rsqrt), Check::Ulp1),
        one("exp", input(0).unary(UnaryOp::Exp), Check::Ulp1),
        one("log", input(0).unary(UnaryOp::Log), Check::Ulp1),
    ]
}

fn typed(kind: ElementKind, shape: &[i64], strides: &[i64], bits: &[u16]) -> TypedBuffer {
    TypedBuffer::new(kind, shape.to_vec(), strides.to_vec(), bytes(bits))
}

fn bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|b| b.to_le_bytes()).collect()
}

fn read_bits(raw: &[u8]) -> Vec<u16> {
    raw.chunks(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[test]
fn half_kernels_agree_with_the_oracle_on_every_device() {
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
            for kind in FORMATS {
                let pl = pool(kind);
                for n_el in [324usize, 323] {
                    for c in cases() {
                        let desc = OperandDesc::new(1, &[n_el as i64], &[1], kind, 256);
                        let operands = vec![desc; c.n_inputs + 1];
                        let cat = if c.n_inputs == 1 {
                            OpCategory::UnaryElementwise
                        } else {
                            OpCategory::BinaryElementwise
                        };
                        let key = structure_key(cat, &operands, ArchSku::Sm89);
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
                        let a: Vec<u16> = (0..n_el).map(|i| pl[i % 18]).collect();
                        let b: Vec<u16> = (0..n_el).map(|i| pl[(i / 18) % 18]).collect();
                        let ins: Vec<&Vec<u16>> = [&a, &b].into_iter().take(c.n_inputs).collect();
                        let bufs: Vec<_> = ins
                            .iter()
                            .map(|v| {
                                let buf = ctx.create_buffer(n_el * 2).unwrap();
                                queue.write_buffer(&buf, 0, &bytes(v)).unwrap();
                                buf
                            })
                            .collect();
                        let out = ctx.create_buffer(n_el * 2).unwrap();
                        // A sentinel no result equals, so a kernel that never wrote is caught.
                        queue
                            .write_buffer(&out, 0, &vec![0xA5u8; n_el * 2])
                            .unwrap();
                        for (i, buf) in bufs.iter().enumerate() {
                            k.set_arg_buffer(i as u32, buf).unwrap();
                        }
                        k.set_arg_buffer(c.n_inputs as u32, &out).unwrap();
                        k.set_arg_i64(c.n_inputs as u32 + 1, n_el as i64).unwrap();
                        // SAFETY: the kernel guards `i < n`; every buffer holds n_el elements.
                        let ev = unsafe { queue.launch(&k, &[n_el], None) }.unwrap();
                        ev.wait().unwrap();
                        let mut raw = vec![0u8; n_el * 2];
                        queue.read_buffer(&out, 0, &mut raw).unwrap();
                        let got = read_bits(&raw);

                        let plan = build_plan(&op, &key);
                        let shape = [n_el as i64];
                        let tb: Vec<TypedBuffer> =
                            ins.iter().map(|v| typed(kind, &shape, &[1], v)).collect();
                        let want = evaluate(&plan, &operands, &tb, &[]);
                        let want = read_bits(&want[0].raw_bytes()[..n_el * 2]);
                        for i in 0..n_el {
                            assert!(
                                agrees(kind, c.check, got[i], want[i]),
                                "{label}: {kind:?} {} n={n_el} i={i} a={:#06x} b={:#06x}: device {:#06x}, oracle {:#06x}\n{}",
                                c.name,
                                a[i],
                                b[i],
                                got[i],
                                want[i],
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
    eprintln!("half kernels: {checks} case runs agree on {ran} device(s)");
}

/// Every 16-bit pattern through decode, an exact scaling and encode.
#[test]
fn every_half_pattern_round_trips_the_codec_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let backend = OpenClC::new();
    let all: Vec<u16> = (0..=u16::MAX).collect();
    let n = all.len();
    let mut ran = 0;
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
            for kind in FORMATS {
                // x * 1 is the identity round trip; x * 0.5 rounds subnormals
                // (ties included); x * 2 overflows to infinity.
                for scale in [1.0, 0.5, 2.0] {
                    let desc = OperandDesc::new(1, &[n as i64], &[1], kind, 256);
                    let key =
                        structure_key(OpCategory::UnaryElementwise, &[desc, desc], ArchSku::Sm89);
                    let op = OpDef::elementwise("scale", 1, &[kind], input(0) * konst(scale));
                    let kernel = generate(&op, &key, &backend);
                    let prog = match Program::build(&ctx, &kernel.source, "") {
                        Ok(p) => p,
                        Err(e) => {
                            panic!("{label}: {kind:?} failed to build:\n{e}\n{}", kernel.source)
                        }
                    };
                    let mut k = prog.kernel(&kernel.name).unwrap();
                    let inb = ctx.create_buffer(n * 2).unwrap();
                    queue.write_buffer(&inb, 0, &bytes(&all)).unwrap();
                    let out = ctx.create_buffer(n * 2).unwrap();
                    queue.write_buffer(&out, 0, &vec![0xA5u8; n * 2]).unwrap();
                    k.set_arg_buffer(0, &inb).unwrap();
                    k.set_arg_buffer(1, &out).unwrap();
                    k.set_arg_i64(2, n as i64).unwrap();
                    // SAFETY: the kernel guards `i < n`; both buffers hold n elements.
                    let ev = unsafe { queue.launch(&k, &[n], None) }.unwrap();
                    ev.wait().unwrap();
                    let mut raw = vec![0u8; n * 2];
                    queue.read_buffer(&out, 0, &mut raw).unwrap();
                    let got = read_bits(&raw);
                    let plan = build_plan(&op, &key);
                    let want = evaluate(
                        &plan,
                        &[desc, desc],
                        &[typed(kind, &[n as i64], &[1], &all)],
                        &[],
                    );
                    let want = read_bits(&want[0].raw_bytes()[..n * 2]);
                    let bad: Vec<usize> = (0..n)
                        .filter(|&i| !agrees(kind, Check::ArithNan, got[i], want[i]))
                        .collect();
                    assert!(
                        bad.is_empty(),
                        "{label}: {kind:?} x*{scale}: {} of {n} patterns differ, first {:#06x}: device {:#06x}, oracle {:#06x}",
                        bad.len(),
                        bad[0],
                        got[bad[0]],
                        want[bad[0]]
                    );
                }
            }
            ran += 1;
        }
    }
    eprintln!("every bf16 and f16 pattern round-trips on {ran} device(s)");
}

/// The strided schedule: the second input is broadcast along axis 0.
#[test]
fn half_strided_kernels_agree_with_the_oracle_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let backend = OpenClC::new();
    let shape = [4i64, 70];
    let n = 280usize;
    let mut ran = 0;
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
            for kind in FORMATS {
                let pl = pool(kind);
                for (name, body, check) in [
                    ("add", input(0) + input(1), Check::ArithNan),
                    ("max", input(0).binary(BinaryOp::Max, input(1)), Check::Bits),
                ] {
                    let a0 = OperandDesc::new(2, &shape, &[70, 1], kind, 256);
                    let a1 = OperandDesc::new(2, &shape, &[0, 1], kind, 256);
                    let o = OperandDesc::new(2, &shape, &[70, 1], kind, 256);
                    let key =
                        structure_key(OpCategory::BinaryElementwise, &[a0, a1, o], ArchSku::Sm89);
                    let op = OpDef::elementwise(name, 2, &[kind], body);
                    let kernel = generate(&op, &key, &backend);
                    let prog = match Program::build(&ctx, &kernel.source, "") {
                        Ok(p) => p,
                        Err(e) => panic!("{label}: strided {name} failed:\n{e}\n{}", kernel.source),
                    };
                    let mut k = prog.kernel(&kernel.name).unwrap();
                    let x: Vec<u16> = (0..n).map(|i| pl[i % 18]).collect();
                    let y: Vec<u16> = (0..70).map(|i| pl[(i / 3) % 18]).collect();
                    let bx = ctx.create_buffer(n * 2).unwrap();
                    queue.write_buffer(&bx, 0, &bytes(&x)).unwrap();
                    let by = ctx.create_buffer(70 * 2).unwrap();
                    queue.write_buffer(&by, 0, &bytes(&y)).unwrap();
                    let out = ctx.create_buffer(n * 2).unwrap();
                    queue.write_buffer(&out, 0, &vec![0xA5u8; n * 2]).unwrap();
                    k.set_arg_buffer(0, &bx).unwrap();
                    k.set_arg_buffer(1, &by).unwrap();
                    k.set_arg_buffer(2, &out).unwrap();
                    // shape, input strides (x dense, y broadcast), output strides, n.
                    let scalars: [i64; 8] = [4, 70, 70, 1, 0, 1, 70, 1];
                    let mut arg = 3;
                    for v in &scalars {
                        k.set_arg_i64(arg, *v).unwrap();
                        arg += 1;
                    }
                    k.set_arg_i64(arg, n as i64).unwrap();
                    // SAFETY: the kernel guards `i < n`; x and out hold n elements and y
                    // holds the 70 of the broadcast row.
                    let ev = unsafe { queue.launch(&k, &[n], None) }.unwrap();
                    ev.wait().unwrap();
                    let mut raw = vec![0u8; n * 2];
                    queue.read_buffer(&out, 0, &mut raw).unwrap();
                    let got = read_bits(&raw);

                    let plan = build_plan(&op, &key);
                    let full: Vec<u16> = (0..n).map(|i| y[i % 70]).collect();
                    let want = evaluate(
                        &plan,
                        &[a0, a1, o],
                        &[
                            typed(kind, &shape, &[70, 1], &x),
                            typed(kind, &shape, &[0, 1], &full),
                        ],
                        &[],
                    );
                    let want = read_bits(&want[0].raw_bytes()[..n * 2]);
                    for i in 0..n {
                        assert!(
                            agrees(kind, check, got[i], want[i]),
                            "{label}: {kind:?} strided {name} i={i} x={:#06x} y={:#06x}: device {:#06x}, oracle {:#06x}\n{}",
                            x[i],
                            y[i % 70],
                            got[i],
                            want[i],
                            kernel.source
                        );
                    }
                }
            }
            ran += 1;
        }
    }
    eprintln!("half strided kernels agree on {ran} device(s)");
}

#[test]
fn half_plans_the_emitter_cannot_check_are_declined_by_name() {
    let backend = OpenClC::new();
    for (kind, tag) in [(ElementKind::Bf16, "bf16"), (ElementKind::F16, "f16")] {
        let desc = OperandDesc::new(1, &[8], &[1], kind, 256);
        // A constant the format cannot hold exactly.
        let key = structure_key(OpCategory::UnaryElementwise, &[desc, desc], ArchSku::Sm89);
        let op = OpDef::elementwise("c", 1, &[kind], input(0) + konst(0.1));
        let err = backend
            .lower(&build_plan(&op, &key))
            .expect_err("an inexact constant lowered");
        assert!(format!("{err:?}").contains(tag), "{err:?}");
        // Control: an exact one lowers.
        let op = OpDef::elementwise("c", 1, &[kind], input(0) + konst(0.5));
        assert!(backend.lower(&build_plan(&op, &key)).is_ok());
        // Reductions are not served yet.
        let a = OperandDesc::new(2, &[4, 6], &[6, 1], kind, 256);
        let o = OperandDesc::new(1, &[4], &[1], kind, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let op = OpDef::reduction("r", 1, &[kind], input(0), ReduceOp::Sum);
        let err = backend
            .lower(&build_plan(&op, &key))
            .expect_err("a half reduction lowered");
        assert!(
            format!("{err:?}").contains(&format!("{tag} reductions")),
            "{err:?}"
        );
    }
}
