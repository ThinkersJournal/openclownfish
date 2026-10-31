//! Emit OpenCL C, build and run it on every device here, and compare with
//! Unpopped's independent f64 oracle. The output buffer starts as NaN so a
//! kernel that never wrote is told apart from one that wrote a wrong value.
//!
//! Devices that cannot make a context or build the source are reported and
//! counted, not hidden: this is a measurement of each vendor stack.

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::ir::{BinaryOp, OpDef, UnaryOp, input, konst};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::{build_plan, generate};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_ne_bytes(*c))
        .collect()
}

/// Distance in representable f32 values (sign-magnitude folded to a line).
fn ulps(a: f32, b: f32) -> u64 {
    fn key(x: f32) -> i64 {
        let i = i64::from(x.to_bits() as i32);
        if i < 0 { i64::from(i32::MIN) - i } else { i }
    }
    key(a).abs_diff(key(b))
}

struct Case {
    name: &'static str,
    op: OpDef,
    cat: OpCategory,
    n_inputs: usize,
    /// Largest accepted ULP distance from the oracle: 0 where IEEE makes the
    /// result exact, 3 for division (OpenCL allows 2.5 ULP by default).
    max_ulps: u64,
    /// Inputs restricted to positive values (sqrt, log, rsqrt).
    positive: bool,
    /// Host f32 arithmetic is a faithful reference (true for IEEE-exact ops).
    /// False for exp/log/rsqrt, whose host libm results are not the standard.
    host_exact: bool,
    /// Inputs are the NaN/inf/signed-zero grid and results are compared by
    /// bit pattern (NaN only by being NaN), not by ULP.
    special: bool,
}

fn cases() -> Vec<Case> {
    let f = [ElementKind::F32];
    vec![
        Case {
            name: "add",
            op: OpDef::elementwise("add", 2, &f, input(0) + input(1)),
            cat: OpCategory::BinaryElementwise,
            n_inputs: 2,
            max_ulps: 0,
            positive: false,
            host_exact: true,
            special: false,
        },
        Case {
            name: "sub",
            op: OpDef::elementwise("sub", 2, &f, input(0) - input(1)),
            cat: OpCategory::BinaryElementwise,
            n_inputs: 2,
            max_ulps: 0,
            positive: false,
            host_exact: true,
            special: false,
        },
        Case {
            name: "mul",
            op: OpDef::elementwise("mul", 2, &f, input(0) * input(1)),
            cat: OpCategory::BinaryElementwise,
            n_inputs: 2,
            max_ulps: 0,
            positive: false,
            host_exact: true,
            special: false,
        },
        // a*b+c: bit-exact only if the compiler did not fuse it.
        Case {
            name: "muladd",
            op: OpDef::elementwise("muladd", 3, &f, input(0) * input(1) + input(2)),
            cat: OpCategory::BinaryElementwise,
            n_inputs: 3,
            max_ulps: 0,
            positive: false,
            host_exact: true,
            special: false,
        },
        Case {
            name: "neg_scale",
            op: OpDef::elementwise(
                "neg_scale",
                1,
                &f,
                input(0).unary(UnaryOp::Neg) * konst(2.5),
            ),
            cat: OpCategory::UnaryElementwise,
            n_inputs: 1,
            max_ulps: 0,
            positive: false,
            host_exact: true,
            special: false,
        },
        Case {
            name: "div",
            op: OpDef::elementwise("div", 2, &f, input(0) / input(1)),
            cat: OpCategory::BinaryElementwise,
            n_inputs: 2,
            max_ulps: 3,
            positive: false,
            host_exact: true,
            special: false,
        },
        Case {
            name: "abs",
            op: OpDef::elementwise("abs", 1, &f, input(0).unary(UnaryOp::Abs)),
            cat: OpCategory::UnaryElementwise,
            n_inputs: 1,
            max_ulps: 0,
            positive: false,
            host_exact: true,
            special: false,
        },
        Case {
            name: "sqr",
            op: OpDef::elementwise("sqr", 1, &f, input(0).unary(UnaryOp::Sqr)),
            cat: OpCategory::UnaryElementwise,
            n_inputs: 1,
            max_ulps: 0,
            positive: false,
            host_exact: true,
            special: false,
        },
        Case {
            name: "recip",
            op: OpDef::elementwise("recip", 1, &f, input(0).unary(UnaryOp::Recip)),
            cat: OpCategory::UnaryElementwise,
            n_inputs: 1,
            max_ulps: 3,
            positive: false,
            host_exact: true,
            special: false,
        },
        Case {
            name: "sqrt",
            op: OpDef::elementwise("sqrt", 1, &f, input(0).unary(UnaryOp::Sqrt)),
            cat: OpCategory::UnaryElementwise,
            n_inputs: 1,
            max_ulps: 3,
            positive: true,
            host_exact: true,
            special: false,
        },
        Case {
            name: "rsqrt",
            op: OpDef::elementwise("rsqrt", 1, &f, input(0).unary(UnaryOp::Rsqrt)),
            cat: OpCategory::UnaryElementwise,
            n_inputs: 1,
            max_ulps: 4,
            positive: true,
            host_exact: false,
            special: false,
        },
        Case {
            name: "exp",
            op: OpDef::elementwise("exp", 1, &f, input(0).unary(UnaryOp::Exp)),
            cat: OpCategory::UnaryElementwise,
            n_inputs: 1,
            max_ulps: 4,
            positive: false,
            host_exact: false,
            special: false,
        },
        Case {
            name: "log",
            op: OpDef::elementwise("log", 1, &f, input(0).unary(UnaryOp::Log)),
            cat: OpCategory::UnaryElementwise,
            n_inputs: 1,
            max_ulps: 4,
            positive: true,
            host_exact: false,
            special: false,
        },
        Case {
            name: "max",
            op: OpDef::elementwise("max", 2, &f, input(0).max(input(1))),
            cat: OpCategory::BinaryElementwise,
            n_inputs: 2,
            max_ulps: 0,
            positive: false,
            host_exact: false,
            special: true,
        },
        Case {
            name: "min",
            op: OpDef::elementwise("min", 2, &f, input(0).min(input(1))),
            cat: OpCategory::BinaryElementwise,
            n_inputs: 2,
            max_ulps: 0,
            positive: false,
            host_exact: false,
            special: true,
        },
        Case {
            name: "fmax_ieee",
            op: OpDef::elementwise(
                "fmax_ieee",
                2,
                &f,
                input(0).binary(BinaryOp::FmaxIeee, input(1)),
            ),
            cat: OpCategory::BinaryElementwise,
            n_inputs: 2,
            max_ulps: 0,
            positive: false,
            host_exact: false,
            special: true,
        },
        Case {
            name: "fmin_ieee",
            op: OpDef::elementwise(
                "fmin_ieee",
                2,
                &f,
                input(0).binary(BinaryOp::FminIeee, input(1)),
            ),
            cat: OpCategory::BinaryElementwise,
            n_inputs: 2,
            max_ulps: 0,
            positive: false,
            host_exact: false,
            special: true,
        },
        // OpenCL allows pow 16 ULP; one more so a stack at the limit is not flaky.
        Case {
            name: "pow",
            op: OpDef::elementwise("pow", 1, &f, input(0).binary(BinaryOp::Pow, konst(2.5))),
            cat: OpCategory::UnaryElementwise,
            n_inputs: 1,
            max_ulps: 17,
            positive: true,
            host_exact: false,
            special: false,
        },
    ]
    .into_iter()
    .chain(special_cases())
    .collect()
}

/// Comparisons and `Select`, over the NaN/inf/zero grid and bit for bit: a
/// comparison is exactly 1.0 or 0.0, and `Select` moves the chosen arm
/// untouched (a NaN or a signed zero arrives as it left).
fn special_cases() -> Vec<Case> {
    let f = [ElementKind::F32];
    let mut v = Vec::new();
    for (name, op) in [
        ("cmp_eq", BinaryOp::CmpEq),
        ("cmp_ne", BinaryOp::CmpNe),
        ("cmp_lt", BinaryOp::CmpLt),
        ("cmp_le", BinaryOp::CmpLe),
        ("cmp_gt", BinaryOp::CmpGt),
        ("cmp_ge", BinaryOp::CmpGe),
    ] {
        v.push(Case {
            name,
            op: OpDef::elementwise(name, 2, &f, input(0).binary(op, input(1))),
            cat: OpCategory::BinaryElementwise,
            n_inputs: 2,
            max_ulps: 0,
            positive: false,
            host_exact: false,
            special: true,
        });
    }
    // cond = input(0): NaN selects the first arm, -0.0 and +0.0 the second.
    v.push(Case {
        name: "select",
        op: OpDef::elementwise("select", 2, &f, input(0).select(input(1), konst(-3.0))),
        cat: OpCategory::BinaryElementwise,
        n_inputs: 2,
        max_ulps: 0,
        positive: false,
        host_exact: false,
        special: true,
    });
    v
}

/// Inputs chosen so rounding differs between fused and unfused arithmetic and
/// between correctly and loosely rounded division.
fn data(n: usize, which: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f32 + 1.0) * 0.173 + which as f32 * 0.37;
            if i % 7 == 3 { -x * 1.0001 } else { x * 1.3 }
        })
        .collect()
}

/// Every pairing of these nine values appears (81 of the first elements), so NaN
/// in either operand, both infinities, and both zeros all meet each other.
const SPECIAL: [f32; 9] = [
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.5,
    -2.5,
];

fn special_data(n: usize, which: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let k = if which == 0 { i % 9 } else { (i / 9) % 9 };
            SPECIAL[k]
        })
        .collect()
}

/// 257 is not a multiple of any common vector width (Scalar schedule); 256 is
/// (Vectorized schedule).
#[test]
fn emitted_kernels_agree_with_the_oracle_on_every_device() {
    for n_el in [257, 256] {
        run_on_every_device(n_el);
    }
}

fn run_on_every_device(n_el: usize) {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let (mut ran, mut skipped) = (0, 0);
    for p in platforms {
        let Ok(devices) = p.devices() else { continue };
        for d in devices {
            let label = format!(
                "{} / {}",
                p.name().unwrap_or_default(),
                d.name().unwrap_or_default()
            );
            let (Ok(ctx), true) = (Context::new(&d), true) else {
                eprintln!("SKIP {label}: no context");
                skipped += 1;
                continue;
            };
            let queue = ctx.create_queue().unwrap();
            for c in cases() {
                let desc = OperandDesc::new(1, &[n_el as i64], &[1], ElementKind::F32, 256);
                let mut operands = vec![desc; c.n_inputs];
                operands.push(desc);
                let key = structure_key(c.cat, &operands, ArchSku::Sm89);
                let kernel = generate(&c.op, &key, &OpenClC);
                let prog = match Program::build(&ctx, &kernel.source, "") {
                    Ok(p) => p,
                    Err(e) => panic!(
                        "{label}: {} failed to build:\n{e}\n{}",
                        c.name, kernel.source
                    ),
                };
                let mut k = prog.kernel(&kernel.name).unwrap();
                let inputs: Vec<Vec<f32>> = (0..c.n_inputs)
                    .map(|w| {
                        let v = if c.special {
                            special_data(n_el, w)
                        } else {
                            data(n_el, w)
                        };
                        if c.positive {
                            v.into_iter().map(|x| x.abs() + 0.01).collect()
                        } else {
                            v
                        }
                    })
                    .collect();
                let bufs: Vec<_> = inputs
                    .iter()
                    .map(|v| {
                        let b = ctx.create_buffer(n_el * 4).unwrap();
                        queue.write_buffer(&b, 0, &bytes(v)).unwrap();
                        b
                    })
                    .collect();
                let out = ctx.create_buffer(n_el * 4).unwrap();
                queue
                    .write_buffer(&out, 0, &bytes(&vec![f32::NAN; n_el]))
                    .unwrap();
                for (i, b) in bufs.iter().enumerate() {
                    k.set_arg_buffer(i as u32, b).unwrap();
                }
                k.set_arg_buffer(c.n_inputs as u32, &out).unwrap();
                k.set_arg_i64(c.n_inputs as u32 + 1, n_el as i64).unwrap();
                // SAFETY: the kernel guards `i < n` and every buffer holds n_el floats.
                let ev = unsafe { queue.launch(&k, &[n_el], None) }.unwrap();
                ev.wait().unwrap();
                let mut raw = vec![0u8; n_el * 4];
                queue.read_buffer(&out, 0, &mut raw).unwrap();
                let got = floats(&raw);

                let plan = build_plan(&c.op, &key);
                let tb: Vec<TypedBuffer> = inputs
                    .iter()
                    .map(|v| TypedBuffer::from_f32(&[n_el as i64], v))
                    .collect();
                let want = evaluate(&plan, &operands, &tb, &[])
                    .into_iter()
                    .next()
                    .unwrap()
                    .to_f64_vec();
                // Host f32 arithmetic, rounded after every operation: what an
                // unfused device must produce. The oracle works in f64 and rounds
                // once, so it can differ from this on a*b+c (double rounding).
                let host: Vec<f32> = (0..n_el)
                    .map(|i| {
                        let x = |k: usize| inputs[k][i];
                        match c.name {
                            "add" => x(0) + x(1),
                            "sub" => x(0) - x(1),
                            "mul" => x(0) * x(1),
                            "muladd" => (x(0) * x(1)) + x(2),
                            "neg_scale" => -x(0) * 2.5,
                            "div" => x(0) / x(1),
                            "abs" => x(0).abs(),
                            "sqr" => x(0) * x(0),
                            "recip" => 1.0 / x(0),
                            "sqrt" => x(0).sqrt(),
                            // exp, log, rsqrt: not host_exact; never compared.
                            _ => f32::NAN,
                        }
                    })
                    .collect();
                if c.special {
                    for (i, g) in got.iter().enumerate() {
                        let w = want[i] as f32;
                        let ok = if w.is_nan() {
                            g.is_nan()
                        } else if c.name.ends_with("_ieee") && w == 0.0 {
                            // IEEE maxNum/minNum leave the sign of a zero tie open.
                            *g == 0.0
                        } else {
                            g.to_bits() == w.to_bits()
                        };
                        assert!(
                            ok,
                            "{label}: {} {:?} {:?} -> {g:?} (bits {:#x}), oracle {w:?} (bits {:#x})",
                            c.name,
                            inputs[0][i],
                            inputs[1][i],
                            g.to_bits(),
                            w.to_bits()
                        );
                    }
                    eprintln!("{label}: {} exact over the NaN/inf/zero grid", c.name);
                    continue;
                }
                let vs_host = got
                    .iter()
                    .zip(&host)
                    .map(|(g, h)| ulps(*g, *h))
                    .max()
                    .unwrap();
                let vs_oracle = got
                    .iter()
                    .zip(&want)
                    .map(|(g, w)| ulps(*g, *w as f32))
                    .max()
                    .unwrap();
                let host_note = if c.host_exact {
                    format!("{vs_host} ulp")
                } else {
                    "n/a".to_owned()
                };
                eprintln!(
                    "{label}: {} vs host-f32 {host_note}, vs oracle {vs_oracle} ulp",
                    c.name
                );
                for (i, g) in got.iter().enumerate() {
                    assert!(
                        !g.is_nan() || want[i].is_nan(),
                        "{label}: {} NEVER-WROTE at index {i}",
                        c.name
                    );
                }
                // Primary check: the unfused host result. With FP_CONTRACT OFF a
                // device must match it exactly (division: OpenCL's 2.5 ULP).
                assert!(
                    !c.host_exact || vs_host <= c.max_ulps,
                    "{label}: {} off by {vs_host} ulp from host f32 (limit {})",
                    c.name,
                    c.max_ulps
                );
                // The oracle rounds once in f64, so a*b+c differs from unfused
                // f32 by design (see `muladd`); every single-rounding op must agree.
                if c.name != "muladd" {
                    assert!(
                        vs_oracle <= c.max_ulps,
                        "{label}: {} off by {vs_oracle} ulp from the oracle (limit {})",
                        c.name,
                        c.max_ulps
                    );
                }
            }
            ran += 1;
        }
    }
    eprintln!("emitted kernels ran on {ran} device(s), skipped {skipped}");
}

#[test]
fn unsupported_plans_are_refused_not_approximated() {
    // tanh is outside this version: must be a typed refusal, never a guess.
    let op = OpDef::elementwise(
        "root",
        1,
        &[ElementKind::F32],
        input(0).unary(UnaryOp::Tanh),
    );
    let desc = OperandDesc::new(1, &[8], &[1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[desc, desc], ArchSku::Sm89);
    let plan = build_plan(&op, &key);
    let err = unpopped::backend::Backend::lower(&OpenClC, &plan).unwrap_err();
    assert!(err.to_string().contains("opencl_c v1"), "{err}");
    assert!(err.to_string().contains("Tanh"), "{err}");
}
