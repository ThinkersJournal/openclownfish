// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Backend::lower_kernel` for `F32` and `F64` kernels, on every OpenCL device here, against
//! Unpopped's launch interpreter (`run_launch`).
//!
//! Three kernels over 128 threads: a Horner loop (`acc = acc * x + y` eight times, so each
//! multiply and add must round on its own: this is what `FP_CONTRACT OFF` protects), a branch
//! that uses `Div`, `Neg`, `Fma`, a `Const` and a comparison, and a `While` that accumulates
//! squares to a limit. The first two take inputs that include NaN, both infinities and both
//! zeros. Results must match bit for bit (two NaNs agree whatever their payload). `F64` runs
//! only on a device that lists `cl_khr_fp64`. The declines are checked in source.

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr, UnaryOp};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const BLOCK: u32 = 32;
const GRID: u32 = 4;
const T: usize = (BLOCK * GRID) as usize;

fn width(e: ElementKind) -> usize {
    if e == ElementKind::F32 { 4 } else { 8 }
}

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(bx(a), bx(b))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(bx(a), bx(b))
}

fn cmp(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, bx(a), bx(b))
}

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn gtid() -> ScalarExpr {
    add(
        mul(
            ScalarExpr::Exec(ExecId::BlockIdx(0)),
            ScalarExpr::Exec(ExecId::BlockDim(0)),
        ),
        ScalarExpr::Exec(ExecId::ThreadIdx(0)),
    )
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![gtid()],
    }
}

fn launch(mut k: KernelFn) -> KernelFn {
    k.launch = Some(LaunchShape::new([BLOCK, 1, 1], [GRID, 1, 1]));
    k
}

/// Locals: 0 x, 1 y, 2 acc, 3 loop counter (I32).
fn kernel(name: &str, e: ElementKind, body: Vec<Stmt>) -> KernelFn {
    let mut stmts = vec![
        Stmt::Load {
            local: 0,
            place: at(0),
        },
        Stmt::Load {
            local: 1,
            place: at(1),
        },
        Stmt::Let {
            local: 2,
            init: ScalarExpr::Const(0.0),
        },
    ];
    stmts.extend(body);
    stmts.push(Stmt::Store {
        place: at(2),
        value: local(2),
    });
    launch(KernelFn::new(
        name,
        e,
        vec![
            Param::buffer("x", e, BufferRole::Read, Some(1)),
            Param::buffer("y", e, BufferRole::Read, Some(1)),
            Param::buffer("out", e, BufferRole::ReadWrite, Some(1)),
        ],
        vec![
            LocalDecl::scalar(e),
            LocalDecl::scalar(e),
            LocalDecl::scalar(e),
            LocalDecl::scalar(ElementKind::I32),
        ],
        Block(stmts),
    ))
}

fn horner(e: ElementKind) -> KernelFn {
    kernel(
        "horner",
        e,
        vec![Stmt::For {
            local: 3,
            start: ScalarExpr::ConstInt(0, ElementKind::I32),
            end: ScalarExpr::ConstInt(8, ElementKind::I32),
            step: ScalarExpr::ConstInt(1, ElementKind::I32),
            body: Block(vec![Stmt::Assign {
                local: 2,
                value: add(mul(local(2), local(0)), local(1)),
            }]),
        }],
    )
}

fn branchy(e: ElementKind) -> KernelFn {
    kernel(
        "branchy",
        e,
        vec![Stmt::If {
            cond: cmp(BinaryOp::CmpLt, local(0), local(1)),
            then: Block(vec![Stmt::Assign {
                local: 2,
                value: add(
                    ScalarExpr::Unary(UnaryOp::Neg, bx(mul(local(0), local(1)))),
                    ScalarExpr::Const(0.1),
                ),
            }]),
            els: Block(vec![Stmt::Assign {
                local: 2,
                value: add(
                    ScalarExpr::Div(bx(local(0)), bx(local(1))),
                    ScalarExpr::Fma(bx(local(0)), bx(local(0)), bx(local(1))),
                ),
            }]),
        }],
    )
}

/// One rounding operation per statement: the form an `F32` kernel may take without `with_fp64`.
fn single(name: &str, e: ElementKind, div: bool) -> KernelFn {
    let value = if div {
        ScalarExpr::Div(bx(local(0)), bx(local(1)))
    } else {
        mul(local(0), local(1))
    };
    kernel(name, e, vec![Stmt::Assign { local: 2, value }])
}

fn while_sum(e: ElementKind) -> KernelFn {
    kernel(
        "while_sum",
        e,
        vec![Stmt::While {
            cond: cmp(BinaryOp::CmpLt, local(2), ScalarExpr::Const(20.0)),
            body: Block(vec![Stmt::Assign {
                local: 2,
                value: add(local(2), mul(local(0), local(0))),
            }]),
        }],
    )
}

/// A deterministic value in `[0, 1)`.
fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

fn bytes(v: f64, e: ElementKind) -> Vec<u8> {
    if e == ElementKind::F32 {
        (v as f32).to_le_bytes().to_vec()
    } else {
        v.to_le_bytes().to_vec()
    }
}

/// `(x, y)` buffers. With `specials`, lanes of `x` carry NaN, the infinities and the zeros;
/// without, `x` is in `[0.5, 2)` so a `While` on its square ends.
fn inputs(e: ElementKind, specials: bool) -> Vec<Vec<u8>> {
    let mut s = 0x1234_5678_9abc_def0u64;
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for t in 0..T {
        let mut x = if specials {
            (unit(&mut s) - 0.5) * 8.0
        } else {
            0.5 + unit(&mut s) * 1.5
        };
        if specials {
            x = match t % 16 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => -0.0,
                4 => 0.0,
                _ => x,
            };
        }
        let mut y = (unit(&mut s) - 0.5) * 8.0;
        if y == 0.0 {
            y = 1.0;
        }
        xs.extend(bytes(x, e));
        ys.extend(bytes(y, e));
    }
    vec![xs, ys, vec![0u8; T * width(e)]]
}

fn oracle(k: &KernelFn, init: &[Vec<u8>], e: ElementKind) -> Vec<u8> {
    let mut args: Vec<KernelArg> = init
        .iter()
        .map(|b| KernelArg::Buffer(TypedBuffer::new(e, vec![T as i64], vec![1], b.clone())))
        .collect();
    run_launch(k, &mut args).expect("the interpreter runs the kernel");
    match args.pop() {
        Some(KernelArg::Buffer(b)) => b.bytes,
        _ => unreachable!(),
    }
}

fn device(
    ctx: &Context,
    queue: &openclownfish::Queue,
    backend: &OpenClC,
    k: &KernelFn,
    init: &[Vec<u8>],
) -> Result<Vec<u8>, String> {
    let g = backend.lower_kernel(k).expect("lowers");
    let prog = Program::build(ctx, &g.source, "").map_err(|e| format!("{e}\n{}", g.source))?;
    let mut kern = prog.kernel(&g.name).unwrap();
    // The buffers must outlive the launch.
    let mut held = Vec::new();
    for (i, b) in init.iter().enumerate() {
        let buf = ctx.create_buffer(b.len()).unwrap();
        queue.write_buffer(&buf, 0, b).unwrap();
        kern.set_arg_buffer(i as u32, &buf).unwrap();
        held.push(buf);
    }
    // SAFETY: every access is by construction inside its buffer; `held` outlives the launch.
    unsafe { queue.launch(&kern, &[T], Some(&[BLOCK as usize])) }
        .unwrap()
        .wait()
        .unwrap();
    let last = init.len() - 1;
    let mut raw = vec![0u8; init[last].len()];
    queue.read_buffer(&held[last], 0, &mut raw).unwrap();
    Ok(raw)
}

/// Bit equality, with any two NaNs equal.
fn same(a: &[u8], b: &[u8], e: ElementKind) -> Result<(), String> {
    let w = width(e);
    for (i, (x, y)) in a.chunks(w).zip(b.chunks(w)).enumerate() {
        let (nx, ny) = if e == ElementKind::F32 {
            (
                f32::from_le_bytes(x.try_into().unwrap()).is_nan(),
                f32::from_le_bytes(y.try_into().unwrap()).is_nan(),
            )
        } else {
            (
                f64::from_le_bytes(x.try_into().unwrap()).is_nan(),
                f64::from_le_bytes(y.try_into().unwrap()).is_nan(),
            )
        };
        if x != y && !(nx && ny) {
            return Err(format!("element {i}: device {x:02x?}, oracle {y:02x?}"));
        }
    }
    Ok(())
}

#[test]
fn float_kernels_agree_with_the_interpreter_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let (mut kernels, mut devices_run, mut f64_skipped) = (0, 0, 0);
    for p in platforms {
        let Ok(devs) = p.devices() else { continue };
        for d in devs {
            let label = format!(
                "{} / {}",
                p.name().unwrap_or_default(),
                d.name().unwrap_or_default()
            );
            let has64 = d
                .extensions()
                .map(|x| x.iter().any(|n| n == "cl_khr_fp64"))
                .unwrap_or(false);
            let Ok(ctx) = Context::new(&d) else { continue };
            let queue = ctx.create_queue().unwrap();
            devices_run += 1;
            // One rounding operation per statement, without `with_fp64`: F32 on every device.
            let plain = OpenClC::new();
            {
                let k = single("single_mul", ElementKind::F32, false);
                let e = ElementKind::F32;
                let data = inputs(e, true);
                let want = oracle(&k, &data, e);
                let got = device(&ctx, &queue, &plain, &k, &data)
                    .unwrap_or_else(|l| panic!("{label}: {} failed to build:\n{l}", k.name));
                kernels += 1;
                if let Err(m) = same(&got, &want, e) {
                    panic!("{label}: {e:?} {} without fp64: {m}", k.name);
                }
            }
            // Evaluated at double and rounded per statement, as the oracle does.
            for e in [ElementKind::F32, ElementKind::F64] {
                if !has64 {
                    f64_skipped += 1;
                    continue;
                }
                let wide = OpenClC::new().with_fp64(true);
                let mut cases: Vec<(KernelFn, bool)> = vec![
                    (horner(e), true),
                    (while_sum(e), false),
                    (single("single_mul", e, false), true),
                ];
                if e == ElementKind::F64 {
                    // `branchy` and `single_div` divide; F32 division is declined.
                    cases.push((branchy(e), true));
                    cases.push((single("single_div", e, true), true));
                }
                for (k, specials) in cases {
                    let data = inputs(e, specials);
                    let want = oracle(&k, &data, e);
                    let got = device(&ctx, &queue, &wide, &k, &data).unwrap_or_else(|l| {
                        panic!(
                            "{label}: {e:?} {} failed to build:
{l}",
                            k.name
                        )
                    });
                    kernels += 1;
                    if let Err(m) = same(&got, &want, e) {
                        panic!("{label}: {e:?} {}: {m}", k.name);
                    }
                }
            }
        }
    }
    eprintln!(
        "{kernels} float kernels agree on {devices_run} device(s); F64 skipped {f64_skipped} time(s)"
    );
}

fn declined(k: &KernelFn, oc: &OpenClC, needle: &str) {
    match oc.lower_kernel(k) {
        Err(unpopped::backend::LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains(needle), "{detail:?} lacks {needle:?}");
        }
        other => panic!("expected a decline mentioning {needle:?}, got {other:?}"),
    }
}

#[test]
fn float_spelling_and_declines() {
    let on = OpenClC::new().with_fp64(true);
    let src = on
        .lower_kernel(&horner(ElementKind::F32))
        .expect("lowers")
        .source;
    assert!(src.contains("#pragma OPENCL FP_CONTRACT OFF"), "{src}");
    // F32 statements are evaluated at double, as the oracle does, so the extension is on.
    assert!(
        src.contains("#pragma OPENCL EXTENSION cl_khr_fp64 : enable"),
        "{src}"
    );
    assert!(src.contains("((double)l2)"), "{src}");
    // Without with_fp64 a statement of two rounding operations is declined, one is not, and
    // a constant that is not exactly an f32 is declined.
    declined(
        &horner(ElementKind::F32),
        &OpenClC::new(),
        "more than one rounding",
    );
    declined(
        &branchy(ElementKind::F32),
        &OpenClC::new(),
        "more than one rounding",
    );
    let src = OpenClC::new()
        .lower_kernel(&single("single_mul", ElementKind::F32, false))
        .expect("lowers")
        .source;
    assert!(!src.contains("double"), "{src}");
    // NVIDIA's native float division is 1 ULP off the oracle, so an F32 Div without with_fp64
    // is declined (with it the division runs at double).
    // NVIDIA's float division is 1 ULP off the oracle, with or without with_fp64.
    for oc in [OpenClC::new(), on] {
        declined(
            &single("single_div", ElementKind::F32, true),
            &oc,
            "F32 Div",
        );
    }
    let mut k = single("single_mul", ElementKind::F32, false);
    k.body.0[3] = Stmt::Assign {
        local: 2,
        value: add(local(0), ScalarExpr::Const(0.1)),
    };
    declined(&k, &OpenClC::new(), "not exactly an f32");
    assert!(src.contains("__global const float* b0"), "{src}");
    let src = on
        .lower_kernel(&branchy(ElementKind::F64))
        .expect("lowers")
        .source;
    assert!(
        src.contains("#pragma OPENCL EXTENSION cl_khr_fp64 : enable"),
        "{src}"
    );
    assert!(src.contains("fma(("), "{src}");
    assert!(src.contains("((double)0.1)"), "{src}");
    // F64 needs the opt-in.
    declined(&branchy(ElementKind::F64), &OpenClC::new(), "with_fp64");
    // A float Max is declined by name.
    let mut k = horner(ElementKind::F32);
    k.body.0.insert(
        3,
        Stmt::Assign {
            local: 2,
            value: cmp(BinaryOp::Max, local(0), local(1)),
        },
    );
    declined(&k, &on, "Max or Min");
    // A float atomic is declined even with the atomics opt-in.
    let mut k = horner(ElementKind::F32);
    k.body.0.insert(
        3,
        Stmt::AtomicRmw {
            op: AtomicOp::Add,
            place: at(2),
            value: local(0),
            old: None,
            order: MemoryOrder::AcqRel,
            scope: unpopped::exec::Scope::Device,
        },
    );
    assert!(
        on.with_c20_atomics(true).lower_kernel(&k).is_err(),
        "a float atomic is declined"
    );
    // A non-finite constant is declined.
    let mut k = horner(ElementKind::F32);
    k.body.0[2] = Stmt::Let {
        local: 2,
        init: ScalarExpr::Const(f64::NAN),
    };
    declined(&k, &on, "non-finite");
}
