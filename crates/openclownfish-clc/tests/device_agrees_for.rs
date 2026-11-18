// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Backend::lower_kernel` for `Stmt::For`, on every OpenCL device here, against Unpopped's
//! launch interpreter (`run_launch`).
//!
//! Each of 128 threads runs a counted loop whose start, end and step come from buffers, over
//! I32, U32, I64 and U64. The cases are ordinary ranges, empty ranges, negative steps (signed),
//! a step larger than the range, and ranges that end next to the type's limits so the oracle's
//! stop-instead-of-wrap rule is exercised. Three kernels: a plain accumulate (order sensitive,
//! so the values and the count both show), one with `Continue` and `Break` in the body (the
//! advance must still happen on `Continue`), and a nested pair. A zero step is a caller
//! precondition and is not run. A mutant that drops the overflow stop is caught by the
//! near-limit cases.

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const BLOCK: u32 = 32;
const GRID: u32 = 4;
const T: usize = (BLOCK * GRID) as usize;

fn width(e: ElementKind) -> usize {
    match e {
        ElementKind::I32 | ElementKind::U32 => 4,
        _ => 8,
    }
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

fn ce(v: i64, e: ElementKind) -> ScalarExpr {
    ScalarExpr::ConstInt(v, e)
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

fn params(e: ElementKind) -> Vec<Param> {
    vec![
        Param::buffer("starts", e, BufferRole::Read, Some(1)),
        Param::buffer("ends", e, BufferRole::Read, Some(1)),
        Param::buffer("steps", e, BufferRole::Read, Some(1)),
        Param::buffer("out", e, BufferRole::ReadWrite, Some(1)),
    ]
}

/// Locals: 0 accumulator, 1 start, 2 end, 3 step, 4 counter, 5 outer counter.
fn kernel(name: &str, e: ElementKind, loop_body: Vec<Stmt>, nested: bool) -> KernelFn {
    let inner = Stmt::For {
        local: 4,
        start: local(1),
        end: local(2),
        step: local(3),
        unroll: Unroll::None,
        body: Block(loop_body),
    };
    let looped = if nested {
        Stmt::For {
            local: 5,
            start: ce(0, e),
            end: ce(3, e),
            step: ce(1, e),
            unroll: Unroll::None,
            body: Block(vec![inner]),
        }
    } else {
        inner
    };
    let body = vec![
        Stmt::Let {
            local: 0,
            init: ce(0, e),
        },
        Stmt::Load {
            local: 1,
            place: at(0),
        },
        Stmt::Load {
            local: 2,
            place: at(1),
        },
        Stmt::Load {
            local: 3,
            place: at(2),
        },
        looped,
        Stmt::Store {
            place: at(3),
            value: local(0),
        },
    ];
    launch(KernelFn::new(
        name,
        e,
        params(e),
        (0..6).map(|_| LocalDecl::scalar(e)).collect(),
        Block(body),
    ))
}

/// `acc = acc * 31 + counter + 1`: order sensitive, so the values and the count both show.
fn fold(e: ElementKind) -> Stmt {
    Stmt::Assign {
        local: 0,
        value: add(add(mul(local(0), ce(31, e)), local(4)), ce(1, e)),
    }
}

fn plain(e: ElementKind) -> KernelFn {
    kernel("for_plain", e, vec![fold(e)], false)
}

/// `Continue` below ten (the advance must still happen) and `Break` from forty.
fn control(e: ElementKind) -> KernelFn {
    let body = vec![
        Stmt::If {
            cond: cmp(BinaryOp::CmpLt, local(4), ce(10, e)),
            then: Block(vec![Stmt::Continue]),
            els: Block(vec![]),
        },
        Stmt::If {
            cond: cmp(BinaryOp::CmpGe, local(4), ce(40, e)),
            then: Block(vec![Stmt::Break]),
            els: Block(vec![]),
        },
        fold(e),
    ];
    kernel("for_control", e, body, false)
}

fn nested(e: ElementKind) -> KernelFn {
    kernel("for_nested", e, vec![fold(e)], true)
}

fn limits(e: ElementKind) -> (i128, i128) {
    match e {
        ElementKind::I32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
        ElementKind::U32 => (0, i128::from(u32::MAX)),
        ElementKind::I64 => (i128::from(i64::MIN), i128::from(i64::MAX)),
        _ => (0, i128::from(u64::MAX)),
    }
}

fn signed(e: ElementKind) -> bool {
    matches!(e, ElementKind::I32 | ElementKind::I64)
}

fn le(v: i128, e: ElementKind) -> Vec<u8> {
    v.to_le_bytes()[..width(e)].to_vec()
}

/// Per-thread (start, end, step); never a zero step.
fn cases(e: ElementKind) -> Vec<[i128; 3]> {
    let (lo, hi) = limits(e);
    let sg = signed(e);
    (0..T as i128)
        .map(|t| match t % 8 {
            0 => [t, t + 20, 1 + t % 5],
            1 if sg => [20, t % 7, -(1 + t % 3)],
            1 => [20, t % 7, 1 + t % 3],
            2 => [hi - t % 6, hi, 1 + t % 4],
            3 if sg => [lo + t % 6, lo, -(1 + t % 4)],
            3 => [hi - 5 - t % 3, hi - 1, 3],
            4 => [5, 5, 1],
            5 if sg => [10, -10, -(1 + t % 4)],
            5 => [2, 40, 7],
            6 => [0, hi, hi / 3 + 1],
            _ => [t, t + 3, 1],
        })
        .collect()
}

fn init(e: ElementKind) -> Vec<Vec<u8>> {
    let cs = cases(e);
    let mut bufs = Vec::new();
    for field in 0..3 {
        bufs.push(cs.iter().flat_map(|c| le(c[field], e)).collect());
    }
    bufs.push(vec![0u8; T * width(e)]);
    bufs
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
    k: &KernelFn,
    init: &[Vec<u8>],
) -> Result<Vec<u8>, String> {
    let g = OpenClC::new().lower_kernel(k).expect("lowers");
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

#[test]
fn for_loops_agree_with_the_interpreter_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let (mut kernels, mut devices_run) = (0, 0);
    for p in platforms {
        let Ok(devs) = p.devices() else { continue };
        for d in devs {
            let label = format!(
                "{} / {}",
                p.name().unwrap_or_default(),
                d.name().unwrap_or_default()
            );
            let Ok(ctx) = Context::new(&d) else { continue };
            let queue = ctx.create_queue().unwrap();
            devices_run += 1;
            for e in [
                ElementKind::I32,
                ElementKind::U32,
                ElementKind::I64,
                ElementKind::U64,
            ] {
                let data = init(e);
                for k in [plain(e), control(e), nested(e)] {
                    let want = oracle(&k, &data, e);
                    let got = device(&ctx, &queue, &k, &data).unwrap_or_else(|l| {
                        panic!("{label}: {e:?} {} failed to build:\n{l}", k.name)
                    });
                    kernels += 1;
                    assert_eq!(got, want, "{label}: {e:?} {}", k.name);
                }
            }
        }
    }
    eprintln!("{kernels} For kernels agree on {devices_run} device(s)");
}

#[test]
fn for_is_spelled_with_the_advance_in_the_header_and_the_limit_stop() {
    let on = OpenClC::new();
    let src = on
        .lower_kernel(&plain(ElementKind::I32))
        .expect("lowers")
        .source;
    assert!(src.contains("for (l4 = "), "{src}");
    assert!(src.contains("fd4 == 0 &&"), "{src}");
    assert!(src.contains("INT_MAX - fs4"), "{src}");
    assert!(src.contains("(int)((uint)l4 + (uint)fs4)"), "{src}");
    let src = on
        .lower_kernel(&plain(ElementKind::U64))
        .expect("lowers")
        .source;
    assert!(src.contains("ULONG_MAX - fs4"), "{src}");
    // A float counter is declined by name.
    let mut k = plain(ElementKind::I32);
    k.locals[4] = LocalDecl::scalar(ElementKind::F32);
    assert!(on.lower_kernel(&k).is_err(), "a float counter is declined");
}
