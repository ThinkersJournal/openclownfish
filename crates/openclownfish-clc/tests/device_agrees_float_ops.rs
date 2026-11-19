// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Backend::lower_kernel` for float `Max`, `Min`, `FmaxIeee`, `FminIeee`, `Abs`, `Sqr` and
//! `Select`, on every OpenCL device here, against Unpopped's launch interpreter. Results must
//! match bit for bit (two NaNs agree whatever their payload). The inputs carry NaN in either
//! operand, both infinities and both zeros, including `(-0, +0)` and `(+0, -0)` ties, which is
//! where OpenCL's own `fmax`/`fmin` differ from the reference. One operation per statement runs
//! without `with_fp64`; composites (more than one rounding) run with it where the device lists
//! `cl_khr_fp64`.

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr, UnaryOp};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
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

/// `(x, y)` buffers: random values in `[-4, 4)`, with lanes of both carrying NaN, the infinities
/// and both zeros (so ties of `-0`/`+0` and NaN in either operand are exercised).
fn inputs(e: ElementKind) -> Vec<Vec<u8>> {
    let mut s = 0x1234_5678_9abc_def0u64;
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for t in 0..T {
        let mut x = (unit(&mut s) - 0.5) * 8.0;
        let mut y = (unit(&mut s) - 0.5) * 8.0;
        x = match t % 16 {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => f64::NEG_INFINITY,
            3 => -0.0,
            4 => 0.0,
            5 => 0.0,
            6 => -0.0,
            _ => x,
        };
        y = match t % 16 {
            7 => f64::NAN,
            8 => f64::INFINITY,
            9 => -0.0,
            10 => 0.0,
            5 => -0.0,
            6 => 0.0,
            11 => f64::NAN,
            _ => y,
        };
        if t % 16 == 11 {
            x = f64::NAN;
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
fn same(a: &[u8], b: &[u8], e: ElementKind, zeros_equal: bool) -> Result<(), String> {
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
        let zero = |z: &[u8]| {
            z.iter()
                .enumerate()
                .all(|(i, b)| *b == 0 || i == w - 1 && *b == 0x80)
        };
        if x != y && !(nx && ny) && !(zeros_equal && zero(x) && zero(y)) {
            return Err(format!("element {i}: device {x:02x?}, oracle {y:02x?}"));
        }
    }
    Ok(())
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

/// `acc = <value>` over locals 0 x and 1 y, stored to `out`.
fn one(name: &str, e: ElementKind, value: ScalarExpr) -> KernelFn {
    let stmts = vec![
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
        Stmt::Assign { local: 2, value },
        Stmt::Store {
            place: at(2),
            value: local(2),
        },
    ];
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
        ],
        Block(stmts),
    ))
}

fn bin(op: BinaryOp) -> ScalarExpr {
    ScalarExpr::Binary(op, bx(local(0)), bx(local(1)))
}

/// Kernels with one rounding operation or fewer, valid without `with_fp64`.
fn singles(e: ElementKind) -> Vec<KernelFn> {
    vec![
        one("f_max", e, bin(BinaryOp::Max)),
        one("f_min", e, bin(BinaryOp::Min)),
        one("f_fmax", e, bin(BinaryOp::FmaxIeee)),
        one("f_fmin", e, bin(BinaryOp::FminIeee)),
        one("f_abs", e, ScalarExpr::Unary(UnaryOp::Abs, bx(local(0)))),
        one("f_sqr", e, ScalarExpr::Unary(UnaryOp::Sqr, bx(local(1)))),
        // The condition is `!= 0`: NaN is true, -0 is false.
        one(
            "f_select_x",
            e,
            ScalarExpr::Select(
                bx(local(0)),
                bx(local(1)),
                bx(ScalarExpr::Unary(UnaryOp::Neg, bx(local(1)))),
            ),
        ),
        one(
            "f_select_lt",
            e,
            ScalarExpr::Select(bx(bin(BinaryOp::CmpLt)), bx(local(0)), bx(local(1))),
        ),
    ]
}

/// Composites with several rounding operations: need `with_fp64`.
fn composites(e: ElementKind) -> Vec<KernelFn> {
    let abs_x = ScalarExpr::Unary(UnaryOp::Abs, bx(local(0)));
    let sqr_y = ScalarExpr::Unary(UnaryOp::Sqr, bx(local(1)));
    vec![
        one(
            "c_abs_plus_sqr",
            e,
            ScalarExpr::Add(bx(abs_x), bx(sqr_y.clone())),
        ),
        one(
            "c_max_of_sqr_and_x",
            e,
            ScalarExpr::Binary(BinaryOp::Max, bx(sqr_y), bx(local(0))),
        ),
        one(
            "c_select_over_sum_fmin",
            e,
            ScalarExpr::Select(
                bx(bin(BinaryOp::CmpGt)),
                bx(ScalarExpr::Add(bx(local(0)), bx(local(1)))),
                bx(ScalarExpr::Binary(
                    BinaryOp::FminIeee,
                    bx(local(0)),
                    bx(local(1)),
                )),
            ),
        ),
    ]
}

fn check(
    label: &str,
    backend: &OpenClC,
    ctx: &Context,
    queue: &openclownfish::Queue,
    e: ElementKind,
    k: &KernelFn,
    mode: &str,
) {
    let data = inputs(e);
    let want = oracle(k, &data, e);
    let got = device(ctx, queue, backend, k, &data)
        .unwrap_or_else(|l| panic!("{label}: {} failed to build:\n{l}", k.name));
    // The oracle spells `FmaxIeee`/`FminIeee` as Rust's `f64::max`/`min`, which may return either
    // zero for `(+0, -0)`; that tie is unspecified there, so only those kernels accept either.
    let zeros_equal = k.name.contains("fmax") || k.name.contains("fmin");
    if let Err(m) = same(&got, &want, e, zeros_equal) {
        panic!("{label}: {e:?} {} {mode}: {m}", k.name);
    }
}

#[test]
fn float_ops_agree_with_the_interpreter_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let (mut checks, mut devices_run) = (0, 0);
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
            let plain = OpenClC::new();
            for k in singles(ElementKind::F32) {
                check(
                    &label,
                    &plain,
                    &ctx,
                    &queue,
                    ElementKind::F32,
                    &k,
                    "without fp64",
                );
                checks += 1;
            }
            if !has64 {
                continue;
            }
            let wide = OpenClC::new().with_fp64(true);
            for e in [ElementKind::F32, ElementKind::F64] {
                for k in singles(e).into_iter().chain(composites(e)) {
                    check(&label, &wide, &ctx, &queue, e, &k, "with fp64");
                    checks += 1;
                }
            }
        }
    }
    eprintln!("float ops: {checks} checks on {devices_run} devices");
}

/// The mutation control for the helpers' tie rule: the reference takes the first operand on a
/// tie, so `max(-0, +0)` is `-0`. OpenCL's own `fmax` is free to return either.
#[test]
fn the_reference_max_of_signed_zeros_is_the_first_operand() {
    let e = ElementKind::F64;
    let k = one("tie", e, bin(BinaryOp::Max));
    let mut data = inputs(e);
    // lane 0: x = -0, y = +0; lane 1: x = +0, y = -0.
    data[0][..16].copy_from_slice(&[(-0.0f64).to_le_bytes(), 0.0f64.to_le_bytes()].concat());
    data[1][..16].copy_from_slice(&[0.0f64.to_le_bytes(), (-0.0f64).to_le_bytes()].concat());
    let out = oracle(&k, &data, e);
    assert_eq!(
        out[..8],
        (-0.0f64).to_le_bytes(),
        "max(-0, +0) is the first operand"
    );
    assert_eq!(
        out[8..16],
        0.0f64.to_le_bytes(),
        "max(+0, -0) is the first operand"
    );
}
