// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Backend::lower_kernel` for the float `Sqrt`, `Exp`, `Log`, `Tanh`, `Sin` and `Cos`, on every
//! OpenCL device here, against Unpopped's launch interpreter. These go through the device's
//! built-ins, which OpenCL bounds at several ULP rather than demanding the oracle's result, so
//! the agreement is a stated tolerance, not bit equality (NaN agrees with NaN, infinities and
//! zeros must match exactly). Tolerances are the OpenCL C 1.2 maxima for `float` (`sqrt` 3,
//! `exp` 3, `log` 3, `tanh` 5, `sin`/`cos` 4) and the `double` ones for F64 (`sqrt` 0, `exp`
//! 3, `log` 3, `tanh` 5, `sin`/`cos` 4). An F32 kernel with `with_fp64` computes at `double` and
//! rounds once, so it must be within 1 ULP of `f32`.

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{ScalarExpr, UnaryOp};
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

/// Distance in representable values (sign-magnitude folded to a line), `None` for exact
/// agreement of special values being required elsewhere.
fn key(bits: u64, width: usize) -> i128 {
    let sign = 1u64 << (width * 8 - 1);
    let mag = (bits & !sign) as i128;
    if bits & sign != 0 { -mag } else { mag }
}

fn lane(b: &[u8], e: ElementKind) -> (f64, u64) {
    if e == ElementKind::F32 {
        let x = f32::from_le_bytes(b.try_into().unwrap());
        (f64::from(x), u64::from(x.to_bits()))
    } else {
        let x = f64::from_le_bytes(b.try_into().unwrap());
        (x, x.to_bits())
    }
}

/// Bit equality (any two NaNs agree): unlike the ULP distance it tells `-0` from `+0`.
fn exact_same(a: &[u8], b: &[u8], e: ElementKind) -> Result<(), String> {
    let w = width(e);
    for (i, (x, y)) in a.chunks(w).zip(b.chunks(w)).enumerate() {
        let (fx, _) = lane(x, e);
        let (fy, _) = lane(y, e);
        if x != y && !(fx.is_nan() && fy.is_nan()) {
            return Err(format!("element {i}: device {x:02x?}, oracle {y:02x?}"));
        }
    }
    Ok(())
}

/// The largest ULP distance, erroring where NaN-ness or finiteness differs.
fn max_ulps(a: &[u8], b: &[u8], e: ElementKind) -> Result<u64, String> {
    let w = width(e);
    let mut worst = 0u64;
    for (i, (x, y)) in a.chunks(w).zip(b.chunks(w)).enumerate() {
        let ((fx, bx_), (fy, by)) = (lane(x, e), lane(y, e));
        if fx.is_nan() || fy.is_nan() {
            if fx.is_nan() != fy.is_nan() {
                return Err(format!("element {i}: device {fx}, oracle {fy}"));
            }
            continue;
        }
        if fx.is_infinite() || fy.is_infinite() {
            if fx != fy {
                return Err(format!("element {i}: device {fx}, oracle {fy}"));
            }
            continue;
        }
        let d = key(bx_, w).abs_diff(key(by, w)) as u64;
        worst = worst.max(d);
    }
    Ok(worst)
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

fn un(op: UnaryOp) -> ScalarExpr {
    ScalarExpr::Unary(op, bx(local(0)))
}

/// `(name, op, tolerance in ULPs of the compute type)`.
fn ops(e: ElementKind, wide: bool) -> Vec<(&'static str, UnaryOp, u64)> {
    use UnaryOp::*;
    // Bit for bit in every mode (they do not round: the oracle's `f64` result is already exact).
    let mut v = vec![
        ("e_floor", Floor, 0),
        ("e_ceil", Ceil, 0),
        ("e_trunc", Trunc, 0),
        ("e_round", Round, 0),
        ("e_relu", Relu, 0),
        ("e_sign", Sign, 0),
        ("e_step", Step, 0),
    ];
    // `1.0 / x` is correctly rounded at F64 (bit for bit); F32 Recip is declined with F32 Div.
    if e == ElementKind::F64 {
        v.push(("e_recip", Recip, 0));
        // `1.0 / sqrt(x)`: both steps correctly rounded, so bit for bit. Sigmoid and Silu carry
        // `exp`, so they get a limit. Gelu stays declined: `1 + erf(x / sqrt 2)` cancels for
        // negative x, so the device's erf error is magnified without bound (7906 ULP measured).
        v.push(("e_rsqrt", Rsqrt, 0));
        v.push(("t_sigmoid", Sigmoid, 4));
        v.push(("t_silu", Silu, 4));
    }
    // The OpenCL C 1.2 maxima for `float` and `double` (identical for these ops except `sqrt`).
    let table: [(&'static str, UnaryOp, u64); 23] = [
        ("t_sqrt", Sqrt, if e == ElementKind::F32 { 3 } else { 0 }),
        ("t_exp", Exp, 3),
        ("t_log", Log, 3),
        ("t_tanh", Tanh, 5),
        ("t_sin", Sin, 4),
        ("t_cos", Cos, 4),
        ("t_exp2", Exp2, 3),
        ("t_expm1", Expm1, 3),
        ("t_log2", Log2, 3),
        ("t_log10", Log10, 3),
        ("t_log1p", Log1p, 2),
        ("t_sinh", Sinh, 4),
        ("t_cosh", Cosh, 4),
        ("t_tan", Tan, 5),
        ("t_asin", Asin, 4),
        ("t_acos", Acos, 4),
        ("t_atan", Atan, 5),
        ("t_asinh", Asinh, 4),
        ("t_acosh", Acosh, 4),
        ("t_atanh", Atanh, 5),
        ("t_cbrt", Cbrt, 4),
        ("t_erf", Erf, 16),
        ("t_erfc", Erfc, 16),
    ];
    for (n, op, tol) in table {
        // Evaluated at double and rounded once to f32: within 1 ULP of f32.
        v.push((
            n,
            op,
            if e == ElementKind::F32 && wide {
                1
            } else {
                tol
            },
        ));
    }
    v
}

fn kernel(name: &str, e: ElementKind, value: ScalarExpr) -> KernelFn {
    let stmts = vec![
        Stmt::Load {
            local: 0,
            place: at(0),
        },
        Stmt::Let {
            local: 1,
            init: ScalarExpr::Const(0.0),
        },
        Stmt::Assign { local: 1, value },
        Stmt::Store {
            place: at(1),
            value: local(1),
        },
    ];
    launch(KernelFn::new(
        name,
        e,
        vec![
            Param::buffer("x", e, BufferRole::Read, Some(1)),
            Param::buffer("out", e, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(e), LocalDecl::scalar(e)],
        Block(stmts),
    ))
}

/// Values in `[-scale/2, scale/2)` with NaN, both infinities and both zeros mixed in.
fn data(e: ElementKind, scale: f64) -> Vec<Vec<u8>> {
    let mut s = 0x0bad_cafe_1234_5678u64;
    let mut xs = Vec::new();
    for t in 0..T {
        let x = (unit(&mut s) - 0.5) * scale;
        let x = match t % 16 {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => f64::NEG_INFINITY,
            3 => -0.0,
            4 => 0.0,
            _ => x,
        };
        xs.extend(bytes(x, e));
    }
    vec![xs, vec![0u8; T * width(e)]]
}

#[test]
fn transcendentals_agree_within_their_stated_ulps_on_every_device() {
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
            let mut modes = vec![(ElementKind::F32, false)];
            if has64 {
                modes.push((ElementKind::F32, true));
                modes.push((ElementKind::F64, true));
            }
            for (e, wide) in modes {
                let backend = OpenClC::new().with_fp64(wide);
                for (name, op, tol) in ops(e, wide) {
                    let k = kernel(name, e, un(op));
                    // A wide range, and one inside the domain of `asin`, `acos` and `atanh`.
                    let mut worst = 0;
                    for scale in [16.0, 2.4] {
                        let init = data(e, scale);
                        let want = oracle(&k, &init, e);
                        let got = device(&ctx, &queue, &backend, &k, &init).unwrap_or_else(|l| {
                            panic!(
                                "{label}: {name} failed to build:
{l}"
                            )
                        });
                        let w = max_ulps(&got, &want, e)
                            .unwrap_or_else(|m| panic!("{label}: {e:?} wide={wide} {name}: {m}"));
                        worst = worst.max(w);
                        if tol == 0 {
                            exact_same(&got, &want, e).unwrap_or_else(|m| {
                                panic!("{label}: {e:?} wide={wide} {name}: {m}")
                            });
                        }
                    }
                    eprintln!("{label}: {e:?} wide={wide} {name}: worst {worst} ULP (limit {tol})");
                    assert!(
                        worst <= tol,
                        "{label}: {e:?} wide={wide} {name}: {worst} ULP > {tol}"
                    );
                    checks += 1;
                }
            }
        }
    }
    eprintln!("transcendentals: {checks} checks on {devices_run} devices");
}

/// The ops this emitter declines by name still say so: a float division at F32, and Lgamma at
/// both widths (the oracle's differs from the device at +inf).
#[test]
fn f32_recip_rsqrt_sigmoid_silu_gelu_and_lgamma_are_declined_by_name() {
    let cases = [
        (UnaryOp::Recip, ElementKind::F32),
        (UnaryOp::Rsqrt, ElementKind::F32),
        (UnaryOp::Sigmoid, ElementKind::F32),
        (UnaryOp::Silu, ElementKind::F32),
        (UnaryOp::Gelu, ElementKind::F32),
        (UnaryOp::Gelu, ElementKind::F64),
        (UnaryOp::Lgamma, ElementKind::F32),
        (UnaryOp::Lgamma, ElementKind::F64),
    ];
    for (op, e) in cases {
        let k = kernel("declined", e, un(op));
        let err = OpenClC::new()
            .with_fp64(true)
            .lower_kernel(&k)
            .expect_err("declined");
        assert!(
            format!("{err}").contains(&format!("{op:?}")),
            "{op:?}: {err}"
        );
    }
}
