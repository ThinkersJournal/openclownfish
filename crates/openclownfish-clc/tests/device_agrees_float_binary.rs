// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Backend::lower_kernel` for the float binary ops `Copysign`, `RemTrunc` (`fmod`) and an F64
//! floored `Rem` (bit for bit), and `Pow` and `Atan2` (the device's built-ins, within OpenCL's
//! per-op ULP maxima: `pow` 16, `atan2` 6; 1 ULP of f32 when an F32 kernel computes at `double`
//! under `with_fp64`), on every OpenCL device here, against Unpopped's launch interpreter. NaN
//! agrees with NaN and infinities must match exactly. `Nextafter` (float) and an F32 `Rem` are
//! declined by name.

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
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

fn bin(op: BinaryOp) -> ScalarExpr {
    ScalarExpr::Binary(op, bx(local(0)), bx(local(1)))
}

fn kernel(name: &str, e: ElementKind, value: ScalarExpr) -> KernelFn {
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

/// `(name, op, tolerance in ULPs of the compute type)` valid for `e` in this mode.
fn ops(e: ElementKind, wide: bool) -> Vec<(&'static str, BinaryOp, u64)> {
    let wide32 = e == ElementKind::F32 && wide;
    let mut v = vec![
        ("b_copysign", BinaryOp::Copysign, 0),
        ("b_fmod", BinaryOp::RemTrunc, 0),
        ("b_pow", BinaryOp::Pow, if wide32 { 1 } else { 16 }),
        ("b_atan2", BinaryOp::Atan2, if wide32 { 1 } else { 6 }),
    ];
    if e == ElementKind::F64 {
        v.push(("b_rem", BinaryOp::Rem, 0));
    }
    v
}

/// `(x, y)`: values in `[-scale/2, scale/2)` with NaN, infinities and both zeros mixed in.
fn data(e: ElementKind, scale: f64) -> Vec<Vec<u8>> {
    let mut s = 0x5eed_1234_9abc_def0u64;
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for t in 0..T {
        let x = (unit(&mut s) - 0.5) * scale;
        let y = (unit(&mut s) - 0.5) * scale;
        let x = match t % 16 {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => f64::NEG_INFINITY,
            3 => -0.0,
            4 => 0.0,
            _ => x,
        };
        let y = match t % 16 {
            5 => f64::NAN,
            6 => f64::INFINITY,
            7 => f64::NEG_INFINITY,
            8 => -0.0,
            9 => 0.0,
            _ => y,
        };
        xs.extend(bytes(x, e));
        ys.extend(bytes(y, e));
    }
    vec![xs, ys, vec![0u8; T * width(e)]]
}

#[test]
fn float_binary_ops_agree_within_their_stated_ulps_on_every_device() {
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
                    let k = kernel(name, e, bin(op));
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
    eprintln!("float binary ops: {checks} checks on {devices_run} devices");
}

/// Declined by name: a float `Nextafter`, and an F32 floored `Rem` (it divides).
#[test]
fn float_nextafter_and_f32_rem_are_declined_by_name() {
    for (e, op, what) in [
        (ElementKind::F32, BinaryOp::Nextafter, "Nextafter"),
        (ElementKind::F64, BinaryOp::Nextafter, "Nextafter"),
        (ElementKind::F32, BinaryOp::Rem, "F32 Rem"),
    ] {
        let k = kernel("declined", e, bin(op));
        let err = OpenClC::new()
            .with_fp64(true)
            .lower_kernel(&k)
            .expect_err("declined");
        assert!(format!("{err}").contains(what), "{op:?} {e:?}: {err}");
    }
}
