// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Backend::lower_kernel` for `F64`-to-`F32` (four rounding modes) and `F32`-to-`F64` `Cast`s, bit
//! for bit against Unpopped's launch interpreter on every device with `cl_khr_fp64`. Inputs hold
//! exact f32 ties, subnormals, overflow, both zeros, NaN (any NaN equals any NaN) and infinities.

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const BLOCK: u32 = 32;
const GRID: u32 = 2;
const T: usize = (BLOCK * GRID) as usize;

fn width(e: ElementKind) -> usize {
    if e == ElementKind::F32 { 4 } else { 8 }
}

fn gtid() -> ScalarExpr {
    ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(
            Box::new(ScalarExpr::Exec(ExecId::BlockIdx(0))),
            Box::new(ScalarExpr::Exec(ExecId::BlockDim(0))),
        )),
        Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
    )
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![gtid()],
    }
}

fn kernel(name: &str, src: ElementKind, to: ElementKind, round: RoundMode) -> KernelFn {
    let stmts = vec![
        Stmt::Load {
            local: 0,
            place: at(0),
        },
        Stmt::Let {
            local: 1,
            init: ScalarExpr::Cast(
                Box::new(ScalarExpr::Local(0)),
                CastSpec {
                    to,
                    round,
                    sat: Saturation::Wrap,
                },
            ),
        },
        Stmt::Store {
            place: at(1),
            value: ScalarExpr::Local(1),
        },
    ];
    let mut k = KernelFn::new(
        name,
        src,
        vec![
            Param::buffer("x", src, BufferRole::Read, Some(1)),
            Param::buffer("out", to, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(src), LocalDecl::scalar(to)],
        Block(stmts),
    );
    k.launch = Some(LaunchShape::new([BLOCK, 1, 1], [GRID, 1, 1]));
    k
}

fn data(src: ElementKind) -> Vec<u8> {
    let mut s = 0x00c0_ffee_1234_5678_u64;
    let f = f64::from(f32::MAX);
    let ulp24 = 2f64.powi(-24);
    let special = [
        0.0,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1.0 + ulp24 * 2.0,
        1.0 + ulp24,
        1.0 + ulp24 * 3.0,
        -(1.0 + ulp24),
        f,
        f * (1.0 + 2f64.powi(-25)),
        f * (1.0 + 2f64.powi(-24)),
        1e39,
        -1e39,
        1e-45,
        7e-46,
        2e-46,
        1.17549435e-38,
        1e-40,
    ];
    let mut out = Vec::new();
    for t in 0..T {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 11) as f64 / (1u64 << 53) as f64;
        let v = if t < special.len() {
            special[t]
        } else {
            (u - 0.5) * 10f64.powi((t % 80) as i32 - 40)
        };
        if src == ElementKind::F32 {
            out.extend((v as f32).to_le_bytes());
        } else {
            out.extend(v.to_le_bytes());
        }
    }
    out
}

fn same(a: &[u8], b: &[u8], e: ElementKind) -> bool {
    let w = width(e);
    a.chunks(w).zip(b.chunks(w)).all(|(x, y)| {
        let nan = |c: &[u8]| {
            if w == 4 {
                f32::from_le_bytes(c.try_into().unwrap()).is_nan()
            } else {
                f64::from_le_bytes(c.try_into().unwrap()).is_nan()
            }
        };
        x == y || (nan(x) && nan(y))
    })
}

#[test]
fn float_to_float_casts_agree_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let backend = OpenClC::new().with_fp64(true);
    let mut checks = 0;
    for p in platforms {
        let Ok(devs) = p.devices() else { continue };
        for d in devs {
            let has64 = d
                .extensions()
                .map(|x| x.iter().any(|n| n == "cl_khr_fp64"))
                .unwrap_or(false);
            let Ok(ctx) = Context::new(&d) else { continue };
            if !has64 {
                continue;
            }
            let queue = ctx.create_queue().unwrap();
            let label = d.name().unwrap_or_default();
            let cases = [
                (ElementKind::F64, ElementKind::F32, RoundMode::NearestEven),
                (ElementKind::F64, ElementKind::F32, RoundMode::TowardZero),
                (ElementKind::F64, ElementKind::F32, RoundMode::TowardNegInf),
                (ElementKind::F64, ElementKind::F32, RoundMode::TowardPosInf),
                (ElementKind::F32, ElementKind::F64, RoundMode::NearestEven),
            ];
            for (src, to, round) in cases {
                let name = format!("f2f_{src:?}_{to:?}_{round:?}");
                let k = kernel(&name, src, to, round);
                let init = [data(src), vec![0u8; T * width(to)]];
                let mut args: Vec<KernelArg> = init
                    .iter()
                    .zip([src, to])
                    .map(|(b, e)| {
                        let n = (b.len() / width(e)) as i64;
                        KernelArg::Buffer(TypedBuffer::new(e, vec![n], vec![1], b.clone()))
                    })
                    .collect();
                run_launch(&k, &mut args).expect("oracle runs");
                let want = match args.pop() {
                    Some(KernelArg::Buffer(b)) => b.bytes,
                    _ => unreachable!(),
                };
                let g = backend.lower_kernel(&k).expect("lowers");
                let prog = Program::build(&ctx, &g.source, "")
                    .unwrap_or_else(|e| panic!("{label} {name}: {e}\n{}", g.source));
                let mut kern = prog.kernel(&g.name).unwrap();
                let mut held = Vec::new();
                for (i, b) in init.iter().enumerate() {
                    let buf = ctx.create_buffer(b.len()).unwrap();
                    queue.write_buffer(&buf, 0, b).unwrap();
                    kern.set_arg_buffer(i as u32, &buf).unwrap();
                    held.push(buf);
                }
                // SAFETY: every access is inside its buffer; `held` outlives the launch.
                unsafe { queue.launch(&kern, &[T], Some(&[BLOCK as usize])) }
                    .unwrap()
                    .wait()
                    .unwrap();
                let mut got = vec![0u8; init[1].len()];
                queue.read_buffer(&held[1], 0, &mut got).unwrap();
                assert!(same(&got, &want, to), "{label}: {name} differs");
                checks += 1;
            }
        }
    }
    eprintln!("float-to-float casts: {checks} checks");
}

#[test]
fn float_cast_rounding_ties_away_is_declined() {
    let b = OpenClC::new().with_fp64(true);
    let k = kernel(
        "d",
        ElementKind::F64,
        ElementKind::F32,
        RoundMode::NearestAway,
    );
    assert!(b.lower_kernel(&k).is_err());
}
