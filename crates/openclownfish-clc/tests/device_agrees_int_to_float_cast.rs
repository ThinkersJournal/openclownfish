// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Backend::lower_kernel` for an integer-to-float `Cast` (`Wrap`, the four directed or even rounding
//! modes; from `I32`, `U32`, `I64`, `U64` to `F32` and `F64`), on every OpenCL device here, against
//! Unpopped's launch interpreter. Results must match bit for bit. The inputs hold values around the
//! 2^24 and 2^53 rounding boundaries (exact ties among them), zero and both extremes of each source.

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
const GRID: u32 = 4;
const T: usize = (BLOCK * GRID) as usize;

fn width(e: ElementKind) -> usize {
    if matches!(e, ElementKind::F32 | ElementKind::I32 | ElementKind::U32) {
        4
    } else {
        8
    }
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

fn ibytes(v: i128, e: ElementKind) -> Vec<u8> {
    let n = width(e);
    v.to_le_bytes()[..n].to_vec()
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

fn oracle(k: &KernelFn, init: &[Vec<u8>], dts: &[ElementKind]) -> Vec<u8> {
    let mut args: Vec<KernelArg> = init
        .iter()
        .zip(dts)
        .map(|(b, e)| {
            let n = (b.len() / width(*e)) as i64;
            KernelArg::Buffer(TypedBuffer::new(*e, vec![n], vec![1], b.clone()))
        })
        .collect();
    run_launch(k, &mut args).expect("the interpreter runs the kernel");
    match args.pop() {
        Some(KernelArg::Buffer(b)) => b.bytes,
        _ => unreachable!(),
    }
}

fn cast_kernel(name: &str, src: ElementKind, to: ElementKind, round: RoundMode) -> KernelFn {
    let stmts = vec![
        Stmt::Load {
            local: 0,
            place: at(0),
        },
        Stmt::Let {
            local: 1,
            init: ScalarExpr::Cast(
                bx(local(0)),
                CastSpec {
                    to,
                    round,
                    sat: Saturation::Wrap,
                },
            ),
        },
        Stmt::Store {
            place: at(1),
            value: local(1),
        },
    ];
    launch(KernelFn::new(
        name,
        src,
        vec![
            Param::buffer("x", src, BufferRole::Read, Some(1)),
            Param::buffer("out", to, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(src), LocalDecl::scalar(to)],
        Block(stmts),
    ))
}

fn data(src: ElementKind) -> Vec<u8> {
    let (lo, hi): (i128, i128) = match src {
        ElementKind::I32 => (i32::MIN.into(), i32::MAX.into()),
        ElementKind::U32 => (0, u32::MAX.into()),
        ElementKind::I64 => (i64::MIN.into(), i64::MAX.into()),
        _ => (0, u64::MAX.into()),
    };
    // Around 2^24 and 2^53 (exact ties and their neighbours), plus the extremes.
    let mut special: Vec<i128> = vec![0, 1, -1, lo, hi, lo + 1, hi - 1];
    for base in [
        1i128 << 24,
        1i128 << 25,
        1i128 << 53,
        1i128 << 54,
        1i128 << 62,
    ] {
        for d in -3..=3 {
            special.push(base + d);
            special.push(-base + d);
        }
    }
    special.retain(|v| (lo..=hi).contains(v));
    let mut s = 0x00c0_ffee_1234_5678_u64;
    let mut xs = Vec::new();
    for t in 0..T {
        let r = (unit(&mut s) * 18446744073709551616.0) as u128 as i128;
        let v = if t < special.len() {
            special[t]
        } else {
            // A random magnitude of random width, signed where the source is.
            let bits = 1 + (unit(&mut s) * 63.0) as u32;
            let m = r & ((1i128 << bits) - 1);
            if lo < 0 && t % 2 == 0 { -m } else { m }
        };
        xs.extend(ibytes(v.clamp(lo, hi), src));
    }
    xs
}

#[test]
fn int_to_float_casts_agree_bit_for_bit_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let rounds = [
        RoundMode::NearestEven,
        RoundMode::TowardZero,
        RoundMode::TowardNegInf,
        RoundMode::TowardPosInf,
    ];
    let sources = [
        ElementKind::I32,
        ElementKind::U32,
        ElementKind::I64,
        ElementKind::U64,
    ];
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
            for (to, wide) in modes {
                let backend = OpenClC::new().with_fp64(wide);
                for src in sources {
                    for round in rounds {
                        let name = format!("i2f_{src:?}_{to:?}_{round:?}");
                        let k = cast_kernel(&name, src, to, round);
                        let init = vec![data(src), vec![0u8; T * width(to)]];
                        let dts = [src, to];
                        let want = oracle(&k, &init, &dts);
                        let got = device(&ctx, &queue, &backend, &k, &init)
                            .unwrap_or_else(|l| panic!("{label}: {name} failed to build:\n{l}"));
                        if got != want {
                            let w = width(to);
                            let i = got
                                .chunks(w)
                                .zip(want.chunks(w))
                                .position(|(a, b)| a != b)
                                .unwrap();
                            let xs = data(src);
                            let ws = width(src);
                            panic!(
                                "{label}: {name} wide={wide}: element {i} of input {:02x?}: device {:02x?}, oracle {:02x?}",
                                &xs[i * ws..(i + 1) * ws],
                                &got[i * w..(i + 1) * w],
                                &want[i * w..(i + 1) * w]
                            );
                        }
                        checks += 1;
                    }
                }
            }
        }
    }
    eprintln!("int-to-float casts: {checks} checks on {devices_run} devices");
}

#[test]
fn int_to_float_rounding_ties_away_is_declined_by_name() {
    let k = cast_kernel(
        "away",
        ElementKind::I32,
        ElementKind::F32,
        RoundMode::NearestAway,
    );
    // Unpopped's own gate may reject it first; either way it must not lower.
    assert!(OpenClC::new().lower_kernel(&k).is_err());
}
