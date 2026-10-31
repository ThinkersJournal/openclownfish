//! Serial per-output reductions on every device against Unpopped's oracle.
//!
//! Arguments follow Baracuda's general reduction nest: the input and output
//! pointers, `shape{d}` per input axis, `s0_{d}` input strides, `so_{j}` output
//! strides (per input axis with keepdim, else per kept axis), then `n_out`.
//! Float data is chosen so every fold is exact in `f32` (quarters, small
//! counts), so the oracle's `f64` accumulator and the device's `float` one agree
//! bit for bit; integer data includes wrap-around.

use openclownfish::{Context, Kernel, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::ir::{OpDef, ReduceOp, input};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::{build_plan, generate};
use unpopped_vocab::{ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, structure_key};

const SHAPE: [i64; 3] = [3, 4, 5];

fn dense(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

fn input_buffer(dtype: ElementKind) -> TypedBuffer {
    let n: usize = SHAPE.iter().product::<i64>() as usize;
    match dtype {
        ElementKind::F32 => {
            let cyc = [1.0f32, 2.0, 0.5, -1.0, 0.25, -2.0, 1.5];
            let v: Vec<f32> = (0..n).map(|i| cyc[i % cyc.len()]).collect();
            TypedBuffer::from_f32(&SHAPE, &v)
        }
        ElementKind::I32 => {
            let cyc = [i32::MAX, 1, -1, 2, 7, -7, i32::MIN, 3, 100_000];
            let v: Vec<i32> = (0..n).map(|i| cyc[i % cyc.len()]).collect();
            TypedBuffer::from_i32(&SHAPE, &v)
        }
        _ => {
            let cyc = [i64::MAX, 1, -1, 2, 7, -7, i64::MIN, 3, 1 << 40];
            let v: Vec<i64> = (0..n).map(|i| cyc[i % cyc.len()]).collect();
            TypedBuffer::from_i64(&SHAPE, &v)
        }
    }
}

struct Case {
    dtype: ElementKind,
    op: ReduceOp,
    axes: Vec<usize>,
    keepdim: bool,
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    let axis_sets: [(&[usize], bool); 8] = [
        (&[2], false),
        (&[0], false),
        (&[1], false),
        (&[0, 2], false),
        (&[1, 2], false),
        (&[2], true),
        (&[0, 1], true),
        (&[0, 1, 2], true),
    ];
    for dtype in [ElementKind::F32, ElementKind::I32, ElementKind::I64] {
        let ops: &[ReduceOp] = if dtype == ElementKind::F32 {
            &[
                ReduceOp::Sum,
                ReduceOp::Max,
                ReduceOp::Min,
                ReduceOp::Prod,
                ReduceOp::Mean,
            ]
        } else {
            &[ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min, ReduceOp::Prod]
        };
        for &op in ops {
            for (axes, keepdim) in axis_sets {
                // Products over many elements leave f32 exactness; fold few.
                if dtype == ElementKind::F32 && op == ReduceOp::Prod && axes.len() > 1 {
                    continue;
                }
                v.push(Case {
                    dtype,
                    op,
                    axes: axes.to_vec(),
                    keepdim,
                });
            }
        }
    }
    v
}

fn name(c: &Case) -> String {
    format!("{:?}/{:?}/{:?}/kd={}", c.dtype, c.op, c.axes, c.keepdim)
}

#[test]
fn reductions_agree_with_the_oracle_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let backend = OpenClC::new();
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
            let all = cases();
            for c in &all {
                let rank = SHAPE.len();
                let mut mask = AxisMask::EMPTY;
                for &a in &c.axes {
                    mask.set(a as u8);
                }
                let kept: Vec<usize> = (0..rank).filter(|a| !c.axes.contains(a)).collect();
                let out_shape: Vec<i64> = if c.keepdim {
                    (0..rank)
                        .map(|a| if c.axes.contains(&a) { 1 } else { SHAPE[a] })
                        .collect()
                } else {
                    kept.iter().map(|&a| SHAPE[a]).collect()
                };
                let out_strides = dense(&out_shape);
                let inp = input_buffer(c.dtype);
                let a = OperandDesc::new(rank, &SHAPE, &inp.strides, c.dtype, 256);
                let o = OperandDesc::new(out_shape.len(), &out_shape, &out_strides, c.dtype, 256);
                let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
                let opdef =
                    OpDef::reduction_axes("r", 1, &[c.dtype], input(0), c.op, mask, c.keepdim);
                let plan = build_plan(&opdef, &key);
                let kernel = generate(&opdef, &key, &backend);
                let prog = match Program::build(&ctx, &kernel.source, "") {
                    Ok(p) => p,
                    Err(e) => panic!(
                        "{label}: {} failed to build:\n{e}\n{}",
                        name(c),
                        kernel.source
                    ),
                };
                let mut k = prog.kernel(&kernel.name).unwrap();
                let w = if c.dtype == ElementKind::I64 { 8 } else { 4 };
                let n_out: i64 = out_shape.iter().product();
                let inb = ctx.create_buffer(inp.bytes.len()).unwrap();
                queue.write_buffer(&inb, 0, &inp.bytes).unwrap();
                let out = ctx.create_buffer(n_out as usize * w).unwrap();
                queue
                    .write_buffer(&out, 0, &vec![0xA5u8; n_out as usize * w])
                    .unwrap();
                k.set_arg_buffer(0, &inb).unwrap();
                k.set_arg_buffer(1, &out).unwrap();
                let mut arg = 2u32;
                let mut push = |k: &mut Kernel, x: i64| {
                    k.set_arg_i64(arg, x).unwrap();
                    arg += 1;
                };
                for &x in &SHAPE {
                    push(&mut k, x);
                }
                for &x in &inp.strides {
                    push(&mut k, x);
                }
                for &x in &out_strides {
                    push(&mut k, x);
                }
                push(&mut k, n_out);
                // SAFETY: one work item per output element, each guarded by `o < n_out`;
                // every input offset is inside the dense input buffer.
                let ev = unsafe { queue.launch(&k, &[n_out as usize], None) }.unwrap();
                ev.wait().unwrap();
                let mut got = vec![0u8; n_out as usize * w];
                queue.read_buffer(&out, 0, &mut got).unwrap();

                let want = evaluate(&plan, &[a, o], std::slice::from_ref(&inp), &[])
                    .remove(0)
                    .bytes;
                let want = &want[..n_out as usize * w];
                if c.op == ReduceOp::Mean {
                    // f32 division vs f64 division rounded to f32: allow 1 ulp.
                    for (g, wnt) in got.chunks(4).zip(want.chunks(4)) {
                        let g = f32::from_le_bytes(g.try_into().unwrap());
                        let wnt = f32::from_le_bytes(wnt.try_into().unwrap());
                        assert!(
                            (i64::from(g.to_bits()) - i64::from(wnt.to_bits())).abs() <= 1,
                            "{label}: {}: kernel {g}, oracle {wnt}",
                            name(c)
                        );
                    }
                } else {
                    assert_eq!(got, want, "{label}: {}\n{}", name(c), kernel.source);
                }
            }
            eprintln!("{label}: {} reduction cases agree", all.len());
            ran += 1;
        }
    }
    eprintln!("reductions ran on {ran} device(s)");
}

#[test]
fn integer_mean_is_refused() {
    let a = OperandDesc::new(2, &[4, 6], &[6, 1], ElementKind::I32, 256);
    let o = OperandDesc::new(1, &[4], &[1], ElementKind::I32, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    let mean = OpDef::reduction("m", 1, &[ElementKind::I32], input(0), ReduceOp::Mean);
    // The plan gate may itself reject integer Mean; if it plans, we must refuse.
    if let Ok(plan) = unpopped::plan::try_build_plan(&mean, &key) {
        assert!(OpenClC::new().lower(&plan).is_err());
    }
}
