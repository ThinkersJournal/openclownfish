//! i8, u8, i16 and u16 serial reductions on every device against Unpopped's oracle,
//! byte for byte (ROADMAP E19; the oracle covers them from Unpopped 0.40.16).
//!
//! A narrow `Sum` and `Prod` wrap at the dtype's width: the pool holds both extremes
//! and values whose sums and products overflow, so a kernel that accumulated in a
//! wider type and stored without wrapping would show up. `Max` and `Min` start from
//! the dtype's extremes, which the unsigned types spell as 0 and the top value.
//! Integer `Mean` stays declined by name (a typed plan-gate decline upstream).

use openclownfish::{Context, Kernel, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::ir::{OpDef, ReduceOp, input};
use unpopped::oracle::evaluate;
use unpopped::{build_plan, generate};
use unpopped_vocab::{ArchSku, AxisMask, OpCategory, OperandDesc, structure_key};

mod common;

const SHAPE: [i64; 3] = [3, 4, 5];

fn dense(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

#[test]
fn narrow_reductions_agree_with_the_oracle_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let axis_sets: [(&[usize], bool); 5] = [
        (&[2], false),
        (&[0], false),
        (&[0, 2], false),
        (&[2], true),
        (&[0, 1, 2], true),
    ];
    let backend = OpenClC::new();
    let mut checks = 0;
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
            for kind in common::NARROW {
                let w = common::width(kind);
                let inp = common::narrow_buffer(kind, &SHAPE).unwrap();
                let inb = ctx.create_buffer(inp.bytes.len()).unwrap();
                queue.write_buffer(&inb, 0, &inp.bytes).unwrap();
                for op in [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Max, ReduceOp::Min] {
                    for (axes, keepdim) in axis_sets {
                        let rank = SHAPE.len();
                        let mut mask = AxisMask::EMPTY;
                        for &a in axes {
                            mask.set(a as u8);
                        }
                        let out_shape: Vec<i64> = if keepdim {
                            (0..rank)
                                .map(|a| if axes.contains(&a) { 1 } else { SHAPE[a] })
                                .collect()
                        } else {
                            (0..rank)
                                .filter(|a| !axes.contains(a))
                                .map(|a| SHAPE[a])
                                .collect()
                        };
                        let out_strides = dense(&out_shape);
                        let a = OperandDesc::new(rank, &SHAPE, &inp.strides, kind, 256);
                        let o =
                            OperandDesc::new(out_shape.len(), &out_shape, &out_strides, kind, 256);
                        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
                        let opdef =
                            OpDef::reduction_axes("r", 1, &[kind], input(0), op, mask, keepdim);
                        let plan = build_plan(&opdef, &key);
                        let kernel = generate(&opdef, &key, &backend);
                        let tag = format!("{kind:?}/{op:?}/{axes:?}/kd={keepdim}");
                        let prog = Program::build(&ctx, &kernel.source, "")
                            .unwrap_or_else(|e| panic!("{label}: {tag}: {e}\n{}", kernel.source));
                        let mut k = prog.kernel(&kernel.name).unwrap();
                        let n_out: i64 = out_shape.iter().product();
                        let nb = n_out as usize * w;
                        let out = ctx.create_buffer(nb).unwrap();
                        queue.write_buffer(&out, 0, &vec![0xA5u8; nb]).unwrap();
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
                        // SAFETY: one work item per output element, each guarded by
                        // `o < n_out`; every input offset is inside the dense input.
                        unsafe { queue.launch(&k, &[n_out as usize], None) }
                            .unwrap()
                            .wait()
                            .unwrap();
                        let mut got = vec![0u8; nb];
                        queue.read_buffer(&out, 0, &mut got).unwrap();
                        let want = evaluate(&plan, &[a, o], std::slice::from_ref(&inp), &[])
                            .remove(0)
                            .bytes;
                        assert_eq!(got, &want[..nb], "{label}: {tag}\n{}", kernel.source);
                        checks += 1;
                    }
                }
            }
            ran += 1;
        }
    }
    eprintln!("narrow reductions: {checks} case runs bit-exact on {ran} device(s)");
}

#[test]
fn narrow_integer_mean_is_declined_by_name_and_sum_lowers() {
    let backend = OpenClC::new();
    for kind in common::NARROW {
        let a = OperandDesc::new(2, &[4, 6], &[6, 1], kind, 256);
        let o = OperandDesc::new(1, &[4], &[1], kind, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        // Control: Sum at the same dtype and shape lowers, so the decline below is
        // about Mean and not the plan.
        let sum = OpDef::reduction("r", 1, &[kind], input(0), ReduceOp::Sum);
        assert!(backend.lower(&build_plan(&sum, &key)).is_ok(), "{kind:?}");
        let mean = OpDef::reduction("m", 1, &[kind], input(0), ReduceOp::Mean);
        // The plan gate may itself reject integer Mean; if it plans, we must refuse.
        if let Ok(plan) = unpopped::plan::try_build_plan(&mean, &key) {
            let err = backend.lower(&plan).expect_err("an integer Mean lowered");
            assert!(
                format!("{err:?}").contains("integer Mean"),
                "{kind:?}: {err:?}"
            );
        }
    }
}
