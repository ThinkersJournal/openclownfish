//! bf16 and f16 serial reductions and scans on every device against Unpopped's oracle.
//!
//! The data is quarters, halves, ones and twos, so every sum, product and prefix is
//! exact in the kernel's `float` accumulator and in the oracle's `f64` one. Each result
//! is then rounded once to the format, to nearest even, by both sides, so they agree
//! bit for bit. `Mean` divides in `float` on the device and in `f64` in the oracle
//! before that one rounding, so it is allowed one step of the format.

use openclownfish::{Context, Kernel, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::ir::{OpDef, ReduceOp, input, reduced};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::{build_plan, generate};
use unpopped_vocab::{ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, structure_key};

const SHAPE: [i64; 3] = [3, 4, 5];
const FORMATS: [ElementKind; 2] = [ElementKind::Bf16, ElementKind::F16];

/// 1, 2, 0.5, -1, 0.25, -2, 1.5 as stored patterns.
fn cycle(kind: ElementKind) -> [u16; 7] {
    match kind {
        ElementKind::Bf16 => [0x3f80, 0x4000, 0x3f00, 0xbf80, 0x3e80, 0xc000, 0x3fc0],
        _ => [0x3c00, 0x4000, 0x3800, 0xbc00, 0x3400, 0xc000, 0x3e00],
    }
}

fn dense(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

fn input_buffer(kind: ElementKind, shape: &[i64]) -> TypedBuffer {
    let n: usize = shape.iter().product::<i64>() as usize;
    let c = cycle(kind);
    let bytes: Vec<u8> = (0..n).flat_map(|i| c[i % 7].to_le_bytes()).collect();
    TypedBuffer::new(kind, shape.to_vec(), dense(shape), bytes)
}

/// Equal, or (for `Mean`) within one step of the format.
fn agrees(mean: bool, got: &[u8], want: &[u8]) -> bool {
    got.chunks(2).zip(want.chunks(2)).all(|(g, w)| {
        let (g, w) = (
            u16::from_le_bytes([g[0], g[1]]),
            u16::from_le_bytes([w[0], w[1]]),
        );
        g == w || (mean && g.abs_diff(w) <= 1 && (g ^ w) & 0x8000 == 0)
    })
}

#[test]
fn half_reductions_and_scans_agree_with_the_oracle_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let backend = OpenClC::new();
    let axis_sets: [(&[usize], bool); 5] = [
        (&[2], false),
        (&[0], false),
        (&[0, 2], false),
        (&[2], true),
        (&[0, 1, 2], true),
    ];
    let (mut checks, mut ran) = (0, 0);
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
            for kind in FORMATS {
                let inp = input_buffer(kind, &SHAPE);
                let inb = ctx.create_buffer(inp.bytes.len()).unwrap();
                queue.write_buffer(&inb, 0, &inp.bytes).unwrap();

                for op in [
                    ReduceOp::Sum,
                    ReduceOp::Prod,
                    ReduceOp::Max,
                    ReduceOp::Min,
                    ReduceOp::Mean,
                ] {
                    for (axes, keepdim) in axis_sets {
                        // A product over many elements leaves the format's range.
                        if op == ReduceOp::Prod && axes.len() > 1 {
                            continue;
                        }
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
                        let nb = n_out as usize * 2;
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
                        assert!(
                            agrees(op == ReduceOp::Mean, &got, &want[..nb]),
                            "{label}: {tag}: got {got:?}, oracle {:?}\n{}",
                            &want[..nb],
                            kernel.source
                        );
                        checks += 1;
                    }
                }

                // Scans along the last axis: every op, forward and reverse, inclusive
                // and exclusive.
                for rop in [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Max, ReduceOp::Min] {
                    for (reverse, exclusive) in
                        [(false, false), (false, true), (true, false), (true, true)]
                    {
                        let shape = [12i64, 5];
                        let sin = input_buffer(kind, &shape);
                        let sb = ctx.create_buffer(sin.bytes.len()).unwrap();
                        queue.write_buffer(&sb, 0, &sin.bytes).unwrap();
                        let a = OperandDesc::new(2, &shape, &dense(&shape), kind, 256);
                        let key = structure_key(OpCategory::Scan, &[a, a], ArchSku::Sm89);
                        let opdef = OpDef::scan(
                            "s",
                            1,
                            &[kind],
                            rop,
                            1,
                            reverse,
                            exclusive,
                            input(0),
                            reduced(0),
                        );
                        let plan = build_plan(&opdef, &key);
                        let kernel = generate(&opdef, &key, &backend);
                        let tag = format!("{kind:?} scan {rop:?} rev={reverse} excl={exclusive}");
                        let prog = Program::build(&ctx, &kernel.source, "")
                            .unwrap_or_else(|e| panic!("{label}: {tag}: {e}\n{}", kernel.source));
                        let mut k = prog.kernel(&kernel.name).unwrap();
                        let nb = (shape[0] * shape[1]) as usize * 2;
                        let out = ctx.create_buffer(nb).unwrap();
                        queue.write_buffer(&out, 0, &vec![0xA5u8; nb]).unwrap();
                        k.set_arg_buffer(0, &sb).unwrap();
                        k.set_arg_buffer(1, &out).unwrap();
                        k.set_arg_i64(2, shape[0]).unwrap();
                        k.set_arg_i64(3, shape[1]).unwrap();
                        // SAFETY: one work item per row, guarded by `row < n_out`.
                        unsafe { queue.launch(&k, &[shape[0] as usize], None) }
                            .unwrap()
                            .wait()
                            .unwrap();
                        let mut got = vec![0u8; nb];
                        queue.read_buffer(&out, 0, &mut got).unwrap();
                        let want = evaluate(&plan, &[a, a], std::slice::from_ref(&sin), &[])
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
    eprintln!("half reductions and scans: {checks} case runs on {ran} device(s)");
}
