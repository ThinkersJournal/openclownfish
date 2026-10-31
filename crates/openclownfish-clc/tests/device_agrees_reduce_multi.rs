//! Reductions over two inputs (dot product, weighted Max/Min, a dot with a fused
//! post-expression) on every device against Unpopped's oracle, through the
//! serial kernel and the block-reduce variant (block sizes 8 and 32).
//!
//! Arguments: the input pointers, the output pointer, `shape{d}`, then `s{k}_{d}`
//! for every input `k` in turn, `so_{j}` and `n_out`. The second input is either
//! dense or broadcast along axis 0 (stride 0, a smaller buffer), so each input is
//! read through its own strides. Float data is exactly representable and short
//! enough that every fold is exact; integer data includes wrap-around.

use openclownfish::{Context, Kernel, Program, Queue, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::build_plan;
use unpopped::ir::{Access, Expr, OpDef, ReduceOp, input, konst, reduced};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped_vocab::{ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, structure_key};

const SHAPES: [&[i64]; 2] = [&[3, 4, 5], &[2, 3, 70]];
const BLOCKS: [usize; 2] = [8, 32];

fn dense(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

/// `n` elements of `dtype`, cycling through `which` of two value tables.
fn buffer(dtype: ElementKind, shape: &[i64], which: usize) -> TypedBuffer {
    let n: usize = shape.iter().product::<i64>() as usize;
    match dtype {
        ElementKind::F32 | ElementKind::F64 => {
            let cyc: [f64; 7] = if which == 0 {
                [1.0, 2.0, 0.5, -1.0, 0.25, -2.0, 1.5]
            } else {
                [0.5, -1.0, 2.0, 0.25, 1.5, 1.0, -0.5]
            };
            if dtype == ElementKind::F32 {
                let v: Vec<f32> = (0..n).map(|i| cyc[i % 7] as f32).collect();
                TypedBuffer::from_f32(shape, &v)
            } else {
                let v: Vec<f64> = (0..n).map(|i| cyc[i % 7]).collect();
                TypedBuffer::from_f64(shape, &v)
            }
        }
        ElementKind::I32 => {
            let cyc: [i32; 9] = if which == 0 {
                [i32::MAX, 1, -1, 2, 7, -7, i32::MIN, 3, 100_000]
            } else {
                [3, -2, 1, 5, -4, 2, 70_000, -9, 11]
            };
            let v: Vec<i32> = (0..n).map(|i| cyc[i % 9]).collect();
            TypedBuffer::from_i32(shape, &v)
        }
        _ => {
            let cyc: [i64; 9] = if which == 0 {
                [i64::MAX, 1, -1, 2, 7, -7, i64::MIN, 3, 1 << 40]
            } else {
                [3, -2, 1, 5, -4, 2, 1 << 33, -9, 11]
            };
            let v: Vec<i64> = (0..n).map(|i| cyc[i % 9]).collect();
            TypedBuffer::from_i64(shape, &v)
        }
    }
}

struct Kind {
    label: &'static str,
    op: ReduceOp,
    body: fn() -> Expr,
    post: Option<fn() -> Expr>,
    float_only: bool,
}

fn kinds() -> Vec<Kind> {
    vec![
        Kind {
            label: "dot",
            op: ReduceOp::Sum,
            body: || input(0) * input(1),
            post: None,
            float_only: false,
        },
        Kind {
            label: "wmax",
            op: ReduceOp::Max,
            body: || input(0) + input(1),
            post: None,
            float_only: false,
        },
        Kind {
            label: "wmin",
            op: ReduceOp::Min,
            body: || input(0) - input(1),
            post: None,
            float_only: false,
        },
        Kind {
            label: "dot*0.5",
            op: ReduceOp::Sum,
            body: || input(0) * input(1),
            post: Some(|| reduced(0) * konst(0.5)),
            float_only: true,
        },
    ]
}

#[derive(Clone)]
struct Case {
    dtype: ElementKind,
    kind: usize,
    shape: &'static [i64],
    axes: Vec<usize>,
    keepdim: bool,
    /// The second input is broadcast along axis 0.
    bcast: bool,
}

fn cases() -> Vec<Case> {
    let sets: [(&[usize], bool); 4] = [
        (&[2], false),
        (&[0], false),
        (&[1, 2], false),
        (&[0, 1, 2], true),
    ];
    let mut v = Vec::new();
    for dtype in [
        ElementKind::F32,
        ElementKind::F64,
        ElementKind::I32,
        ElementKind::I64,
    ] {
        let float = matches!(dtype, ElementKind::F32 | ElementKind::F64);
        for (ki, k) in kinds().iter().enumerate() {
            if k.float_only && !float {
                continue;
            }
            for shape in SHAPES {
                for (axes, keepdim) in sets {
                    for bcast in [false, true] {
                        v.push(Case {
                            dtype,
                            kind: ki,
                            shape,
                            axes: axes.to_vec(),
                            keepdim,
                            bcast,
                        });
                    }
                }
            }
        }
    }
    v
}

fn name(c: &Case) -> String {
    format!(
        "{:?}/{}/{:?}/{:?}/kd={}/bcast={}",
        c.dtype,
        kinds()[c.kind].label,
        c.shape,
        c.axes,
        c.keepdim,
        c.bcast
    )
}

fn width(d: ElementKind) -> usize {
    if matches!(d, ElementKind::I64 | ElementKind::F64) {
        8
    } else {
        4
    }
}

fn opdef_for(c: &Case) -> OpDef {
    let k = &kinds()[c.kind];
    let mut mask = AxisMask::EMPTY;
    for &a in &c.axes {
        mask.set(a as u8);
    }
    let mut op = OpDef::reduction_axes("r", 2, &[c.dtype], (k.body)(), k.op, mask, c.keepdim);
    if let (Some(post), Access::Reduction { post: slot, .. }) = (k.post, &mut op.access) {
        *slot = post().0;
    }
    op
}

#[allow(clippy::too_many_arguments)]
fn run(
    ctx: &Context,
    queue: &Queue,
    source: &str,
    kernel_name: &str,
    opts: &str,
    c: &Case,
    inputs: &[(&TypedBuffer, Vec<i64>)],
    out_strides: &[i64],
    n_out: i64,
    block: Option<usize>,
) -> Vec<u8> {
    let prog = match Program::build(ctx, source, opts) {
        Ok(p) => p,
        Err(e) => panic!("{} failed to build ({opts}):\n{e}\n{source}", name(c)),
    };
    let mut k = prog.kernel(kernel_name).unwrap();
    let w = width(c.dtype);
    let mut bufs = Vec::new();
    for (b, _) in inputs {
        let buf = ctx.create_buffer(b.bytes.len()).unwrap();
        queue.write_buffer(&buf, 0, &b.bytes).unwrap();
        bufs.push(buf);
    }
    let out_len = n_out as usize * w;
    let out = ctx.create_buffer(out_len).unwrap();
    queue.write_buffer(&out, 0, &vec![0xA5u8; out_len]).unwrap();
    for (i, b) in bufs.iter().enumerate() {
        k.set_arg_buffer(i as u32, b).unwrap();
    }
    k.set_arg_buffer(bufs.len() as u32, &out).unwrap();
    let mut arg = bufs.len() as u32 + 1;
    let mut push = |k: &mut Kernel, x: i64| {
        k.set_arg_i64(arg, x).unwrap();
        arg += 1;
    };
    for &x in c.shape {
        push(&mut k, x);
    }
    for (_, strides) in inputs {
        for &x in strides {
            push(&mut k, x);
        }
    }
    for &x in out_strides {
        push(&mut k, x);
    }
    push(&mut k, n_out);
    // SAFETY: one work item (or work-group) per output element, each guarded by
    // `o < n_out`; every input offset is inside its buffer (the broadcast input
    // has stride 0 along axis 0 and a buffer of the remaining extents).
    let ev = unsafe {
        match block {
            Some(b) => queue.launch(&k, &[n_out as usize * b], Some(&[b])),
            None => queue.launch(&k, &[n_out as usize], None),
        }
    }
    .unwrap();
    ev.wait().unwrap();
    let mut got = vec![0u8; out_len];
    queue.read_buffer(&out, 0, &mut got).unwrap();
    got
}

#[test]
fn multi_input_reductions_agree_with_the_oracle_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let mut ran = 0;
    for p in platforms {
        let Ok(devices) = p.devices() else { continue };
        for d in devices {
            let label = format!(
                "{} / {}",
                p.name().unwrap_or_default(),
                d.name().unwrap_or_default()
            );
            let Ok(exts) = d.extensions() else { continue };
            let has_fp64 = exts.iter().any(|e| e == "cl_khr_fp64");
            let backend = OpenClC::for_extensions(exts.iter().map(String::as_str));
            let Ok(ctx) = Context::new(&d) else { continue };
            let queue = ctx.create_queue().unwrap();
            let all = cases();
            let mut done = 0;
            for c in &all {
                if c.dtype == ElementKind::F64 && !has_fp64 {
                    continue;
                }
                done += 1;
                let rank = c.shape.len();
                let kept: Vec<usize> = (0..rank).filter(|a| !c.axes.contains(a)).collect();
                let out_shape: Vec<i64> = if c.keepdim {
                    (0..rank)
                        .map(|a| if c.axes.contains(&a) { 1 } else { c.shape[a] })
                        .collect()
                } else {
                    kept.iter().map(|&a| c.shape[a]).collect()
                };
                let out_strides = dense(&out_shape);
                let n_out: i64 = out_shape.iter().product();
                let in0 = buffer(c.dtype, c.shape, 0);
                let (in1, s1) = if c.bcast {
                    let small = &c.shape[1..];
                    let mut s = vec![0];
                    s.extend(dense(small));
                    (buffer(c.dtype, small, 1), s)
                } else {
                    (buffer(c.dtype, c.shape, 1), dense(c.shape))
                };
                let a0 = OperandDesc::new(rank, c.shape, &in0.strides, c.dtype, 256);
                let a1 = OperandDesc::new(rank, c.shape, &s1, c.dtype, 256);
                let o = OperandDesc::new(out_shape.len(), &out_shape, &out_strides, c.dtype, 256);
                let key = structure_key(OpCategory::Reduction, &[a0, a1, o], ArchSku::Sm89);
                let opdef = opdef_for(c);
                let plan = build_plan(&opdef, &key);
                let w = width(c.dtype);
                let want = evaluate(&plan, &[a0, a1, o], &[in0.clone(), in1.clone()], &[])
                    .remove(0)
                    .bytes;
                let want = &want[..n_out as usize * w];
                let ins = [(&in0, in0.strides.clone()), (&in1, s1.clone())];
                let serial = backend.lower(&plan).unwrap();
                let got = run(
                    &ctx,
                    &queue,
                    &serial.source,
                    &serial.name,
                    "",
                    c,
                    &ins,
                    &out_strides,
                    n_out,
                    None,
                );
                assert_eq!(got, want, "{label}: {} serial\n{}", name(c), serial.source);
                let variants = backend.lower_variants(&plan).unwrap();
                assert_eq!(variants.len(), 1, "{}", name(c));
                let kernel = &variants[0].kernels[0];
                for block in BLOCKS {
                    let got = run(
                        &ctx,
                        &queue,
                        &kernel.source,
                        &kernel.name,
                        &format!("-D OCF_BLOCK={block}"),
                        c,
                        &ins,
                        &out_strides,
                        n_out,
                        Some(block),
                    );
                    assert_eq!(
                        got,
                        want,
                        "{label}: {} B={block}\n{}",
                        name(c),
                        kernel.source
                    );
                }
            }
            eprintln!(
                "{label}: {done} of {} multi-input reduction cases agree (fp64: {has_fp64})",
                all.len()
            );
            ran += 1;
        }
    }
    eprintln!("multi-input reductions ran on {ran} device(s)");
}
