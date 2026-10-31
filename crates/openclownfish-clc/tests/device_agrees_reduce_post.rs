//! Fused reduction post-expressions on every device against Unpopped's oracle,
//! through both the serial kernel and the block-reduce variant (block sizes 8
//! and 32). The post sees the finalized fold (after `Mean`'s division) as
//! `Reduced(0)`, may use constants, and may not read an input.
//!
//! Float data is exactly representable and short enough that every fold is
//! exact; posts that round (`sqrt`, `Mean`'s division) are compared within a few
//! ulp, the rest bit for bit. Integer data includes wrap-around.

use openclownfish::{Context, Kernel, Program, Queue, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::build_plan;
use unpopped::ir::{Access, Expr, OpDef, ReduceOp, UnaryOp, input, konst, reduced};
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

fn input_buffer(dtype: ElementKind, shape: &[i64]) -> TypedBuffer {
    let n: usize = shape.iter().product::<i64>() as usize;
    match dtype {
        ElementKind::F32 => {
            let cyc = [1.0f32, 2.0, 0.5, -1.0, 0.25, -2.0, 1.5];
            let v: Vec<f32> = (0..n).map(|i| cyc[i % cyc.len()]).collect();
            TypedBuffer::from_f32(shape, &v)
        }
        ElementKind::F64 => {
            let cyc = [1.0f64, 2.0, 0.5, -1.0, 0.25, -2.0, 1.5];
            let v: Vec<f64> = (0..n).map(|i| cyc[i % cyc.len()]).collect();
            TypedBuffer::from_f64(shape, &v)
        }
        ElementKind::I32 => {
            let cyc = [i32::MAX, 1, -1, 2, 7, -7, i32::MIN, 3, 100_000];
            let v: Vec<i32> = (0..n).map(|i| cyc[i % cyc.len()]).collect();
            TypedBuffer::from_i32(shape, &v)
        }
        _ => {
            let cyc = [i64::MAX, 1, -1, 2, 7, -7, i64::MIN, 3, 1 << 40];
            let v: Vec<i64> = (0..n).map(|i| cyc[i % cyc.len()]).collect();
            TypedBuffer::from_i64(shape, &v)
        }
    }
}

/// One fused-post shape: the fold body, the op and the post, per element kind.
struct Post {
    label: &'static str,
    op: ReduceOp,
    body: fn() -> Expr,
    post: fn() -> Expr,
    float_only: bool,
    /// Compared within this many ulp (0: bit for bit).
    ulps: u64,
}

fn posts() -> Vec<Post> {
    vec![
        Post {
            label: "scale+1",
            op: ReduceOp::Sum,
            body: || input(0),
            post: || reduced(0) * konst(0.5) + konst(1.0),
            float_only: true,
            ulps: 0,
        },
        Post {
            label: "neg(sqr)",
            op: ReduceOp::Max,
            body: || input(0),
            post: || reduced(0).unary(UnaryOp::Sqr).unary(UnaryOp::Neg),
            float_only: true,
            ulps: 0,
        },
        Post {
            label: "abs(min)",
            op: ReduceOp::Min,
            body: || input(0),
            post: || reduced(0).unary(UnaryOp::Abs),
            float_only: true,
            ulps: 0,
        },
        Post {
            label: "norm2",
            op: ReduceOp::Sum,
            body: || input(0) * input(0),
            post: || reduced(0).sqrt(),
            float_only: true,
            ulps: 4,
        },
        Post {
            label: "mean*2",
            op: ReduceOp::Mean,
            body: || input(0),
            post: || reduced(0) * konst(2.0),
            float_only: true,
            ulps: 4,
        },
        Post {
            label: "red*red",
            op: ReduceOp::Sum,
            body: || input(0),
            post: || reduced(0) * reduced(0),
            float_only: false,
            ulps: 0,
        },
        Post {
            label: "red+red",
            op: ReduceOp::Max,
            body: || input(0),
            post: || reduced(0) + reduced(0),
            float_only: false,
            ulps: 0,
        },
    ]
}

#[derive(Clone)]
struct Case {
    dtype: ElementKind,
    post: usize,
    shape: &'static [i64],
    axes: Vec<usize>,
    keepdim: bool,
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
        for (pi, p) in posts().iter().enumerate() {
            if p.float_only && !float {
                continue;
            }
            for shape in SHAPES {
                for (axes, keepdim) in sets {
                    v.push(Case {
                        dtype,
                        post: pi,
                        shape,
                        axes: axes.to_vec(),
                        keepdim,
                    });
                }
            }
        }
    }
    v
}

fn name(c: &Case) -> String {
    format!(
        "{:?}/{}/{:?}/{:?}/kd={}",
        c.dtype,
        posts()[c.post].label,
        c.shape,
        c.axes,
        c.keepdim
    )
}

fn width(d: ElementKind) -> usize {
    if matches!(d, ElementKind::I64 | ElementKind::F64) {
        8
    } else {
        4
    }
}

fn ulp_distance(a: &[u8], b: &[u8]) -> u64 {
    if a.len() == 4 {
        let (a, b) = (
            i64::from(u32::from_le_bytes(a.try_into().unwrap())),
            i64::from(u32::from_le_bytes(b.try_into().unwrap())),
        );
        a.abs_diff(b)
    } else {
        let (a, b) = (
            u64::from_le_bytes(a.try_into().unwrap()),
            u64::from_le_bytes(b.try_into().unwrap()),
        );
        a.abs_diff(b)
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    ctx: &Context,
    queue: &Queue,
    source: &str,
    kernel_name: &str,
    opts: &str,
    c: &Case,
    out_strides: &[i64],
    n_out: i64,
    inp: &TypedBuffer,
    block: Option<usize>,
) -> Vec<u8> {
    let prog = match Program::build(ctx, source, opts) {
        Ok(p) => p,
        Err(e) => panic!("{} failed to build ({opts}):\n{e}\n{source}", name(c)),
    };
    let mut k = prog.kernel(kernel_name).unwrap();
    let w = width(c.dtype);
    let inb = ctx.create_buffer(inp.bytes.len()).unwrap();
    queue.write_buffer(&inb, 0, &inp.bytes).unwrap();
    let out_len = n_out as usize * w;
    let out = ctx.create_buffer(out_len).unwrap();
    queue.write_buffer(&out, 0, &vec![0xA5u8; out_len]).unwrap();
    k.set_arg_buffer(0, &inb).unwrap();
    k.set_arg_buffer(1, &out).unwrap();
    let mut arg = 2u32;
    let mut push = |k: &mut Kernel, x: i64| {
        k.set_arg_i64(arg, x).unwrap();
        arg += 1;
    };
    for &x in c.shape {
        push(&mut k, x);
    }
    for &x in &inp.strides {
        push(&mut k, x);
    }
    for &x in out_strides {
        push(&mut k, x);
    }
    push(&mut k, n_out);
    // SAFETY: one work item (or work-group) per output element, each guarded by
    // `o < n_out`; every input offset is inside the input buffer.
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

fn for_each_device(mut f: impl FnMut(&str, &Context, &Queue, &OpenClC, bool)) {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
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
            f(&label, &ctx, &queue, &backend, has_fp64);
        }
    }
}

fn opdef_for(c: &Case) -> OpDef {
    let p = &posts()[c.post];
    let mut mask = AxisMask::EMPTY;
    for &a in &c.axes {
        mask.set(a as u8);
    }
    let mut op = OpDef::reduction_axes("r", 1, &[c.dtype], (p.body)(), p.op, mask, c.keepdim);
    if let Access::Reduction { post, .. } = &mut op.access {
        *post = (p.post)().0;
    }
    op
}

#[test]
fn fused_post_expressions_agree_with_the_oracle_on_every_device() {
    let mut ran = 0;
    for_each_device(|label, ctx, queue, backend, has_fp64| {
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
            let inp = input_buffer(c.dtype, c.shape);
            let a = OperandDesc::new(rank, c.shape, &inp.strides, c.dtype, 256);
            let o = OperandDesc::new(out_shape.len(), &out_shape, &out_strides, c.dtype, 256);
            let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
            let opdef = opdef_for(c);
            let plan = build_plan(&opdef, &key);
            let w = width(c.dtype);
            let want = evaluate(&plan, &[a, o], std::slice::from_ref(&inp), &[])
                .remove(0)
                .bytes;
            let want = &want[..n_out as usize * w];
            let ulps = posts()[c.post].ulps;
            let check = |got: &[u8], what: &str, src: &str| {
                if ulps == 0 {
                    assert_eq!(got, want, "{label}: {} {what}\n{src}", name(c));
                } else {
                    for (g, x) in got.chunks(w).zip(want.chunks(w)) {
                        assert!(
                            ulp_distance(g, x) <= ulps,
                            "{label}: {} {what}: {g:?} vs oracle {x:?}\n{src}",
                            name(c)
                        );
                    }
                }
            };
            let serial = backend.lower(&plan).unwrap();
            let got = run(
                ctx,
                queue,
                &serial.source,
                &serial.name,
                "",
                c,
                &out_strides,
                n_out,
                &inp,
                None,
            );
            check(&got, "serial", &serial.source);
            let variants = backend.lower_variants(&plan).unwrap();
            assert_eq!(variants.len(), 1, "{}", name(c));
            let kernel = &variants[0].kernels[0];
            for block in BLOCKS {
                let got = run(
                    ctx,
                    queue,
                    &kernel.source,
                    &kernel.name,
                    &format!("-D OCF_BLOCK={block}"),
                    c,
                    &out_strides,
                    n_out,
                    &inp,
                    Some(block),
                );
                check(&got, &format!("B={block}"), &kernel.source);
            }
        }
        eprintln!(
            "{label}: {done} of {} fused-post cases agree (fp64: {has_fp64})",
            all.len()
        );
        ran += 1;
    });
    eprintln!("fused post-expressions ran on {ran} device(s)");
}

/// Unpopped's plan gate rejects it (by panic) before the emitter sees it, so
/// the emitter's own `Input` guard is a second layer that no plan reaches.
#[test]
#[should_panic(expected = "post-expr must not read Input")]
fn the_plan_gate_rejects_a_post_that_reads_an_input() {
    let a = OperandDesc::new(2, &[4, 6], &[6, 1], ElementKind::F32, 256);
    let o = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    let mut op = OpDef::reduction("s", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
    if let Access::Reduction { post, .. } = &mut op.access {
        *post = (reduced(0) + input(0)).0;
    }
    let _ = build_plan(&op, &key);
}
