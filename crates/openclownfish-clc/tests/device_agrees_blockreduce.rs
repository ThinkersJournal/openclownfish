//! The block-reduce variant on every device: against Unpopped's oracle, and
//! (for Max/Min over NaNs and signed zeros, and empty extents) bit for bit
//! against the serial kernel on the same device. One work-group per output
//! element, at block sizes 8 and 32, so reduced extents both shorter and longer
//! than a block, and blocks with empty chunks, are exercised.
//!
//! Arguments are the serial reduction's: input and output pointers, `shape{d}`,
//! `s0_{d}`, `so_{j}`, `n_out`. Float data is exactly representable and short
//! enough that every fold is exact in `f32`, so reassociation cannot change a
//! bit; integer data includes wrap-around.

use openclownfish::{Context, Kernel, Program, Queue, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::{Backend, VariantFidelity};
use unpopped::build_plan;
use unpopped::ir::{OpDef, ReduceOp, input};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped_vocab::{ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, structure_key};

const SMALL: [i64; 3] = [3, 4, 5];
const LONG: [i64; 3] = [2, 3, 70];
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

#[derive(Clone)]
struct Case {
    dtype: ElementKind,
    op: ReduceOp,
    shape: &'static [i64],
    axes: Vec<usize>,
    keepdim: bool,
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    let small_sets: [(&[usize], bool); 6] = [
        (&[2], false),
        (&[0], false),
        (&[1], false),
        (&[0, 2], false),
        (&[2], true),
        (&[0, 1, 2], true),
    ];
    let long_sets: [(&[usize], bool); 4] = [
        (&[2], false),
        (&[1, 2], false),
        (&[0], true),
        (&[0, 1, 2], true),
    ];
    for dtype in [
        ElementKind::F32,
        ElementKind::F64,
        ElementKind::I32,
        ElementKind::I64,
    ] {
        let float = matches!(dtype, ElementKind::F32 | ElementKind::F64);
        let ops: &[ReduceOp] = if float {
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
            for (shape, sets) in [(&SMALL[..], &small_sets[..]), (&LONG[..], &long_sets[..])] {
                for &(axes, keepdim) in sets {
                    // Float products leave f32 exactness quickly: fold few.
                    if float && op == ReduceOp::Prod && (axes.len() > 1 || shape[axes[0]] > 5) {
                        continue;
                    }
                    let shape: &'static [i64] = if shape.len() == 3 && shape[2] == 5 {
                        &SMALL
                    } else {
                        &LONG
                    };
                    v.push(Case {
                        dtype,
                        op,
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
        "{:?}/{:?}/{:?}/{:?}/kd={}",
        c.dtype, c.op, c.shape, c.axes, c.keepdim
    )
}

struct Geometry {
    key: unpopped_vocab::StructureKey,
    a: OperandDesc,
    o: OperandDesc,
    opdef: OpDef,
    out_shape: Vec<i64>,
    out_strides: Vec<i64>,
    n_out: i64,
}

fn geometry(c: &Case, in_strides: &[i64]) -> Geometry {
    let rank = c.shape.len();
    let mut mask = AxisMask::EMPTY;
    for &a in &c.axes {
        mask.set(a as u8);
    }
    let kept: Vec<usize> = (0..rank).filter(|a| !c.axes.contains(a)).collect();
    let out_shape: Vec<i64> = if c.keepdim {
        (0..rank)
            .map(|a| if c.axes.contains(&a) { 1 } else { c.shape[a] })
            .collect()
    } else {
        kept.iter().map(|&a| c.shape[a]).collect()
    };
    let out_strides = dense(&out_shape);
    let a = OperandDesc::new(rank, c.shape, in_strides, c.dtype, 256);
    let o = OperandDesc::new(out_shape.len(), &out_shape, &out_strides, c.dtype, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    let opdef = OpDef::reduction_axes("r", 1, &[c.dtype], input(0), c.op, mask, c.keepdim);
    let n_out = out_shape.iter().product();
    Geometry {
        key,
        a,
        o,
        opdef,
        out_shape,
        out_strides,
        n_out,
    }
}

fn width(d: ElementKind) -> usize {
    if matches!(d, ElementKind::I64 | ElementKind::F64) {
        8
    } else {
        4
    }
}

/// Builds `source` with `opts`, runs `name` over `inp` and returns the output bytes.
/// `block` is `Some(B)` for a work-group-per-output launch, `None` for serial.
#[allow(clippy::too_many_arguments)]
fn run(
    ctx: &Context,
    queue: &Queue,
    source: &str,
    kernel_name: &str,
    opts: &str,
    c: &Case,
    g: &Geometry,
    inp: &[u8],
    in_strides: &[i64],
    block: Option<usize>,
) -> Vec<u8> {
    let prog = match Program::build(ctx, source, opts) {
        Ok(p) => p,
        Err(e) => panic!("{} failed to build ({opts}):\n{e}\n{source}", name(c)),
    };
    let mut k = prog.kernel(kernel_name).unwrap();
    let w = width(c.dtype);
    let inb = ctx.create_buffer(inp.len().max(w)).unwrap();
    if !inp.is_empty() {
        queue.write_buffer(&inb, 0, inp).unwrap();
    }
    let out_len = (g.n_out as usize * w).max(w);
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
    for &x in in_strides {
        push(&mut k, x);
    }
    for &x in &g.out_strides {
        push(&mut k, x);
    }
    push(&mut k, g.n_out);
    // SAFETY: one work item (or work-group) per output element, each guarded by
    // `o < n_out`; every input offset is inside the input buffer.
    let ev = unsafe {
        match block {
            Some(b) => queue.launch(&k, &[g.n_out as usize * b], Some(&[b])),
            None => queue.launch(&k, &[g.n_out as usize], None),
        }
    }
    .unwrap();
    ev.wait().unwrap();
    let mut got = vec![0u8; out_len];
    queue.read_buffer(&out, 0, &mut got).unwrap();
    got.truncate(g.n_out as usize * w);
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

#[test]
fn block_reductions_agree_with_the_oracle_on_every_device() {
    let mut ran = 0;
    for_each_device(|label, ctx, queue, backend, has_fp64| {
        let all = cases();
        let mut done = 0;
        for c in &all {
            if c.dtype == ElementKind::F64 && !has_fp64 {
                continue;
            }
            done += 1;
            let inp = input_buffer(c.dtype, c.shape);
            let g = geometry(c, &inp.strides);
            let plan = build_plan(&g.opdef, &g.key);
            let variants = backend.lower_variants(&plan).unwrap();
            assert_eq!(variants.len(), 1, "{}", name(c));
            let v = &variants[0];
            assert_eq!(v.tag, "blockreduce");
            let float = matches!(c.dtype, ElementKind::F32 | ElementKind::F64);
            let reassociates =
                float && matches!(c.op, ReduceOp::Sum | ReduceOp::Mean | ReduceOp::Prod);
            assert_eq!(
                v.fidelity,
                if reassociates {
                    VariantFidelity::DeterministicallyDivergent
                } else {
                    VariantFidelity::BitIdentical
                },
                "{}",
                name(c)
            );
            let kernel = &v.kernels[0];
            let want = evaluate(&plan, &[g.a, g.o], std::slice::from_ref(&inp), &[])
                .remove(0)
                .bytes;
            let w = width(c.dtype);
            let want = &want[..g.n_out as usize * w];
            for block in BLOCKS {
                let got = run(
                    ctx,
                    queue,
                    &kernel.source,
                    &kernel.name,
                    &format!("-D OCF_BLOCK={block}"),
                    c,
                    &g,
                    &inp.bytes,
                    &inp.strides,
                    Some(block),
                );
                if c.op == ReduceOp::Mean && c.dtype == ElementKind::F32 {
                    // f32 division vs f64 division rounded to f32: allow 1 ulp.
                    for (a, b) in got.chunks(4).zip(want.chunks(4)) {
                        let a = f32::from_le_bytes(a.try_into().unwrap());
                        let b = f32::from_le_bytes(b.try_into().unwrap());
                        assert!(
                            (i64::from(a.to_bits()) - i64::from(b.to_bits())).abs() <= 1,
                            "{label}: {} B={block}: kernel {a}, oracle {b}",
                            name(c)
                        );
                    }
                } else {
                    assert_eq!(
                        got,
                        want,
                        "{label}: {} B={block}\n{}",
                        name(c),
                        kernel.source
                    );
                }
            }
        }
        eprintln!(
            "{label}: {done} of {} block-reduce cases agree (fp64: {has_fp64})",
            all.len()
        );
        ran += 1;
    });
    eprintln!("block reductions ran on {ran} device(s)");
}

/// Max/Min pick an operand, so the tree must reproduce the serial fold's bits
/// even where operands compare equal or unordered: signed zeros, NaN payloads.
#[test]
fn block_max_min_match_the_serial_kernel_bit_for_bit() {
    let mut ran = 0;
    for_each_device(|label, ctx, queue, backend, _| {
        let pattern = [
            -0.0f32,
            0.0,
            f32::from_bits(0x7fc0_0001),
            1.0,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_0002),
            0.0,
            -0.0,
            f32::INFINITY,
            2.0,
            -0.0,
        ];
        let shape: &'static [i64] = &[3, 70];
        for op in [ReduceOp::Max, ReduceOp::Min] {
            // Rotate the pattern per run so every position takes a turn.
            for rot in 0..pattern.len() {
                let n = (shape[0] * shape[1]) as usize;
                let v: Vec<f32> = (0..n).map(|i| pattern[(i + rot) % pattern.len()]).collect();
                let inp = TypedBuffer::from_f32(shape, &v);
                let c = Case {
                    dtype: ElementKind::F32,
                    op,
                    shape,
                    axes: vec![1],
                    keepdim: false,
                };
                let g = geometry(&c, &inp.strides);
                let plan = build_plan(&g.opdef, &g.key);
                let serial = backend.lower(&plan).unwrap();
                let variants = backend.lower_variants(&plan).unwrap();
                let kernel = &variants[0].kernels[0];
                let want = run(
                    ctx,
                    queue,
                    &serial.source,
                    &serial.name,
                    "",
                    &c,
                    &g,
                    &inp.bytes,
                    &inp.strides,
                    None,
                );
                for block in BLOCKS {
                    let got = run(
                        ctx,
                        queue,
                        &kernel.source,
                        &kernel.name,
                        &format!("-D OCF_BLOCK={block}"),
                        &c,
                        &g,
                        &inp.bytes,
                        &inp.strides,
                        Some(block),
                    );
                    assert_eq!(got, want, "{label}: {op:?} rot={rot} B={block}");
                }
            }
        }
        ran += 1;
    });
    eprintln!("block max/min bit-compared on {ran} device(s)");
}

/// An empty reduced extent leaves the serial seed (0 for floats, the dtype
/// extreme for integers), not the tree's starting identity.
#[test]
fn empty_reduced_extent_matches_the_serial_kernel() {
    let mut ran = 0;
    for_each_device(|label, ctx, queue, backend, _| {
        let shape: &'static [i64] = &[3, 0];
        for dtype in [ElementKind::F32, ElementKind::I32] {
            for op in [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Max, ReduceOp::Min] {
                let c = Case {
                    dtype,
                    op,
                    shape,
                    axes: vec![1],
                    keepdim: false,
                };
                let strides = dense(shape);
                let g = geometry(&c, &strides);
                let Ok(plan) = unpopped::plan::try_build_plan(&g.opdef, &g.key) else {
                    return;
                };
                let serial = backend.lower(&plan).unwrap();
                let variants = backend.lower_variants(&plan).unwrap();
                let kernel = &variants[0].kernels[0];
                let want = run(
                    ctx,
                    queue,
                    &serial.source,
                    &serial.name,
                    "",
                    &c,
                    &g,
                    &[],
                    &strides,
                    None,
                );
                let got = run(
                    ctx,
                    queue,
                    &kernel.source,
                    &kernel.name,
                    "-D OCF_BLOCK=8",
                    &c,
                    &g,
                    &[],
                    &strides,
                    Some(8),
                );
                assert_eq!(got, want, "{label}: {dtype:?}/{op:?} empty extent");
                assert_eq!(g.out_shape, vec![3]);
            }
        }
        ran += 1;
    });
    eprintln!("empty-extent reductions checked on {ran} device(s)");
}

#[test]
fn only_a_reduction_or_scan_has_a_block_variant() {
    let a = OperandDesc::new(2, &[4, 6], &[6, 1], ElementKind::F32, 256);
    let o = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    let sum = OpDef::reduction("s", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
    let plan = build_plan(&sum, &key);
    let v = OpenClC::new().lower_variants(&plan).unwrap();
    assert_eq!(v.len(), 1);
    assert_eq!(v[0].tag, "blockreduce");
    let e = OperandDesc::new(1, &[8], &[1], ElementKind::F32, 256);
    let ekey = structure_key(OpCategory::BinaryElementwise, &[e, e], ArchSku::Sm89);
    let add = OpDef::elementwise("a", 2, &[ElementKind::F32], input(0) + input(1));
    let eplan = build_plan(&add, &ekey);
    assert!(OpenClC::new().lower_variants(&eplan).unwrap().is_empty());
}
