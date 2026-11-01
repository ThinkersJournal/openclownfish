//! The block-scan variant on every device against Unpopped's oracle, at block
//! sizes 8 and 32 so rows longer than a block exercise the chunk carry. One
//! work-group per row.
//!
//! Arguments follow Baracuda's scan convention: the input and output pointers,
//! then `n_out` rows and the row length `k`. Float data is exactly
//! representable and short enough that every prefix is exact in `f32`, so the
//! oracle's `f64` prefix and the device's native one agree bit for bit; integer
//! data includes wrap-around. Every op runs forward and reverse, inclusive and
//! exclusive, on rank 2 and rank 3.

mod common;

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::{Backend, VariantFidelity};
use unpopped::build_plan;
use unpopped::ir::{OpDef, ReduceOp, input, reduced};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const SHAPES: [&[i64]; 4] = [&[4, 7], &[3, 19], &[2, 70], &[2, 3, 37]];
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
    if let Some(b) = common::narrow_buffer(dtype, shape) {
        return b;
    }
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
    reverse: bool,
    exclusive: bool,
    /// `pre = input(0) + input(0)` instead of the identity.
    doubled: bool,
    shape: &'static [i64],
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    for dtype in [
        ElementKind::F32,
        ElementKind::F64,
        ElementKind::I32,
        ElementKind::I64,
    ]
    .into_iter()
    .chain(common::NARROW)
    {
        for op in [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Max, ReduceOp::Min] {
            for shape in SHAPES {
                for reverse in [false, true] {
                    for exclusive in [false, true] {
                        v.push(Case {
                            dtype,
                            op,
                            reverse,
                            exclusive,
                            doubled: false,
                            shape,
                        });
                    }
                }
            }
        }
        v.push(Case {
            dtype,
            op: ReduceOp::Sum,
            reverse: false,
            exclusive: false,
            doubled: true,
            shape: SHAPES[0],
        });
    }
    v
}

fn name(c: &Case) -> String {
    format!(
        "{:?}/{:?}/rev={}/excl={}/pre2={}/{:?}",
        c.dtype, c.op, c.reverse, c.exclusive, c.doubled, c.shape
    )
}

#[test]
fn block_scans_agree_with_the_oracle_on_every_device() {
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
                let inp = input_buffer(c.dtype, c.shape);
                let a = OperandDesc::new(rank, c.shape, &dense(c.shape), c.dtype, 256);
                let key = structure_key(OpCategory::Scan, &[a, a], ArchSku::Sm89);
                let pre = if c.doubled {
                    input(0) + input(0)
                } else {
                    input(0)
                };
                let opdef = OpDef::scan(
                    "s",
                    1,
                    &[c.dtype],
                    c.op,
                    (rank - 1) as u8,
                    c.reverse,
                    c.exclusive,
                    pre,
                    reduced(0),
                );
                let plan = build_plan(&opdef, &key);
                let variants = backend.lower_variants(&plan).unwrap();
                assert_eq!(variants.len(), 1, "{}", name(c));
                let v = &variants[0];
                assert_eq!(v.tag, "blockscan");
                let float_sum_prod = matches!(c.dtype, ElementKind::F32 | ElementKind::F64)
                    && matches!(c.op, ReduceOp::Sum | ReduceOp::Prod);
                assert_eq!(
                    v.fidelity,
                    if float_sum_prod {
                        VariantFidelity::DeterministicallyDivergent
                    } else {
                        VariantFidelity::BitIdentical
                    },
                    "{}",
                    name(c)
                );
                let kernel = &v.kernels[0];
                let w = common::width(c.dtype);
                let len = *c.shape.last().unwrap();
                let n_el: i64 = c.shape.iter().product();
                let n_out = n_el / len;
                let want = evaluate(&plan, &[a, a], std::slice::from_ref(&inp), &[])
                    .remove(0)
                    .bytes;
                for block in BLOCKS {
                    let prog = match Program::build(
                        &ctx,
                        &kernel.source,
                        &format!("-D OCF_BLOCK={block}"),
                    ) {
                        Ok(p) => p,
                        Err(e) => panic!(
                            "{label}: {} B={block} failed to build:
{e}
{}",
                            name(c),
                            kernel.source
                        ),
                    };
                    let mut k = prog.kernel(&kernel.name).unwrap();
                    let inb = ctx.create_buffer(inp.bytes.len()).unwrap();
                    queue.write_buffer(&inb, 0, &inp.bytes).unwrap();
                    let out = ctx.create_buffer(n_el as usize * w).unwrap();
                    queue
                        .write_buffer(&out, 0, &vec![0xA5u8; n_el as usize * w])
                        .unwrap();
                    k.set_arg_buffer(0, &inb).unwrap();
                    k.set_arg_buffer(1, &out).unwrap();
                    k.set_arg_i64(2, n_out).unwrap();
                    k.set_arg_i64(3, len).unwrap();
                    // SAFETY: one work-group per row, guarded by `row < n_out`;
                    // each touches only the `len` elements of its own row.
                    let ev = unsafe { queue.launch(&k, &[n_out as usize * block], Some(&[block])) }
                        .unwrap();
                    ev.wait().unwrap();
                    let mut got = vec![0u8; n_el as usize * w];
                    queue.read_buffer(&out, 0, &mut got).unwrap();
                    assert_eq!(
                        got,
                        &want[..n_el as usize * w],
                        "{label}: {} B={block}
{}",
                        name(c),
                        kernel.source
                    );
                }
            }
            eprintln!(
                "{label}: {done} of {} scan cases agree (fp64: {has_fp64})",
                all.len()
            );
            ran += 1;
        }
    }
    eprintln!("scans ran on {ran} device(s)");
}

#[test]
fn only_a_serial_scan_has_a_block_variant() {
    let a = OperandDesc::new(2, &[2, 8], &[8, 1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::Scan, &[a, a], ArchSku::Sm89);
    let op = OpDef::scan_simple("m", &[ElementKind::F32], ReduceOp::Max, 1, false, false);
    let plan = build_plan(&op, &key);
    assert_eq!(OpenClC::new().lower_variants(&plan).unwrap().len(), 1);
    let e = OperandDesc::new(1, &[8], &[1], ElementKind::F32, 256);
    let ekey = structure_key(OpCategory::BinaryElementwise, &[e, e], ArchSku::Sm89);
    let add = OpDef::elementwise("a", 2, &[ElementKind::F32], input(0) + input(1));
    let eplan = build_plan(&add, &ekey);
    assert!(OpenClC::new().lower_variants(&eplan).unwrap().is_empty());
}
