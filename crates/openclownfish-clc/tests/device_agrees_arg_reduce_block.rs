//! The block-reduce variant of `ArgMax` and `ArgMin` on every device, at three block sizes: the
//! kernel, Unpopped's oracle and an independent host fold written here from the rule must all
//! give the same indices. The reduced extents are 6 and 36, so some blocks leave work items with
//! an empty chunk and the ties and NaNs straddle chunk boundaries at B = 4.
//!
//! The rule is KISS-OPS-6.11-0007's: NaN is the greatest value and ties go to the lower
//! index, so `ArgMax` of a run with a NaN is the first NaN's index, `ArgMin` never takes a NaN
//! while a number exists, and a row of only NaN gives 0 for both. The index is the position in
//! the row-major order of the reduced axes (one axis: that axis's coordinate; all axes: the
//! flat index), and the output is an `I64` whatever the input type. The corpus has ties, a NaN
//! first, a NaN in the middle with larger numbers after it, an all-negative row, equal zeros and
//! an all-NaN row, so a seed of 0, a last-wins tie and a skipped NaN each change an answer.

use openclownfish::{Context, Kernel, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::{Backend, VariantFidelity};
use unpopped::ir::{OpDef, ReduceOp, input};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::build_plan_for;
use unpopped_vocab::{ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, structure_key};

const BLOCKS: [usize; 3] = [4, 16, 64];
const ROWS: i64 = 6;
const COLS: i64 = 6;

fn corpus() -> Vec<f64> {
    let n = f64::NAN;
    vec![
        3.0, 9.0, 9.0, 1.0, 1.0, 2.0, // ties: the first 9 and the first 1
        n, 1.0, 2.0, 3.0, -4.0, 5.0, // a NaN first
        1.0, 4.0, n, 7.0, n, 2.0, // a NaN in the middle, larger numbers after it
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, // all negative
        0.0, -0.0, 0.0, -0.0, 0.0, -0.0, // equal zeros
        n, n, n, n, n, n, // all NaN
    ]
}

fn is_float(k: ElementKind) -> bool {
    matches!(
        k,
        ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
    )
}

fn is_unsigned(k: ElementKind) -> bool {
    matches!(
        k,
        ElementKind::U8 | ElementKind::U16 | ElementKind::U32 | ElementKind::U64
    )
}

fn width(k: ElementKind) -> usize {
    match k {
        ElementKind::I8 | ElementKind::U8 => 1,
        ElementKind::I16 | ElementKind::U16 | ElementKind::F16 | ElementKind::Bf16 => 2,
        ElementKind::I64 | ElementKind::U64 | ElementKind::F64 => 8,
        _ => 4,
    }
}

/// The corpus as this dtype holds it: an integer has no NaN (it becomes 0) and an unsigned one
/// no sign (the magnitude is taken), so the host fold reads these values back.
fn values(k: ElementKind) -> Vec<f64> {
    corpus()
        .into_iter()
        .map(|v| {
            if is_float(k) {
                v
            } else if v.is_nan() {
                0.0
            } else if is_unsigned(k) {
                v.abs()
            } else {
                v
            }
        })
        .collect()
}

/// A small finite value (an integer up to 9, or zero) as an f16 pattern; exact.
fn f16_bits(v: f64) -> u16 {
    if v.is_nan() {
        return 0x7e00;
    }
    let sign = if v.is_sign_negative() { 0x8000u16 } else { 0 };
    if v == 0.0 {
        return sign;
    }
    let b = (v.abs() as f32).to_bits();
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    sign | ((exp as u16) << 10) | ((b >> 13) & 0x3ff) as u16
}

fn bf16_bits(v: f64) -> u16 {
    if v.is_nan() {
        return 0x7fc0;
    }
    ((v as f32).to_bits() >> 16) as u16
}

fn input_buffer(k: ElementKind) -> TypedBuffer {
    let bytes: Vec<u8> = values(k)
        .iter()
        .flat_map(|&v| match k {
            ElementKind::F32 => (v as f32).to_le_bytes().to_vec(),
            ElementKind::F64 => v.to_le_bytes().to_vec(),
            ElementKind::F16 => f16_bits(v).to_le_bytes().to_vec(),
            ElementKind::Bf16 => bf16_bits(v).to_le_bytes().to_vec(),
            _ => (v as i64 as u64).to_le_bytes()[..width(k)].to_vec(),
        })
        .collect();
    TypedBuffer::new(k, vec![ROWS, COLS], vec![COLS, 1], bytes)
}

/// The index of the extreme of `xs` under the rule, from its definition.
fn arg_of(xs: &[f64], max: bool) -> i64 {
    // NaN is the greatest value, and only a strictly better element replaces the best.
    let greater = |a: f64, b: f64| match (a.is_nan(), b.is_nan()) {
        (true, true) => false,
        (true, false) => true,
        (false, true) => false,
        (false, false) => a > b,
    };
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate().skip(1) {
        let better = if max {
            greater(x, xs[best])
        } else {
            greater(xs[best], x)
        };
        if better {
            best = i;
        }
    }
    best as i64
}

/// The answer, row-major over the reduced set.
fn reference(k: ElementKind, max: bool, axes: &[usize]) -> Vec<i64> {
    let v = values(k);
    let cell = |r: i64, c: i64| v[(r * COLS + c) as usize];
    match axes {
        [1] => (0..ROWS)
            .map(|r| arg_of(&(0..COLS).map(|c| cell(r, c)).collect::<Vec<_>>(), max))
            .collect(),
        [0] => (0..COLS)
            .map(|c| arg_of(&(0..ROWS).map(|r| cell(r, c)).collect::<Vec<_>>(), max))
            .collect(),
        _ => vec![arg_of(&v, max)],
    }
}

fn dense(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

const KINDS: [ElementKind; 11] = [
    ElementKind::F32,
    ElementKind::F64,
    ElementKind::F16,
    ElementKind::Bf16,
    ElementKind::I8,
    ElementKind::U8,
    ElementKind::I16,
    ElementKind::U16,
    ElementKind::I32,
    ElementKind::I64,
    ElementKind::U64,
];

fn arg_op(k: ElementKind, rop: ReduceOp, mask: AxisMask, keepdim: bool) -> OpDef {
    let mut op = OpDef::reduction_axes("r", 1, &[k], input(0), rop, mask, keepdim);
    op.out_dtype = Some(ElementKind::I64);
    op
}

#[test]
fn block_arg_reductions_agree_with_the_oracle_and_the_rule_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let axis_sets: [(&[usize], bool); 3] = [(&[1], false), (&[0], false), (&[0, 1], true)];
    let (mut checks, mut ran) = (0, 0);
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
            for kind in KINDS {
                if kind == ElementKind::F64 && !has_fp64 {
                    continue;
                }
                let inp = input_buffer(kind);
                let inb = ctx.create_buffer(inp.bytes.len()).unwrap();
                queue.write_buffer(&inb, 0, &inp.bytes).unwrap();
                for rop in [ReduceOp::ArgMax, ReduceOp::ArgMin] {
                    for (axes, keepdim) in axis_sets {
                        let mut mask = AxisMask::EMPTY;
                        for &a in axes {
                            mask.set(a as u8);
                        }
                        let out_shape: Vec<i64> = if keepdim {
                            vec![1, 1]
                        } else {
                            (0..2)
                                .filter(|a| !axes.contains(a))
                                .map(|a| [ROWS, COLS][a])
                                .collect()
                        };
                        let out_strides = dense(&out_shape);
                        let a = OperandDesc::new(2, &[ROWS, COLS], &inp.strides, kind, 256);
                        let o = OperandDesc::new(
                            out_shape.len(),
                            &out_shape,
                            &out_strides,
                            ElementKind::I64,
                            256,
                        );
                        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
                        let opdef = arg_op(kind, rop, mask, keepdim);
                        let plan = build_plan_for(&opdef, &key, &backend);
                        let tag = format!("{kind:?}/{rop:?}/{axes:?}");
                        let variants = backend.lower_variants(&plan).unwrap();
                        assert_eq!(variants.len(), 1, "{label}: {tag}");
                        let v = &variants[0];
                        assert_eq!(v.tag, "blockreduce", "{tag}");
                        assert_eq!(v.fidelity, VariantFidelity::BitIdentical, "{tag}");
                        let kernel = &v.kernels[0];
                        let n_out: i64 = out_shape.iter().product();
                        let nb = n_out as usize * 8;
                        let idx = |b: &[u8]| -> Vec<i64> {
                            b.chunks(8)
                                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                                .collect()
                        };
                        let oracle = evaluate(&plan, &[a, o], std::slice::from_ref(&inp), &[])
                            .remove(0)
                            .bytes;
                        let want = reference(kind, rop == ReduceOp::ArgMax, axes);
                        assert_eq!(idx(&oracle[..nb]), want, "{label}: oracle vs rule, {tag}");
                        for block in BLOCKS {
                            let prog = Program::build(
                                &ctx,
                                &kernel.source,
                                &format!("-D OCF_BLOCK={block}"),
                            )
                            .unwrap_or_else(|e| {
                                panic!(
                                    "{label}: {tag} B={block}: {e}
{}",
                                    kernel.source
                                )
                            });
                            let mut k = prog.kernel(&kernel.name).unwrap();
                            let out = ctx.create_buffer(nb).unwrap();
                            queue.write_buffer(&out, 0, &vec![0xA5u8; nb]).unwrap();
                            k.set_arg_buffer(0, &inb).unwrap();
                            k.set_arg_buffer(1, &out).unwrap();
                            let mut arg = 2u32;
                            let mut push = |k: &mut Kernel, x: i64| {
                                k.set_arg_i64(arg, x).unwrap();
                                arg += 1;
                            };
                            for x in [ROWS, COLS] {
                                push(&mut k, x);
                            }
                            for &x in &inp.strides {
                                push(&mut k, x);
                            }
                            for &x in &out_strides {
                                push(&mut k, x);
                            }
                            push(&mut k, n_out);
                            // SAFETY: one work-group per output element, guarded by
                            // `o < n_out`; every input offset is inside the dense input.
                            unsafe { queue.launch(&k, &[n_out as usize * block], Some(&[block])) }
                                .unwrap()
                                .wait()
                                .unwrap();
                            let mut raw = vec![0u8; nb];
                            queue.read_buffer(&out, 0, &mut raw).unwrap();
                            assert_eq!(
                                idx(&raw),
                                want,
                                "{label}: {tag} B={block}
{}",
                                kernel.source
                            );
                            checks += 1;
                        }
                    }
                }
            }
            ran += 1;
        }
    }
    eprintln!("block arg reductions: {checks} case runs on {ran} device(s)");
}
