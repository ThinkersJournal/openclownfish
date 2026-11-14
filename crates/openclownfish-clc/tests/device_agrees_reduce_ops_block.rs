//! The block-reduce variant of `And`, `Or`, `Xor`, `MaxNum` and `MinNum` on every device
//! against Unpopped's oracle, at three block sizes (the extents here are 5 to 60 elements, so
//! some blocks hold one chunk item and some leave work items with an empty chunk).
//!
//! The bit folds are integer only (a float one is declined by the plan gate and by this
//! emitter). `MaxNum` and `MinNum` skip a NaN element and replace a NaN accumulator, so a
//! row of only NaN stays NaN, where `Max` and `Min` keep a NaN once seen: the float data
//! carries NaNs and one all-NaN row, and `Max` and `Min` run beside them as the control that
//! the two rules really differ. Two NaNs agree whatever their payload.

use openclownfish::{Context, Kernel, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::{Backend, VariantFidelity};
use unpopped::ir::{OpDef, ReduceOp, input};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::build_plan_for;
use unpopped_vocab::{ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, structure_key};

const SHAPE: [i64; 3] = [3, 4, 5];
const BLOCKS: [usize; 3] = [4, 16, 64];

const INTS: [ElementKind; 8] = [
    ElementKind::I8,
    ElementKind::U8,
    ElementKind::I16,
    ElementKind::U16,
    ElementKind::I32,
    ElementKind::U32,
    ElementKind::I64,
    ElementKind::U64,
];

fn width(k: ElementKind) -> usize {
    match k {
        ElementKind::I8 | ElementKind::U8 => 1,
        ElementKind::I16 | ElementKind::U16 | ElementKind::F16 | ElementKind::Bf16 => 2,
        ElementKind::I64 | ElementKind::U64 | ElementKind::F64 => 8,
        _ => 4,
    }
}

fn dense(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

/// Nine values per integer type, as two's-complement patterns widened to `u64`: both
/// extremes, small values, and values that set and clear different bits.
fn int_pool(k: ElementKind) -> [u64; 9] {
    let s = |v: i64| v as u64;
    match k {
        ElementKind::I8 => [127, 1, 0xff, 2, 7, 0xf9, 0x80, 3, 100].map(|v| v as u64),
        ElementKind::U8 => [255, 1, 2, 7, 200, 128, 3, 100, 127].map(|v| v as u64),
        ElementKind::I16 => [32767, 1, -1, 2, 7, -7, -32768, 3, 12345].map(s),
        ElementKind::U16 => [65535, 1, 2, 7, 40000, 32768, 3, 100, 12345].map(|v| v as u64),
        ElementKind::I32 => [
            i64::from(i32::MAX),
            1,
            -1,
            2,
            7,
            -7,
            i64::from(i32::MIN),
            3,
            100_000,
        ]
        .map(s),
        ElementKind::U32 => [
            u64::from(u32::MAX),
            1,
            2,
            7,
            1 << 31,
            3,
            100_000,
            255,
            12345,
        ],
        ElementKind::I64 => [i64::MAX, 1, -1, 2, 7, -7, i64::MIN, 3, 1 << 40].map(s),
        _ => [u64::MAX, 1, 2, 7, 1 << 63, 3, 1 << 40, 255, 12345],
    }
}

fn int_buffer(k: ElementKind) -> TypedBuffer {
    let n: usize = SHAPE.iter().product::<i64>() as usize;
    let p = int_pool(k);
    let w = width(k);
    let bytes: Vec<u8> = (0..n)
        .flat_map(|i| p[(i * 5 + i / 9) % 9].to_le_bytes()[..w].to_vec())
        .collect();
    TypedBuffer::new(k, SHAPE.to_vec(), dense(&SHAPE), bytes)
}

/// 1, 2, 0.5, -1, 0.25, -2, 1.5 and a NaN, in each float format. The first five elements
/// (one whole row of the last axis) and every seventh from index 3 are NaN.
fn float_buffer(k: ElementKind) -> TypedBuffer {
    let n: usize = SHAPE.iter().product::<i64>() as usize;
    let bytes: Vec<u8> = (0..n)
        .flat_map(|i| {
            let nan = i < 5 || i % 7 == 3;
            let c = i % 7;
            match k {
                ElementKind::F32 => {
                    let v = [1.0f32, 2.0, 0.5, -1.0, 0.25, -2.0, 1.5][c];
                    (if nan { f32::NAN } else { v }).to_le_bytes().to_vec()
                }
                ElementKind::F64 => {
                    let v = [1.0f64, 2.0, 0.5, -1.0, 0.25, -2.0, 1.5][c];
                    (if nan { f64::NAN } else { v }).to_le_bytes().to_vec()
                }
                ElementKind::Bf16 => {
                    let v = [0x3f80u16, 0x4000, 0x3f00, 0xbf80, 0x3e80, 0xc000, 0x3fc0][c];
                    (if nan { 0x7fc0u16 } else { v }).to_le_bytes().to_vec()
                }
                _ => {
                    let v = [0x3c00u16, 0x4000, 0x3800, 0xbc00, 0x3400, 0xc000, 0x3e00][c];
                    (if nan { 0x7e00u16 } else { v }).to_le_bytes().to_vec()
                }
            }
        })
        .collect();
    TypedBuffer::new(k, SHAPE.to_vec(), dense(&SHAPE), bytes)
}

fn is_nan(k: ElementKind, e: &[u8]) -> bool {
    match k {
        ElementKind::F32 => f32::from_le_bytes([e[0], e[1], e[2], e[3]]).is_nan(),
        ElementKind::F64 => f64::from_le_bytes(e.try_into().unwrap()).is_nan(),
        ElementKind::Bf16 => u16::from_le_bytes([e[0], e[1]]) & 0x7fff > 0x7f80,
        ElementKind::F16 => u16::from_le_bytes([e[0], e[1]]) & 0x7fff > 0x7c00,
        _ => false,
    }
}

fn agrees(k: ElementKind, got: &[u8], want: &[u8]) -> bool {
    let w = width(k);
    got.chunks(w)
        .zip(want.chunks(w))
        .all(|(g, x)| g == x || (is_nan(k, g) && is_nan(k, x)))
}

struct Case {
    kind: ElementKind,
    op: ReduceOp,
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    for kind in INTS {
        for op in [
            ReduceOp::And,
            ReduceOp::Or,
            ReduceOp::Xor,
            ReduceOp::MaxNum,
            ReduceOp::MinNum,
        ] {
            v.push(Case { kind, op });
        }
    }
    for kind in [
        ElementKind::F32,
        ElementKind::F64,
        ElementKind::Bf16,
        ElementKind::F16,
    ] {
        for op in [
            ReduceOp::MaxNum,
            ReduceOp::MinNum,
            ReduceOp::Max,
            ReduceOp::Min,
        ] {
            v.push(Case { kind, op });
        }
    }
    v
}

#[test]
fn block_bit_and_num_reductions_agree_with_the_oracle_on_every_device() {
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
            for c in cases() {
                if c.kind == ElementKind::F64 && !has_fp64 {
                    continue;
                }
                let w = width(c.kind);
                let inp = if INTS.contains(&c.kind) {
                    int_buffer(c.kind)
                } else {
                    float_buffer(c.kind)
                };
                let inb = ctx.create_buffer(inp.bytes.len()).unwrap();
                queue.write_buffer(&inb, 0, &inp.bytes).unwrap();
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
                    let a = OperandDesc::new(rank, &SHAPE, &inp.strides, c.kind, 256);
                    let o =
                        OperandDesc::new(out_shape.len(), &out_shape, &out_strides, c.kind, 256);
                    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
                    let opdef =
                        OpDef::reduction_axes("r", 1, &[c.kind], input(0), c.op, mask, keepdim);
                    let plan = build_plan_for(&opdef, &key, &backend);
                    let tag = format!("{:?}/{:?}/{axes:?}/kd={keepdim}", c.kind, c.op);
                    let variants = backend.lower_variants(&plan).unwrap();
                    assert_eq!(variants.len(), 1, "{label}: {tag}");
                    let v = &variants[0];
                    assert_eq!(v.tag, "blockreduce", "{tag}");
                    let float_num = !INTS.contains(&c.kind)
                        && matches!(c.op, ReduceOp::MaxNum | ReduceOp::MinNum);
                    assert_eq!(
                        v.fidelity,
                        if float_num {
                            VariantFidelity::DeterministicallyDivergent
                        } else {
                            VariantFidelity::BitIdentical
                        },
                        "{tag}"
                    );
                    let kernel = &v.kernels[0];
                    let n_out: i64 = out_shape.iter().product();
                    let nb = n_out as usize * w;
                    let want = evaluate(&plan, &[a, o], std::slice::from_ref(&inp), &[])
                        .remove(0)
                        .bytes;
                    for block in BLOCKS {
                        let prog =
                            Program::build(&ctx, &kernel.source, &format!("-D OCF_BLOCK={block}"))
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
                        // SAFETY: one work-group per output element, guarded by `o < n_out`;
                        // every input offset is inside the dense input.
                        unsafe { queue.launch(&k, &[n_out as usize * block], Some(&[block])) }
                            .unwrap()
                            .wait()
                            .unwrap();
                        let mut got = vec![0u8; nb];
                        queue.read_buffer(&out, 0, &mut got).unwrap();
                        assert!(
                            agrees(c.kind, &got, &want[..nb]),
                            "{label}: {tag} B={block}: got {got:?}, oracle {:?}
{}",
                            &want[..nb],
                            kernel.source
                        );
                        checks += 1;
                    }
                }
            }
            ran += 1;
        }
    }
    eprintln!("bit and num reductions: {checks} case runs on {ran} device(s)");
}
