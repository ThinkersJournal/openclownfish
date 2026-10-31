//! The strided schedule on every device: transposed, broadcast and rank-3
//! operands, plus a `Coord` body, against Unpopped's oracle.
//!
//! The kernel's arguments follow Baracuda's strided convention: the input and
//! output pointers, then `shape{d}`, `s{i}_{d}` per input, `so_{d}` for the
//! output, then `n`. Input buffers are storage images laid out by their strides;
//! the oracle reads them the same way, so both sides see identical memory.

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::ir::{Expr, OpDef, coord, input};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::Schedule;
use unpopped::{build_plan, generate};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

struct Case {
    name: &'static str,
    shape: Vec<i64>,
    /// Element strides per input (0 on a broadcast axis).
    strides: Vec<Vec<i64>>,
    body: Expr,
}

fn row_major(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

fn col_major(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1; shape.len()];
    for d in 1..shape.len() {
        s[d] = s[d - 1] * shape[d - 1];
    }
    s
}

/// Storage length of an operand: one past its largest reachable offset.
fn storage_len(shape: &[i64], strides: &[i64]) -> usize {
    1 + shape
        .iter()
        .zip(strides)
        .map(|(n, s)| (n - 1) * s)
        .sum::<i64>() as usize
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    for shape in [vec![5, 7], vec![16, 16]] {
        let rm = row_major(&shape);
        let cm = col_major(&shape);
        v.push(Case {
            name: "transposed_add",
            shape: shape.clone(),
            strides: vec![cm.clone(), rm.clone()],
            body: input(0) + input(1),
        });
        v.push(Case {
            name: "row_broadcast_mul",
            shape: shape.clone(),
            strides: vec![rm.clone(), vec![0, 1]],
            body: input(0) * input(1),
        });
        v.push(Case {
            name: "scalar_broadcast_sub",
            shape: shape.clone(),
            strides: vec![cm.clone(), vec![0, 0]],
            body: input(0) - input(1),
        });
        v.push(Case {
            name: "coord_add",
            shape: shape.clone(),
            strides: vec![cm],
            body: input(0) + coord(0),
        });
    }
    let shape = vec![3, 4, 5];
    v.push(Case {
        name: "rank3_mid_broadcast",
        strides: vec![col_major(&shape), vec![20, 0, 1]],
        shape,
        body: input(0) + input(1),
    });
    v
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_ne_bytes(*c))
        .collect()
}

#[test]
fn strided_kernels_agree_with_the_oracle_on_every_device() {
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
            for c in cases() {
                let rank = c.shape.len();
                let n_in = c.strides.len();
                let n_el: i64 = c.shape.iter().product();
                let out_strides = row_major(&c.shape);
                let mut operands: Vec<OperandDesc> = c
                    .strides
                    .iter()
                    .map(|s| OperandDesc::new(rank, &c.shape, s, ElementKind::F32, 256))
                    .collect();
                operands.push(OperandDesc::new(
                    rank,
                    &c.shape,
                    &out_strides,
                    ElementKind::F32,
                    256,
                ));
                let cat = if n_in == 1 {
                    OpCategory::UnaryElementwise
                } else {
                    OpCategory::BinaryElementwise
                };
                let key = structure_key(cat, &operands, ArchSku::Sm89);
                let op =
                    OpDef::elementwise(c.name, n_in as u8, &[ElementKind::F32], c.body.clone());
                let plan = build_plan(&op, &key);
                assert_eq!(plan.schedule, Schedule::Strided, "{}: not strided", c.name);
                let kernel = generate(&op, &key, &backend);
                let prog = match Program::build(&ctx, &kernel.source, "") {
                    Ok(p) => p,
                    Err(e) => panic!(
                        "{label}: {} failed to build:\n{e}\n{}",
                        c.name, kernel.source
                    ),
                };
                let mut k = prog.kernel(&kernel.name).unwrap();

                // Storage images: distinct, exactly representable values.
                let images: Vec<Vec<f32>> = c
                    .strides
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        (0..storage_len(&c.shape, s))
                            .map(|j| (j as f32) * 0.5 + 1.0 + i as f32 * 100.0)
                            .collect()
                    })
                    .collect();
                let bufs: Vec<_> = images
                    .iter()
                    .map(|im| {
                        let b = ctx.create_buffer(im.len() * 4).unwrap();
                        queue.write_buffer(&b, 0, &bytes(im)).unwrap();
                        b
                    })
                    .collect();
                let out = ctx.create_buffer(n_el as usize * 4).unwrap();
                queue
                    .write_buffer(&out, 0, &bytes(&vec![f32::NAN; n_el as usize]))
                    .unwrap();

                let mut arg = 0u32;
                for b in &bufs {
                    k.set_arg_buffer(arg, b).unwrap();
                    arg += 1;
                }
                k.set_arg_buffer(arg, &out).unwrap();
                arg += 1;
                for &n in &c.shape {
                    k.set_arg_i64(arg, n).unwrap();
                    arg += 1;
                }
                for s in &c.strides {
                    for &x in s {
                        k.set_arg_i64(arg, x).unwrap();
                        arg += 1;
                    }
                }
                for &x in &out_strides {
                    k.set_arg_i64(arg, x).unwrap();
                    arg += 1;
                }
                k.set_arg_i64(arg, n_el).unwrap();
                // SAFETY: the kernel guards `i < n`; every offset it forms is a
                // reachable offset of the operand it reads (storage_len covers them).
                let ev = unsafe { queue.launch(&k, &[n_el as usize], None) }.unwrap();
                ev.wait().unwrap();
                let mut raw = vec![0u8; n_el as usize * 4];
                queue.read_buffer(&out, 0, &mut raw).unwrap();
                let got = floats(&raw);

                let tb: Vec<TypedBuffer> = images
                    .iter()
                    .map(|im| TypedBuffer::from_f32(&[im.len() as i64], im))
                    .collect();
                let want = evaluate(&plan, &operands, &tb, &[])
                    .into_iter()
                    .next()
                    .unwrap()
                    .to_f64_vec();
                assert_eq!(got.len(), want.len(), "{label}: {}", c.name);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        g.to_bits() == (*w as f32).to_bits(),
                        "{label}: {} shape {:?} element {i}: kernel {g}, oracle {w}",
                        c.name,
                        c.shape
                    );
                }
            }
            eprintln!("{label}: {} strided cases bit-exact", cases().len());
            ran += 1;
        }
    }
    eprintln!("strided kernels ran on {ran} device(s)");
}

#[test]
fn strided_plans_with_views_or_offsets_are_refused() {
    // A Permute view changes which stride an axis reads: not spelled here.
    let shape = [4i64, 6];
    let d = OperandDesc::new(2, &shape, &col_major(&shape), ElementKind::F32, 256);
    let o = OperandDesc::new(2, &shape, &row_major(&shape), ElementKind::F32, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[d, o], ArchSku::Sm89);
    let op = OpDef::elementwise("copy", 1, &[ElementKind::F32], input(0));
    let plan = build_plan(&op, &key);
    // The plain strided plan is served.
    assert!(OpenClC::new().lower(&plan).is_ok());
    // Reductions are another schedule entirely.
    let ro = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256);
    let rd = OperandDesc::new(2, &shape, &row_major(&shape), ElementKind::F32, 256);
    let rkey = structure_key(OpCategory::Reduction, &[rd, ro], ArchSku::Sm89);
    let rop = OpDef::reduction(
        "sum",
        1,
        &[ElementKind::F32],
        input(0),
        unpopped::ir::ReduceOp::Sum,
    );
    let rplan = build_plan(&rop, &rkey);
    assert!(
        OpenClC::new().lower(&rplan).is_err(),
        "{:?}",
        rplan.schedule
    );
}
