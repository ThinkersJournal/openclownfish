//! OpenCL C reductions and scans: lifted into Unpopped's IR, emitted again, and
//! run on every device against the original source kernel run as written.
//!
//! The source kernels are the naive one-work-item accumulator loops the lifter
//! recognises. Data is exactly representable, so a prefix or fold in any order
//! is exact and the two kernels must agree bit for bit.

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::{OPENCL_C, OpenClC};
use unpopped::convert::{lift, lift_reduction, lift_scan};
use unpopped::ir::Access;
use unpopped::{build_plan, generate};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const F32: &[ElementKind] = &[ElementKind::F32];
const N: usize = 13;

struct Case {
    name: &'static str,
    scan: bool,
    /// The body of the loop and the accumulator's initial value.
    init: &'static str,
    step: &'static str,
}

const CASES: [Case; 7] = [
    Case {
        name: "sum",
        scan: false,
        init: "0.0f",
        step: "acc += in0[i];",
    },
    Case {
        name: "prod",
        scan: false,
        init: "1.0f",
        step: "acc *= in0[i];",
    },
    Case {
        name: "max",
        scan: false,
        init: "in0[0]",
        step: "acc = fmax(acc, in0[i]);",
    },
    Case {
        name: "min",
        scan: false,
        init: "in0[0]",
        step: "acc = fmin(acc, in0[i]);",
    },
    Case {
        name: "sum_sq",
        scan: false,
        init: "0.0f",
        step: "acc += in0[i] * in0[i];",
    },
    Case {
        name: "cumsum",
        scan: true,
        init: "0.0f",
        step: "acc += in0[i];",
    },
    Case {
        name: "cummax",
        scan: true,
        init: "in0[0]",
        step: "acc = fmax(acc, in0[i]);",
    },
];

fn source(c: &Case) -> String {
    let store_in = if c.scan { " out[i] = acc;" } else { "" };
    let store_out = if c.scan { "" } else { " out[0] = acc;" };
    format!(
        "__kernel void k(__global const float* in0, __global float* out, long n) {{ \
         float acc = {}; for (long i = 0; i < n; ++i) {{ {}{store_in} }}{store_out} }}",
        c.init, c.step
    )
}

fn data() -> Vec<f32> {
    let cyc = [1.0f32, 2.0, 0.5, -1.0, 0.25, -2.0, 1.5];
    (0..N).map(|i| cyc[i % cyc.len()]).collect()
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

#[test]
fn lifted_reductions_and_scans_match_the_source_kernels_on_every_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let backend = OpenClC::new();
    let input = data();
    let n = N as i64;
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
            let inb = ctx.create_buffer(N * 4).unwrap();
            queue.write_buffer(&inb, 0, &bytes(&input)).unwrap();
            for c in &CASES {
                let src = source(c);
                let lifted = if c.scan {
                    lift_scan(&OPENCL_C, &src, c.name, F32)
                } else {
                    lift_reduction(&OPENCL_C, &src, c.name, F32)
                }
                .unwrap_or_else(|e| panic!("{}: does not lift: {e:?}\n{src}", c.name))
                .op;
                assert!(
                    matches!(
                        (&lifted.access, c.scan),
                        (Access::Scan { .. }, true) | (Access::Reduction { .. }, false)
                    ),
                    "{}: wrong access {:?}",
                    c.name,
                    lifted.access
                );
                // The unknown-class entry point agrees with the explicit one.
                assert_eq!(
                    format!(
                        "{:?}",
                        lift(&OPENCL_C, &src, c.name, F32).unwrap().op.access
                    ),
                    format!("{:?}", lifted.access),
                    "{}",
                    c.name
                );

                // Reference: the source kernel as written, one work item.
                let out_len = if c.scan { N } else { 1 };
                let run_src = {
                    let prog = Program::build(&ctx, &src, "").unwrap();
                    let mut k = prog.kernel("k").unwrap();
                    let out = ctx.create_buffer(out_len * 4).unwrap();
                    queue
                        .write_buffer(&out, 0, &vec![0xA5u8; out_len * 4])
                        .unwrap();
                    k.set_arg_buffer(0, &inb).unwrap();
                    k.set_arg_buffer(1, &out).unwrap();
                    k.set_arg_i64(2, n).unwrap();
                    // SAFETY: one work item reads `n` elements of `in0` and writes
                    // `out_len` of `out`, both sized for that.
                    unsafe { queue.launch(&k, &[1], None) }
                        .unwrap()
                        .wait()
                        .unwrap();
                    let mut raw = vec![0u8; out_len * 4];
                    queue.read_buffer(&out, 0, &mut raw).unwrap();
                    raw
                };

                // Emitted from the lifted op.
                let a = OperandDesc::new(1, &[n], &[1], ElementKind::F32, 256);
                let (cat, o) = if c.scan {
                    (OpCategory::Scan, a)
                } else {
                    (
                        OpCategory::Reduction,
                        OperandDesc::new(1, &[1], &[1], ElementKind::F32, 256),
                    )
                };
                let key = structure_key(cat, &[a, o], ArchSku::Sm89);
                let _ = build_plan(&lifted, &key);
                let kernel = generate(&lifted, &key, &backend);
                let prog = Program::build(&ctx, &kernel.source, "")
                    .unwrap_or_else(|e| panic!("{label}: {}: {e}\n{}", c.name, kernel.source));
                let mut k = prog.kernel(&kernel.name).unwrap();
                let out = ctx.create_buffer(out_len * 4).unwrap();
                queue
                    .write_buffer(&out, 0, &vec![0x5Au8; out_len * 4])
                    .unwrap();
                k.set_arg_buffer(0, &inb).unwrap();
                k.set_arg_buffer(1, &out).unwrap();
                if c.scan {
                    k.set_arg_i64(2, 1).unwrap(); // rows
                    k.set_arg_i64(3, n).unwrap(); // row length
                } else {
                    k.set_arg_i64(2, n).unwrap(); // shape0
                    k.set_arg_i64(3, 1).unwrap(); // s0_0
                    k.set_arg_i64(4, 1).unwrap(); // n_out
                }
                // SAFETY: one work item for the single row/output; it reads `n`
                // elements of `in0` and writes at most `out_len` of `out`.
                unsafe { queue.launch(&k, &[1], None) }
                    .unwrap()
                    .wait()
                    .unwrap();
                let mut got = vec![0u8; out_len * 4];
                queue.read_buffer(&out, 0, &mut got).unwrap();
                assert_eq!(got, run_src, "{label}: {}\n{}", c.name, kernel.source);
            }
            eprintln!(
                "{label}: {} lifted kernels agree with their source",
                CASES.len()
            );
            ran += 1;
        }
    }
    eprintln!("lifted reductions ran on {ran} device(s)");
}
