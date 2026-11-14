//! The native `f16` path (`cl_khr_fp16`, `OpenClC::with_native_half`) on every device that
//! lists the extension, against both Unpopped's oracle and the integer codec on the same device.
//!
//! The native path only swaps how a kernel decodes and encodes a stored pattern (`as_half` and
//! a `half` conversion instead of integer arithmetic); storage is still a `ushort` and the
//! arithmetic still runs in `float`, rounded once to nearest even at the store. So every one of
//! the 65536 patterns, sent through scales that round subnormals, overflow, and through `x * y`
//! and `x + y` over a bijective pairing of all patterns (so ordinary values round inexactly),
//! must give the oracle's bits, and the codec's, except that two NaNs agree whatever their
//! payload. A device that does not list the extension is skipped, not failed: that is the point
//! of the gate.

use openclownfish::{Context, Kernel, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::ir::{OpDef, ReduceOp, input, konst};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::{build_plan, generate};
use unpopped_vocab::AxisMask;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const KIND: ElementKind = ElementKind::F16;

fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bits(b: &[u8]) -> Vec<u16> {
    b.chunks(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn is_nan(x: u16) -> bool {
    x & 0x7fff > 0x7c00
}

fn agree(a: u16, b: u16) -> bool {
    a == b || (is_nan(a) && is_nan(b))
}

fn typed(v: &[u16]) -> TypedBuffer {
    TypedBuffer::new(KIND, vec![v.len() as i64], vec![1], bytes(v))
}

/// Runs `op` over `inputs` on one device and returns the stored bits.
fn run(
    ctx: &Context,
    queue: &openclownfish::Queue,
    backend: &OpenClC,
    op: &OpDef,
    inputs: &[Vec<u16>],
    label: &str,
) -> (Vec<u16>, String) {
    let n = inputs[0].len();
    let desc = OperandDesc::new(1, &[n as i64], &[1], KIND, 256);
    let descs: Vec<OperandDesc> = (0..=inputs.len()).map(|_| desc).collect();
    let cat = if inputs.len() == 1 {
        OpCategory::UnaryElementwise
    } else {
        OpCategory::BinaryElementwise
    };
    let key = structure_key(cat, &descs, ArchSku::Sm89);
    let kernel = generate(op, &key, backend);
    let prog = Program::build(ctx, &kernel.source, "")
        .unwrap_or_else(|e| panic!("{label}: failed to build:\n{e}\n{}", kernel.source));
    let mut k = prog.kernel(&kernel.name).unwrap();
    // The buffers must outlive the launch.
    let mut held = Vec::new();
    for (i, v) in inputs.iter().enumerate() {
        let b = ctx.create_buffer(n * 2).unwrap();
        queue.write_buffer(&b, 0, &bytes(v)).unwrap();
        k.set_arg_buffer(i as u32, &b).unwrap();
        held.push(b);
    }
    let out = ctx.create_buffer(n * 2).unwrap();
    queue.write_buffer(&out, 0, &vec![0xA5u8; n * 2]).unwrap();
    k.set_arg_buffer(inputs.len() as u32, &out).unwrap();
    k.set_arg_i64(inputs.len() as u32 + 1, n as i64).unwrap();
    // SAFETY: the kernel guards `i < n`; every buffer holds n elements.
    unsafe { queue.launch(&k, &[n], None) }
        .unwrap()
        .wait()
        .unwrap();
    let mut raw = vec![0u8; n * 2];
    queue.read_buffer(&out, 0, &mut raw).unwrap();
    (bits(&raw), kernel.source)
}

fn oracle(op: &OpDef, inputs: &[Vec<u16>]) -> Vec<u16> {
    let n = inputs[0].len();
    let desc = OperandDesc::new(1, &[n as i64], &[1], KIND, 256);
    let descs: Vec<OperandDesc> = (0..=inputs.len()).map(|_| desc).collect();
    let cat = if inputs.len() == 1 {
        OpCategory::UnaryElementwise
    } else {
        OpCategory::BinaryElementwise
    };
    let key = structure_key(cat, &descs, ArchSku::Sm89);
    let plan = build_plan(op, &key);
    let ins: Vec<TypedBuffer> = inputs.iter().map(|v| typed(v)).collect();
    bits(&evaluate(&plan, &descs, &ins, &[]).remove(0).raw_bytes()[..n * 2])
}

#[test]
fn native_half_agrees_with_the_oracle_and_the_codec_on_every_device_that_lists_it() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let all: Vec<u16> = (0..=u16::MAX).collect();
    // 7919 is odd, so this is a bijection of the patterns: every pattern appears once on each side.
    let paired: Vec<u16> = (0..65536u32)
        .map(|i| ((i * 7919 + 13) & 0xffff) as u16)
        .collect();
    let cases: Vec<(String, OpDef, Vec<Vec<u16>>)> = vec![
        (
            "x*1".into(),
            OpDef::elementwise("s", 1, &[KIND], input(0) * konst(1.0)),
            vec![all.clone()],
        ),
        (
            "x*0.5".into(),
            OpDef::elementwise("s", 1, &[KIND], input(0) * konst(0.5)),
            vec![all.clone()],
        ),
        (
            "x*0.25".into(),
            OpDef::elementwise("s", 1, &[KIND], input(0) * konst(0.25)),
            vec![all.clone()],
        ),
        (
            "x*2".into(),
            OpDef::elementwise("s", 1, &[KIND], input(0) * konst(2.0)),
            vec![all.clone()],
        ),
        (
            "x*y".into(),
            OpDef::elementwise("m", 2, &[KIND], input(0) * input(1)),
            vec![all.clone(), paired.clone()],
        ),
        (
            "x+y".into(),
            OpDef::elementwise("a", 2, &[KIND], input(0) + input(1)),
            vec![all.clone(), paired.clone()],
        ),
    ];
    let native = OpenClC::new().with_native_half(true);
    let codec = OpenClC::new();
    let (mut checked, mut skipped, mut ran) = (0, 0, 0);
    for p in platforms {
        let Ok(devices) = p.devices() else { continue };
        for d in devices {
            let label = format!(
                "{} / {}",
                p.name().unwrap_or_default(),
                d.name().unwrap_or_default()
            );
            let Ok(exts) = d.extensions() else { continue };
            let Ok(ctx) = Context::new(&d) else { continue };
            let queue = ctx.create_queue().unwrap();
            ran += 1;
            if !OpenClC::lists_native_half(exts.iter().map(String::as_str)) {
                skipped += 1;
                // The gate's other side: the codec path builds and agrees here without it.
                let (name, op, ins) = &cases[4];
                let (got, _) = run(&ctx, &queue, &codec, op, ins, &format!("{label}: {name}"));
                let want = oracle(op, ins);
                assert!(
                    got.iter().zip(&want).all(|(a, b)| agree(*a, *b)),
                    "{label}: codec {name}"
                );
                continue;
            }
            for (name, op, ins) in &cases {
                let tag = format!("{label}: {name}");
                let (got, src) = run(&ctx, &queue, &native, op, ins, &tag);
                assert!(src.contains("cl_khr_fp16"), "{tag}: no pragma\n{src}");
                assert!(src.contains("as_half"), "{tag}: no native decode\n{src}");
                let (old, _) = run(&ctx, &queue, &codec, op, ins, &tag);
                let want = oracle(op, ins);
                let n = want.len();
                let bad_oracle: Vec<usize> = (0..n).filter(|&i| !agree(got[i], want[i])).collect();
                assert!(
                    bad_oracle.is_empty(),
                    "{tag}: {} of {n} differ from the oracle, first at {}: inputs {:?}, native {:#06x}, oracle {:#06x}",
                    bad_oracle.len(),
                    bad_oracle[0],
                    ins.iter().map(|v| v[bad_oracle[0]]).collect::<Vec<_>>(),
                    got[bad_oracle[0]],
                    want[bad_oracle[0]]
                );
                let bad_codec: Vec<usize> = (0..n).filter(|&i| !agree(got[i], old[i])).collect();
                assert!(
                    bad_codec.is_empty(),
                    "{tag}: {} of {n} differ from the codec, first at {}: native {:#06x}, codec {:#06x}",
                    bad_codec.len(),
                    bad_codec[0],
                    got[bad_codec[0]],
                    old[bad_codec[0]]
                );
                checked += 1;
            }
        }
    }
    eprintln!("native half: {checked} sweeps agree; {skipped} of {ran} device(s) lack cl_khr_fp16");
}

#[test]
fn native_half_is_opt_in_and_only_for_f16() {
    let key_for = |kind| {
        let d = OperandDesc::new(1, &[8], &[1], kind, 256);
        structure_key(OpCategory::UnaryElementwise, &[d, d], ArchSku::Sm89)
    };
    let gen_src = |b: &OpenClC, kind| {
        let op = OpDef::elementwise("s", 1, &[kind], input(0) * konst(1.0));
        generate(&op, &key_for(kind), b).source
    };
    // The default backend, and one built from the extension list, use the integer codec.
    let exts = ["cl_khr_fp16", "cl_khr_fp64"];
    for b in [OpenClC::new(), OpenClC::for_extensions(exts)] {
        let s = gen_src(&b, ElementKind::F16);
        assert!(!s.contains("cl_khr_fp16") && !s.contains("as_half"), "{s}");
    }
    assert!(OpenClC::lists_native_half(exts));
    assert!(!OpenClC::lists_native_half(["cl_khr_fp64"]));
    let native = OpenClC::new().with_native_half(true);
    let s = gen_src(&native, ElementKind::F16);
    assert!(s.contains("#pragma OPENCL EXTENSION cl_khr_fp16 : enable"));
    assert!(s.contains("(float)as_half(") && s.contains("as_ushort((half)"));
    // bf16 has no native type: it keeps its codec and asks for no extension.
    let b = gen_src(&native, ElementKind::Bf16);
    assert!(!b.contains("cl_khr_fp16") && !b.contains("as_half"), "{b}");
    // Storage is unchanged: a `ushort` pattern either way.
    assert!(
        s.contains("ushort* out") && s.contains("const ushort* in0"),
        "{s}"
    );
}

/// Quarters, halves, ones and twos as f16 patterns, so every sum is exact in `float`.
fn exact_cycle() -> [u16; 7] {
    [0x3c00, 0x4000, 0x3800, 0xbc00, 0x3400, 0xc000, 0x3e00]
}

#[test]
fn native_half_reductions_and_their_block_variant_build_and_agree_on_a_device_that_lists_it() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let shape = [4i64, 16];
    let n_in = 64usize;
    let data: Vec<u16> = (0..n_in).map(|i| exact_cycle()[i % 7]).collect();
    let inp = TypedBuffer::new(KIND, shape.to_vec(), vec![16, 1], bytes(&data));
    let a = OperandDesc::new(2, &shape, &[16, 1], KIND, 256);
    let o = OperandDesc::new(1, &[4], &[1], KIND, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    let mut mask = AxisMask::EMPTY;
    mask.set(1);
    let native = OpenClC::new().with_native_half(true);
    let (mut done, mut ran) = (0, 0);
    for p in platforms {
        let Ok(devices) = p.devices() else { continue };
        for d in devices {
            let Ok(exts) = d.extensions() else { continue };
            if !OpenClC::lists_native_half(exts.iter().map(String::as_str)) {
                continue;
            }
            let label = format!(
                "{} / {}",
                p.name().unwrap_or_default(),
                d.name().unwrap_or_default()
            );
            let Ok(ctx) = Context::new(&d) else { continue };
            let queue = ctx.create_queue().unwrap();
            ran += 1;
            let inb = ctx.create_buffer(n_in * 2).unwrap();
            queue.write_buffer(&inb, 0, &bytes(&data)).unwrap();
            for rop in [ReduceOp::Sum, ReduceOp::Max] {
                let op = OpDef::reduction_axes("r", 1, &[KIND], input(0), rop, mask, false);
                let plan = build_plan(&op, &key);
                let want = bits(
                    &evaluate(&plan, &[a, o], std::slice::from_ref(&inp), &[])
                        .remove(0)
                        .raw_bytes()[..8],
                );
                let serial = generate(&op, &key, &native);
                let variants = native.lower_variants(&plan).unwrap();
                assert_eq!(variants.len(), 1, "{label}: {rop:?}");
                let block = &variants[0].kernels[0];
                for (kernel, group, opts) in [
                    (&serial, None, String::new()),
                    (block, Some(16usize), "-D OCF_BLOCK=16".to_owned()),
                ] {
                    let tag = format!(
                        "{label}: {rop:?} {}",
                        if group.is_some() { "block" } else { "serial" }
                    );
                    assert!(
                        kernel.source.contains("cl_khr_fp16"),
                        "{tag}
{}",
                        kernel.source
                    );
                    let prog = Program::build(&ctx, &kernel.source, &opts).unwrap_or_else(|e| {
                        panic!(
                            "{tag}: {e}
{}",
                            kernel.source
                        )
                    });
                    let mut k = prog.kernel(&kernel.name).unwrap();
                    let out = ctx.create_buffer(8).unwrap();
                    queue.write_buffer(&out, 0, &[0xA5u8; 8]).unwrap();
                    k.set_arg_buffer(0, &inb).unwrap();
                    k.set_arg_buffer(1, &out).unwrap();
                    let mut arg = 2u32;
                    let mut push = |k: &mut Kernel, x: i64| {
                        k.set_arg_i64(arg, x).unwrap();
                        arg += 1;
                    };
                    for x in [4i64, 16, 16, 1, 1] {
                        push(&mut k, x);
                    }
                    push(&mut k, 4);
                    // SAFETY: one work item (or group) per output element, guarded by `o < n_out`.
                    match group {
                        None => unsafe { queue.launch(&k, &[4], None) },
                        Some(b) => unsafe { queue.launch(&k, &[4 * b], Some(&[b])) },
                    }
                    .unwrap()
                    .wait()
                    .unwrap();
                    let mut raw = vec![0u8; 8];
                    queue.read_buffer(&out, 0, &mut raw).unwrap();
                    assert_eq!(
                        bits(&raw),
                        want,
                        "{tag}
{}",
                        kernel.source
                    );
                    done += 1;
                }
            }
        }
    }
    eprintln!("native half reductions: {done} kernels agree on {ran} device(s)");
}
