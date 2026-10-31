//! OpenCL C source lifted into Unpopped's IR, and round-tripped to the emitter.

use openclownfish_clc::{OPENCL_C, OpenClC};
use unpopped::convert::lift_elementwise;
use unpopped::generate;
use unpopped::ir::{BinaryOp, OpDef, UnaryOp, input, konst};
use unpopped::lift::LiftError;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const F32: &[ElementKind] = &[ElementKind::F32];

fn lift(src: &str) -> Result<OpDef, LiftError> {
    lift_elementwise(&OPENCL_C, src, "lifted", F32).map(|l| l.op)
}

#[test]
fn lifts_a_binary_kernel_with_a_local_index() {
    let op = lift(
        "__kernel void add(__global const float* in0, __global const float* in1, \
         __global float* out) { size_t i = get_global_id(0); out[i] = in0[i] + in1[i]; }",
    )
    .unwrap();
    assert_eq!(op.body, (input(0) + input(1)).0);
}

#[test]
fn lifts_a_guarded_kernel_with_float_literals() {
    let op = lift(
        "__kernel void k(__global const float* in0, __global float* out, long n) { \
         long i = (long)get_global_id(0); if (i < n) { out[i] = -in0[i] * 2.5f; } }",
    )
    .unwrap();
    assert_eq!(
        op.body,
        (input(0).unary(UnaryOp::Neg) * konst(2.5)).0,
        "the 2.5f suffix must not change the value"
    );
}

#[test]
fn lifts_fabs_and_fmax() {
    let op = lift(
        "__kernel void k(__global const float* in0, __global const float* in1, \
         __global float* out) { int i = get_global_id(0); out[i] = fmax(fabs(in0[i]), in1[i]); }",
    )
    .unwrap();
    assert!(
        format!("{:?}", op.body).contains("FmaxIeee"),
        "{:?}",
        op.body
    );
}

/// Source -> IR -> OpenCL C: the emitted kernel equals the one from a
/// hand-built op, so the two halves of the backend agree.
#[test]
fn lifted_op_emits_the_same_kernel_as_the_hand_built_one() {
    let lifted = lift(
        "__kernel void k(__global const float* in0, __global const float* in1, \
         __global float* out) { size_t i = get_global_id(0); out[i] = in0[i] * in1[i]; }",
    )
    .unwrap();
    let built = OpDef::elementwise("lifted", 2, F32, input(0) * input(1));
    let d = OperandDesc::new(1, &[8], &[1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);
    assert_eq!(
        generate(&lifted, &key, &OpenClC::new()).source,
        generate(&built, &key, &OpenClC::new()).source
    );
}

#[test]
fn work_group_constructs_are_refused_as_inexpressible() {
    for src in [
        "__kernel void k(__global float* out) { barrier(CLK_LOCAL_MEM_FENCE); out[0] = 1.0f; }",
        "__kernel void k(__local float* t, __global float* out) { out[0] = t[0]; }",
        "__kernel void k(__global float* out) { int i = get_local_id(0); out[i] = 1.0f; }",
        "__kernel void k(__global int* out) { atomic_add(out, 1); }",
    ] {
        assert!(
            matches!(lift(src), Err(LiftError::Inexpressible(_))),
            "{src}: {:?}",
            lift(src)
        );
    }
}

#[test]
fn not_a_kernel_and_not_elementwise_are_distinct_refusals() {
    assert!(matches!(
        lift("void f(float* out) { out[0] = 1.0f; }"),
        Err(LiftError::NotAKernel)
    ));
    assert!(matches!(
        lift("__kernel void k(__global float* out) { out[0] = 1.0f; }"),
        Err(LiftError::NotElementwise)
    ));
}

fn unary_kernel(call: &str) -> String {
    format!(
        "__kernel void k(__global const float* in0, __global float* out) {{ \
         int i = get_global_id(0); out[i] = {call}(in0[i]); }}"
    )
}

/// OpenCL's overloaded math names lift to the matching IR operators (Unpopped
/// 0.35.3 per-frontend tables).
#[test]
fn overloaded_math_names_lift_to_their_operators() {
    for (call, op) in [
        ("sqrt", UnaryOp::Sqrt),
        ("exp", UnaryOp::Exp),
        ("log", UnaryOp::Log),
        ("rsqrt", UnaryOp::Rsqrt),
        ("fabs", UnaryOp::Abs),
        ("tanh", UnaryOp::Tanh),
    ] {
        let got = lift(&unary_kernel(call)).unwrap();
        assert_eq!(got.body, input(0).unary(op).0, "{call}");
    }
}

#[test]
fn pow_lifts_and_fmax_stays_nan_suppressing() {
    let src = "__kernel void k(__global const float* in0, __global const float* in1, \
               __global float* out) { int i = get_global_id(0); out[i] = pow(in0[i], in1[i]); }";
    let got = lift(src).unwrap();
    assert_eq!(got.body, input(0).binary(BinaryOp::Pow, input(1)).0);
    let src = src.replace("pow(", "fmax(");
    let got = lift(&src).unwrap();
    assert_eq!(got.body, input(0).binary(BinaryOp::FmaxIeee, input(1)).0);
}

/// Reduced-accuracy built-ins, CUDA spellings and the integer `abs` are not
/// OpenCL float math: each stays a typed refusal rather than a guess.
#[test]
fn approximate_and_foreign_spellings_are_refused() {
    for call in ["native_exp", "half_sqrt", "expf", "sqrtf", "abs"] {
        let err = lift(&unary_kernel(call)).unwrap_err();
        assert!(matches!(err, LiftError::Unrecognized(_)), "{call}: {err:?}");
    }
}

/// Source -> IR -> OpenCL C for a lifted `sqrt`: the same kernel as the
/// hand-built op, so lifting and emitting agree on the operator.
#[test]
fn lifted_sqrt_emits_the_same_kernel_as_the_hand_built_one() {
    let lifted = lift(&unary_kernel("sqrt")).unwrap();
    let built = OpDef::elementwise("lifted", 1, F32, input(0).unary(UnaryOp::Sqrt));
    let d = OperandDesc::new(1, &[8], &[1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[d, d], ArchSku::Sm89);
    assert_eq!(
        generate(&lifted, &key, &OpenClC::new()).source,
        generate(&built, &key, &OpenClC::new()).source
    );
}
