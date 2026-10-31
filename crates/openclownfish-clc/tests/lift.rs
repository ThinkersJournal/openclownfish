//! OpenCL C source lifted into Unpopped's IR, and round-tripped to the emitter.

use openclownfish_clc::{OPENCL_C, OpenClC};
use unpopped::convert::lift_elementwise;
use unpopped::generate;
use unpopped::ir::{OpDef, UnaryOp, input, konst};
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
        generate(&lifted, &key, &OpenClC).source,
        generate(&built, &key, &OpenClC).source
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

/// Pins the current limit so it is visible, not silent: Unpopped's walker maps
/// CUDA spellings (`sqrtf`) and has no per-frontend table, so OpenCL's
/// overloaded `sqrt` is a typed refusal until Unpopped adds one.
#[test]
fn overloaded_math_names_are_refused_until_unpopped_maps_them() {
    let err = lift(
        "__kernel void k(__global const float* in0, __global float* out) { \
         int i = get_global_id(0); out[i] = sqrt(in0[i]); }",
    )
    .unwrap_err();
    assert!(matches!(err, LiftError::Unrecognized(_)), "{err:?}");
}
