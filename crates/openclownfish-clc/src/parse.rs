//! OpenCL C source to Unpopped IR: a `Frontend` over the C grammar.

use tree_sitter::{Parser, Tree};
use unpopped::convert::Frontend;
use unpopped::ir::{BinaryOp, UnaryOp};

/// Parse OpenCL C into a tree-sitter CST with the C++ grammar. OpenCL C is C99
/// plus qualifiers (`__kernel`, `__global`), which the grammar tolerates as
/// unknown identifiers; the walker only needs the store and its expression.
pub fn parse_opencl_c(src: &str) -> Option<Tree> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_cpp::LANGUAGE.into())
        .ok()?;
    parser.parse(src, None)
}

/// Constructs the IR cannot express; refused before parsing.
const RESIDUE: &[&str] = &[
    "barrier",
    "mem_fence",
    "read_mem_fence",
    "write_mem_fence",
    "atomic_",
    "atom_",
    "work_group_",
    "sub_group_",
    "async_work_group_copy",
    "async_work_group_strided_copy",
    "wait_group_events",
    "prefetch",
    "__local",
    " local ",
    "image2d_t",
    "image3d_t",
    "sampler_t",
    " pipe ",
    "enqueue_kernel",
    "printf",
    "get_local_id",
    "get_group_id",
    "get_local_size",
    "get_num_groups",
];

/// OpenCL C's overloaded math built-ins. There is no `expf`: `exp` takes a
/// `float` or a `double` by overload, and the IR dtype comes from the plan.
///
/// Deliberately absent: `native_*` and `half_*` (reduced accuracy, which the IR
/// can only carry as a qualifier), and `abs`, which is integer-only in OpenCL C.
fn unary_fn(name: &str) -> Option<UnaryOp> {
    Some(match name {
        "exp" => UnaryOp::Exp,
        "exp2" => UnaryOp::Exp2,
        "log" => UnaryOp::Log,
        "log2" => UnaryOp::Log2,
        "sqrt" => UnaryOp::Sqrt,
        "rsqrt" => UnaryOp::Rsqrt,
        "fabs" => UnaryOp::Abs,
        "tanh" => UnaryOp::Tanh,
        "sin" => UnaryOp::Sin,
        "cos" => UnaryOp::Cos,
        "floor" => UnaryOp::Floor,
        "ceil" => UnaryOp::Ceil,
        "trunc" => UnaryOp::Trunc,
        _ => return None,
    })
}

/// `fmax`/`fmin` are IEEE maxNum/minNum (NaN-suppressing), so they map to the
/// IR's `FmaxIeee`/`FminIeee`, never to the NaN-propagating `Max`/`Min`. The
/// gentype `max`/`min` are not mapped: their NaN behaviour is not specified.
fn binary_fn(name: &str) -> Option<BinaryOp> {
    Some(match name {
        "fmax" => BinaryOp::FmaxIeee,
        "fmin" => BinaryOp::FminIeee,
        "pow" => BinaryOp::Pow,
        "atan2" => BinaryOp::Atan2,
        "copysign" => BinaryOp::Copysign,
        _ => return None,
    })
}

/// OpenCL C: `__kernel` functions with `out` / `in{K}` buffer names and OpenCL's
/// own math-call tables.
pub const OPENCL_C: Frontend =
    Frontend::new("opencl-c", "__kernel", RESIDUE, "out", "in", parse_opencl_c)
        .with_intrinsics(unary_fn, binary_fn);
