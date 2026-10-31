//! OpenCL C source to Unpopped IR: a `Frontend` over the C grammar.

use tree_sitter::{Parser, Tree};
use unpopped::convert::Frontend;

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

/// OpenCL C: `__kernel` functions with `out` / `in{K}` buffer names.
pub const OPENCL_C: Frontend =
    Frontend::new("opencl-c", "__kernel", RESIDUE, "out", "in", parse_opencl_c);
