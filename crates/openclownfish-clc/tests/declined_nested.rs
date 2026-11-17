// SPDX-License-Identifier: MIT OR Apache-2.0
//! End to end: a declined NVIDIA-patent statement nested in `If`, `For` and `While` is refused by
//! name through `Backend::lower_kernel`, not only when it is a bare statement (pm-kernels finding on
//! #60). Needs no OpenCL device: lowering is text. Not a clearance.

use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::exec::LaunchShape;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Stmt, Unroll};
use unpopped_vocab::ElementKind;

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, ElementKind::I32)
}

fn kernel(body: Vec<Stmt>) -> KernelFn {
    let e = ElementKind::I32;
    let mut k = KernelFn::new(
        "declined_nested",
        e,
        vec![Param::buffer("out", e, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(e)],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([4, 1, 1], [1, 1, 1]));
    k
}

fn refused(body: Vec<Stmt>) -> String {
    match OpenClC::new().lower_kernel(&kernel(body)) {
        Ok(_) => panic!("a declined statement lowered"),
        Err(e) => e.to_string(),
    }
}

fn declined() -> Stmt {
    Stmt::WgmmaWait { pending: 0 }
}

fn assert_by_name(msg: &str) {
    assert!(
        msg.contains("WgmmaWait")
            && msg.contains("declined by name")
            && msg.contains("US12204897B2"),
        "{msg}"
    );
}

#[test]
fn bare_statement_is_declined_by_name() {
    // Control: the same error shape at the top level, so the nested cases test recursion only.
    assert_by_name(&refused(vec![declined()]));
}

#[test]
fn nested_in_if_then_and_else_is_declined_by_name() {
    assert_by_name(&refused(vec![Stmt::If {
        cond: ci(1),
        then: Block(vec![declined()]),
        els: Block(vec![]),
    }]));
    assert_by_name(&refused(vec![Stmt::If {
        cond: ci(1),
        then: Block(vec![]),
        els: Block(vec![declined()]),
    }]));
}

#[test]
fn nested_in_for_and_while_is_declined_by_name() {
    assert_by_name(&refused(vec![Stmt::For {
        local: 0,
        start: ci(0),
        end: ci(2),
        step: ci(1),
        unroll: Unroll::None,
        body: Block(vec![declined()]),
    }]));
    assert_by_name(&refused(vec![Stmt::While {
        cond: ci(0),
        body: Block(vec![declined()]),
    }]));
}

#[test]
fn nested_two_deep_is_declined_by_name() {
    assert_by_name(&refused(vec![Stmt::If {
        cond: ci(1),
        then: Block(vec![Stmt::While {
            cond: ci(0),
            body: Block(vec![declined()]),
        }]),
        els: Block(vec![]),
    }]));
}
