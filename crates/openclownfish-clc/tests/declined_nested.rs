// SPDX-License-Identifier: MIT OR Apache-2.0
//! End to end: a declined NVIDIA-patent statement nested in `If`, `For` and `While` is refused by
//! name through `Backend::lower_kernel`, not only when it is a bare statement (pm-kernels finding on
//! #60). Needs no OpenCL device: lowering is text. Not a clearance.

use openclownfish_clc::OpenClC;
use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ReduceOp;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, Stmt, Unroll,
};
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
        vec![LocalDecl::scalar(e), LocalDecl::scalar(e)],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([4, 1, 1], [1, 1, 1]));
    // A `Shared` allocation so a valid `CopyAsync` destination exists (the checker runs first).
    k.allocs = vec![AllocDecl::new(
        "tile",
        MemorySpace::Shared,
        e,
        Extent::Static(8),
    )];
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

/// Puts a statement into a kernel body at some depth.
type Wrap = Box<dyn Fn(Stmt) -> Vec<Stmt>>;

/// Every board 434 C statement (CireSnave 2026-11-13) with the name and patent its refusal must carry.
fn board_434c() -> Vec<(&'static str, &'static str, Stmt)> {
    let at = || Place::Logical {
        buffer: 0,
        coords: vec![ci(0)],
    };
    vec![
        (
            "CopyAsync",
            "US12118382B2",
            Stmt::CopyAsync {
                dst: Place::Alloc {
                    alloc: 0,
                    index: ci(0),
                },
                src: at(),
                elems: 1,
                valid: None,
                cache: Default::default(),
                l2_prefetch: None,
            },
        ),
        ("AsyncCommit", "US11294713B2", Stmt::AsyncCommit),
        ("AsyncWait", "US12118382B2", Stmt::AsyncWait { pending: 0 }),
        (
            "Collective",
            "US8539204B2",
            Stmt::Collective {
                kind: CollectiveKind::Reduce(ReduceOp::Sum),
                value: ci(1),
                result: 0,
                scope: Scope::Block,
            },
        ),
        // `CollectiveTuple` is not here: its combine function needs a module, and `check_kernel`
        // (which `lower_kernel` runs first) refuses it alone, so it cannot reach the decline through
        // `lower_kernel`. It is covered by the unit test on `patent_declined`.
    ]
}

#[test]
fn board_434c_statements_are_declined_by_name_at_every_depth() {
    for (name, patent, s) in board_434c() {
        let wrap: [(&str, Wrap); 4] = [
            ("bare", Box::new(|s| vec![s])),
            (
                "if",
                Box::new(|s| {
                    vec![Stmt::If {
                        cond: ci(1),
                        then: Block(vec![s]),
                        els: Block(vec![]),
                    }]
                }),
            ),
            (
                "while",
                Box::new(|s| {
                    vec![Stmt::While {
                        cond: ci(0),
                        body: Block(vec![s]),
                    }]
                }),
            ),
            (
                "for-in-if",
                Box::new(|s| {
                    vec![Stmt::If {
                        cond: ci(1),
                        then: Block(vec![Stmt::For {
                            local: 1,
                            start: ci(0),
                            end: ci(2),
                            step: ci(1),
                            unroll: Unroll::None,
                            body: Block(vec![s]),
                        }]),
                        els: Block(vec![]),
                    }]
                }),
            ),
        ];
        for (shape, f) in &wrap {
            let msg = refused(f(s.clone()));
            assert!(
                msg.contains(&format!("the statement {name}:"))
                    && msg.contains("declined by name")
                    && msg.contains(patent)
                    && msg.contains("board 434 C"),
                "{name} {shape}: {msg}"
            );
        }
    }
}
