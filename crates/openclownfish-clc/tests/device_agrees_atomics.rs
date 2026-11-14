//! `Backend::lower_kernel` for the statement IR: the OpenCL C 2.0 atomics on every device that
//! reports OpenCL C 2.0, against Unpopped's launch interpreter (`run_launch`).
//!
//! Three kernel shapes: every `AtomicOp` on a global cell from 128 threads (final value equal to
//! the oracle's; for `Exchange`, whose winner is order-dependent, a member of the written set;
//! for `Add` of one, the returned old values a permutation of 0..128), a block-scope counter in
//! `Shared` memory with barriers, and a compare-exchange loop that reaches the maximum. A device
//! below OpenCL C 2.0 is skipped (that is the gate); a 64-bit build failure is reported, not
//! hidden. The declines are checked in source, with no device.

use openclownfish::{Context, Program, Runtime};
use openclownfish_clc::OpenClC;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const BLOCK: u32 = 32;
const GRID: u32 = 4;
const T: usize = (BLOCK * GRID) as usize;
const OPTS: &str = "-cl-std=CL2.0";

fn width(e: ElementKind) -> usize {
    match e {
        ElementKind::I32 | ElementKind::U32 => 4,
        _ => 8,
    }
}

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(bx(a), bx(b))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(bx(a), bx(b))
}

fn cmp(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, bx(a), bx(b))
}

fn cu(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, ElementKind::U32)
}

fn ce(v: i64, e: ElementKind) -> ScalarExpr {
    ScalarExpr::ConstInt(v, e)
}

fn cast(x: ScalarExpr, to: ElementKind) -> ScalarExpr {
    ScalarExpr::Cast(
        bx(x),
        CastSpec {
            to,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

fn gtid() -> ScalarExpr {
    add(
        mul(
            ScalarExpr::Exec(ExecId::BlockIdx(0)),
            ScalarExpr::Exec(ExecId::BlockDim(0)),
        ),
        ScalarExpr::Exec(ExecId::ThreadIdx(0)),
    )
}

fn cell0() -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![cu(0)],
    }
}

fn slot() -> Place {
    Place::Logical {
        buffer: 1,
        coords: vec![gtid()],
    }
}

fn rmw(op: AtomicOp, place: Place, value: ScalarExpr, old: Option<u32>, scope: Scope) -> Stmt {
    Stmt::AtomicRmw {
        op,
        place,
        value,
        old,
        order: MemoryOrder::AcqRel,
        scope,
    }
}

fn launch(mut k: KernelFn) -> KernelFn {
    k.launch = Some(LaunchShape::new([BLOCK, 1, 1], [GRID, 1, 1]));
    k
}

/// 128 threads, each `old = op(cell, value)` and `out[gtid] = old`.
fn global_rmw(e: ElementKind, op: AtomicOp, value: ScalarExpr) -> KernelFn {
    let body = vec![
        Stmt::Let {
            local: 0,
            init: cast(gtid(), e),
        },
        rmw(op, cell0(), value, Some(1), Scope::Device),
        Stmt::Store {
            place: slot(),
            value: ScalarExpr::Local(1),
        },
    ];
    launch(KernelFn::new(
        "rmw",
        e,
        vec![
            Param::buffer("cell", e, BufferRole::ReadWrite, Some(1)),
            Param::buffer("out", e, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(e), LocalDecl::scalar(e)],
        Block(body),
    ))
}

/// Per block: thread 0 zeroes a `Shared` cell, a barrier, every thread adds one at block scope,
/// a barrier, thread 0 stores the count.
fn shared_count() -> KernelFn {
    let e = ElementKind::U32;
    let tid = || ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let shared = || Place::Alloc {
        alloc: 0,
        index: cu(0),
    };
    let body = vec![
        Stmt::If {
            cond: cmp(BinaryOp::CmpEq, tid(), cu(0)),
            then: Block(vec![Stmt::Store {
                place: shared(),
                value: cu(0),
            }]),
            els: Block(vec![]),
        },
        Stmt::Barrier {
            scope: Scope::Block,
            kind: unpopped::kernel::BarrierKind::Full,
        },
        rmw(AtomicOp::Add, shared(), cu(1), None, Scope::Block),
        Stmt::Barrier {
            scope: Scope::Block,
            kind: unpopped::kernel::BarrierKind::Full,
        },
        Stmt::If {
            cond: cmp(BinaryOp::CmpEq, tid(), cu(0)),
            then: Block(vec![
                Stmt::Load {
                    local: 0,
                    place: shared(),
                },
                Stmt::Store {
                    place: Place::Logical {
                        buffer: 0,
                        coords: vec![ScalarExpr::Exec(ExecId::BlockIdx(0))],
                    },
                    value: ScalarExpr::Local(0),
                },
            ]),
            els: Block(vec![]),
        },
    ];
    let mut k = launch(KernelFn::new(
        "shared_count",
        e,
        vec![Param::buffer("out", e, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(e)],
        Block(body),
    ));
    k.allocs = vec![AllocDecl::new(
        "count",
        MemorySpace::Shared,
        e,
        Extent::Static(1),
    )];
    k
}

/// An atomic maximum built from a compare-exchange loop.
fn cas_max() -> KernelFn {
    let e = ElementKind::I32;
    let body = vec![
        Stmt::Let {
            local: 0,
            init: add(mul(cast(gtid(), e), ce(7919, e)), ce(13, e)),
        },
        // An atomic read (fetch-or of zero): a plain load of an atomic cell is a data race.
        rmw(AtomicOp::Or, cell0(), ce(0, e), Some(1), Scope::Device),
        Stmt::While {
            cond: cmp(BinaryOp::CmpLt, ScalarExpr::Local(1), ScalarExpr::Local(0)),
            body: Block(vec![
                Stmt::AtomicCas {
                    place: cell0(),
                    expected: ScalarExpr::Local(1),
                    desired: ScalarExpr::Local(0),
                    old: Some(2),
                    order: MemoryOrder::AcqRel,
                    scope: Scope::Device,
                },
                Stmt::If {
                    cond: cmp(BinaryOp::CmpEq, ScalarExpr::Local(2), ScalarExpr::Local(1)),
                    then: Block(vec![Stmt::Break]),
                    els: Block(vec![Stmt::Assign {
                        local: 1,
                        value: ScalarExpr::Local(2),
                    }]),
                },
            ]),
        },
    ];
    launch(KernelFn::new(
        "cas_max",
        e,
        vec![Param::buffer("cell", e, BufferRole::ReadWrite, Some(1))],
        vec![
            LocalDecl::scalar(e),
            LocalDecl::scalar(e),
            LocalDecl::scalar(e),
        ],
        Block(body),
    ))
}

fn le(v: i128, e: ElementKind) -> Vec<u8> {
    v.to_le_bytes()[..width(e)].to_vec()
}

fn rd(b: &[u8], i: usize, e: ElementKind) -> i128 {
    let w = width(e);
    let c = &b[i * w..(i + 1) * w];
    match e {
        ElementKind::I32 => i128::from(i32::from_le_bytes(c.try_into().unwrap())),
        ElementKind::U32 => i128::from(u32::from_le_bytes(c.try_into().unwrap())),
        ElementKind::I64 => i128::from(i64::from_le_bytes(c.try_into().unwrap())),
        _ => i128::from(u64::from_le_bytes(c.try_into().unwrap())),
    }
}

/// The oracle's buffers after the launch.
fn oracle(k: &KernelFn, init: &[Vec<u8>], kinds: &[ElementKind]) -> Vec<Vec<u8>> {
    let mut args: Vec<KernelArg> = init
        .iter()
        .zip(kinds)
        .map(|(b, e)| {
            KernelArg::Buffer(TypedBuffer::new(
                *e,
                vec![(b.len() / width(*e)) as i64],
                vec![1],
                b.clone(),
            ))
        })
        .collect();
    run_launch(k, &mut args).expect("the interpreter runs the kernel");
    args.into_iter()
        .map(|a| match a {
            KernelArg::Buffer(b) => b.bytes,
            _ => unreachable!(),
        })
        .collect()
}

/// The same launch on a device; `None` when the program does not build (the log is returned).
fn device(
    ctx: &Context,
    queue: &openclownfish::Queue,
    k: &KernelFn,
    init: &[Vec<u8>],
) -> Result<Vec<Vec<u8>>, String> {
    let backend = OpenClC::new().with_c20_atomics(true);
    let g = backend.lower_kernel(k).expect("lowers");
    let prog = Program::build(ctx, &g.source, OPTS).map_err(|e| format!("{e}\n{}", g.source))?;
    let mut kern = prog.kernel(&g.name).unwrap();
    // The buffers must outlive the launch.
    let mut held = Vec::new();
    for (i, b) in init.iter().enumerate() {
        let buf = ctx.create_buffer(b.len()).unwrap();
        queue.write_buffer(&buf, 0, b).unwrap();
        kern.set_arg_buffer(i as u32, &buf).unwrap();
        held.push(buf);
    }
    // SAFETY: every access is by construction inside its buffer; `held` outlives the launch.
    unsafe { queue.launch(&kern, &[T], Some(&[BLOCK as usize])) }
        .unwrap()
        .wait()
        .unwrap();
    let mut out = Vec::new();
    for (b, buf) in init.iter().zip(&held) {
        let mut raw = vec![0u8; b.len()];
        queue.read_buffer(buf, 0, &mut raw).unwrap();
        out.push(raw);
    }
    Ok(out)
}

fn c20(version: &str) -> bool {
    version
        .split("OpenCL C ")
        .nth(1)
        .and_then(|v| v.split(' ').next())
        .and_then(|v| v.split('.').next())
        .and_then(|m| m.parse::<u32>().ok())
        .is_some_and(|m| m >= 2)
}

#[test]
fn atomics_agree_with_the_interpreter_on_every_opencl_c_2_device() {
    let Ok(rt) = Runtime::load() else { return };
    let Ok(platforms) = rt.platforms() else {
        return;
    };
    let (mut kernels, mut devices_run, mut devices_skipped) = (0, 0, 0);
    for p in platforms {
        let Ok(devs) = p.devices() else { continue };
        for d in devs {
            let label = format!(
                "{} / {}",
                p.name().unwrap_or_default(),
                d.name().unwrap_or_default()
            );
            let Ok(ver) = d.opencl_c_version() else {
                continue;
            };
            if !c20(&ver) {
                devices_skipped += 1;
                continue;
            }
            let Ok(ctx) = Context::new(&d) else { continue };
            let queue = ctx.create_queue().unwrap();
            devices_run += 1;
            for e in [
                ElementKind::I32,
                ElementKind::U32,
                ElementKind::I64,
                ElementKind::U64,
            ] {
                let w = width(e);
                let all_ones = -1i128;
                let ops: [(AtomicOp, i128, &str); 7] = [
                    (AtomicOp::Add, 0, "add"),
                    (AtomicOp::Max, 0, "max"),
                    (AtomicOp::Min, 0x7fff_ffff, "min"),
                    (AtomicOp::And, all_ones, "and"),
                    (AtomicOp::Or, 0, "or"),
                    (AtomicOp::Xor, 0, "xor"),
                    (AtomicOp::Exchange, 0, "exchange"),
                ];
                for (op, init_cell, name) in ops {
                    let value = add(mul(ScalarExpr::Local(0), ce(37, e)), ce(11, e));
                    let k = global_rmw(e, op, value);
                    let init = vec![le(init_cell, e), vec![0u8; T * w]];
                    let want = oracle(&k, &init, &[e, e]);
                    let got = match device(&ctx, &queue, &k, &init) {
                        Ok(g) => g,
                        Err(log) if w == 8 => {
                            eprintln!("{label}: {e:?} {name}: 64-bit atomics do not build:\n{log}");
                            continue;
                        }
                        Err(log) => panic!("{label}: {e:?} {name} failed to build:\n{log}"),
                    };
                    kernels += 1;
                    if op == AtomicOp::Exchange {
                        let written: Vec<i128> = (0..T)
                            .map(|t| rd(&le((t as i128) * 37 + 11, e), 0, e))
                            .collect();
                        assert!(
                            written.contains(&rd(&got[0], 0, e)),
                            "{label}: {e:?} exchange final is not a written value"
                        );
                    } else {
                        assert_eq!(
                            rd(&got[0], 0, e),
                            rd(&want[0], 0, e),
                            "{label}: {e:?} {name} final cell"
                        );
                    }
                }
                // A counter: the old values are a permutation, so each add was atomic.
                let k = global_rmw(e, AtomicOp::Add, ce(1, e));
                let init = vec![le(0, e), vec![0u8; T * w]];
                if let Ok(got) = device(&ctx, &queue, &k, &init) {
                    kernels += 1;
                    let mut olds: Vec<i128> = (0..T).map(|i| rd(&got[1], i, e)).collect();
                    olds.sort_unstable();
                    assert_eq!(
                        olds,
                        (0..T as i128).collect::<Vec<_>>(),
                        "{label}: {e:?} add old values"
                    );
                    assert_eq!(rd(&got[0], 0, e), T as i128, "{label}: {e:?} add total");
                }
            }
            // Block scope in Shared memory, with barriers.
            let k = shared_count();
            let init = vec![vec![0xFFu8; GRID as usize * 4]];
            let want = oracle(&k, &init, &[ElementKind::U32]);
            let got = device(&ctx, &queue, &k, &init).unwrap_or_else(|l| panic!("{label}: {l}"));
            kernels += 1;
            assert_eq!(got, want, "{label}: shared block counter");
            // A compare-exchange loop reaches the maximum.
            let k = cas_max();
            let init = vec![le(i128::from(i32::MIN), ElementKind::I32)];
            let want = oracle(&k, &init, &[ElementKind::I32]);
            let got = device(&ctx, &queue, &k, &init).unwrap_or_else(|l| panic!("{label}: {l}"));
            kernels += 1;
            assert_eq!(got, want, "{label}: cas max loop");
        }
    }
    eprintln!(
        "{kernels} kernels agree on {devices_run} OpenCL C 2.0 device(s); {devices_skipped} below 2.0 skipped"
    );
}

fn declined(r: Result<unpopped::backend::GeneratedKernel, LowerError>, needle: &str) {
    match r {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains(needle), "{detail:?} lacks {needle:?}");
        }
        other => panic!("expected a decline mentioning {needle:?}, got {other:?}"),
    }
}

#[test]
fn atomics_and_everything_outside_the_subset_are_declined_by_name() {
    let on = OpenClC::new().with_c20_atomics(true);
    let off = OpenClC::new();
    let e = ElementKind::I32;
    let k = global_rmw(e, AtomicOp::Add, ce(1, e));
    // Off by default: the declined statement is named, and so is the flag.
    declined(off.lower_kernel(&k), "with_c20_atomics");
    let src = on.lower_kernel(&k).expect("lowers").source;
    assert!(
        src.contains("atomic_fetch_add_explicit((volatile __global atomic_int*)&b0[(long)("),
        "{src}"
    );
    assert!(
        src.contains("memory_order_acq_rel, memory_scope_device"),
        "{src}"
    );
    // Scopes other than Block and Device.
    for s in [Scope::Subgroup, Scope::System] {
        let mut k = global_rmw(e, AtomicOp::Add, ce(1, e));
        if let Stmt::AtomicRmw { scope, .. } = &mut k.body.0[1] {
            *scope = s;
        }
        let r = on.lower_kernel(&k);
        assert!(r.is_err(), "scope {s:?} must be declined");
    }
    // Floats and 16-bit elements.
    for bad in [ElementKind::F32, ElementKind::F16] {
        let k = global_rmw(bad, AtomicOp::Add, ScalarExpr::Const(1.0));
        let r = on.lower_kernel(&k);
        assert!(r.is_err(), "a {bad:?} atomic must be declined: {r:?}");
    }
    // The peer and multicast places.
    let mut k = global_rmw(e, AtomicOp::Add, ce(1, e));
    if let Stmt::AtomicRmw { place, .. } = &mut k.body.0[1] {
        *place = Place::Multicast {
            buffer: 0,
            coords: vec![cu(0)],
        };
    }
    assert!(
        on.lower_kernel(&k).is_err(),
        "a multicast place is declined"
    );
    // No launch shape.
    let mut k = global_rmw(e, AtomicOp::Add, ce(1, e));
    k.launch = None;
    assert!(
        on.lower_kernel(&k).is_err(),
        "a kernel without a launch is declined"
    );
    // The block-scope kernel uses `__local` and the work-group scope.
    let src = on.lower_kernel(&shared_count()).expect("lowers").source;
    assert!(src.contains("__local uint a0[1];"), "{src}");
    assert!(src.contains("(volatile __local atomic_uint*)"), "{src}");
    assert!(src.contains("memory_scope_work_group"), "{src}");
    assert!(
        src.contains("barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);"),
        "{src}"
    );
    // The compare-exchange loop spells a failure order that does not release.
    let src = on.lower_kernel(&cas_max()).expect("lowers").source;
    assert!(
        src.contains("memory_order_acq_rel, memory_order_acquire, memory_scope_device"),
        "{src}"
    );
}
