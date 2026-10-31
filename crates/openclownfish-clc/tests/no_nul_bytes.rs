//! No tracked text file may contain a raw NUL byte. Editing tooling can turn a
//! backslash-zero escape into a real NUL that still compiles, so scan the index.

use std::path::{Path, PathBuf};
use std::process::Command;

const TEXT_EXTS: &[&str] = &[
    "rs", "md", "toml", "cl", "yml", "yaml", "json", "txt", "lock",
];

fn has_nul(bytes: &[u8]) -> bool {
    bytes.contains(&0u8)
}

fn tracked_text_files() -> (PathBuf, Vec<PathBuf>) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(&root)
        .output()
        .expect("git ls-files must run");
    assert!(
        out.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let files = out
        .stdout
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| PathBuf::from(String::from_utf8(s.to_vec()).expect("utf-8 path")))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| TEXT_EXTS.contains(&e))
        })
        .collect();
    (root, files)
}

#[test]
fn scanner_detects_an_injected_nul() {
    assert!(has_nul(b"let s = \"a\0b\";"));
    assert!(!has_nul(b"let s = \"a\\0b\";"));
}

#[test]
fn no_tracked_text_file_contains_a_nul_byte() {
    let (root, files) = tracked_text_files();
    // Population control: the scan must have seen the repo, not an empty list.
    assert!(
        files.len() >= 20,
        "only {} tracked text files found",
        files.len()
    );
    assert!(
        files.iter().any(|p| p.ends_with("Cargo.toml")),
        "control: Cargo.toml not listed"
    );
    let bad: Vec<_> = files
        .iter()
        .filter(|p| has_nul(&std::fs::read(root.join(p)).unwrap_or_default()))
        .collect();
    assert!(bad.is_empty(), "raw NUL byte in: {bad:?}");
}
