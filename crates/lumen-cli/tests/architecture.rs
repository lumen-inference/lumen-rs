//! Workspace-wide architectural invariants (`docs/EN/architecture.md` Section 9), enforced from
//! the first commit as the roadmap requires:
//!
//! 1. model family names appear only in `lumen-models` and `lumen-chat`;
//! 2. the crate dependency graph follows the table in Section 2 (each crate may only depend on
//!    crates listed above it);
//! 3. backend crates never use a wildcard arm when matching on `Op`.
//!
//! These live in `lumen-cli` (the top of the graph) because they inspect every other crate.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn crate_dirs() -> Vec<(String, PathBuf)> {
    let mut out: Vec<_> = fs::read_dir(workspace_root().join("crates"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("Cargo.toml").exists())
        .map(|p| (p.file_name().unwrap().to_str().unwrap().to_string(), p))
        .collect();
    out.sort();
    out
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Source text with comments removed (strings, including raw strings, are kept intact).
fn strip_comments(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let next = b.get(i + 1).copied();
        if c == '/' && next == Some('/') {
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
        } else if c == '/' && next == Some('*') {
            let mut depth = 0;
            while i < b.len() {
                if b[i] == '/' && b.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if b[i] == '*' && b.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
        } else if c == 'r' && matches!(next, Some('"' | '#')) && {
            let mut j = i + 1;
            while b.get(j) == Some(&'#') {
                j += 1;
            }
            b.get(j) == Some(&'"')
        } {
            // raw string r#"..."#
            let mut hashes = 0;
            let mut j = i + 1;
            while b[j] == '#' {
                hashes += 1;
                j += 1;
            }
            j += 1; // opening quote
            loop {
                if j >= b.len() {
                    break;
                }
                if b[j] == '"' && (0..hashes).all(|h| b.get(j + 1 + h) == Some(&'#')) {
                    j += 1 + hashes;
                    break;
                }
                j += 1;
            }
            out.extend(&b[i..j.min(b.len())]);
            i = j;
        } else if c == '"' {
            let start = i;
            i += 1;
            while i < b.len() && b[i] != '"' {
                if b[i] == '\\' {
                    i += 1;
                }
                i += 1;
            }
            i = (i + 1).min(b.len());
            out.extend(&b[start..i]);
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

#[test]
fn model_family_names_only_in_models_and_chat() {
    let allowed = ["lumen-models", "lumen-chat"];
    let this_file = Path::new(file!()).file_name().unwrap().to_owned();
    let forbidden = [
        "qwen", "gemma", "llama", "mistral", "mixtral", "deepseek", "gpt_oss", "gpt-oss", "kimi",
        "glm",
    ];

    let mut violations = Vec::new();
    for (name, dir) in crate_dirs() {
        if allowed.contains(&name.as_str()) {
            continue;
        }
        let mut files = Vec::new();
        rust_files(&dir.join("src"), &mut files);
        rust_files(&dir.join("tests"), &mut files);
        for file in files {
            if file.file_name() == Some(&this_file) {
                continue;
            }
            let code = strip_comments(&fs::read_to_string(&file).unwrap()).to_ascii_lowercase();
            for token in forbidden {
                if code.contains(token) {
                    violations.push(format!(
                        "'{token}' in {}",
                        file.strip_prefix(workspace_root()).unwrap().display()
                    ));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "model family names outside lumen-models/lumen-chat:\n{}",
        violations.join("\n")
    );
}

/// Crate order of `docs/EN/architecture.md` Section 2. A crate may depend only on crates
/// that appear before it.
const ALLOWED_ORDER: &[&str] = &[
    "lumen-core",
    "lumen-kabi",
    "lumen-config",
    "lumen-mem",
    "lumen-tensor",
    "lumen-ltq",
    "lumen-format",
    "lumen-backend",
    "lumen-ref",
    "lumen-cpu",
    "lumen-cuda-kernels",
    "lumen-cuda",
    "lumen-models",
    "lumen-exec",
    "lumen-media",
    "lumen-chat",
    "lumen-engine",
    "lumen-quantize",
    "lumen-server",
    "lumen-cli",
];

#[test]
fn dependency_graph_follows_the_documented_order() {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let out = Command::new(cargo)
        .args([
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--manifest-path",
        ])
        .arg(workspace_root().join("Cargo.toml"))
        .output()
        .expect("cargo metadata");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let meta: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();

    let rank: BTreeMap<&str, usize> = ALLOWED_ORDER
        .iter()
        .enumerate()
        .map(|(i, n)| (*n, i))
        .collect();
    let packages = meta["packages"].as_array().unwrap();

    let mut violations = Vec::new();
    let mut seen = 0;
    for pkg in packages {
        let name = pkg["name"].as_str().unwrap();
        let Some(&my_rank) = rank.get(name) else {
            violations.push(format!(
                "crate '{name}' is not in the documented crate table"
            ));
            continue;
        };
        seen += 1;
        for dep in pkg["dependencies"].as_array().unwrap() {
            let dep_name = dep["name"].as_str().unwrap();
            if rank
                .get(dep_name)
                .is_some_and(|&dep_rank| dep_rank >= my_rank)
            {
                violations.push(format!(
                    "{name} -> {dep_name} points up/sideways in the DAG"
                ));
            }
        }
    }
    assert_eq!(
        seen,
        ALLOWED_ORDER.len(),
        "documented crates missing from the workspace"
    );
    assert!(violations.is_empty(), "{}", violations.join("\n"));
}

/// Finds `match op`-style blocks and reports a `_ =>` arm directly inside them.
fn wildcard_op_matches(code: &str) -> Vec<usize> {
    let b = code.as_bytes();
    let mut found = Vec::new();
    for (start, _) in code.match_indices("match ") {
        let head_end = code[start..].find('{').map(|i| start + i);
        let Some(open) = head_end else { continue };
        let scrutinee = code[start + 6..open].trim();
        if !matches!(scrutinee, "op" | "&op" | "*op" | "self.op" | "&self.op") {
            continue;
        }
        let mut depth = 0usize;
        let mut i = open;
        while i < b.len() {
            match b[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                b'_' if depth == 1 => {
                    let before = code[..i].trim_end().chars().last();
                    let after = code[i + 1..].trim_start();
                    if after.starts_with("=>") && matches!(before, Some(',' | '{' | '|')) {
                        found.push(code[..i].matches('\n').count() + 1);
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }
    found
}

#[test]
fn backends_never_use_wildcard_arms_on_op() {
    let backends = [
        "lumen-ref",
        "lumen-cpu",
        "lumen-cuda",
        "lumen-cuda-kernels",
        "lumen-backend",
    ];
    let mut violations = Vec::new();
    for (name, dir) in crate_dirs() {
        if !backends.contains(&name.as_str()) {
            continue;
        }
        let mut files = Vec::new();
        rust_files(&dir.join("src"), &mut files);
        for file in files {
            let code = strip_comments(&fs::read_to_string(&file).unwrap());
            for line in wildcard_op_matches(&code) {
                violations.push(format!(
                    "{}: wildcard arm in `match op` (stripped line {line})",
                    file.display()
                ));
            }
        }
    }
    assert!(violations.is_empty(), "{}", violations.join("\n"));
}

#[test]
fn scanner_self_checks() {
    // The comment stripper keeps strings (even with `//` inside) and drops comments.
    let src = "let a = \"qwen // x\"; // gemma\n/* llama */ let b = r#\"mistral \"//\" \"#;";
    let code = strip_comments(src).to_ascii_lowercase();
    assert!(code.contains("qwen // x") && code.contains("mistral"));
    assert!(!code.contains("gemma") && !code.contains("llama"));

    // The wildcard detector flags `_ =>` in `match op` but not in other matches or nested ones.
    assert_eq!(
        wildcard_op_matches("fn f(op: &Op) { match op { A => 1, _ => 2 } }").len(),
        1
    );
    assert!(
        wildcard_op_matches(
            "fn f(op: &Op) { match op { A => 1, B { .. } => match x { _ => 2 } } }"
        )
        .is_empty()
    );
    assert!(wildcard_op_matches("fn f() { match other { _ => 2 } }").is_empty());
    assert!(wildcard_op_matches("fn f(op: &Op) { match op { A | B => 1, C => 2 } }").is_empty());
}
