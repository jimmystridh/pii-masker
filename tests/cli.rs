use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

const TEST_WEIGHTS_ENV_VAR: &str = "PII_MASKER_TEST_MODEL_WEIGHTS";

fn local_weights() -> Option<PathBuf> {
    if let Ok(path) = std::env::var(TEST_WEIGHTS_ENV_VAR) {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    let repo_local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("model/model.safetensors");
    if repo_local.exists() {
        return Some(repo_local);
    }

    None
}

fn run_cli(input: &str, weights: &PathBuf) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_pii-mask"))
        .env("PII_MASKER_MODEL_WEIGHTS", weights)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn pii-mask");

    child
        .stdin
        .as_mut()
        .expect("stdin handle")
        .write_all(input.as_bytes())
        .expect("write stdin");

    child.wait_with_output().expect("collect output")
}

#[test]
fn masks_stdin_text() {
    let Some(weights) = local_weights() else {
        eprintln!("Skipping CLI model test because no test weights were configured.");
        return;
    };

    let output = run_cli("John Doe lives at 1234 Elm St.", &weights);
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        String::from_utf8(output.stdout).expect("utf-8 stdout"),
        "John Doe lives at [ADDRESS].\n"
    );
}
