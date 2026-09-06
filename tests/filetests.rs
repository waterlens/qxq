//! Runs every `tests/**/*.qxq` file through the freshly built `qxq` binary, one libtest
//! trial per file, so `cargo test` is the single entry point for the whole suite.

use libtest_mimic::{Arguments, Failed, Trial};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn collect_qxq_files(dir: &Path, files: &mut Vec<PathBuf>) {
  let entries =
    fs::read_dir(dir).unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()));
  for entry in entries {
    let path = entry.expect("failed to read directory entry").path();
    if path.is_dir() {
      collect_qxq_files(&path, files);
    } else if path.extension().is_some_and(|ext| ext == "qxq") {
      files.push(path);
    }
  }
}

/// Runs `qxq <mode> <file>` with the binary's directory first on PATH, so the
/// `{qxq|cargo run --}` choice in RUN: lines resolves to this build of `qxq`.
fn run_qxq(mode: &str, file: &Path) -> Result<(), Failed> {
  let exe = Path::new(env!("CARGO_BIN_EXE_qxq"));
  let mut paths = vec![exe.parent().expect("binary has a parent directory").to_path_buf()];
  if let Some(path) = env::var_os("PATH") {
    paths.extend(env::split_paths(&path));
  }
  let path = env::join_paths(paths).map_err(|e| Failed::from(e.to_string()))?;

  let output = Command::new(exe)
    .arg(mode)
    .arg(file)
    .env("PATH", path)
    .output()
    .map_err(|e| Failed::from(format!("failed to run {}: {e}", exe.display())))?;
  if output.status.success() {
    return Ok(());
  }
  Err(Failed::from(format!(
    "qxq {mode} {} exited with {}\n--- stdout ---\n{}--- stderr ---\n{}",
    file.display(),
    output.status,
    String::from_utf8_lossy(&output.stdout),
    String::from_utf8_lossy(&output.stderr)
  )))
}

fn trial_for(tests_dir: &Path, file: PathBuf) -> Trial {
  let name =
    file.strip_prefix(tests_dir).expect("test file is below tests/").to_string_lossy().into_owned();
  let content =
    fs::read_to_string(&file).unwrap_or_else(|e| panic!("failed to read {}: {e}", file.display()));
  match qxq::expect::directives(&content) {
    Err(e) => {
      let message = e.to_string();
      Trial::test(name, move || Err(Failed::from(message)))
    }
    Ok(directives) => match directives.skip {
      Some(reason) => {
        Trial::test(format!("{name} (SKIP: {reason})"), || Ok(())).with_ignored_flag(true)
      }
      None => {
        let mode = if directives.has_run { "--test-expect" } else { "--inspect" };
        Trial::test(name, move || run_qxq(mode, &file))
      }
    },
  }
}

fn main() {
  let mut args = Arguments::from_args();
  // The 7_io/file_edit_* tests write to fixed paths under /tmp, so run the trials
  // one at a time unless --test-threads is given explicitly.
  if args.test_threads.is_none() {
    args.test_threads = Some(1);
  }

  let tests_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
  let mut files = Vec::new();
  collect_qxq_files(&tests_dir, &mut files);
  files.sort();

  let trials = files.into_iter().map(|file| trial_for(&tests_dir, file)).collect();
  libtest_mimic::run(&args, trials).exit();
}
