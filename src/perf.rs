use std::{
  ffi::{OsStr, OsString},
  fs,
  path::{Path, PathBuf},
  process::{Command, Stdio},
  sync::atomic::{AtomicU64, Ordering},
  time::{SystemTime, UNIX_EPOCH},
};

use crate::diagnostic::{Diagnostic, Result};

const LINUX_EVENTS: &str =
  "cycles,instructions,branches,branch-misses,cache-references,cache-misses";

static RUN_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Platform {
  Macos,
  Linux,
  Unsupported,
}

impl Platform {
  pub fn current() -> Self {
    if cfg!(target_os = "macos") {
      Self::Macos
    } else if cfg!(target_os = "linux") {
      Self::Linux
    } else {
      Self::Unsupported
    }
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactPaths {
  pub artifact: PathBuf,
  pub run_dir: PathBuf,
  pub stat: Option<PathBuf>,
  pub command: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfilerStepKind {
  Record,
  Stat,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfilerCommand {
  pub program: OsString,
  pub args: Vec<OsString>,
}

impl ProfilerCommand {
  fn new(program: impl Into<OsString>) -> Self {
    Self { program: program.into(), args: Vec::new() }
  }

  fn arg(&mut self, arg: impl Into<OsString>) {
    self.args.push(arg.into());
  }

  fn args<I, S>(&mut self, args: I)
  where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
  {
    self.args.extend(args.into_iter().map(Into::into));
  }

  pub fn display(&self) -> String {
    let mut parts = Vec::with_capacity(self.args.len() + 1);
    parts.push(shell_quote(&self.program));
    parts.extend(self.args.iter().map(|arg| shell_quote(arg.as_os_str())));
    format!("{} >/dev/null", parts.join(" "))
  }

  fn command(&self) -> Command {
    let mut command = Command::new(&self.program);
    command.args(&self.args);
    command.stdin(Stdio::inherit());
    command.stderr(Stdio::inherit());
    command.stdout(Stdio::null());
    command
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfilerStep {
  pub kind: ProfilerStepKind,
  pub required: bool,
  pub command: ProfilerCommand,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfilePlan {
  pub platform: Platform,
  pub paths: ArtifactPaths,
  pub steps: Vec<ProfilerStep>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfileReport {
  pub artifact: PathBuf,
  pub commands: Vec<String>,
}

pub fn default_root() -> PathBuf {
  PathBuf::from("/tmp/qxq-perf")
}

pub fn new_run_id() -> String {
  let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
  let counter = RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
  format!("{}-{}-{}", now.as_nanos(), std::process::id(), counter)
}

pub fn repl_source_path(root: &Path, run_id: &str) -> PathBuf {
  root.join(format!("{run_id}.qxq"))
}

pub fn artifact_paths(
  root: &Path,
  run_id: &str,
  platform: Platform,
  diag: &Diagnostic,
) -> Result<ArtifactPaths> {
  match platform {
    Platform::Macos => Ok(ArtifactPaths {
      artifact: root.join(format!("{run_id}.trace")),
      run_dir: root.to_path_buf(),
      stat: None,
      command: None,
    }),
    Platform::Linux => {
      let run_dir = root.join(run_id);
      Ok(ArtifactPaths {
        artifact: run_dir.join("perf.data"),
        run_dir: run_dir.clone(),
        stat: Some(run_dir.join("perf-stat.txt")),
        command: Some(run_dir.join("command.txt")),
      })
    }
    Platform::Unsupported => diag.fail("--perf is only supported on macOS and Linux"),
  }
}

pub fn build_plan(
  platform: Platform,
  root: &Path,
  run_id: &str,
  target_exe: &Path,
  target_args: &[OsString],
  diag: &Diagnostic,
) -> Result<ProfilePlan> {
  let paths = artifact_paths(root, run_id, platform, diag)?;
  let target = target_command(target_exe, target_args);
  let steps = match platform {
    Platform::Macos => vec![ProfilerStep {
      kind: ProfilerStepKind::Record,
      required: true,
      command: macos_record_command(&paths.artifact, &target),
    }],
    Platform::Linux => vec![
      ProfilerStep {
        kind: ProfilerStepKind::Record,
        required: true,
        command: linux_record_command(&paths.artifact, &target),
      },
      ProfilerStep {
        kind: ProfilerStepKind::Stat,
        required: false,
        command: linux_stat_command(paths.stat.as_ref().unwrap(), &target),
      },
    ],
    Platform::Unsupported => return diag.fail("--perf is only supported on macOS and Linux"),
  };

  Ok(ProfilePlan { platform, paths, steps })
}

pub fn run_profile(plan: &ProfilePlan, diag: &Diagnostic) -> Result<ProfileReport> {
  diag.context(
    fs::create_dir_all(&plan.paths.run_dir),
    format!("failed to create {}", plan.paths.run_dir.display()),
  )?;
  write_command_file(plan, diag)?;

  for step in &plan.steps {
    let mut command = step.command.command();
    match command.status() {
      Ok(status) if status.success() => {}
      Ok(status) if step.required => {
        return diag.fail(format!(
          "profiling failed with status {status}\nartifact: {}\ncommand: {}",
          plan.paths.artifact.display(),
          step.command.display()
        ));
      }
      Ok(status) => {
        eprintln!(
          "Warning: optional profiling step failed with status {status}: {}",
          step.command.display()
        );
      }
      Err(err) if step.required => {
        return diag.fail(format!(
          "failed to launch profiler: {err}\nartifact: {}\ncommand: {}",
          plan.paths.artifact.display(),
          step.command.display()
        ));
      }
      Err(err) => {
        eprintln!(
          "Warning: optional profiling step could not be launched: {err}: {}",
          step.command.display()
        );
      }
    }
  }

  Ok(ProfileReport {
    artifact: plan.paths.artifact.clone(),
    commands: plan.steps.iter().map(|step| step.command.display()).collect(),
  })
}

fn write_command_file(plan: &ProfilePlan, diag: &Diagnostic) -> Result<()> {
  let Some(path) = &plan.paths.command else {
    return Ok(());
  };
  let commands =
    plan.steps.iter().map(|step| step.command.display()).collect::<Vec<_>>().join("\n");
  diag.context(
    fs::write(path, format!("{commands}\n")),
    format!("failed to write {}", path.display()),
  )?;
  Ok(())
}

fn target_command(target_exe: &Path, target_args: &[OsString]) -> Vec<OsString> {
  let mut target = Vec::with_capacity(target_args.len() + 1);
  target.push(target_exe.as_os_str().to_os_string());
  target.extend(target_args.iter().cloned());
  target
}

fn macos_record_command(artifact: &Path, target: &[OsString]) -> ProfilerCommand {
  let mut command = ProfilerCommand::new("xcrun");
  command.args(["xctrace", "record", "--template", "CPU Counters", "--output"]);
  command.arg(artifact.as_os_str());
  command.args(["--launch", "--"]);
  command.args(target.iter().cloned());
  command
}

fn linux_record_command(artifact: &Path, target: &[OsString]) -> ProfilerCommand {
  let mut command = ProfilerCommand::new("perf");
  command.args(["record", "-g", "-e", LINUX_EVENTS, "-o"]);
  command.arg(artifact.as_os_str());
  command.arg("--");
  command.args(target.iter().cloned());
  command
}

fn linux_stat_command(stat: &Path, target: &[OsString]) -> ProfilerCommand {
  let mut command = ProfilerCommand::new("perf");
  command.args(["stat", "-d", "-o"]);
  command.arg(stat.as_os_str());
  command.arg("--");
  command.args(target.iter().cloned());
  command
}

fn shell_quote(arg: &OsStr) -> String {
  let text = arg.to_string_lossy();
  if text.is_empty() {
    return "''".to_string();
  }
  if text.bytes().all(is_shell_safe_byte) {
    return text.into_owned();
  }
  format!("'{}'", text.replace('\'', "'\\''"))
}

fn is_shell_safe_byte(byte: u8) -> bool {
  byte.is_ascii_alphanumeric()
    || matches!(byte, b'_' | b'-' | b'.' | b'/' | b':' | b'=' | b',' | b'+')
}

#[cfg(test)]
mod tests {
  use super::*;

  fn os_args(args: &[&str]) -> Vec<OsString> {
    args.iter().map(OsString::from).collect()
  }

  #[test]
  fn macos_artifact_path_is_trace_bundle() {
    let diag = Diagnostic::new();
    let paths =
      artifact_paths(Path::new("/tmp/qxq-perf"), "run-1", Platform::Macos, &diag).unwrap();
    assert_eq!(paths.artifact, PathBuf::from("/tmp/qxq-perf/run-1.trace"));
    assert_eq!(paths.run_dir, PathBuf::from("/tmp/qxq-perf"));
    assert!(paths.stat.is_none());
    assert!(paths.command.is_none());
  }

  #[test]
  fn linux_artifact_paths_are_inside_run_directory() {
    let diag = Diagnostic::new();
    let paths =
      artifact_paths(Path::new("/tmp/qxq-perf"), "run-1", Platform::Linux, &diag).unwrap();
    assert_eq!(paths.run_dir, PathBuf::from("/tmp/qxq-perf/run-1"));
    assert_eq!(paths.artifact, PathBuf::from("/tmp/qxq-perf/run-1/perf.data"));
    assert_eq!(paths.stat, Some(PathBuf::from("/tmp/qxq-perf/run-1/perf-stat.txt")));
    assert_eq!(paths.command, Some(PathBuf::from("/tmp/qxq-perf/run-1/command.txt")));
  }

  #[test]
  fn builds_macos_xctrace_command() {
    let diag = Diagnostic::new();
    let plan = build_plan(
      Platform::Macos,
      Path::new("/tmp/qxq-perf"),
      "run-1",
      Path::new("/bin/qxq"),
      &os_args(&["tests/a file.qxq"]),
      &diag,
    )
    .unwrap();

    assert_eq!(plan.steps.len(), 1);
    let command = &plan.steps[0].command;
    assert_eq!(command.program, OsString::from("xcrun"));
    assert_eq!(
      command.args,
      os_args(&[
        "xctrace",
        "record",
        "--template",
        "CPU Counters",
        "--output",
        "/tmp/qxq-perf/run-1.trace",
        "--launch",
        "--",
        "/bin/qxq",
        "tests/a file.qxq",
      ])
    );
    assert_eq!(
      command.display(),
      "xcrun xctrace record --template 'CPU Counters' --output /tmp/qxq-perf/run-1.trace --launch -- /bin/qxq 'tests/a file.qxq' >/dev/null"
    );
  }

  #[test]
  fn builds_linux_perf_record_and_stat_commands() {
    let diag = Diagnostic::new();
    let plan = build_plan(
      Platform::Linux,
      Path::new("/tmp/qxq-perf"),
      "run-1",
      Path::new("/bin/qxq"),
      &os_args(&["--load", "program.qxc"]),
      &diag,
    )
    .unwrap();

    assert_eq!(plan.steps.len(), 2);
    assert_eq!(plan.steps[0].kind, ProfilerStepKind::Record);
    assert!(plan.steps[0].required);
    assert_eq!(plan.steps[0].command.program, OsString::from("perf"));
    assert_eq!(
      plan.steps[0].command.args,
      os_args(&[
        "record",
        "-g",
        "-e",
        LINUX_EVENTS,
        "-o",
        "/tmp/qxq-perf/run-1/perf.data",
        "--",
        "/bin/qxq",
        "--load",
        "program.qxc",
      ])
    );

    assert_eq!(plan.steps[1].kind, ProfilerStepKind::Stat);
    assert!(!plan.steps[1].required);
    assert_eq!(plan.steps[1].command.program, OsString::from("perf"));
    assert_eq!(
      plan.steps[1].command.args,
      os_args(&[
        "stat",
        "-d",
        "-o",
        "/tmp/qxq-perf/run-1/perf-stat.txt",
        "--",
        "/bin/qxq",
        "--load",
        "program.qxc",
      ])
    );
  }
}
