use bumpalo::Bump;
use clap::{ArgGroup, Parser};
use qxq::diagnostic::Result;
use qxq::*;
use rustyline::{
  config::Configurer, error::ReadlineError, Cmd, ConditionalEventHandler, DefaultEditor, Event,
  EventContext, EventHandler, KeyEvent, Movement, RepeatCount,
};
use serde::{Deserialize, Serialize};
use std::{
  fs,
  io::{self, IsTerminal, Write},
  rc::Rc,
  time::{Duration, Instant},
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(group(
  ArgGroup::new("mode")
    .required(false)
    .args(["check_expect", "test_expect", "update_expect", "dump", "load"]),
))]
struct Cli {
  #[arg(long)]
  check_expect: Option<String>,

  #[arg(long)]
  test_expect: Option<String>,

  #[arg(long)]
  update_expect: Option<String>,

  #[arg(long)]
  skip_multiple_expect: bool,

  #[arg(long, value_name = "OUTPUT_FILE")]
  dump: Option<String>,

  #[arg(long)]
  load: bool,

  #[arg(long)]
  inspect: bool,

  #[arg(long)]
  no_tree: bool,

  #[arg(value_name = "INPUT_FILES")]
  input_files: Vec<String>,
}

fn show_message() {
  println!(
    r#"
 ________       ___    ___  ________      
|\   __  \     |\  \  /  /||\   __  \     
\ \  \|\  \    \ \  \/  / /\ \  \|\  \    
 \ \  \\\  \    \ \    / /  \ \  \\\  \   
  \ \  \\\  \    /     \/    \ \  \\\  \  
   \ \_____  \  /  /\   \     \ \_____  \ 
    \|___| \__\/__/ /\ __\     \|___| \__\
          \|__||__|/ \|__|           \|__|
"#
  );
  println!("QxQ REPL Version 0.1.0");
  println!("Copyright (c) 2024-{} waterlens", chrono::Local::now().format("%Y"));
  println!();
}

fn print_tree<T: std::fmt::Display>(tree: T) {
  println!("--- Syntax Tree ---");
  println!("{tree}");
}

fn print_thunks(image: &bytecode::BytecodeImage) {
  println!("--- Thunk ---");
  print!(
    "{}",
    image.thunks.iter().map(|t| t.to_string()).collect::<Vec<_>>().join("\n--- Thunk ---\n")
  );
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ReplConfig {
  elapsed: bool,
  inspect: bool,
}

fn detect_colon_command(line: &str) -> Option<&str> {
  let rest = line.strip_prefix(':')?;
  if rest.trim().is_empty() {
    return Some("");
  }
  if rest.starts_with(char::is_whitespace) {
    return None;
  }
  Some(rest)
}

fn apply_repl_command(config: &ReplConfig, cmd: &str) -> Result<ReplConfig> {
  let mut value = serde_json::to_value(config)?;
  ssof::apply_str(&mut value, cmd)?;
  Ok(serde_json::from_value(value)?)
}

struct ReplInterruptHandler;

fn repl_interrupt_command(line: &str) -> Cmd {
  if line.is_empty() {
    Cmd::Interrupt
  } else {
    Cmd::Kill(Movement::WholeBuffer)
  }
}

impl ConditionalEventHandler for ReplInterruptHandler {
  fn handle(
    &self,
    _evt: &Event,
    _n: RepeatCount,
    _positive: bool,
    ctx: &EventContext,
  ) -> Option<Cmd> {
    Some(repl_interrupt_command(ctx.line()))
  }
}

struct ReplTimings {
  parse: Duration,
  codegen: Duration,
  finalize: Duration,
  execute: Option<Duration>,
  total: Duration,
}

fn print_repl_timings(t: &ReplTimings) {
  color_print::cprintln!("<dim>elapsed:</dim>");
  color_print::cprintln!("<dim>  parse:</dim>    <cyan>{:.6}s</cyan>", t.parse.as_secs_f64());
  color_print::cprintln!("<dim>  codegen:</dim>  <cyan>{:.6}s</cyan>", t.codegen.as_secs_f64());
  color_print::cprintln!("<dim>  finalize:</dim> <cyan>{:.6}s</cyan>", t.finalize.as_secs_f64());
  if let Some(exec) = t.execute {
    color_print::cprintln!("<dim>  execute:</dim>  <cyan>{:.6}s</cyan>", exec.as_secs_f64());
  }
  color_print::cprintln!("<dim>  total:</dim>    <cyan>{:.6}s</cyan>", t.total.as_secs_f64());
}

fn run_repl(diag: Rc<diagnostic::Diagnostic>) -> Result<()> {
  show_message();
  let mut rl = DefaultEditor::new()?;
  let history_path = dirs::home_dir().map(|f| f.join(".qxq_history"));

  rl.set_history_ignore_space(true);
  rl.set_max_history_size(1024)?;
  rl.bind_sequence(KeyEvent::ctrl('C'), EventHandler::Conditional(Box::new(ReplInterruptHandler)));

  history_path.as_ref().inspect(|path| {
    if let Ok(true) = path.try_exists() {
      rl.load_history(path).expect("unable to load history");
    }
  });

  let mut config = ReplConfig::default();

  loop {
    let readline = rl.readline("> ");
    match readline {
      Ok(line) => {
        if line.trim().is_empty() {
          continue;
        }
        if let Some(cmd) = detect_colon_command(&line) {
          if cmd.is_empty() {
            diag.report("empty command");
            continue;
          }
          match apply_repl_command(&config, cmd) {
            Ok(candidate) => config = candidate,
            Err(e) => diag.report(&format!("invalid config: {e}")),
          }
          continue;
        }
        let total_start = Instant::now();
        let arena = Bump::new();
        let parse_start = Instant::now();
        let parser = parser::Parser::new(&arena, Rc::clone(&diag), &line);
        match parser.parse() {
          Ok(tree) => {
            let parse_dur = parse_start.elapsed();
            if config.inspect {
              print_tree(&tree);
            }
            let codegen_start = Instant::now();
            let mut codegen = codegen::CodeGenCtx::new(&arena, Rc::clone(&diag), tree);
            let mut bc = bytecode::BytecodeCtx::new();
            match codegen.emit_tree(&mut bc) {
              Ok(()) => {
                let codegen_dur = codegen_start.elapsed();
                let finalize_start = Instant::now();
                let image = bc.finalize();
                let finalize_dur = finalize_start.elapsed();
                if config.inspect {
                  print_thunks(&image);
                  if config.elapsed {
                    print_repl_timings(&ReplTimings {
                      parse: parse_dur,
                      codegen: codegen_dur,
                      finalize: finalize_dur,
                      execute: None,
                      total: total_start.elapsed(),
                    });
                  }
                } else {
                  let exec_start = Instant::now();
                  match runtime::execute(image, Rc::clone(&diag)) {
                    Ok(result) => {
                      let exec_dur = exec_start.elapsed();
                      println!("{result}");
                      if config.elapsed {
                        print_repl_timings(&ReplTimings {
                          parse: parse_dur,
                          codegen: codegen_dur,
                          finalize: finalize_dur,
                          execute: Some(exec_dur),
                          total: total_start.elapsed(),
                        });
                      }
                    }
                    Err(e) => diag.report_err(&e),
                  }
                }
              }
              Err(e) => diag.report_err(&e),
            }
            rl.add_history_entry(line.as_str())?;
          }
          Err(e) => diag.report_err(&e),
        }
      }
      Err(ReadlineError::Interrupted) => break,
      Err(ReadlineError::Eof) => break,
      Err(err) => {
        diag.report_err(&err.into());
        break;
      }
    }
  }

  history_path.inspect(|path| rl.save_history(path).expect("unable to save history"));
  Ok(())
}

fn run(cli: Cli, diag: Rc<diagnostic::Diagnostic>) -> Result<()> {
  if let Some(file) = cli.check_expect {
    return expect::run_check(&file);
  }

  if let Some(file) = cli.test_expect {
    return expect::run_test_file(&file);
  }

  if let Some(file) = cli.update_expect {
    match expect::update_expectations(&file, cli.skip_multiple_expect)? {
      expect::UpdateStatus::Updated => return Ok(()),
      expect::UpdateStatus::Skipped => std::process::exit(2),
    }
  }

  if !cli.input_files.is_empty() {
    for file_path in cli.input_files {
      if cli.load {
        let image = if file_path == "-" {
          let mut loader = loader::Loader::new(io::stdin().lock(), Rc::clone(&diag));
          loader.load()?
        } else {
          let file =
            diag.context(fs::File::open(&file_path), format!("failed to open {}", file_path))?;
          let mut loader = loader::Loader::new(file, Rc::clone(&diag));
          loader.load()?
        };
        if cli.inspect {
          print_thunks(&image);
        } else {
          println!("{}", runtime::execute(image, Rc::clone(&diag))?);
        }
        continue;
      }
      let content =
        diag.context(fs::read_to_string(&file_path), format!("failed to read {}", file_path))?;
      let arena = Bump::new();
      let parser = parser::Parser::new(&arena, Rc::clone(&diag), &content);
      let tree = diag.context(parser.parse(), format!("failed to parse {}", file_path))?;

      if cli.inspect && cli.dump.is_none() && !cli.no_tree {
        print_tree(&tree);
      }

      let mut codegen = codegen::CodeGenCtx::new(&arena, Rc::clone(&diag), tree);
      let mut bc = bytecode::BytecodeCtx::new();
      codegen.emit_tree(&mut bc)?;
      let image = bc.finalize();

      if let Some(ref dump_path) = cli.dump {
        let dumper = dumper::Dumper::new(image, Rc::clone(&diag));
        let data = dumper.dump()?;
        if dump_path == "-" {
          if io::stdout().is_terminal() {
            return diag
              .fail("refusing to dump binary data to a terminal. use redirection or a file.");
          }
          io::stdout().write_all(&data)?;
        } else {
          diag.context(fs::write(dump_path, &data), format!("failed to write {}", dump_path))?;
        }
      } else {
        if cli.inspect {
          print_thunks(&image);
        } else {
          println!("{}", runtime::execute(image, Rc::clone(&diag))?);
        }
      }
    }
  } else if cli.dump.is_none() {
    run_repl(diag)?;
  } else {
    return diag.fail("no input files provided for dump.");
  }
  Ok(())
}

fn main() {
  let cli = Cli::parse();
  let diag = Rc::new(diagnostic::Diagnostic::new());

  if let Err(e) = run(cli, Rc::clone(&diag)) {
    diag.report_err(&e);
    std::process::exit(1);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn detect_colon_commands() {
    assert_eq!(detect_colon_command(":+elapsed"), Some("+elapsed"));
    assert_eq!(detect_colon_command(":-elapsed"), Some("-elapsed"));
    assert_eq!(detect_colon_command(":+inspect"), Some("+inspect"));
    assert_eq!(detect_colon_command(":-inspect"), Some("-inspect"));
    assert_eq!(detect_colon_command(":"), Some(""));
    assert_eq!(detect_colon_command(":   "), Some(""));
  }

  #[test]
  fn reject_space_after_colon() {
    assert_eq!(detect_colon_command(": +elapsed"), None);
    assert_eq!(detect_colon_command(": x 1 + 1"), None);
  }

  #[test]
  fn reject_leading_space() {
    assert_eq!(detect_colon_command(" :+elapsed"), None);
    assert_eq!(detect_colon_command("  :+elapsed"), None);
  }

  #[test]
  fn normal_lines_not_commands() {
    assert_eq!(detect_colon_command("1 + 2"), None);
    assert_eq!(detect_colon_command("let x = 1"), None);
    assert_eq!(detect_colon_command(""), None);
  }

  #[test]
  fn config_enable_disable_elapsed() {
    let config = ReplConfig::default();
    assert!(!config.elapsed);

    let config = apply_repl_command(&config, "+elapsed").unwrap();
    assert!(config.elapsed);

    let config = apply_repl_command(&config, "-elapsed").unwrap();
    assert!(!config.elapsed);
  }

  #[test]
  fn config_enable_disable_inspect() {
    let config = ReplConfig::default();
    assert!(!config.inspect);

    let config = apply_repl_command(&config, "+inspect").unwrap();
    assert!(config.inspect);

    let config = apply_repl_command(&config, "-inspect").unwrap();
    assert!(!config.inspect);
  }

  #[test]
  fn config_reject_non_boolean() {
    let config = ReplConfig::default();
    assert!(apply_repl_command(&config, "elapsed=42").is_err());
    assert!(apply_repl_command(&config, "inspect=hello").is_err());
  }

  #[test]
  fn config_reject_unknown_option() {
    let config = ReplConfig::default();
    assert!(apply_repl_command(&config, "+verbose").is_err());
  }

  #[test]
  fn config_failed_patch_does_not_mutate() {
    let config = ReplConfig::default();
    let _ = apply_repl_command(&config, "+verbose");
    assert!(!config.elapsed);
    assert!(!config.inspect);
  }

  #[test]
  fn repl_interrupt_clears_non_empty_input() {
    assert_eq!(repl_interrupt_command("1 + 2"), Cmd::Kill(Movement::WholeBuffer));
  }

  #[test]
  fn repl_interrupt_exits_empty_input() {
    assert_eq!(repl_interrupt_command(""), Cmd::Interrupt);
  }
}
