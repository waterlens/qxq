use bumpalo::Bump;
use clap::{ArgGroup, Parser};
use qxq::diagnostic::Result;
use qxq::*;
use rustyline::{DefaultEditor, config::Configurer, error::ReadlineError};
use serde_json::Value;
use std::{
  fs,
  io::{self, IsTerminal, Write},
  rc::Rc,
  time::Instant,
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

fn validate_repl_config(config: &Value) -> std::result::Result<(), String> {
  let obj = config.as_object().ok_or("config must be an object")?;
  for (key, val) in obj {
    match key.as_str() {
      "elapsed" => {
        if !val.is_boolean() {
          return Err(format!("elapsed must be a boolean, got {val}"));
        }
      }
      _ => return Err(format!("unknown option: {key}")),
    }
  }
  Ok(())
}

fn run_repl(diag: Rc<diagnostic::Diagnostic>) -> Result<()> {
  show_message();
  let mut rl = DefaultEditor::new()?;
  let history_path = dirs::home_dir().map(|f| f.join(".qxq_history"));

  rl.set_history_ignore_space(true);
  rl.set_max_history_size(1024)?;

  history_path.as_ref().inspect(|path| {
    if let Ok(true) = path.try_exists() {
      rl.load_history(path).expect("unable to load history");
    }
  });

  let mut config: Value = serde_json::from_str(r#"{"elapsed": false}"#).unwrap();

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
          let mut candidate = config.clone();
          if let Err(e) = ssof::apply_str(&mut candidate, cmd) {
            diag.report(&format!("invalid command: {e}"));
            continue;
          }
          if let Err(e) = validate_repl_config(&candidate) {
            diag.report(&format!("invalid config: {e}"));
            continue;
          }
          config = candidate;
          continue;
        }
        let elapsed_enabled = config["elapsed"].as_bool().unwrap_or(false);
        let start = Instant::now();
        let arena = Bump::new();
        let parser = parser::Parser::new(&arena, Rc::clone(&diag), &line);
        match parser.parse() {
          Ok(tree) => {
            let mut codegen = codegen::CodeGenCtx::new(&arena, Rc::clone(&diag), tree);
            let mut bc = bytecode::BytecodeCtx::new();
            match codegen.emit_tree(&mut bc) {
              Ok(()) => {
                let image = bc.finalize();
                match runtime::execute(image, Rc::clone(&diag)) {
                  Ok(result) => {
                    let duration = start.elapsed();
                    println!("{result}");
                    if elapsed_enabled {
                      color_print::cprintln!(
                        "<dim>elapsed:</dim> <cyan>{:.6}s</cyan>",
                        duration.as_secs_f64()
                      );
                    }
                  }
                  Err(e) => diag.report_err(&e),
                }
              }
              Err(e) => diag.report_err(&e),
            }
            rl.add_history_entry(line.as_str())?;
          }
          Err(e) => diag.report_err(&e),
        }
      }
      Err(ReadlineError::Interrupted) => {
        println!("Interrupted");
        break;
      }
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

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn detect_colon_commands() {
    assert_eq!(detect_colon_command(":+elapsed"), Some("+elapsed"));
    assert_eq!(detect_colon_command(":-elapsed"), Some("-elapsed"));
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
  fn config_enable_disable() {
    let mut config: Value = serde_json::from_str(r#"{"elapsed": false}"#).unwrap();
    ssof::apply_str(&mut config, "+elapsed").unwrap();
    validate_repl_config(&config).unwrap();
    assert_eq!(config["elapsed"], true);

    ssof::apply_str(&mut config, "-elapsed").unwrap();
    validate_repl_config(&config).unwrap();
    assert_eq!(config["elapsed"], false);
  }

  #[test]
  fn config_reject_non_boolean_elapsed() {
    let mut config: Value = serde_json::from_str(r#"{"elapsed": false}"#).unwrap();
    ssof::apply_str(&mut config, "elapsed=42").unwrap();
    assert!(validate_repl_config(&config).is_err());
  }

  #[test]
  fn config_reject_unknown_option() {
    let mut config: Value = serde_json::from_str(r#"{"elapsed": false}"#).unwrap();
    ssof::apply_str(&mut config, "+verbose").unwrap();
    assert!(validate_repl_config(&config).is_err());
  }

  #[test]
  fn config_failed_patch_does_not_mutate() {
    let config: Value = serde_json::from_str(r#"{"elapsed": false}"#).unwrap();
    let mut candidate = config.clone();
    ssof::apply_str(&mut candidate, "+verbose").unwrap();
    let _ = validate_repl_config(&candidate);
    assert_eq!(config["elapsed"], false);
  }
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
