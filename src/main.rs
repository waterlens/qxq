use anyhow::Result;
use bumpalo::Bump;
use qxq::*;
use rustyline::{DefaultEditor, config::Configurer, error::ReadlineError};
use std::{fs, path::Path};

fn process_content(content: &str) -> Result<()> {
  let arena = Bump::new();
  let parser = parser::Parser::new(&arena, content);
  let tree = parser.parse()?;
  std::println!("--- Syntax Tree ---");
  std::println!("{tree}");
  let mut codegen = codegen::CodeGenCtx::new(&arena, tree);
  let mut bc = bytecode::BytecodeCtx::new();
  codegen.emit_tree(&mut bc);
  let image = bc.finalize();
  std::println!("--- Thunk ---");
  std::print!(
    "{}",
    image.thunks.iter().map(|t| t.to_string()).collect::<Vec<_>>().join("\n--- Thunk ---\n")
  );
  Ok(())
}

fn process_file<P: AsRef<Path>>(file: P) -> Result<()> {
  let content = fs::read_to_string(file)?;
  process_content(&content)
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

fn run_repl() -> Result<()> {
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

  loop {
    let readline = rl.readline("> ");
    match readline {
      Ok(line) => {
        if line.trim().is_empty() {
          continue;
        }
        if let Err(e) = process_content(&line) {
          eprintln!("Error: {}", e);
        } else {
          rl.add_history_entry(line.as_str())?;
        }
      }
      Err(ReadlineError::Interrupted) => {
        println!("Interrupted");
        break;
      }
      Err(ReadlineError::Eof) => {
        break;
      }
      Err(err) => {
        println!("Error: {:?}", err);
        break;
      }
    }
  }

  history_path.inspect(|path| rl.save_history(path).expect("unable to save history"));
  Ok(())
}

fn main() -> Result<()> {
  let args: Vec<String> = std::env::args().collect();
  if args.len() > 1 {
    if args[1] == "--check-expect" {
      if args.len() < 3 {
        anyhow::bail!("Usage: qxq --check-expect <filename>");
      }
      return expect::run_check(&args[2]);
    }
    if args[1] == "--test-expect" {
      if args.len() < 3 {
        anyhow::bail!("Usage: qxq --test-expect <filename>");
      }
      return expect::run_test_file(&args[2]);
    }
    if args[1] == "--update-expect" {
      if args.len() < 3 {
        anyhow::bail!("Usage: qxq --update-expect <filename> [--skip-multiple-expect]");
      }
      let skip_multi = args.iter().any(|a| a == "--skip-multiple-expect");
      match expect::update_expectations(&args[2], skip_multi)? {
        expect::UpdateStatus::Updated => return Ok(()),
        expect::UpdateStatus::Skipped => std::process::exit(2),
      }
    }
    let mut success = true;
    for arg in &args[1..] {
      if let Err(e) = process_file(arg) {
        eprintln!("Error processing file {}: {}", arg, e);
        success = false;
      }
    }
    if !success {
      std::process::exit(1);
    }
  } else {
    run_repl()?;
  }
  Ok(())
}
