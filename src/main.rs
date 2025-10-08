use anyhow::Result;
use bumpalo::Bump;
use qxq::*;
use rustyline::{error::ReadlineError, DefaultEditor};
use std::{fs, path::Path};

fn process_content(content: &str) -> Result<()> {
  let arena = Bump::new();
  let ctx = codegen::CodeGenCtx::new(&arena);
  let mut parser = parser::Parser::new(&arena, content, ctx);
  let tree = parser.parse()?;
  std::println!("--- Syntax Tree ---");
  std::println!("{tree}");
  std::println!("{}", parser.to_codegen());
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
  loop {
    let readline = rl.readline("> ");
    match readline {
      Ok(line) => {
        let _ = rl.add_history_entry(line.as_str());
        if line.trim().is_empty() {
          continue;
        }
        if let Err(e) = process_content(&line) {
          eprintln!("Error: {}", e);
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
  Ok(())
}

fn main() -> Result<()> {
  let args: Vec<String> = std::env::args().collect();
  if args.len() > 1 {
    for arg in &args[1..] {
      if let Err(e) = process_file(arg) {
        eprintln!("Error processing file {}: {}", arg, e);
      }
    }
  } else {
    run_repl()?;
  }
  Ok(())
}
