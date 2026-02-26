use bumpalo::Bump;
use clap::{ArgGroup, Parser};
use qxq::diagnostic::Result;
use qxq::*;
use rustyline::{DefaultEditor, config::Configurer, error::ReadlineError};
use std::{
  fs,
  io::{self, IsTerminal, Write},
  rc::Rc,
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

  loop {
    let readline = rl.readline("> ");
    match readline {
      Ok(line) => {
        if line.trim().is_empty() {
          continue;
        }
        let arena = Bump::new();
        let parser = parser::Parser::new(&arena, &line);
        match parser.parse() {
          Ok(tree) => {
            let mut codegen = codegen::CodeGenCtx::new(&arena, tree, Rc::clone(&diag));
            let mut bc = bytecode::BytecodeCtx::new();
            codegen.emit_tree(&mut bc);
            let image = bc.finalize();

            print_thunks(&image);
            rl.add_history_entry(line.as_str())?;
          }
          Err(e) => diag.report_anyhow(&e),
        }
      }
      Err(ReadlineError::Interrupted) => {
        println!("Interrupted");
        break;
      }
      Err(ReadlineError::Eof) => break,
      Err(err) => {
        diag.report_anyhow(&err.into());
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
            diag.enrich(fs::File::open(&file_path), format!("failed to open {}", file_path))?;
          let mut loader = loader::Loader::new(file, Rc::clone(&diag));
          loader.load()?
        };
        print_thunks(&image);
        continue;
      }
      let content =
        diag.enrich(fs::read_to_string(&file_path), format!("failed to read {}", file_path))?;
      let arena = Bump::new();
      let parser = parser::Parser::new(&arena, &content);
      let tree = diag.enrich(parser.parse(), format!("failed to parse {}", file_path))?;

      if cli.dump.is_none() && !cli.no_tree {
        print_tree(&tree);
      }

      let mut codegen = codegen::CodeGenCtx::new(&arena, tree, Rc::clone(&diag));
      let mut bc = bytecode::BytecodeCtx::new();
      codegen.emit_tree(&mut bc);
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
          diag.enrich(fs::write(dump_path, &data), format!("failed to write {}", dump_path))?;
        }
      } else {
        print_thunks(&image);
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
    diag.report_anyhow(&e);
    std::process::exit(1);
  }
}
