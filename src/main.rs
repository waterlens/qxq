use anyhow::Result;
use bumpalo::Bump;
use qxq::*;
use std::{fs, path::Path};

fn process_file<P: AsRef<Path>>(file: P) -> Result<()> {
  let content = fs::read_to_string(file)?;
  let arena = Bump::new();
  let ctx = codegen::CodeGenCtx::new(&arena);
  let mut parser = parser::Parser::new(&arena, &content, ctx);
  let tree = parser.parse()?;
  std::println!("--- Syntax Tree ---");
  std::println!("{tree}");
  std::println!("{}", parser.to_codegen());
  Ok(())
}

fn main() -> Result<()> {
  let args = std::env::args();
  let mut first = true;
  for arg in args {
    if !first {
      process_file(arg)?;
    }
    first = if first { false } else { first };
  }
  Ok(())
}
