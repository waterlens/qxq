use anyhow::Result;
use bumpalo::Bump;
use qxq::*;
use std::{fs, path::Path};

fn process_file<P: AsRef<Path>>(file: P) -> Result<()> {
  let content = fs::read_to_string(file)?;
  let arena = Bump::new();
  let mut parser = parser::Parser::new(&arena, &content);
  let tree = parser.parse()?;
  std::println!("{}", tree);
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
