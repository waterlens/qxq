use bumpalo::Bump;

use crate::{bytecode::*, parser::ExprRef};

pub struct BytecodeGen<'a> {
  arena: Bump,
  tree: ExprRef<'a>,
}

impl<'a> BytecodeGen<'a> {}
