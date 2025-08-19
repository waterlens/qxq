use bumpalo::Bump;

use crate::tokenizer::TokenStr;

pub struct CodeGenCtx<'a> {
  arena: &'a Bump,
}

impl<'a> CodeGenCtx<'a> {
  pub fn new(arena: &'a Bump) -> Self {
    Self { arena }
  }

  pub fn emit_ident_def(&mut self, name: TokenStr<'a>) {}

  pub fn emit_ident_use(&mut self, name: TokenStr<'a>) {}

  pub fn emit_ident_use_pre(&mut self, name: TokenStr<'a>) {}

  pub fn emit_int_literal(&mut self, n: i128) {}

  pub fn emit_str_literal(&mut self, s: &str) {}

  pub fn emit_tuple_pre(&mut self, n: usize) {}

  // depending on right side. this may be an operator object form or an operator apply form.
  pub fn emit_op_obj_pre(&mut self, op: &str) {}

  pub fn emit_prefix_op(&mut self, op: &str) {}

  pub fn emit_fn(&mut self) {}
}
