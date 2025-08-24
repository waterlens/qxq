use bumpalo::Bump;
use indexmap::IndexMap;
use smallvec::SmallVec;

use crate::{diagnostic::Diagnostic, parser::ExprRef, tokenizer::TokenStr};

pub struct CodeGenCtx<'a> {
  arena: &'a Bump,
  diagnostic: Diagnostic,
  vstack: Vec<IndexMap<TokenStr<'a>, usize>>,
  ostack: SmallVec<[u8; 8]>,
}

macro_rules! vstack_top {
  ($self:ident) => {
    $self.vstack.last_mut().unwrap()
  };
}

impl<'a> CodeGenCtx<'a> {
  pub fn new(arena: &'a Bump) -> Self {
    let vstack = vec![];
    let diagnostic = Diagnostic::new();
    let ostack = SmallVec::new();
    Self { arena, diagnostic, vstack, ostack }
  }

  pub fn emit_param_def(&mut self, name: TokenStr<'a>) {
    let vstack_top = vstack_top!(self);
    if vstack_top.contains_key(&name) {
      self.diagnostic.error(&format!("parameter {} already defined", name.as_ref()));
    }
    vstack_top.insert(name, vstack_top.len());
  }

  pub fn emit_ident_def(&mut self, name: TokenStr<'a>) {
    let vstack_top = vstack_top!(self);
    if vstack_top.contains_key(&name) {
      self.diagnostic.error(&format!("variable {} already defined", name.as_ref()));
    }
    vstack_top.insert(name, vstack_top.len());
  }

  pub fn emit_ident_use(&mut self, name: TokenStr<'a>) {}

  pub fn emit_ident_use_pre(&mut self, name: TokenStr<'a>) {}

  pub fn emit_int_literal(&mut self, n: i128) {}

  pub fn emit_str_literal(&mut self, s: &str) {}

  pub fn emit_tuple_pre(&mut self, n: usize) {}

  pub fn emit_op_obj(&mut self, op: &str) {}

  pub fn emit_prefix_op(&mut self, op: &str) {}

  pub fn emit_postfix_op(&mut self, op: &str) {}

  pub fn emit_infix_op(&mut self, op: &str) {}

  pub fn emit_op_apply(&mut self, op: &str, args: &[ExprRef<'a>]) {}

  pub fn emit_fn(&mut self, params: &[ExprRef<'a>]) {}

  pub fn emit_bind(&mut self, is_rec: bool, name: TokenStr<'a>, expr: ExprRef<'a>) {}

  pub fn emit_if(&mut self, c: ExprRef<'a>, t: ExprRef<'a>, f: ExprRef<'a>) {}

  pub fn emit_apply(&mut self, func: ExprRef<'a>, args: &[ExprRef<'a>]) {}
}
