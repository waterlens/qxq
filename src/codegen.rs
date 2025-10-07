use std::{
  fmt::{self, Display},
  hint::black_box,
};

use bumpalo::Bump;
use indexmap::IndexMap;
use smallvec::{smallvec, SmallVec};

use crate::{
  bytecode::{Bytecode, BytecodeCtx, Op},
  diagnostic::Diagnostic,
  parser::ExprRef,
  tokenizer::TokenStr,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum ValPos {
  Reg(u8),
  IConst(u16),
  SConst(u16),
  Upvalue { idx: u8, level: u16 },
}

pub struct ConstantPool {
  ipool: IndexMap<i128, usize>,
  spool: IndexMap<String, usize>,
}

impl Default for ConstantPool {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstantPool {
  pub fn new() -> Self {
    Self { ipool: IndexMap::new(), spool: IndexMap::new() }
  }

  pub fn add_int(&mut self, n: i128) -> usize {
    let id = self.ipool.len();
    *self.ipool.entry(n).or_insert(id)
  }

  pub fn add_str(&mut self, s: &str) -> usize {
    let id = self.spool.len();
    *self.spool.entry(s.to_string()).or_insert(id)
  }
}

impl Display for ConstantPool {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "--- Integer Constants ---")?;
    let mut int_vec: Vec<_> = self.ipool.iter().collect();
    int_vec.sort_by_key(|(_, &v)| v);
    for (val, idx) in int_vec {
      writeln!(f, "@{}: {}", idx, val)?;
    }

    writeln!(f, "--- String Constants ---")?;
    let mut str_vec: Vec<_> = self.spool.iter().collect();
    str_vec.sort_by_key(|(_, &v)| v);
    for (val, idx) in str_vec {
      writeln!(f, "@{}: \"{}\"", idx, val)?;
    }

    Ok(())
  }
}

pub struct CodeGenCtx<'a> {
  arena: &'a Bump,
  diagnostic: Diagnostic,
  vstack: Vec<IndexMap<TokenStr<'a>, usize>>,
  ostack: SmallVec<[ValPos; 8]>,
  dstack: SmallVec<[ValPos; 8]>,
  constant_pool: ConstantPool,
  bc: BytecodeCtx,
}

macro_rules! vstack_top {
  ($self:ident) => {
    $self.vstack.last_mut().unwrap()
  };
}

macro_rules! ostack_push {
  ($self:ident, $value:expr) => {
    $self.ostack.push($value)
  };
}

macro_rules! ostack_pop2 {
  ($self:ident) => {{
    let size = $self.ostack.len();
    if size < 2 {
      $self.diagnostic.error("not enough operands on the stack");
    }
    // SAFETY: we know that the stack has at least 2 elements
    unsafe {
      let opr1 = *$self.ostack.get_unchecked(size - 2);
      let opr2 = *$self.ostack.get_unchecked(size - 1);
      $self.ostack.truncate(size - 2);
      (opr1, opr2)
    }
  }};
}

macro_rules! allocate {
  ($self:ident) => {{
    let vstack_top = vstack_top!($self);
    let next_reg = vstack_top.len();
    vstack_top.insert(TokenStr($self.arena.alloc_str(&format!("<tmp{}>", next_reg))), next_reg);
    next_reg as u8
  }};
}

macro_rules! impl_infix_op {
  ($self:ident, $dst:ident, $operands:ident, $opDD:path, $opDC:path) => {
    match $operands {
      (ValPos::Reg(r1), ValPos::Reg(r2)) => {
        $self.bc.push($opDD(Op::xyz($dst, r1, r2)));
        ostack_push!($self, ValPos::Reg($dst));
      }
      (ValPos::Reg(r1), ValPos::IConst(i)) => {
        $self.bc.push($opDC(Op::xyz($dst, r1, i as u16 as u8)));
        ostack_push!($self, ValPos::Reg($dst));
      }
      (ValPos::IConst(i1), ValPos::IConst(i2)) => {
        $self.bc.push(Bytecode::LoadC(Op::ab($dst, i1)));
        $self.bc.push($opDC(Op::xyz($dst, $dst, i2 as u16 as u8)));
        ostack_push!($self, ValPos::Reg($dst));
      }
      (ValPos::IConst(i), ValPos::Reg(r)) => {
        $self.bc.push(Bytecode::LoadC(Op::ab($dst, i)));
        $self.bc.push($opDD(Op::xyz($dst, $dst, r)));
        ostack_push!($self, ValPos::Reg($dst));
      }
      _ => $self.diagnostic.error(&format!("unsupported operands: {:?}", $operands)),
    }
  };
}

impl<'a> CodeGenCtx<'a> {
  fn codegen_action(&self) {
    black_box(0);
  }

  pub fn new(arena: &'a Bump) -> Self {
    let vstack = vec![IndexMap::new()];
    let diagnostic = Diagnostic::new();
    let ostack = smallvec![];
    let dstack = smallvec![];
    let constant_pool = ConstantPool::new();
    let bc = BytecodeCtx::new();
    Self { arena, diagnostic, vstack, ostack, dstack, constant_pool, bc }
  }

  pub fn bytecode_ctx(&self) -> &BytecodeCtx {
    &self.bc
  }

  pub fn emit_param_def(&mut self, name: TokenStr<'a>) {
    self.codegen_action();
    let vstack_top = vstack_top!(self);
    if vstack_top.contains_key(&name) {
      self.diagnostic.error(&format!("parameter {} already defined", name.as_ref()));
    }
    vstack_top.insert(name, vstack_top.len());
  }

  pub fn emit_ident_def(&mut self, name: TokenStr<'a>) {
    self.codegen_action();
    let vstack_top = vstack_top!(self);
    if vstack_top.contains_key(&name) {
      self.diagnostic.error(&format!("variable {} already defined", name.as_ref()));
    }
    vstack_top.insert(name, vstack_top.len());
  }

  pub fn emit_ident_use(&mut self, name: TokenStr<'a>) {
    self.codegen_action();
    let vstack_top = vstack_top!(self);
    if let Some(idx) = vstack_top.get(&name) {
      ostack_push!(self, ValPos::Reg(*idx as u8));
    } else {
      // the value may exist in the outer scope
      // look up the stack for the value
      for (level, vstack) in self.vstack.iter().rev().skip(1).enumerate() {
        if let Some(idx) = vstack.get(&name) {
          ostack_push!(self, ValPos::Upvalue { idx: *idx as u8, level: level as u16 });
          return;
        }
      }
      self.diagnostic.error(&format!("variable {} not defined", name.as_ref()));
    }
  }

  pub fn emit_ident_use_pre(&mut self, name: TokenStr<'a>) {}

  pub fn emit_int_literal(&mut self, n: i128) {
    self.codegen_action();
    let id = self.constant_pool.add_int(n);
    ostack_push!(self, ValPos::IConst(id as u16));
  }

  pub fn emit_str_literal(&mut self, s: &str) {
    self.codegen_action();
    let id = self.constant_pool.add_str(s);
    ostack_push!(self, ValPos::SConst(id as u16));
  }

  pub fn emit_tuple_pre(&mut self, n: usize) {}

  pub fn emit_op_obj(&mut self, op: &str) {}

  pub fn emit_prefix_op_pre(&mut self, op: &str) {}

  pub fn emit_postfix_op(&mut self, op: &str) {}

  pub fn emit_infix_op_pre(&mut self, op: &str) {}

  pub fn emit_infix_op(&mut self, op: &str) {
    use ValPos::*;
    self.codegen_action();
    let dst = match self.dstack.last() {
      Some(Reg(r)) => *r,
      Some(Upvalue { idx, level }) => todo!(),
      None => allocate!(self),
      dst => self.diagnostic.error(&format!("unsupported destination: {dst:?}")),
    };
    let operands = ostack_pop2!(self);
    match op {
      "+" => impl_infix_op!(self, dst, operands, Bytecode::AddDD, Bytecode::AddDC),
      "-" => impl_infix_op!(self, dst, operands, Bytecode::SubDD, Bytecode::SubDC),
      "*" => impl_infix_op!(self, dst, operands, Bytecode::MulDD, Bytecode::MulDC),
      "/" => impl_infix_op!(self, dst, operands, Bytecode::DivDD, Bytecode::DivDC),
      _ => self.diagnostic.error("unsupported operator"),
    }
  }

  pub fn emit_op_apply(&mut self, op: &str, args: &[ExprRef<'a>]) {}

  pub fn emit_fn(&mut self, params: &[ExprRef<'a>]) {}

  pub fn emit_bind(&mut self, is_rec: bool, name: TokenStr<'a>, expr: ExprRef<'a>) {}

  pub fn emit_if(&mut self, c: ExprRef<'a>, t: ExprRef<'a>, f: ExprRef<'a>) {}

  pub fn emit_apply(&mut self, func: ExprRef<'a>, args: &[ExprRef<'a>]) {}
}

impl<'a> Display for CodeGenCtx<'a> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "{}", self.constant_pool)?;
    writeln!(f, "--- Bytecode ---")?;
    writeln!(f, "{}", self.bc)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::parser::Parser;
  fn test_codegen(source: &str, expected_bytecode_str: &str) {
    let arena = Bump::new();
    let ctx = CodeGenCtx::new(&arena);
    let mut parser = Parser::new(&arena, source, ctx);
    parser.parse().unwrap();

    assert_eq!(parser.to_codegen().to_string(), expected_bytecode_str);
  }

  #[test]
  fn test_parse_expressions() {
    test_codegen("1 + 2 * 3 - 4 / 5", "");
  }
}
