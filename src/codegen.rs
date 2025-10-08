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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum ValDesc {
  Slot(u8),
  Imm8(i8),
  IConst(u16),
  SConst(u16),
  Upvalue { idx: u8, level: u16 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum DestDesc {
  UnusedSlot(u8),
  ClaimedSlot(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum ValKind<'a> {
  Named(TokenStr<'a>),
  Temporary,
}

struct ValInfo<'a> {
  kind: ValKind<'a>,
}

struct Frame<'a> {
  regs: SmallVec<[ValInfo<'a>; 8]>,
  symbols: IndexMap<TokenStr<'a>, usize>,
}

struct Stack<'a> {
  frames: Vec<Frame<'a>>,
}

impl<'a> Frame<'a> {
  fn new() -> Self {
    Self { regs: smallvec![], symbols: IndexMap::new() }
  }
}

impl<'a> Stack<'a> {
  fn new() -> Self {
    Self { frames: vec![Frame::new()] }
  }
}

struct OperandStacks {
  operands: SmallVec<[ValDesc; 8]>,
  dest_hints: SmallVec<[DestDesc; 8]>,
}

impl OperandStacks {
  fn new() -> Self {
    Self { operands: smallvec![], dest_hints: smallvec![] }
  }
}

pub struct CodeGenCtx<'a> {
  arena: &'a Bump,
  diagnostic: Diagnostic,
  stack_frame: Stack<'a>,
  operand_stacks: OperandStacks,
  constant_pool: ConstantPool,
  bc: BytecodeCtx,
}

macro_rules! frame_top {
  ($self:ident) => {
    &mut $self.stack_frame.frames.last_mut().unwrap()
  };
}

macro_rules! symbols_top {
  ($self:ident) => {
    &mut frame_top!($self).symbols
  };
}

macro_rules! free_reg {
  ($self:ident) => {
    $self.stack_frame.frames.last().unwrap().regs.len()
  };
}

macro_rules! operand_push {
  ($self:ident, $value:expr) => {
    $self.operand_stacks.operands.push($value)
  };
}

macro_rules! dest_push {
  ($self:ident, $value:expr) => {
    $self.operand_stacks.dest_hints.push($value)
  };
}

macro_rules! dest_pop {
  ($self:ident) => {
    $self.operand_stacks.dest_hints.pop()
  };
}

macro_rules! dest_peek {
  ($self:ident) => {
    $self.operand_stacks.dest_hints.last()
  };
}

macro_rules! reg_push {
  ($self:ident, $value:expr) => {
    frame_top!($self).regs.push($value)
  };
}

macro_rules! reg_pop {
  ($self:ident) => {
    frame_top!($self).regs.pop()
  };
}

macro_rules! reg_top {
  ($self:ident) => {
    frame_top!($self).regs.last()
  };
}

macro_rules! operand_pop {
  ($self:ident) => {
    $self.operand_stacks.operands.pop()
  };
}

macro_rules! operand_pop2 {
  ($self:ident) => {{
    let size = $self.operand_stacks.operands.len();
    if size < 2 {
      $self.diagnostic.error("not enough operands on the stack");
    }
    // SAFETY: we know that the stack has at least 2 elements
    unsafe {
      let opr1 = *$self.operand_stacks.operands.get_unchecked(size - 2);
      let opr2 = *$self.operand_stacks.operands.get_unchecked(size - 1);
      $self.operand_stacks.operands.truncate(size - 2);
      (opr1, opr2)
    }
  }};
}

macro_rules! allocate_named {
  ($self:ident, $fmt:expr) => {{
    let next_reg = free_reg!($self);
    let symbols_top = symbols_top!($self);
    let name = TokenStr::new($self.arena.alloc_str($fmt(next_reg)));
    symbols_top.insert(name, next_reg);
    reg_push!($self, ValInfo { kind: ValKind::Named(name) });
    next_reg as u8
  }};
}

macro_rules! allocate_temporary {
  ($self:ident) => {{
    let next_reg = free_reg!($self);
    reg_push!($self, ValInfo { kind: ValKind::Temporary });
    next_reg as u8
  }};
}

macro_rules! reify_temporary {
  ($self:ident, $name:expr, $reg:expr) => {{
    let symbols_top = symbols_top!($self);
    symbols_top.insert($name, $reg);
  }};
}

macro_rules! deallocate_temporary {
  ($self:ident, $reg:expr) => {{
    if $reg as usize == free_reg!($self) - 1 {
      if let Some(reg) = reg_top!($self) {
        if reg.kind == ValKind::Temporary {
          reg_pop!($self);
        }
      }
    }
  }};
}

macro_rules! impl_infix_op {
  ($self:ident, $operands:ident, ($($opDD:tt)*), ($($opDI:tt)*)) => {{
    let dst;
    use DestDesc::*;
    let mut claim_dest = None;
    match $operands {
      (ValDesc::Slot(r1), ValDesc::Slot(r2)) => {
        match dest_peek!($self) {
          Some(UnusedSlot(r)) => {
            dst = *r;
            $self.bc.push($($opDD)*(Op::xyz(dst, r1, r2)));
            deallocate_temporary!($self, r1);
            deallocate_temporary!($self, r2);
            claim_dest = Some(*r);
          },
          Some(ClaimedSlot(_)) | None => {
            dst = std::cmp::min(r1, r2);
            $self.bc.push($($opDD)*(Op::xyz(dst, r1, r2)));
            deallocate_temporary!($self, std::cmp::max(r1, r2));
          }
        }
      }
      (ValDesc::Slot(r1), ValDesc::IConst(i)) => {
        match dest_peek!($self) {
          Some(UnusedSlot(r)) => {
            dst = *r;
            deallocate_temporary!($self, r1);
            claim_dest = Some(*r);
          }
          Some(ClaimedSlot(_)) | None => { dst = r1; }
        }
        $self.bc.push(Bytecode::LoadC(Op::ab(dst, i)));
        $self.bc.push($($opDD)*(Op::xyz(dst, r1, dst)));
      }
      (ValDesc::IConst(i1), ValDesc::IConst(i2)) => {
        match dest_peek!($self) {
          Some(UnusedSlot(r)) => {
            dst = *r;
            claim_dest = Some(*r);
          }
          Some(ClaimedSlot(_)) | None => dst = allocate_temporary!($self),
        }
        let dst2 = allocate_temporary!($self);
        $self.bc.push(Bytecode::LoadC(Op::ab(dst, i1)));
        $self.bc.push(Bytecode::LoadC(Op::ab(dst2, i2)));
        $self.bc.push($($opDD)*(Op::xyz(dst, dst, dst2)));
        deallocate_temporary!($self, dst2);
      }
      (ValDesc::IConst(i), ValDesc::Slot(r1)) => {
        match dest_peek!($self) {
          Some(UnusedSlot(r)) => {
            dst = *r;
            claim_dest = Some(*r);
          }
          Some(ClaimedSlot(_)) | None => {
            dst = allocate_temporary!($self);
          }
        }
        $self.bc.push(Bytecode::LoadC(Op::ab(dst, i)));
        $self.bc.push($($opDD)*(Op::xyz(dst, dst, r1)));
        deallocate_temporary!($self, r1);
      }
      _ => $self.diagnostic.error(&format!("unsupported operands: {:?}", $operands)),
    }
    if let Some(r) = claim_dest {
      dest_pop!($self);
      dest_push!($self, DestDesc::ClaimedSlot(r));
    }
    operand_push!($self, ValDesc::Slot(dst));
  }};
}

impl<'a> CodeGenCtx<'a> {
  fn codegen_action(&self) {
    black_box(0);
  }

  pub fn new(arena: &'a Bump) -> Self {
    let diagnostic = Diagnostic::new();
    let stack_frame = Stack::new();
    let operand_stacks = OperandStacks::new();
    let constant_pool = ConstantPool::new();
    let bc = BytecodeCtx::new();
    Self { arena, diagnostic, stack_frame, operand_stacks, constant_pool, bc }
  }

  pub fn bytecode_ctx(&self) -> &BytecodeCtx {
    &self.bc
  }

  pub fn emit_param_def(&mut self, name: TokenStr<'a>) {
    self.codegen_action();
    let symbols_top = symbols_top!(self);
    if symbols_top.contains_key(&name) {
      self.diagnostic.error(&format!("parameter {} already defined", name.as_ref()));
    }
    symbols_top.insert(name, symbols_top.len());
  }

  pub fn emit_ident_def(&mut self, name: TokenStr<'a>) {
    self.codegen_action();
    let symbols_top = symbols_top!(self);
    if symbols_top.contains_key(&name) {
      self.diagnostic.error(&format!("variable {} already defined", name.as_ref()));
    }
    symbols_top.insert(name, symbols_top.len());
  }

  pub fn emit_ident_use(&mut self, name: TokenStr<'a>) {
    self.codegen_action();
    let symbols_top = symbols_top!(self);
    if let Some(idx) = symbols_top.get(&name) {
      operand_push!(self, ValDesc::Slot(*idx as u8));
    } else {
      // the value may exist in the outer scope
      // look up the stack for the value
      for (level, frame) in self.stack_frame.frames.iter().rev().skip(1).enumerate() {
        if let Some(idx) = frame.symbols.get(&name) {
          operand_push!(self, ValDesc::Upvalue { idx: *idx as u8, level: level as u16 });
          return;
        }
      }
      self.diagnostic.error(&format!("variable {} not defined", name.as_ref()));
    }
  }

  pub fn emit_int_literal(&mut self, n: i128) {
    self.codegen_action();
    let id = self.constant_pool.add_int(n);
    operand_push!(self, ValDesc::IConst(id as u16));
  }

  pub fn emit_str_literal(&mut self, s: &str) {
    self.codegen_action();
    let id = self.constant_pool.add_str(s);
    operand_push!(self, ValDesc::SConst(id as u16));
  }

  pub fn emit_tuple_pre(&mut self, n: usize) {}

  pub fn emit_op_obj(&mut self, op: &str) {}

  pub fn emit_prefix_op_pre(&mut self, op: &str) {}

  pub fn emit_postfix_op(&mut self, op: &str) {}

  pub fn emit_infix_op_pre(&mut self, op: &str) {}

  pub fn emit_infix_op(&mut self, op: &str) {
    self.codegen_action();
    let operands = operand_pop2!(self);
    match op {
      "+" => impl_infix_op!(self, operands, (Bytecode::AddDD), (Bytecode::AddDI)),
      "-" => impl_infix_op!(self, operands, (Bytecode::SubDD), (Bytecode::SubDI)),
      "*" => impl_infix_op!(self, operands, (Bytecode::MulDD), (Bytecode::MulDI)),
      "/" => impl_infix_op!(self, operands, (Bytecode::DivDD), (Bytecode::DivDI)),
      _ => self.diagnostic.error("unsupported operator"),
    }
  }

  pub fn emit_op_apply(&mut self, op: &str) {}

  pub fn emit_fn(&mut self, params: &[ExprRef<'a>]) {}

  pub fn emit_binder(&mut self, is_rec: bool, name: TokenStr<'a>) {
    self.codegen_action();
    let dst =
      if is_rec { allocate_named!(self, |_| name.as_ref()) } else { allocate_temporary!(self) };
    dest_push!(self, DestDesc::UnusedSlot(dst));
  }

  pub fn emit_binding_end(&mut self, is_rec: bool, name: TokenStr<'a>) {
    self.codegen_action();

    match dest_pop!(self) {
      Some(DestDesc::ClaimedSlot(dst)) => {
        if is_rec {
          reify_temporary!(self, name, dst as usize);
        }
      }
      Some(DestDesc::UnusedSlot(dst)) => {
        deallocate_temporary!(self, dst as usize);
      }
      None => unreachable!("at least one destination hint must be present"),
    }
  }

  pub fn emit_if_cond(&mut self) {}

  pub fn emit_if_then(&mut self) {}

  pub fn emit_if_else(&mut self) {}

  pub fn emit_if_done(&mut self) {}

  pub fn emit_apply(&mut self, func: ExprRef<'a>) {}
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
