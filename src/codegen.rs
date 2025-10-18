use std::fmt::{self, Display};

use bumpalo::Bump;
use indexmap::IndexMap;

use crate::{
  bytecode::{Bytecode, BytecodeCtx, Label, Op},
  diagnostic::Diagnostic,
  parser::{Expr, ExprRef, ExprsRef, SynTree},
  tokenizer::{Paired, TokenStr},
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

  pub fn add_int(&mut self, n: i128) -> u16 {
    let id = self.ipool.len();
    *self.ipool.entry(n).or_insert(id) as u16
  }

  pub fn add_str(&mut self, s: &str) -> u16 {
    let id = self.spool.len();
    *self.spool.entry(s.to_string()).or_insert(id) as u16
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
enum Control {
  Return(u8),
  Pos(Label),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum Location {
  Slot(u8),
  Upvalue { idx: u8, level: u16 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum Value {
  IConst(u16),
  SConst(u16),
  Loc(Location),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DataDest {
  Effect,
  Loc(Location),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ControlDest {
  Uncond(Control),
  Branch(Control, Control),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Test {
  EqI(u8, u16),
}

struct ValInfo<'a> {
  name: Option<&'a str>,
}

struct Frame<'a> {
  regs: Vec<ValInfo<'a>>,
  symbols: IndexMap<TokenStr<'a>, u8>,
}

struct Stack<'a> {
  frames: Vec<Frame<'a>>,
}

impl<'a> Frame<'a> {
  fn new() -> Self {
    Self { regs: vec![], symbols: IndexMap::new() }
  }
}

impl<'a> Stack<'a> {
  fn new() -> Self {
    Self { frames: vec![Frame::new()] }
  }
}

struct OperandStacks {
  operands: Vec<Value>,
}

impl OperandStacks {
  fn new() -> Self {
    Self { operands: vec![] }
  }
}

pub struct CodeGenCtx<'a> {
  arena: &'a Bump,
  diagnostic: Diagnostic,
  stack_frame: Stack<'a>,
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

macro_rules! allocate_named {
  ($self:ident, $fmt:expr) => {{
    let next_reg = free_reg!($self);
    let symbols_top = symbols_top!($self);
    let name = TokenStr::new($self.arena.alloc_str($fmt(next_reg)));
    symbols_top.insert(name, next_reg);
    reg_push!($self, ValInfo { name: Some(name.0) });
    next_reg as u8
  }};
}

macro_rules! allocate_temporary {
  ($self:ident) => {{
    let next_reg = free_reg!($self);
    reg_push!($self, ValInfo { name: None });
    next_reg as u8
  }};
}

impl<'a> CodeGenCtx<'a> {
  pub fn new(arena: &'a Bump) -> Self {
    let diagnostic = Diagnostic::new();
    let stack_frame = Stack::new();
    let constant_pool = ConstantPool::new();
    let bc = BytecodeCtx::new();
    Self { arena, diagnostic, stack_frame, constant_pool, bc }
  }

  fn get_value(&mut self, opr: Value) -> u8 {
    use Location::*;
    use Value::*;
    match opr {
      Value::IConst(idx) | Value::SConst(idx) => {
        let r = allocate_temporary!(self);
        self.bc.push(Bytecode::LoadC(Op::ab(r, idx)));
        r
      }
      Loc(Slot(r)) => r,
      Loc(Upvalue { idx, level }) => todo!(),
    }
  }

  fn set_location(&mut self, loc: Location, opr: Value) {}

  fn emit_jump(&mut self, l: Control) {
    use Control::*;
    match l {
      Return(r) => self.bc.push(Bytecode::Ret(Op::r#as(r as u32))),
      Pos(l) => {
        self.bc.push_relocate(l);
        self.bc.push(Bytecode::Jmp(Op::a(0)))
      }
    }
  }

  fn emit_test(&mut self, test: Test, c1: Control, c2: Control, next: Control) {
    use Test::*;
    let gen1 = |s: &mut Self, l1: Label, c2: Control| {
      match test {
        EqI(r, imm) => s.bc.push(Bytecode::CmpNeDI(Op::ab(r, imm as u16))),
      }
      s.bc.push_relocate(l1);
      s.bc.push(Bytecode::Jmp(Op::a(0)));
      s.emit_jump(c2);
    };
    let gen2 = |s: &mut Self, c1: Control, l2: Label| {
      match test {
        EqI(r, imm) => s.bc.push(Bytecode::CmpEqDI(Op::ab(r, imm as u16))),
      }
      s.bc.push_relocate(l2);
      s.bc.push(Bytecode::Jmp(Op::a(0)));
      s.emit_jump(c1);
    };
    match (c1, c2) {
      (Control::Pos(l1), Control::Return(_)) => gen1(self, l1, c2),
      (Control::Pos(l1), Control::Pos(l2)) => {
        if c2 == next {
          gen1(self, l1, c2)
        } else {
          gen2(self, c1, l2)
        }
      }
      (Control::Return(_), Control::Pos(l2)) => gen2(self, c1, l2),
      _ => unreachable!("return on both branches"),
    }
  }

  fn emit_store(&mut self, opr: Value, data: DataDest, control: ControlDest, next: Control) {
    use ControlDest::*;
    use DataDest::*;
    match (data, control) {
      (Effect, Uncond(l)) => self.emit_jump(l),
      (Effect, Branch(l1, l2)) => {
        let r = self.get_value(opr);
        self.emit_test(Test::EqI(r, 0), l1, l2, next);
      }
      (Loc(loc), Uncond(l)) => {
        self.set_location(loc, opr);
        self.emit_jump(l);
      }
      (Loc(loc), Branch(l1, l2)) => {
        let r = self.get_value(opr);
        self.set_location(loc, opr);
        self.emit_test(Test::EqI(r, 0), l1, l2, next);
      }
    }
  }

  fn emit_expr(&mut self, expr: ExprRef<'a>, data: DataDest, control: ControlDest, next: Control) {
    use Expr::*;
    match expr {
      IntLiteral(i) => {
        let idx = self.constant_pool.add_int(*i);
        self.emit_store(Value::IConst(idx as u16), data, control, next);
      }
      StrLiteral(s) => {
        let idx = self.constant_pool.add_str(*s);
        self.emit_store(Value::SConst(idx as u16), data, control, next);
      }
      Ident(token_str) => {
        let r = *symbols_top!(self).get(token_str).unwrap_or_else(|| {
          self.diagnostic.error(&format!("undeclared identifier: {}", token_str.0))
        });
        self.emit_store(Value::Loc(Location::Slot(r)), data, control, next);
      }
      Op(token_str) => todo!(),
      OpApply { op, pair, args } => todo!(),
      Apply { func, pair, args } => todo!(),
      Bind { rec, name, expr } => todo!(),
      Fn { params, body } => todo!(),
      Block(exprs) => match exprs {
        [] => todo!("empty block"),
        [expr] => self.emit_expr(expr, data, control, next),
        [exprs @ .., last_expr] => {
          for expr in exprs {
            let l = self.bc.fresh_label();
            let c = Control::Pos(l);
            self.emit_expr(expr, DataDest::Effect, ControlDest::Uncond(c), c);
            self.bc.push_label(l);
          }
          self.emit_expr(last_expr, data, control, next);
        }
      },
      If(c, t, f) => {
        let l1 = self.bc.fresh_label();
        let l2 = self.bc.fresh_label();
        let c1 = Control::Pos(l1);
        let c2 = Control::Pos(l2);
        self.emit_expr(c, DataDest::Effect, ControlDest::Branch(c1, c2), c1);
        self.bc.push_label(l1);
        self.emit_expr(t, data, control, c2);
        self.bc.push_label(l2);
        self.emit_expr(f, data, control, next);
      }
      Tuple(exprs) => todo!(
        
      ),
    }
  }

  pub fn emit_tree(&mut self, tree: &SynTree<'a>) {}
}

impl<'a> Display for CodeGenCtx<'a> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "{}", self.constant_pool)?;
    writeln!(f, "--- Bytecode ---")?;
    write!(f, "{}", self.bc)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::parser::Parser;
  fn test_codegen(source: &str, expected_bytecode_str: &str) {
    let arena = Bump::new();
    let mut parser = Parser::new(&arena, source);
    let mut ctx = CodeGenCtx::new(&arena);
    let tree = parser.parse().unwrap();

    assert_eq!(ctx.to_string(), expected_bytecode_str);
  }

  #[test]
  fn test_parse_expressions() {
    test_codegen("1 + 2 * 3 - 4 / 5", "");
  }
}
