use std::{
  fmt::{self, Display},
  rc::Rc,
};

use bumpalo::Bump;
use indexmap::IndexMap;

use crate::{
  bytecode::{Bytecode, BytecodeCtx, Label, Operands, Operator, UnfinishedBytecode},
  diagnostic::Diagnostic,
  parser::{Expr, ExprRef, ExprsRef, SynTree},
  tokenizer::{Paired, TokenStr},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstantId(u16);

pub struct ConstantPool {
  diagnostic: Rc<Diagnostic>,
  ipool: IndexMap<i128, ConstantId>,
  spool: IndexMap<String, ConstantId>,
}

impl ConstantId {
  pub fn new(id: u16) -> Self {
    Self(id)
  }
}

impl Display for ConstantId {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "@{}", self.0)
  }
}

impl From<u16> for ConstantId {
  fn from(id: u16) -> Self {
    ConstantId::new(id)
  }
}

impl TryFrom<usize> for ConstantId {
  type Error = ();
  fn try_from(value: usize) -> Result<Self, Self::Error> {
    if value < u16::MAX as usize {
      Ok(ConstantId::new(value as u16))
    } else {
      Err(())
    }
  }
}

impl ConstantPool {
  pub fn new(diagnostic: Rc<Diagnostic>) -> Self {
    Self { diagnostic, ipool: IndexMap::new(), spool: IndexMap::new() }
  }

  pub fn add_int(&mut self, n: i128) -> ConstantId {
    let id = self.ipool.len();
    let id = id.try_into().unwrap_or_else(|_| self.diagnostic.error("constant id overflow"));
    *self.ipool.entry(n).or_insert(id)
  }

  pub fn add_str(&mut self, s: &str) -> ConstantId {
    let id = self.spool.len();
    let id = id.try_into().unwrap_or_else(|_| self.diagnostic.error("constant id overflow"));
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
pub struct RegId(u8);

impl RegId {
  pub fn new(id: u8) -> Self {
    Self(id)
  }
}

impl From<u8> for RegId {
  fn from(id: u8) -> Self {
    RegId::new(id)
  }
}

impl From<RegId> for crate::bytecode::Op8 {
  fn from(id: RegId) -> Self {
    id.0.into()
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Control {
  Return(RegId),
  Pos(Label),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum Location {
  Temporary,
  Slot(RegId),
  Upvalue { idx: u8, level: u16 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum Value {
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
  EqI(RegId, u16),
}

struct ValInfo<'a> {
  name: Option<&'a str>,
  index: RegId,
}

struct Frame<'a> {
  regs: Vec<ValInfo<'a>>,
  symbols: IndexMap<TokenStr<'a>, RegId>,
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

pub struct CodeGenCtx<'a> {
  arena: &'a Bump,
  diagnostic: Rc<Diagnostic>,
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

macro_rules! reg_top {
  ($self:ident) => {
    $self.stack_frame.frames.last().unwrap().regs.last().unwrap()
  };
}

macro_rules! reg_pop {
  ($self:ident) => {
    $self.stack_frame.frames.last_mut().unwrap().regs.pop().unwrap()
  };
}

impl<'a> CodeGenCtx<'a> {
  pub fn new(arena: &'a Bump) -> Self {
    let diagnostic = Rc::new(Diagnostic::new());
    let stack_frame = Stack::new();
    let constant_pool = ConstantPool::new(diagnostic.clone());
    let bc = BytecodeCtx::new();
    Self { arena, diagnostic, stack_frame, constant_pool, bc }
  }

  fn allocate_temporary(&mut self) -> RegId {
    let next_reg = free_reg!(self);
    if next_reg >= u8::MAX as usize {
      self.diagnostic.error("register id overflow");
    }
    reg_push!(self, ValInfo { name: None, index: (next_reg as u8).into() });
    (next_reg as u8).into()
  }

  fn allocate_named(&mut self, name: &TokenStr<'a>) -> RegId {
    let next_reg = free_reg!(self);
    if next_reg >= u8::MAX as usize {
      self.diagnostic.error("register id overflow");
    }
    let symbols_top = symbols_top!(self);
    if symbols_top.contains_key(name) {
      self.diagnostic.error(&format!("redefined variable {}", name.0))
    }
    symbols_top.insert(*name, (next_reg as u8).into());
    reg_push!(self, ValInfo { name: Some(name.0), index: (next_reg as u8).into() });
    (next_reg as u8).into()
  }

  fn get_value(&mut self, opr: Value) -> RegId {
    use Location::*;
    use Value::*;
    match opr {
      Loc(Slot(r)) => r,
      Loc(Temporary) => {
        let r = reg_top!(self).index;
        reg_pop!(self);
        r
      }
      Loc(Upvalue { idx, level }) => todo!(),
    }
  }

  fn set_location(&mut self, loc: Location, opr: Value) {
    use Location::*;
    match (loc, opr) {
      (Temporary, Value::Loc(Temporary)) => (),
      (Temporary, Value::Loc(Upvalue { idx, level })) => todo!(),
      (Temporary, Value::Loc(Slot(r))) => {
        let r2 = self.allocate_temporary();
        self.bc.push(Bytecode::mov(r.into(), r2.into()));
      }
      (Slot(r), Value::Loc(Slot(r2))) if r == r2 => (),
      (Slot(r), _) => {
        let r2 = self.get_value(opr);
        self.bc.push(Bytecode::mov(r.into(), r2.into()));
      }
      (Upvalue { idx, level }, _) => todo!(),
    }
  }

  fn emit_jump(&mut self, l: Control) {
    use Control::*;
    match l {
      Return(r) => self.bc.push(Bytecode::retn(r.into(), 0.into())),
      Pos(l) => {
        self.bc.push_relocate(l);
        self.bc.push(Bytecode::jmp(0u16.into()))
      }
    }
  }

  fn emit_test(&mut self, test: Test, c1: Control, c2: Control, next: Control) {
    use Test::*;
    let gen1 = |s: &mut Self, l1: Label, c2: Control| {
      match test {
        EqI(r, imm) => s.bc.push(Bytecode::cmpnedi(r.into(), imm.into())),
      }
      s.bc.push_relocate(l1);
      s.bc.push(Bytecode::jmp(0u16.into()));
      s.emit_jump(c2);
    };
    let gen2 = |s: &mut Self, c1: Control, l2: Label| {
      match test {
        EqI(r, imm) => s.bc.push(Bytecode::cmpeqdi(r.into(), imm.into())),
      }
      s.bc.push_relocate(l2);
      s.bc.push(Bytecode::jmp(0u16.into()));
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
      (Effect, Uncond(c)) => {
        if c != next {
          self.emit_jump(c);
        }
      }
      (Effect, Branch(l1, l2)) => {
        let r = self.get_value(opr);
        self.emit_test(Test::EqI(r, 0), l1, l2, next);
      }
      (Loc(loc), Uncond(c)) => {
        self.set_location(loc, opr);
        if c != next {
          self.emit_jump(c);
        }
      }
      (Loc(loc), Branch(l1, l2)) => {
        let r = self.get_value(opr);
        self.set_location(loc, opr);
        self.emit_test(Test::EqI(r, 0), l1, l2, next);
      }
    }
  }

  fn emit_binary_op_with_slots(
    &mut self,
    op: &'a str,
    opr1: RegId,
    opr2: RegId,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) {
    let ubc = match op {
      "+" => Operator::AddDD,
      "-" => Operator::SubDD,
      "*" => Operator::MulDD,
      "/" => Operator::DivDD,
      _ => unreachable!("unknown binary operator: {}", op),
    };
    match data {
      DataDest::Loc(Location::Slot(r)) => {
        self.bc.push(UnfinishedBytecode::new(ubc).fill_operands(Operands::ABC(Operands::abc(
          r.into(),
          opr1.into(),
          opr2.into(),
        ))));
        self.emit_store(Value::Loc(Location::Slot(r)), data, control, next);
      }
      DataDest::Effect | DataDest::Loc(Location::Temporary) => {
        let r = self.allocate_temporary();
        self.bc.push(UnfinishedBytecode::new(ubc).fill_operands(Operands::XYZ(Operands::xyz(
          r.into(),
          opr1.into(),
          opr2.into(),
        ))));
        self.emit_store(Value::Loc(Location::Temporary), data, control, next);
      }
      DataDest::Loc(Location::Upvalue { idx, level }) => {}
    }
  }

  fn emit_op(
    &mut self,
    op: ExprRef<'a>,
    _pair: Option<Paired>,
    args: ExprsRef<'a>,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) {
    if let Expr::Op(op_str) = op {
      match op_str.0 {
        "+" => {
          if args.len() == 2 {
            let l1 = self.bc.fresh_label();
            self.emit_expr(
              args[0],
              DataDest::Loc(Location::Temporary),
              ControlDest::Uncond(Control::Pos(l1)),
              Control::Pos(l1),
            );
            self.bc.push_label(l1);
            let l2 = self.bc.fresh_label();
            self.emit_expr(
              args[1],
              DataDest::Loc(Location::Temporary),
              ControlDest::Uncond(Control::Pos(l2)),
              Control::Pos(l2),
            );
            self.bc.push_label(l2);
            let r2 = self.get_value(Value::Loc(Location::Temporary));
            let r1 = self.get_value(Value::Loc(Location::Temporary));
            self.emit_binary_op_with_slots("+", r1, r2, data, control, next);
          } else {
            self.diagnostic.error("expected two arguments for addition");
          }
        }
        "-" => {
          if args.len() == 2 {
            let l1 = self.bc.fresh_label();
            self.emit_expr(
              args[0],
              DataDest::Loc(Location::Temporary),
              ControlDest::Uncond(Control::Pos(l1)),
              Control::Pos(l1),
            );
            self.bc.push_label(l1);
            let l2 = self.bc.fresh_label();
            self.emit_expr(
              args[1],
              DataDest::Loc(Location::Temporary),
              ControlDest::Uncond(Control::Pos(l2)),
              Control::Pos(l2),
            );
            self.bc.push_label(l2);
            let r2 = self.get_value(Value::Loc(Location::Temporary));
            let r1 = self.get_value(Value::Loc(Location::Temporary));
            self.emit_binary_op_with_slots("-", r1, r2, data, control, next);
          } else {
            self.diagnostic.error("expected two arguments for subtraction");
          }
        }
        "*" => {
          if args.len() == 2 {
            let l1 = self.bc.fresh_label();
            self.emit_expr(
              args[0],
              DataDest::Loc(Location::Temporary),
              ControlDest::Uncond(Control::Pos(l1)),
              Control::Pos(l1),
            );
            self.bc.push_label(l1);
            let l2 = self.bc.fresh_label();
            self.emit_expr(
              args[1],
              DataDest::Loc(Location::Temporary),
              ControlDest::Uncond(Control::Pos(l2)),
              Control::Pos(l2),
            );
            self.bc.push_label(l2);
            let r2 = self.get_value(Value::Loc(Location::Temporary));
            let r1 = self.get_value(Value::Loc(Location::Temporary));
            self.emit_binary_op_with_slots("*", r1, r2, data, control, next);
          } else {
            self.diagnostic.error("expected two arguments for multiplication");
          }
        }
        "/" => {
          if args.len() == 2 {
            let l1 = self.bc.fresh_label();
            self.emit_expr(
              args[0],
              DataDest::Loc(Location::Temporary),
              ControlDest::Uncond(Control::Pos(l1)),
              Control::Pos(l1),
            );
            self.bc.push_label(l1);
            let l2 = self.bc.fresh_label();
            self.emit_expr(
              args[1],
              DataDest::Loc(Location::Temporary),
              ControlDest::Uncond(Control::Pos(l2)),
              Control::Pos(l2),
            );
            self.bc.push_label(l2);
            let r2 = self.get_value(Value::Loc(Location::Temporary));
            let r1 = self.get_value(Value::Loc(Location::Temporary));
            self.emit_binary_op_with_slots("/", r1, r2, data, control, next);
          } else {
            self.diagnostic.error("expected two arguments for division");
          }
        }
        _ => self.diagnostic.error(&format!("unknown operator: {}", op_str.0)),
      }
    } else {
      self.diagnostic.error("expected operator");
    }
  }

  fn emit_expr(&mut self, expr: ExprRef<'a>, data: DataDest, control: ControlDest, next: Control) {
    use Expr::*;
    match expr {
      IntLiteral(i) => {
        let idx = self.constant_pool.add_int(*i);
        let r = self.allocate_temporary();
        self.bc.push(Bytecode::loadc(r.into(), idx.0.into()));
        self.emit_store(Value::Loc(Location::Temporary), data, control, next);
      }
      StrLiteral(s) => {
        let idx = self.constant_pool.add_str(s);
        let r = self.allocate_temporary();
        self.bc.push(Bytecode::loadc(r.into(), idx.0.into()));
        self.emit_store(Value::Loc(Location::Temporary), data, control, next);
      }
      Ident(token_str) => {
        let r = *symbols_top!(self).get(token_str).unwrap_or_else(|| {
          self.diagnostic.error(&format!("undeclared identifier: {}", token_str.0))
        });
        self.emit_store(Value::Loc(Location::Slot(r)), data, control, next);
      }
      Op(token_str) => todo!(),
      OpApply { op, pair, args } => self.emit_op(op, *pair, args, data, control, next),
      Apply { func, pair, args } => todo!(),
      Bind { rec, name, expr } => {
        let r = self.allocate_named(name);
        self.emit_expr(expr, DataDest::Loc(Location::Slot(r)), ControlDest::Uncond(next), next);
      }
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
        self.emit_expr(t, data, control, next);
        self.bc.push_label(l2);
        self.emit_expr(f, data, control, next);
      }
      Tuple(exprs) => {
        let mut elems_regs = Vec::with_capacity(exprs.len());
        elems_regs.fill_with(|| self.allocate_temporary());
        for (elem, r) in (*exprs).iter().zip(elems_regs.into_iter()) {
          let l = self.bc.fresh_label();
          let c = Control::Pos(l);
          self.emit_expr(elem, DataDest::Loc(Location::Slot(r)), ControlDest::Uncond(c), c);
          self.bc.push_label(l);
        }
        self.bc.push(Bytecode::nop()); // MAKE TUPLE
        match control {
          ControlDest::Uncond(l) => self.emit_jump(l),
          _ => self.diagnostic.error("tuple in conditional expression"),
        }
      }
    }
  }

  pub fn emit_tree(&mut self, tree: &SynTree<'a>) {
    self.emit_expr(
      tree.root,
      DataDest::Effect,
      ControlDest::Uncond(Control::Return(0.into())),
      Control::Return(0.into()),
    );
  }
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
  #[allow(unused)]
  fn test_codegen(source: &str, expected_bytecode_str: &str) {
    let arena = Bump::new();
    let mut parser = Parser::new(&arena, source);
    let mut ctx = CodeGenCtx::new(&arena);
    let tree = parser.parse().unwrap();
    ctx.emit_tree(&tree);

    assert_eq!(ctx.to_string(), expected_bytecode_str);
  }
}
