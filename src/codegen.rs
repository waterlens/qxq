use std::{
  fmt::{self, Display},
  rc::Rc,
};

use bumpalo::Bump;
use indexmap::{IndexMap, IndexSet};
use slotmap::SlotMap;

use crate::{
  bytecode::{Bytecode, BytecodeCtx, Label},
  diagnostic::Diagnostic,
  parser::{Expr, ExprRef, ExprsRef, Info, InfoKey, SynTree},
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
      writeln!(f, "{}: {}", idx, val)?;
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
pub struct FreeVarId(u16);

impl FreeVarId {
  pub fn new(id: u16) -> Self {
    Self(id)
  }
}

impl From<u16> for FreeVarId {
  fn from(id: u16) -> Self {
    FreeVarId::new(id)
  }
}

impl From<FreeVarId> for crate::bytecode::Op16 {
  fn from(id: FreeVarId) -> Self {
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
  FreeVar(FreeVarId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum Value<'a> {
  Loc(Location),
  IntLiteral(i128),
  StrLiteral(&'a str),
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
}

struct Stack<'a> {
  frames: Vec<Frame<'a>>,
}

impl<'a> Frame<'a> {
  fn new() -> Self {
    Self { regs: vec![] }
  }
}

impl<'a> Stack<'a> {
  fn new() -> Self {
    Self { frames: vec![Frame::new()] }
  }
}

struct Scope<'a> {
  diagnostic: Rc<Diagnostic>,
  symbols: IndexMap<TokenStr<'a>, Vec<Location>>,
  bound: Vec<IndexSet<TokenStr<'a>>>,
}

impl<'a> Scope<'a> {
  fn new(diagnostic: Rc<Diagnostic>) -> Self {
    Self { diagnostic, symbols: IndexMap::new(), bound: vec![] }
  }

  fn enter(&mut self) {
    self.bound.push(IndexSet::new());
  }

  fn enter_function(&mut self, freevars: &[TokenStr<'a>]) {
    for (i, name) in freevars.iter().enumerate() {
      let i: u16 =
        i.try_into().unwrap_or_else(|_| self.diagnostic.error("free variable id overflow"));
      self.symbols.entry(*name).or_insert(vec![]).push(Location::FreeVar(i.into()));
    }
    let freevars = IndexSet::from_iter(freevars.iter().cloned());
    self.bound.push(freevars);
  }

  fn leave(&mut self) {
    let bound_vars = self.bound.pop().expect("bound stack underflow: check if enter was called");
    for var in bound_vars {
      self
        .symbols
        .get_mut(&var)
        .expect("bound variable not found in the map")
        .pop()
        .expect("bound variable has no bindings");
    }
  }

  fn insert_slot(&mut self, name: &TokenStr<'a>, reg: RegId) {
    self.symbols.entry(*name).or_insert(vec![]).push(Location::Slot(reg));
    self.bound.last_mut().unwrap().insert(*name);
  }

  fn get_bound_in_nth_nested_scope(&self, name: &TokenStr<'a>, n: usize) -> Option<Location> {
    let i = self.bound.len() as isize - n as isize - 1;
    if i >= 0 && self.bound[i as usize].contains(name) {
      self.symbols.get(name).and_then(|locs| locs.last().copied())
    } else {
      None
    }
  }

  fn get_bound(&self, name: &TokenStr<'a>) -> Option<Location> {
    self.get_bound_in_nth_nested_scope(name, 0)
  }
}

pub struct CodeGenCtx<'a> {
  diagnostic: Rc<Diagnostic>,
  stack_frame: Stack<'a>,
  scope: Scope<'a>,
  pub constant_pool: ConstantPool,
  tree: ExprRef<'a, InfoKey>,
  information: SlotMap<InfoKey, Info<'a>>,
}

macro_rules! frame_top {
  ($self:ident) => {
    &mut $self.stack_frame.frames.last_mut().unwrap()
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
  pub fn new(_arena: &'a Bump, tree: SynTree<'a, InfoKey>) -> Self {
    let diagnostic = Rc::new(Diagnostic::new());
    let stack_frame = Stack::new();
    let scope = Scope::new(Rc::clone(&diagnostic));
    let constant_pool = ConstantPool::new(Rc::clone(&diagnostic));
    let information = tree.information;
    let tree = tree.root;
    Self { diagnostic, stack_frame, scope, constant_pool, tree, information }
  }

  fn allocate_temporary(&mut self) -> RegId {
    let next_reg = free_reg!(self);
    if next_reg >= u8::MAX as usize {
      self.diagnostic.error("register id overflow");
    }
    reg_push!(self, ValInfo { name: None, index: (next_reg as u8).into() });
    (next_reg as u8).into()
  }

  fn enter_new_frame(&mut self) {
    self.stack_frame.frames.push(Frame::new());
  }

  fn leave_frame(&mut self) {
    self.stack_frame.frames.pop().unwrap();
  }

  fn update_symbols(&mut self, name: &TokenStr<'a>, reg: RegId) {
    self.scope.insert_slot(name, reg);
  }

  fn allocate_named(&mut self, name: &TokenStr<'a>) -> RegId {
    let next_reg = free_reg!(self);
    if next_reg >= u8::MAX as usize {
      self.diagnostic.error("register id overflow");
    }
    reg_push!(self, ValInfo { name: Some(name.0), index: (next_reg as u8).into() });
    (next_reg as u8).into()
  }

  fn get_temporary(&mut self) -> RegId {
    let val_info = reg_top!(self);
    if val_info.name.is_some() {
      self.diagnostic.error("no temporary variable on the stack");
    }
    let r = val_info.index;
    reg_pop!(self);
    r
  }

  fn reify_int_literal(&mut self, bc: &mut BytecodeCtx, r: RegId, i: i128) {
    let idx = self.constant_pool.add_int(i);
    bc.push(Bytecode::loadc(r.into(), idx.0.into()));
  }

  fn reify_string_literal(&mut self, bc: &mut BytecodeCtx, r: RegId, s: &str) {
    let idx = self.constant_pool.add_str(s);
    bc.push(Bytecode::loadc(r.into(), idx.0.into()));
  }

  fn reify_freevar(&mut self, bc: &mut BytecodeCtx, r: RegId, i: FreeVarId) {
    bc.push(Bytecode::loadf(r.into(), i.0.into()));
  }

  fn get_value(&mut self, bc: &mut BytecodeCtx, opr: Value) -> RegId {
    use Location::*;
    use Value::*;
    match opr {
      Loc(Slot(r)) => r,
      Loc(FreeVar(i)) => {
        let r = self.allocate_temporary();
        self.reify_freevar(bc, r, i);
        self.get_temporary()
      }
      Loc(Temporary) => self.get_temporary(),
      IntLiteral(i) => {
        let r = self.allocate_temporary();
        self.reify_int_literal(bc, r, i);
        self.get_temporary()
      }
      StrLiteral(s) => {
        let r = self.allocate_temporary();
        self.reify_string_literal(bc, r, s);
        self.get_temporary()
      }
    }
  }

  fn set_location(&mut self, bc: &mut BytecodeCtx, loc: Location, opr: Value) {
    use Location::*;
    match (loc, opr) {
      (Temporary, Value::Loc(Temporary)) => (),
      (Temporary, Value::Loc(Slot(r))) => {
        let r2 = self.allocate_temporary();
        bc.push(Bytecode::mov(r2.into(), r.into()));
      }
      (Temporary, Value::IntLiteral(i)) => {
        let r = self.allocate_temporary();
        self.reify_int_literal(bc, r, i);
      }
      (Temporary, Value::StrLiteral(i)) => {
        let r = self.allocate_temporary();
        self.reify_string_literal(bc, r, i);
      }
      (Temporary, Value::Loc(FreeVar(i))) => {
        let r = self.allocate_temporary();
        self.reify_freevar(bc, r, i);
      }
      (Slot(r), Value::Loc(Slot(r2))) if r == r2 => (),
      (Slot(r), Value::Loc(Slot(r2))) => {
        bc.push(Bytecode::mov(r.into(), r2.into()));
      }
      (Slot(r), Value::IntLiteral(i)) => self.reify_int_literal(bc, r, i),
      (Slot(r), Value::StrLiteral(i)) => self.reify_string_literal(bc, r, i),
      (Slot(r), Value::Loc(Temporary)) => {
        let r2 = self.get_temporary();
        bc.push(Bytecode::mov(r.into(), r2.into()));
      }
      (Slot(r), Value::Loc(FreeVar(i))) => self.reify_freevar(bc, r, i),
      (FreeVar(i), Value::Loc(FreeVar(j))) if i == j => {}
      (FreeVar(i), _) => {
        let r = self.get_value(bc, opr);
        bc.push(Bytecode::setf(i.into(), r.into()));
      }
    }
  }

  fn emit_jump(&mut self, bc: &mut BytecodeCtx, l: Control) {
    use Control::*;
    match l {
      Return(r) => bc.push(Bytecode::retn(r.into(), 0.into())),
      Pos(l) => {
        bc.push_relocate(l);
        bc.push(Bytecode::jmp(0i16.into()))
      }
    }
  }

  fn emit_test(
    &mut self,
    bc: &mut BytecodeCtx,
    test: Test,
    c1: Control,
    c2: Control,
    next: Control,
  ) {
    use Test::*;
    let gen1 = |s: &mut Self, bc: &mut BytecodeCtx, l1: Label, c2: Control| {
      match test {
        EqI(r, imm) => bc.push(Bytecode::cmpeqdi(r.into(), imm.into())),
      }
      bc.push_relocate(l1);
      bc.push(Bytecode::jmp(0i16.into()));
      if c2 != next {
        s.emit_jump(bc, c2);
      }
    };
    let gen2 = |s: &mut Self, bc: &mut BytecodeCtx, c1: Control, l2: Label| {
      match test {
        EqI(r, imm) => bc.push(Bytecode::cmpnedi(r.into(), imm.into())),
      }
      bc.push_relocate(l2);
      bc.push(Bytecode::jmp(0i16.into()));
      if c1 != next {
        s.emit_jump(bc, c1);
      }
    };
    match (c1, c2) {
      (Control::Pos(l1), Control::Return(_)) => gen1(self, bc, l1, c2),
      (Control::Pos(l1), Control::Pos(l2)) => {
        if c2 == next {
          gen1(self, bc, l1, c2)
        } else {
          gen2(self, bc, c1, l2)
        }
      }
      (Control::Return(_), Control::Pos(l2)) => gen2(self, bc, c1, l2),
      _ => unreachable!("return on both branches"),
    }
  }

  fn emit_store(
    &mut self,
    bc: &mut BytecodeCtx,
    opr: Value,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) {
    use ControlDest::*;
    use DataDest::*;
    match (data, control) {
      (Effect, Uncond(c)) => {
        if c != next {
          self.emit_jump(bc, c);
        }
      }
      (Effect, Branch(l1, l2)) => {
        let r = self.get_value(bc, opr);
        self.emit_test(bc, Test::EqI(r, 1), l1, l2, next);
      }
      (Loc(loc), Uncond(c)) => {
        self.set_location(bc, loc, opr);
        if c != next {
          self.emit_jump(bc, c);
        }
      }
      (Loc(loc), Branch(l1, l2)) => {
        let r = self.get_value(bc, opr);
        self.set_location(bc, loc, Value::Loc(Location::Slot(r)));
        self.emit_test(bc, Test::EqI(r, 1), l1, l2, next);
      }
    }
  }

  fn emit_binary_op_with_slots(
    &mut self,
    bc: &mut BytecodeCtx,
    op: &'a str,
    opr1: RegId,
    opr2: RegId,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) {
    let make_bc = |op_str: &str, dst: RegId, o1: RegId, o2: RegId| -> Bytecode {
      let dst = dst.into();
      let o1 = o1.into();
      let o2 = o2.into();
      match op_str {
        "+" => Bytecode::adddd(dst, o1, o2),
        "-" => Bytecode::subdd(dst, o1, o2),
        "*" => Bytecode::muldd(dst, o1, o2),
        "/" => Bytecode::divdd(dst, o1, o2),
        _ => unreachable!("unknown binary operator: {}", op_str),
      }
    };

    match data {
      DataDest::Loc(Location::Slot(r)) => {
        bc.push(make_bc(op, r, opr1, opr2));
        self.emit_store(bc, Value::Loc(Location::Slot(r)), data, control, next);
      }
      DataDest::Effect
      | DataDest::Loc(Location::Temporary)
      | DataDest::Loc(Location::FreeVar(_)) => {
        let r = self.allocate_temporary();
        bc.push(make_bc(op, r, opr1, opr2));
        self.emit_store(bc, Value::Loc(Location::Temporary), data, control, next);
      }
    }
  }

  fn emit_op(
    &mut self,
    bc: &mut BytecodeCtx,
    op: ExprRef<'a, InfoKey>,
    _pair: Option<Paired>,
    args: ExprsRef<'a, InfoKey>,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) {
    if let Expr::Op(op_str, _) = op {
      match op_str.0 {
        "+" | "-" | "*" | "/" => {
          if args.len() == 2 {
            let l1 = bc.fresh_label();
            let mv1 = self.emit_expr_maybe_value(
              bc,
              args[0],
              DataDest::Loc(Location::Temporary),
              ControlDest::Uncond(Control::Pos(l1)),
              Control::Pos(l1),
            );
            bc.push_label(l1);
            let l2 = bc.fresh_label();
            let mv2 = self.emit_expr_maybe_value(
              bc,
              args[1],
              DataDest::Loc(Location::Temporary),
              ControlDest::Uncond(Control::Pos(l2)),
              Control::Pos(l2),
            );
            bc.push_label(l2);
            let r2 = match mv2 {
              Some(v) => self.get_value(bc, v),
              None => self.get_temporary(),
            };
            let r1 = match mv1 {
              Some(v) => self.get_value(bc, v),
              None => self.get_temporary(),
            };
            self.emit_binary_op_with_slots(bc, "+", r1, r2, data, control, next);
          } else {
            self.diagnostic.error("expected two arguments for addition");
          }
        }
        _ => self.diagnostic.error(&format!("unknown operator: {}", op_str.0)),
      }
    } else {
      self.diagnostic.error("expected operator");
    }
  }

  fn emit_expr_maybe_value<'b>(
    &'b mut self,
    bc: &mut BytecodeCtx,
    expr: ExprRef<'a, InfoKey>,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) -> Option<Value<'a>> {
    use Expr::*;
    match expr {
      IntLiteral(i, _) => Some(Value::IntLiteral(*i)),
      StrLiteral(s, _) => Some(Value::StrLiteral(s)),
      Ident(token_str, _) => {
        let loc = self.scope.get_bound(token_str).unwrap_or_else(|| {
          self.diagnostic.error(&format!("undeclared identifier: {}", token_str.0))
        });
        Some(Value::Loc(loc))
      }
      _ => {
        self.emit_expr(bc, expr, data, control, next);
        None
      }
    }
  }

  fn emit_expr(
    &mut self,
    bc: &mut BytecodeCtx,
    expr: ExprRef<'a, InfoKey>,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) {
    use Expr::*;
    match expr {
      IntLiteral(i, _) => {
        self.emit_store(bc, Value::IntLiteral(*i), data, control, next);
      }
      StrLiteral(s, _) => {
        self.emit_store(bc, Value::StrLiteral(s), data, control, next);
      }
      Ident(token_str, _) => {
        let loc = self.scope.get_bound(token_str).unwrap_or_else(|| {
          self.diagnostic.error(&format!("undeclared identifier: {}", token_str.0))
        });
        self.emit_store(bc, Value::Loc(loc), data, control, next);
      }
      Op(op_str, _) => self
        .diagnostic
        .error(&format!("use operator `{}` as a first-class value is not supported yet", op_str.0)),
      OpApply { op, pair, args, info: _ } => self.emit_op(bc, op, *pair, args, data, control, next),
      Apply { func, pair: _, args, info: _ } => {
        let func_reg = self.allocate_temporary();
        let l = bc.fresh_label();
        let c = Control::Pos(l);
        self.emit_expr(
          bc,
          func,
          DataDest::Loc(Location::Slot(func_reg)),
          ControlDest::Uncond(c),
          c,
        );
        bc.push_label(l);
        let mut args_regs = Vec::with_capacity(args.len());
        args_regs.resize_with(args.len(), || self.allocate_temporary());
        for (elem, r) in (*args).iter().zip(args_regs.into_iter()) {
          let l = bc.fresh_label();
          let c = Control::Pos(l);
          self.emit_expr(
            bc,
            elem,
            DataDest::Loc(Location::Slot(r)),
            ControlDest::Uncond(c),
            Control::Pos(l),
          );
          bc.push_label(l);
        }
        let args_len: u16 = args
          .len()
          .try_into()
          .unwrap_or_else(|_| self.diagnostic.error("argument length overflow"));
        bc.push(Bytecode::apply(func_reg.into(), args_len.into()));
        self.emit_store(bc, Value::Loc(Location::Slot(func_reg)), data, control, next);
      }
      Bind { rec, name, expr, info: _ } => {
        let r = self.allocate_named(name);
        if *rec {
          self.update_symbols(name, r);
        }
        self.emit_expr(bc, expr, DataDest::Loc(Location::Slot(r)), ControlDest::Uncond(next), next);
        if !*rec {
          self.update_symbols(name, r);
        }
      }
      Fn { params, body, info } => {
        let freevars = &self.information.get(*info).unwrap().freevars;
        self.scope.enter_function(freevars);
        self.enter_new_frame();
        for param in *params {
          let reg = self.allocate_temporary();
          self.update_symbols(param, reg);
        }
        bc.push_thunk("fn");
        self.emit_expr(
          bc,
          body,
          DataDest::Effect,
          ControlDest::Uncond(Control::Return(0.into())),
          Control::Return(0.into()),
        );
        bc.pop_thunk();
        self.leave_frame();
        self.scope.leave();
      }
      Block(exprs, _) => match exprs {
        [] => todo!("empty block"),
        [expr] => self.emit_expr(bc, expr, data, control, next),
        [exprs @ .., last_expr] => {
          for expr in exprs {
            let l = bc.fresh_label();
            let c = Control::Pos(l);
            self.emit_expr(bc, expr, DataDest::Effect, ControlDest::Uncond(c), c);
            bc.push_label(l);
          }
          self.emit_expr(bc, last_expr, data, control, next);
        }
      },
      If(c, t, f, _) => {
        let l1 = bc.fresh_label();
        let l2 = bc.fresh_label();
        let c1 = Control::Pos(l1);
        let c2 = Control::Pos(l2);
        self.emit_expr(bc, c, DataDest::Effect, ControlDest::Branch(c1, c2), c1);
        bc.push_label(l1);
        self.emit_expr(bc, t, data, control, c2);
        bc.push_label(l2);
        self.emit_expr(bc, f, data, control, next);
      }
      Tuple(exprs, _) => {
        let mut elems_regs = Vec::with_capacity(exprs.len());
        elems_regs.resize_with(exprs.len(), || self.allocate_temporary());
        for (elem, r) in (*exprs).iter().zip(elems_regs.into_iter()) {
          let l = bc.fresh_label();
          let c = Control::Pos(l);
          self.emit_expr(
            bc,
            elem,
            DataDest::Loc(Location::Slot(r)),
            ControlDest::Uncond(c),
            Control::Pos(l),
          );
          bc.push_label(l);
        }
        bc.push(Bytecode::nop()); // MAKE TUPLE
        match control {
          ControlDest::Uncond(l) => self.emit_jump(bc, l),
          _ => self.diagnostic.error("tuple in conditional expression"),
        }
      }
    }
  }

  pub fn emit_tree(&mut self, bc: &mut BytecodeCtx) {
    self.scope.enter();
    self.emit_expr(
      bc,
      self.tree,
      DataDest::Effect,
      ControlDest::Uncond(Control::Return(0.into())),
      Control::Return(0.into()),
    );
    self.scope.leave();
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
    let tree = parser.parse().unwrap();
    let mut ctx = CodeGenCtx::new(&arena, tree);
    let mut bc = BytecodeCtx::new();
    ctx.emit_tree(&mut bc);
    let thunks = bc.finalize();
    let output = thunks.into_iter().map(|t| t.to_string()).collect::<Vec<_>>().join("\n");
    assert_eq!(output, expected_bytecode_str);
  }

  #[test]
  fn test1() {
    test_codegen(
      r#"
    let f = 1; (1, f)
    "#,
      "",
    );
  }
}
