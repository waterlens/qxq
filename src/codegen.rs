use std::rc::Rc;

use bumpalo::Bump;
use indexmap::{IndexMap, IndexSet};
use slotmap::SlotMap;

use crate::{
  bytecode::{Bytecode, BytecodeCtx, FloatBits, FreeVarId, Label, Location, RegId, Tag},
  diagnostic::Diagnostic,
  parser::{Expr, ExprRef, ExprsRef, Info, InfoKey, SynTree},
  tokenizer::{Paired, TokenStr},
};

pub struct CodeGenCtx<'a> {
  diagnostic: Rc<Diagnostic>,
  stack_frame: Stack<'a>,
  scope: Scope<'a>,
  tree: ExprRef<'a, InfoKey>,
  information: SlotMap<InfoKey, Info<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Control {
  Return,
  Pos(Label),
  End,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum Value<'a> {
  Loc(Location),
  Unit,
  BoolLiteral(bool),
  IntLiteral(i64),
  FloatLiteral(FloatBits),
  StrLiteral(&'a str),
  Test(Test),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DataDest {
  RetValue,
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
  #[allow(unused)]
  EqImm(RegId, u16),
  Equal(RegId, RegId),
  NotEq(RegId, RegId),
  Less(RegId, RegId),
  Greater(RegId, RegId),
  LessOrEqual(RegId, RegId),
  GreaterOrEqual(RegId, RegId),
  NotF(RegId),
}

struct ValInfo<'a> {
  name: Option<&'a str>,
  index: RegId,
}

struct Frame<'a> {
  regs: Vec<ValInfo<'a>>,
  max_regs: usize,
}

struct Stack<'a> {
  frames: Vec<Frame<'a>>,
}

impl<'a> Frame<'a> {
  fn new() -> Self {
    Self { regs: vec![], max_regs: 0 }
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
        i.try_into().unwrap_or_else(|_| self.diagnostic.fatal("free variable id overflow"));
      self.symbols.entry(*name).or_insert(vec![]).push(Location::FreeVar(FreeVarId(i)));
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

pub struct Fusion {
  enabled: bool,
  position: u32,
}

impl Fusion {
  fn new(enabled: bool) -> Self {
    Self { enabled, position: 0 }
  }

  fn enabled() -> Self {
    Self::new(true)
  }

  fn disabled() -> Self {
    Self::new(false)
  }
}

impl Default for Fusion {
  fn default() -> Self {
    Self::disabled()
  }
}

macro_rules! frame_top {
  ($self:ident) => {
    $self.stack_frame.frames.last_mut().unwrap()
  };
}

macro_rules! free_reg {
  ($self:ident) => {
    $self.stack_frame.frames.last().unwrap().regs.len()
  };
}

macro_rules! reg_push {
  ($self:ident, $value:expr) => {{
    let frame = frame_top!($self);
    frame.regs.push($value);
    frame.max_regs = frame.max_regs.max(frame.regs.len());
  }};
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
  pub fn new(_arena: &'a Bump, diagnostic: Rc<Diagnostic>, tree: SynTree<'a, InfoKey>) -> Self {
    let stack_frame = Stack::new();
    let scope = Scope::new(Rc::clone(&diagnostic));
    let information = tree.information;
    let tree = tree.root;
    Self { diagnostic, stack_frame, scope, tree, information }
  }

  fn allocate_temporary(&mut self) -> RegId {
    let next_reg = free_reg!(self);
    if next_reg >= u8::MAX as usize {
      self.diagnostic.fatal("register id overflow");
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

  fn allocate_named(&mut self, name: &'a str) -> RegId {
    let next_reg = free_reg!(self);
    if next_reg >= u8::MAX as usize {
      self.diagnostic.fatal("register id overflow");
    }
    reg_push!(self, ValInfo { name: Some(name), index: (next_reg as u8).into() });
    (next_reg as u8).into()
  }

  fn peek_temporary(&self) -> RegId {
    let val_info = reg_top!(self);
    if val_info.name.is_some() {
      self.diagnostic.ice("no temporary variable on the stack");
    }
    val_info.index
  }

  fn get_temporary(&mut self) -> RegId {
    let r = self.peek_temporary();
    reg_pop!(self);
    r
  }

  fn reify_int_literal(&mut self, bc: &mut BytecodeCtx, r: RegId, i: i64) {
    let idx = bc.add_int(i);
    bc.push(Bytecode::loadc(r.into(), idx.0.into()));
  }

  fn reify_float_literal(&mut self, bc: &mut BytecodeCtx, r: RegId, f: f64) {
    let idx = bc.add_float(f);
    bc.push(Bytecode::loadc(r.into(), idx.0.into()));
  }

  fn reify_string_literal(&mut self, bc: &mut BytecodeCtx, r: RegId, s: &str) {
    let idx = bc.add_str(s.to_string());
    bc.push(Bytecode::loadc(r.into(), idx.0.into()));
  }

  fn reify_freevar(&mut self, bc: &mut BytecodeCtx, r: RegId, i: FreeVarId) {
    bc.push(Bytecode::loadf(r.into(), i.into()));
  }

  fn reify_closure(&mut self, bc: &mut BytecodeCtx, r: RegId, id: usize) {
    bc.push(Bytecode::clos(
      r.into(),
      id.try_into()
        .unwrap_or_else(|_| self.diagnostic.fatal("exceeding maximium number of functions")),
    ))
  }

  fn reify_test(&mut self, bc: &mut BytecodeCtx, r: RegId, test: Test, fuse_br: &mut Fusion) {
    fuse_br.position = bc.pc();
    bc.push(if fuse_br.enabled {
      Bytecode::setcondj(r.into(), 0.into())
    } else {
      Bytecode::setcond(r.into(), 0.into())
    });
    // if fuse_br enabled, the actual conditional instruction is emitted by the other procedure
    if !fuse_br.enabled {
      self.emit_test(bc, test, false);
    }
  }

  fn wrap_object(&mut self, bc: &mut BytecodeCtx, tag: Tag, start_field: RegId, len: usize) {
    bc.push(Bytecode::wobj(
      start_field.into(),
      tag.into(),
      len.try_into().unwrap_or_else(|_| self.diagnostic.fatal("too many elements")),
    ));
  }

  fn make_object(&mut self, bc: &mut BytecodeCtx, dst: RegId, tag: Tag, src: RegId) {
    bc.push(Bytecode::mobj(dst.into(), tag.into(), src.into()));
  }

  fn get_value_may_have_effect(&mut self, bc: &mut BytecodeCtx, opr: Value) {
    use Location::*;
    use Value::*;
    match opr {
      Loc(Temporary) => {
        self.get_temporary();
      }
      Loc(_) => (),
      Unit | BoolLiteral(_) | IntLiteral(_) | FloatLiteral(_) | StrLiteral(_) => (),
      Test(test) => {
        let r = self.allocate_temporary();
        self.reify_test(bc, r, test, &mut Fusion::disabled());
        self.get_temporary();
      }
    }
  }

  fn get_value(&mut self, bc: &mut BytecodeCtx, opr: Value, fuse_br: &mut Fusion) -> RegId {
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
      Unit => {
        let r = self.allocate_temporary();
        self.make_object(bc, r, Tag::UNIT, 0.into());
        self.get_temporary()
      }
      BoolLiteral(b) => {
        let r = self.allocate_temporary();
        self.make_object(bc, r, if b { Tag::TRUE } else { Tag::FALSE }, 0.into());
        self.get_temporary()
      }
      IntLiteral(i) => {
        let r = self.allocate_temporary();
        self.reify_int_literal(bc, r, i);
        self.get_temporary()
      }
      FloatLiteral(f) => {
        let r = self.allocate_temporary();
        self.reify_float_literal(bc, r, f.0);
        self.get_temporary()
      }
      StrLiteral(s) => {
        let r = self.allocate_temporary();
        self.reify_string_literal(bc, r, s);
        self.get_temporary()
      }
      Test(test) => {
        let r = self.allocate_temporary();
        self.reify_test(bc, r, test, fuse_br);
        self.get_temporary()
      }
    }
  }

  fn is_register_destination(&mut self, loc: Location) -> Option<RegId> {
    use Location::*;
    match loc {
      Temporary => Some(self.allocate_temporary()),
      Slot(r) => Some(r),
      FreeVar(_) => None,
    }
  }

  // invariant:
  // - if `opr` is a value, then `set_location` must return a reg
  // - if `set_location` returns a reg, then it must be safe to
  //   immediately apply `get_value` with `opr` after this procedure
  // - if `loc` is temporary, it always returns a reg
  fn set_location(
    &mut self,
    bc: &mut BytecodeCtx,
    loc: Location,
    opr: Value,
    fuse_br: &mut Fusion,
  ) -> Option<RegId> {
    use Location::*;

    match (loc, opr) {
      (Temporary, Value::Loc(Temporary)) => Some(self.peek_temporary()),
      (Slot(r), Value::Loc(Slot(r2))) if r == r2 => Some(r),
      (FreeVar(i), Value::Loc(FreeVar(j))) if i == j => None,
      (FreeVar(i), _) => {
        let r = self.get_value(bc, opr, &mut Fusion::disabled());
        bc.push(Bytecode::setf(i.into(), r.into()));
        Some(r)
      }
      (_, _) => match self.is_register_destination(loc) {
        Some(r) => {
          match opr {
            Value::Loc(Slot(r2)) => bc.push(Bytecode::mov(r.into(), r2.into())),
            Value::Loc(Temporary) => {
              let r2 = self.get_temporary();
              bc.push(Bytecode::mov(r.into(), r2.into()));
            }
            Value::Loc(FreeVar(i)) => self.reify_freevar(bc, r, i),
            Value::Unit => self.make_object(bc, r, Tag::UNIT, 0.into()),
            Value::BoolLiteral(b) => {
              self.make_object(bc, r, if b { Tag::TRUE } else { Tag::FALSE }, 0.into())
            }
            Value::IntLiteral(i) => self.reify_int_literal(bc, r, i),
            Value::FloatLiteral(f) => self.reify_float_literal(bc, r, f.0),
            Value::StrLiteral(s) => self.reify_string_literal(bc, r, s),
            Value::Test(t) => self.reify_test(bc, r, t, fuse_br),
          }
          Some(r)
        }
        None => unreachable!(
          "all non-register-destination cases should be handled in the outer pattern match"
        ),
      },
    }
  }

  // emit a control: either be a jump or returning a value
  fn emit_jump(&mut self, bc: &mut BytecodeCtx, l: Control, v: Option<RegId>) {
    use Control::*;
    match l {
      Pos(l) => {
        bc.push_relocate(l);
        bc.push(Bytecode::jmp(0i16.into()))
      }
      Return => {
        let Some(r) = v else {
          self.diagnostic.ice("return without a value");
        };
        bc.push(Bytecode::ret(r.into()))
      }
      End => unreachable!("end control can only be used for hinting"),
    }
  }

  fn emit_test(&mut self, bc: &mut BytecodeCtx, test: Test, negate_cond: bool) {
    use self::Test::*;
    if !negate_cond {
      match test {
        EqImm(r, imm) => bc.push(Bytecode::cmpeqdi(r.into(), imm.into())),
        Equal(r1, r2) => bc.push(Bytecode::cmpeqdd(r1.into(), r2.into())),
        NotEq(r1, r2) => bc.push(Bytecode::cmpnedd(r1.into(), r2.into())),
        Less(r1, r2) => bc.push(Bytecode::cmpltdd(r1.into(), r2.into())),
        Greater(r1, r2) => bc.push(Bytecode::cmpgtdd(r1.into(), r2.into())),
        LessOrEqual(r1, r2) => bc.push(Bytecode::cmpledd(r1.into(), r2.into())),
        GreaterOrEqual(r1, r2) => bc.push(Bytecode::cmpgedd(r1.into(), r2.into())),
        NotF(r) => bc.push(Bytecode::cmpnotf(r.into(), 0.into())),
      }
    } else {
      match test {
        EqImm(r, imm) => bc.push(Bytecode::cmpnedi(r.into(), imm.into())),
        Equal(r1, r2) => bc.push(Bytecode::cmpnedd(r1.into(), r2.into())),
        NotEq(r1, r2) => bc.push(Bytecode::cmpeqdd(r1.into(), r2.into())),
        Less(r1, r2) => bc.push(Bytecode::cmpgedd(r1.into(), r2.into())),
        Greater(r1, r2) => bc.push(Bytecode::cmpledd(r1.into(), r2.into())),
        LessOrEqual(r1, r2) => bc.push(Bytecode::cmpgtdd(r1.into(), r2.into())),
        GreaterOrEqual(r1, r2) => bc.push(Bytecode::cmpltdd(r1.into(), r2.into())),
        NotF(r) => bc.push(Bytecode::cmpnotf(r.into(), u16::MAX.into())),
      }
    }
  }

  fn emit_forward_test(
    &mut self,
    bc: &mut BytecodeCtx,
    value: Option<RegId>,
    test: Test,
    c1: Control,
    l2: Label,
    next: Control,
  ) {
    self.emit_test(bc, test, false);
    bc.push_relocate(l2);
    bc.push(Bytecode::jmp(0i16.into()));
    if c1 != next {
      self.emit_jump(bc, c1, value);
    }
  }

  fn emit_backward_test(
    &mut self,
    bc: &mut BytecodeCtx,
    value: Option<RegId>,
    test: Test,
    l1: Label,
    c2: Control,
    next: Control,
  ) {
    debug_assert!(Self::can_emit_backward_test(test));
    self.emit_test(bc, test, true);
    bc.push_relocate(l1);
    bc.push(Bytecode::jmp(0i16.into()));
    if c2 != next {
      self.emit_jump(bc, c2, value);
    }
  }

  fn can_emit_backward_test(test: Test) -> bool {
    matches!(test, Test::EqImm(..) | Test::Equal(..) | Test::NotEq(..) | Test::NotF(..))
  }

  fn emit_safe_test(
    &mut self,
    bc: &mut BytecodeCtx,
    value: Option<RegId>,
    test: Test,
    c1: Control,
    c2: Control,
    next: Control,
  ) {
    let skip = bc.fresh_label();
    self.emit_test(bc, test, false);
    bc.push_relocate(skip);
    bc.push(Bytecode::jmp(0i16.into()));
    self.emit_jump(bc, c1, value);
    bc.push_label(skip);
    if c2 != next {
      self.emit_jump(bc, c2, value);
    }
  }

  fn reverse_setcond(&mut self, bc: &mut BytecodeCtx, fuse_br: Fusion) {
    if fuse_br.enabled {
      let pc = fuse_br.position;
      assert!(bc.reverse_setcond(pc), "not a setcond")
    }
  }

  // invariant: value must be some if either control is a return
  fn emit_branch(
    &mut self,
    bc: &mut BytecodeCtx,
    value: Option<RegId>,
    test: Test,
    fuse_br: Fusion,
    c1: Control,
    c2: Control,
    next: Control,
  ) {
    match (c1, c2) {
      (Control::Return, Control::Pos(l2)) => self.emit_forward_test(bc, value, test, c1, l2, next),
      (Control::Pos(l1), Control::Pos(l2)) => {
        if c1 == next {
          self.emit_forward_test(bc, value, test, c1, l2, next);
        } else if Self::can_emit_backward_test(test) {
          self.emit_backward_test(bc, value, test, l1, c2, next);
          self.reverse_setcond(bc, fuse_br);
        } else {
          self.emit_safe_test(bc, value, test, c1, c2, next);
        }
      }
      (Control::Pos(l1), Control::Return) => {
        if Self::can_emit_backward_test(test) {
          self.emit_backward_test(bc, value, test, l1, c2, next);
          self.reverse_setcond(bc, fuse_br);
        } else {
          self.emit_safe_test(bc, value, test, c1, c2, next);
        }
      }
      _ => unreachable!("return on both branches"),
    }
  }

  fn emit_value_with_branch(
    &mut self,
    bc: &mut BytecodeCtx,
    opr: Value,
    dest: Option<Location>,
    c1: Control,
    c2: Control,
    next: Control,
  ) {
    use self::Test::*;
    use Value::*;
    match opr {
      Value::Test(test) => {
        if let Some(loc) = dest {
          // not for Effect
          let mut fu = Fusion::enabled();
          let r = self.set_location(bc, loc, opr, &mut fu);
          self.emit_branch(bc, Some(r.expect("set_location: invariant")), test, fu, c1, c2, next);
        } else if matches!((c1, c2), (Control::Return, _) | (_, Control::Return)) {
          let mut fu = Fusion::enabled();
          let r = self.get_value(bc, opr, &mut fu);
          self.emit_branch(bc, Some(r), test, fu, c1, c2, next);
        } else {
          self.emit_branch(bc, None, test, Fusion::disabled(), c1, c2, next);
        }
      }
      Loc(_) | Unit | BoolLiteral(_) | IntLiteral(_) | FloatLiteral(_) | StrLiteral(_) => {
        let mut disable_fu = Fusion::disabled();
        if let Some(loc) = dest {
          // not for Effect
          let r = match self.set_location(bc, loc, opr, &mut disable_fu) {
            Some(r) => r,
            None => {
              // this doesn't procedure duplicate code because of `set_location`'s invariant
              self.get_value(bc, opr, &mut disable_fu)
            }
          };
          self.emit_branch(bc, None, NotF(r), disable_fu, c1, c2, next);
        } else {
          let r = self.get_value(bc, opr, &mut disable_fu);
          // pass r anyway to avoid the check of return
          self.emit_branch(bc, Some(r), NotF(r), disable_fu, c1, c2, next);
        };
      }
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
        self.get_value_may_have_effect(bc, opr);
        if c != next {
          self.emit_jump(bc, c, None);
        }
      }
      (Effect, Branch(c1, c2)) => {
        self.emit_value_with_branch(bc, opr, None, c1, c2, next);
      }
      (Loc(loc), Uncond(c)) => {
        self.set_location(bc, loc, opr, &mut Fusion::disabled());
        if c != next {
          self.emit_jump(bc, c, None);
        }
      }
      (Loc(loc), Branch(c1, c2)) => {
        self.emit_value_with_branch(bc, opr, Some(loc), c1, c2, next);
      }
      (RetValue, Uncond(c)) => {
        if c != Control::Return {
          self.diagnostic.ice("data destination: return value; control destination: not return");
        }
        let r = self.get_value(bc, opr, &mut Fusion::disabled());
        self.emit_jump(bc, c, Some(r));
      }
      (RetValue, Branch(_l1, _l2)) => {
        self.diagnostic.ice("data destination: return value; control destination: branch");
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
        "%" => Bytecode::remdd(dst, o1, o2),
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
      | DataDest::Loc(Location::FreeVar(_))
      | DataDest::RetValue => {
        let r = self.allocate_temporary();
        bc.push(make_bc(op, r, opr1, opr2));
        self.emit_store(bc, Value::Loc(Location::Temporary), data, control, next);
      }
    }
  }

  fn emit_unary_op_with_slots(
    &mut self,
    bc: &mut BytecodeCtx,
    op: &'a str,
    opr1: RegId,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) {
    let make_bc = |op_str: &str, dst: RegId, o1: RegId| -> Bytecode {
      let dst = dst.into();
      let o1 = o1.into();
      match op_str {
        "-" => Bytecode::negd(dst, o1),
        _ => unreachable!("unknown unary operator: {}", op_str),
      }
    };

    match data {
      DataDest::Loc(Location::Slot(r)) => {
        bc.push(make_bc(op, r, opr1));
        self.emit_store(bc, Value::Loc(Location::Slot(r)), data, control, next);
      }
      DataDest::Effect
      | DataDest::Loc(Location::Temporary)
      | DataDest::Loc(Location::FreeVar(_))
      | DataDest::RetValue => {
        let r = self.allocate_temporary();
        bc.push(make_bc(op, r, opr1));
        self.emit_store(bc, Value::Loc(Location::Temporary), data, control, next);
      }
    }
  }

  fn eval_any_loc_args(
    &mut self,
    bc: &mut BytecodeCtx,
    args: ExprsRef<'a, InfoKey>,
  ) -> (Box<[RegId]>, usize) {
    let len = args.len();
    let mut res_regs = vec![RegId::new(0); len];
    let mut to_reify = vec![];
    let mut complex_indices = vec![];

    // Phase 1: Probe and emit complex expressions.
    for (i, arg) in args.iter().enumerate() {
      let l = bc.fresh_label();
      let v = self.emit_expr_maybe_value(
        bc,
        arg,
        DataDest::Loc(Location::Temporary),
        ControlDest::Uncond(Control::Pos(l)),
        Control::Pos(l),
      );
      bc.push_label(l);

      match v {
        Some(Value::Loc(Location::Slot(r))) => {
          // Already in a register, no move needed.
          res_regs[i] = r;
        }
        Some(v_other) => {
          // Literals or free variables that need to be loaded.
          to_reify.push((i, v_other));
        }
        None => {
          // Complex expressions that resulted in a temporary on the stack.
          complex_indices.push(i);
        }
      }
    }

    let n_reified = to_reify.len();
    let n_complex = complex_indices.len();

    // Phase 2: Allocate and load simple values.
    // These will sit on top of any complex expression results on the stack.
    for (arg_idx, v) in to_reify {
      let r = self.allocate_temporary();
      self.set_location(bc, Location::Slot(r), v, &mut Fusion::disabled());
      res_regs[arg_idx] = r;
    }

    // Phase 3: Map complex expression results to their stack registers.
    // They are located below the n_reified new temporaries.
    let current_regs = &self.stack_frame.frames.last().unwrap().regs;
    let current_len = current_regs.len();
    for (nth, arg_idx) in complex_indices.into_iter().enumerate() {
      let stack_idx = current_len - n_reified - n_complex + nth;
      res_regs[arg_idx] = current_regs[stack_idx].index;
    }

    (res_regs.into_boxed_slice(), n_reified + n_complex)
  }

  // NOTE: after calling this, emitting a multi-instructions sequence
  // with destination overlapping with this region is strongly not
  // recommended, unless guaranteeing that the argument overwritten
  // mustn't be used after. Fused instrutions sequence is an exception,
  // as they are executed in one cycle (e.g. fused conditional set).
  fn clean_any_loc_args(&mut self, n_to_pop: usize) {
    for _ in 0..n_to_pop {
      self.get_temporary();
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
        "+" | "-" | "*" | "/" | "%" => {
          if args.len() == 2 {
            let (regs, n_temps) = self.eval_any_loc_args(bc, args);
            let (r1, r2) = (regs[0], regs[1]);
            self.clean_any_loc_args(n_temps);
            self.emit_binary_op_with_slots(bc, op_str.0, r1, r2, data, control, next);
          } else if op_str.0 == "-" && args.len() == 1 {
            let (regs, n_temps) = self.eval_any_loc_args(bc, args);
            let r1 = regs[0];
            self.clean_any_loc_args(n_temps);
            self.emit_unary_op_with_slots(bc, op_str.0, r1, data, control, next);
          } else {
            self.diagnostic.fatal("expected two arguments for binary arithmetic operators");
          }
        }
        "<" | "<=" | ">" | ">=" | "==" | "!=" => {
          if args.len() == 2 {
            let (regs, n_temps) = self.eval_any_loc_args(bc, args);
            let (r1, r2) = (regs[0], regs[1]);
            let test = match op_str.0 {
              "<" => Test::Less(r1, r2),
              ">" => Test::Greater(r1, r2),
              "<=" => Test::LessOrEqual(r1, r2),
              ">=" => Test::GreaterOrEqual(r1, r2),
              "==" => Test::Equal(r1, r2),
              "!=" => Test::NotEq(r1, r2),
              _ => unreachable!(),
            };
            self.clean_any_loc_args(n_temps);
            self.emit_store(bc, Value::Test(test), data, control, next);
          } else {
            self.diagnostic.fatal("expected two arguments for comparison operator")
          }
        }
        _ => self.diagnostic.fatal(&format!("unknown operator: {}", op_str.0)),
      }
    } else {
      self.diagnostic.fatal("expected operator");
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
      BoolLiteral(b, _) => Some(Value::BoolLiteral(*b)),
      IntLiteral(i, _) => Some(Value::IntLiteral(*i)),
      FloatLiteral(f, _) => Some(Value::FloatLiteral(*f)),
      StrLiteral(s, _) => Some(Value::StrLiteral(s)),
      Ident(token_str, _) => {
        let loc = self.scope.get_bound(token_str).unwrap_or_else(|| {
          self.diagnostic.fatal(&format!("undeclared identifier: {}", token_str.0))
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
      BoolLiteral(b, _) => {
        self.emit_store(bc, Value::BoolLiteral(*b), data, control, next);
      }
      IntLiteral(i, _) => {
        self.emit_store(bc, Value::IntLiteral(*i), data, control, next);
      }
      FloatLiteral(f, _) => {
        self.emit_store(bc, Value::FloatLiteral(*f), data, control, next);
      }
      StrLiteral(s, _) => {
        self.emit_store(bc, Value::StrLiteral(s), data, control, next);
      }
      Ident(token_str, _) => {
        let loc = self.scope.get_bound(token_str).unwrap_or_else(|| {
          self.diagnostic.fatal(&format!("undeclared identifier: {}", token_str.0))
        });
        self.emit_store(bc, Value::Loc(loc), data, control, next);
      }
      Op(op_str, _) => self
        .diagnostic
        .fatal(&format!("use operator `{}` as a first-class value is not supported yet", op_str.0)),
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
          .unwrap_or_else(|_| self.diagnostic.fatal("argument length overflow"));
        for _ in 0..args_len {
          reg_pop!(self);
        }
        bc.push(Bytecode::apply(func_reg.into(), args_len.into()));
        self.emit_store(bc, Value::Loc(Location::Temporary), data, control, next);
      }
      Bind { rec, name, expr, info: _ } => {
        let r = self.allocate_named(name.0);
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
        let mut fvlocs = vec![];
        for fv in freevars {
          let loc = self.scope.get_bound(fv).unwrap_or_else(|| {
            self.diagnostic.fatal(&format!("unable to find captured variable: {}", fv.0))
          });
          assert!(!matches!(loc, Location::Temporary));
          fvlocs.push(loc);
        }
        self.scope.enter_function(freevars);
        self.enter_new_frame();
        for param in *params {
          let reg = self.allocate_temporary();
          self.update_symbols(param, reg);
        }
        bc.push_thunk("fn", fvlocs.into_boxed_slice(), params.len() as u8);
        self.emit_expr(
          bc,
          body,
          DataDest::RetValue,
          ControlDest::Uncond(Control::Return),
          Control::End,
        );
        bc.set_nregs(frame_top!(self).max_regs as u8);
        let id = bc.pop_thunk();
        self.leave_frame();
        self.scope.leave();
        match data {
          DataDest::Loc(slot @ Location::Slot(r)) => {
            self.reify_closure(bc, r, id);
            self.emit_store(bc, Value::Loc(slot), data, control, next)
          }
          DataDest::Effect
          | DataDest::Loc(Location::Temporary)
          | DataDest::Loc(Location::FreeVar(_))
          | DataDest::RetValue => {
            let r = self.allocate_temporary();
            self.reify_closure(bc, r, id);
            self.emit_store(bc, Value::Loc(Location::Temporary), data, control, next)
          }
        }
      }
      Block(exprs, _) => match exprs {
        [] => self.emit_store(bc, Value::Unit, data, control, next),
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
        let len = exprs.len();
        let mut elems_regs = Vec::with_capacity(len);
        elems_regs.resize_with(exprs.len(), || self.allocate_temporary());
        if let Some(&tuple_reg) = elems_regs.first() {
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
          for _ in 1..len {
            reg_pop!(self);
          }
          match data {
            DataDest::Loc(slot @ Location::Slot(r)) if r == tuple_reg => {
              self.wrap_object(bc, Tag::TUPLE, tuple_reg, len);
              self.emit_store(bc, Value::Loc(slot), data, control, next)
            }
            DataDest::Effect
            | DataDest::Loc(Location::Slot(_))
            | DataDest::Loc(Location::Temporary)
            | DataDest::Loc(Location::FreeVar(_))
            | DataDest::RetValue => {
              self.wrap_object(bc, Tag::TUPLE, tuple_reg, len);
              self.emit_store(bc, Value::Loc(Location::Temporary), data, control, next)
            }
          }
        } else {
          self.emit_store(bc, Value::Unit, data, control, next)
        };
      }
    }
  }

  pub fn emit_tree(&mut self, bc: &mut BytecodeCtx) {
    self.scope.enter();
    self.emit_expr(
      bc,
      self.tree,
      DataDest::RetValue,
      ControlDest::Uncond(Control::Return),
      Control::End,
    );
    self.scope.leave();
    bc.set_nregs(frame_top!(self).max_regs as u8);
  }
}

#[cfg(test)]
mod tests {
  use std::rc::Rc;

  use super::*;
  use crate::parser::Parser;
  #[allow(unused)]
  fn test_codegen(source: &str, expected_bytecode_str: &str) {
    let arena = Bump::new();
    let diag = Rc::new(Diagnostic::new());
    let mut parser = Parser::new(&arena, diag, source);
    let tree = parser.parse().unwrap();
    let diag = Rc::new(Diagnostic::new());
    let mut ctx = CodeGenCtx::new(&arena, diag, tree);
    let mut bc = BytecodeCtx::new();
    ctx.emit_tree(&mut bc);
    let image = bc.finalize();
    let output = image.thunks.into_iter().map(|t| t.to_string()).collect::<Vec<_>>().join("\n");
    assert_eq!(output, expected_bytecode_str);
  }

  #[test]
  fn test1() {
    test_codegen(
      r#"
    let f = 1; (1, f)
    "#,
      "thunk::__top_thunk__ params::0 regs::3 captured::[]\nconstants::[\n  @0: 1\n]\nloadc        r0, @0\nloadc        r1, @0\nmove         r2, r0\nwrap         r1, k1, #2\nret          r1\n",
    );
  }
}
