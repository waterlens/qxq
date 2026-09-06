use std::rc::Rc;

use bumpalo::Bump;
use indexmap::{IndexMap, IndexSet};
use slotmap::SlotMap;

use crate::{
  bytecode::{
    Bytecode, BytecodeCtx, ConstantId, FloatBits, FreeVarId, Label, Location, Op8, RegId,
    SmallConstantId, Tag, TrapId, TypeDesc,
  },
  diagnostic::{Diagnostic, Result},
  parser::{Expr, ExprRef, ExprsRef, Info, InfoKey, Init, SynTree},
  tokenizer::{Paired, TokenStr},
  val,
};

fn fits_i16(i: i64) -> bool {
  (i16::MIN as i64..=i16::MAX as i64).contains(&i)
}

fn fits_u16(i: i64) -> bool {
  (0..=u16::MAX as i64).contains(&i)
}

fn fits_i32(i: i64) -> bool {
  (i32::MIN as i64..=i32::MAX as i64).contains(&i)
}

fn fits_safe_integer(i: i64) -> bool {
  (val::MIN_SAFE_INTEGER..=val::MAX_SAFE_INTEGER).contains(&i)
}

pub struct CodeGenCtx<'a> {
  diagnostic: Rc<Diagnostic>,
  stack_frame: Stack<'a>,
  scope: Scope<'a>,
  tree: ExprRef<'a, InfoKey>,
  information: SlotMap<InfoKey, Info<'a>>,
  /// Binders of the recursion group whose bodies are being compiled: the
  /// methods of one struct type.
  rec_group: Vec<TokenStr<'a>>,
  /// The receiver through which method references are lowered.
  self_expr: ExprRef<'a, InfoKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Binding {
  Var(Location),
  /// A struct type: its runtime value and the image description it constructs.
  Type(Location, u16),
  /// A method of the recursion group, reached through the current `self`.
  Method,
}

impl Binding {
  fn loc(self) -> Option<Location> {
    match self {
      Binding::Var(loc) | Binding::Type(loc, _) => Some(loc),
      Binding::Method => None,
    }
  }

  fn captured(self, id: FreeVarId) -> Self {
    match self {
      Binding::Var(_) => Binding::Var(Location::FreeVar(id)),
      Binding::Type(_, t) => Binding::Type(Location::FreeVar(id), t),
      Binding::Method => Binding::Method,
    }
  }
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
  EqualImm(RegId, i16),
  NotEqImm(RegId, i16),
  EqualConst(RegId, ConstantId),
  NotEqConst(RegId, ConstantId),
  LessConst(RegId, ConstantId),
  LessOrEqualConst(RegId, ConstantId),
  GreaterConst(RegId, ConstantId),
  GreaterOrEqualConst(RegId, ConstantId),
  Equal(RegId, RegId),
  NotEq(RegId, RegId),
  Less(RegId, RegId),
  Greater(RegId, RegId),
  LessOrEqual(RegId, RegId),
  GreaterOrEqual(RegId, RegId),
  NotF(RegId),
}

enum CmpOperand {
  Imm(i16),
  Const(ConstantId),
}

enum ArithOperand {
  Reg(RegId),
  Const(SmallConstantId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SourceBuiltin {
  PrintRaw,
  PrintRawHex,
  AssertEq,
  PrintObject,
  PrintHeapStat,
  Open,
  Close,
  Edit,
}

impl SourceBuiltin {
  fn from_name(name: &str) -> Option<Self> {
    match name {
      "print_raw" => Some(Self::PrintRaw),
      "print_raw_hex" => Some(Self::PrintRawHex),
      "assert_eq" => Some(Self::AssertEq),
      "print_object" => Some(Self::PrintObject),
      "print_heap_stat" => Some(Self::PrintHeapStat),
      "open" => Some(Self::Open),
      "close" => Some(Self::Close),
      "edit" => Some(Self::Edit),
      _ => None,
    }
  }

  fn name(self) -> &'static str {
    match self {
      Self::PrintRaw => "print_raw",
      Self::PrintRawHex => "print_raw_hex",
      Self::AssertEq => "assert_eq",
      Self::PrintObject => "print_object",
      Self::PrintHeapStat => "print_heap_stat",
      Self::Open => "open",
      Self::Close => "close",
      Self::Edit => "edit",
    }
  }

  fn arity(self) -> usize {
    match self {
      Self::PrintHeapStat => 0,
      Self::PrintRaw | Self::PrintRawHex | Self::PrintObject | Self::Open | Self::Close => 1,
      Self::AssertEq | Self::Edit => 2,
    }
  }

  fn trap_id(self) -> TrapId {
    match self {
      Self::PrintRaw => TrapId::PRINT_REGS,
      Self::PrintRawHex => TrapId::PRINT_REGS_HEX,
      Self::AssertEq => TrapId::ASSERT_EQ,
      Self::PrintObject => TrapId::PRINT_OBJ,
      Self::PrintHeapStat => TrapId::HEAP_STAT,
      Self::Open => TrapId::FILE_OPEN,
      Self::Close => TrapId::FILE_CLOSE,
      Self::Edit => TrapId::FILE_EDIT,
    }
  }
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

/// Lexical bindings. Each function owns one layer; anything from an enclosing
/// function is visible only when captured into that layer.
struct Scope<'a> {
  diagnostic: Rc<Diagnostic>,
  symbols: IndexMap<TokenStr<'a>, Vec<Binding>>,
  bound: Vec<IndexSet<TokenStr<'a>>>,
}

impl<'a> Scope<'a> {
  fn new(diagnostic: Rc<Diagnostic>) -> Self {
    Self { diagnostic, symbols: IndexMap::new(), bound: vec![] }
  }

  fn enter(&mut self) {
    self.bound.push(IndexSet::new());
  }

  fn enter_function(
    &mut self,
    self_name: &Option<TokenStr<'a>>,
    captured: &[(TokenStr<'a>, Binding)],
    rec_group: &[TokenStr<'a>],
  ) -> Result<()> {
    self.enter();
    if let Some(self_name) = self_name {
      self.insert(self_name, Binding::Var(Location::FreeVar(FreeVarId(0))));
    }
    for (i, (name, binding)) in captured.iter().enumerate() {
      let i: u16 = i.try_into().map_err(|_| self.diagnostic.error("free variable id overflow"))?;
      self.insert(name, binding.captured(FreeVarId(i + 1)));
    }
    for name in rec_group {
      self.insert(name, Binding::Method);
    }
    Ok(())
  }

  fn leave(&mut self) {
    for name in self.bound.pop().expect("bound stack underflow: check if enter was called") {
      self.symbols.get_mut(&name).and_then(Vec::pop).expect("bound variable has no bindings");
    }
  }

  fn insert(&mut self, name: &TokenStr<'a>, binding: Binding) {
    let bindings = self.symbols.entry(*name).or_default();
    // A name bound twice in one layer keeps only the later binding: the earlier
    // one is unreachable, and `leave` pops one binding per name.
    if !self.bound.last_mut().unwrap().insert(*name) {
      bindings.pop();
    }
    bindings.push(binding);
  }

  fn get_bound(&self, name: &TokenStr<'a>) -> Option<Binding> {
    if self.bound.last()?.contains(name) {
      self.symbols.get(name).and_then(|bindings| bindings.last().copied())
    } else {
      None
    }
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
  pub fn new(arena: &'a Bump, diagnostic: Rc<Diagnostic>, tree: SynTree<'a, InfoKey>) -> Self {
    let stack_frame = Stack::new();
    let scope = Scope::new(Rc::clone(&diagnostic));
    let mut information = tree.information;
    let self_expr =
      &*arena.alloc(Expr::Ident(TokenStr::new("self"), information.insert(Info::default())));
    let tree = tree.root;
    Self { diagnostic, stack_frame, scope, tree, information, rec_group: vec![], self_expr }
  }

  fn allocate_temporary(&mut self) -> Result<RegId> {
    let next_reg = free_reg!(self);
    if next_reg >= u8::MAX as usize {
      return self.diagnostic.fatal("register id overflow");
    }
    reg_push!(self, ValInfo { name: None, index: (next_reg as u8).into() });
    Ok((next_reg as u8).into())
  }

  fn enter_new_frame(&mut self) {
    self.stack_frame.frames.push(Frame::new());
  }

  fn leave_frame(&mut self) {
    self.stack_frame.frames.pop().unwrap();
  }

  fn update_symbols(&mut self, name: &TokenStr<'a>, reg: RegId) {
    self.scope.insert(name, Binding::Var(Location::Slot(reg)));
  }

  fn allocate_named(&mut self, name: &'a str) -> Result<RegId> {
    let next_reg = free_reg!(self);
    if next_reg >= u8::MAX as usize {
      return self.diagnostic.fatal("register id overflow");
    }
    reg_push!(self, ValInfo { name: Some(name), index: (next_reg as u8).into() });
    Ok((next_reg as u8).into())
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
    if fits_i16(i) {
      bc.push(Bytecode::loadi(r.into(), (i as i16).into()));
      return;
    }
    if fits_u16(i) {
      bc.push(Bytecode::loadui(r.into(), (i as u16).into()));
      return;
    }
    if i < i32::MIN as i64 || i > i32::MAX as i64 {
      if (val::MIN_SAFE_INTEGER..=val::MAX_SAFE_INTEGER).contains(&i) {
        self.reify_float_literal(bc, r, i as f64);
        return;
      }
      self
        .diagnostic
        .report(&format!("integer constant {} cannot be represented as i32 or f64", i));
    }
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
    bc.push(Bytecode::loadfree(r.into(), i.into()));
  }

  fn reify_closure(&mut self, bc: &mut BytecodeCtx, r: RegId, id: usize) -> Result<()> {
    let id =
      id.try_into().map_err(|_| self.diagnostic.error("exceeding maximium number of functions"))?;
    bc.push(Bytecode::clos(r.into(), id));
    Ok(())
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

  fn wrap_object(
    &mut self,
    bc: &mut BytecodeCtx,
    tag: Tag,
    start_field: RegId,
    len: usize,
  ) -> Result<()> {
    let len = len.try_into().map_err(|_| self.diagnostic.error("too many elements"))?;
    bc.push(Bytecode::wobj(start_field.into(), tag.into(), len));
    Ok(())
  }

  fn reify_raw_value(&mut self, bc: &mut BytecodeCtx, dst: RegId, value: val::Val) {
    let raw = u16::try_from(value.raw()).expect("raw immediate value should fit in 16 bits");
    bc.push(Bytecode::loadr(dst.into(), raw.into()));
  }

  fn emit_empty_object(
    &mut self,
    bc: &mut BytecodeCtx,
    tag: Tag,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) -> Result<()> {
    self.emit_with_dest(bc, data, control, next, |s, bc, r| s.wrap_object(bc, tag, r, 0))
  }

  /// Emits a value-producing instruction into the destination register, or into
  /// a fresh temporary when the destination is not a register.
  fn emit_with_dest(
    &mut self,
    bc: &mut BytecodeCtx,
    data: DataDest,
    control: ControlDest,
    next: Control,
    emit: impl FnOnce(&mut Self, &mut BytecodeCtx, RegId) -> Result<()>,
  ) -> Result<()> {
    match data {
      DataDest::Loc(slot @ Location::Slot(r)) => {
        emit(self, bc, r)?;
        self.emit_store(bc, Value::Loc(slot), data, control, next)
      }
      _ => {
        let r = self.allocate_temporary()?;
        emit(self, bc, r)?;
        self.emit_store(bc, Value::Loc(Location::Temporary), data, control, next)
      }
    }
  }

  /// Emits `expr` into register `r`, falling through to the next instruction.
  fn emit_expr_to(
    &mut self,
    bc: &mut BytecodeCtx,
    expr: ExprRef<'a, InfoKey>,
    r: RegId,
  ) -> Result<()> {
    let l = bc.fresh_label();
    let c = Control::Pos(l);
    self.emit_expr(bc, expr, DataDest::Loc(Location::Slot(r)), ControlDest::Uncond(c), c)?;
    bc.push_label(l);
    Ok(())
  }

  fn get_value_may_have_effect(&mut self, bc: &mut BytecodeCtx, opr: Value) -> Result<()> {
    use Location::*;
    use Value::*;
    match opr {
      Loc(Temporary) => {
        self.get_temporary();
      }
      Loc(_) => (),
      Unit | BoolLiteral(_) | IntLiteral(_) | FloatLiteral(_) | StrLiteral(_) => (),
      Test(test) => {
        let r = self.allocate_temporary()?;
        self.reify_test(bc, r, test, &mut Fusion::disabled());
        self.get_temporary();
      }
    }
    Ok(())
  }

  fn get_value(&mut self, bc: &mut BytecodeCtx, opr: Value, fuse_br: &mut Fusion) -> Result<RegId> {
    use Location::*;
    use Value::*;
    match opr {
      Loc(Slot(r)) => Ok(r),
      Loc(FreeVar(i)) => {
        let r = self.allocate_temporary()?;
        self.reify_freevar(bc, r, i);
        Ok(self.get_temporary())
      }
      Loc(Temporary) => Ok(self.get_temporary()),
      Unit => {
        let r = self.allocate_temporary()?;
        self.reify_raw_value(bc, r, val::Val::null());
        Ok(self.get_temporary())
      }
      BoolLiteral(b) => {
        let r = self.allocate_temporary()?;
        self.reify_raw_value(bc, r, val::Val::from_bool(b));
        Ok(self.get_temporary())
      }
      IntLiteral(i) => {
        let r = self.allocate_temporary()?;
        self.reify_int_literal(bc, r, i);
        Ok(self.get_temporary())
      }
      FloatLiteral(f) => {
        let r = self.allocate_temporary()?;
        self.reify_float_literal(bc, r, f.0);
        Ok(self.get_temporary())
      }
      StrLiteral(s) => {
        let r = self.allocate_temporary()?;
        self.reify_string_literal(bc, r, s);
        Ok(self.get_temporary())
      }
      Test(test) => {
        let r = self.allocate_temporary()?;
        self.reify_test(bc, r, test, fuse_br);
        Ok(self.get_temporary())
      }
    }
  }

  fn is_register_destination(&mut self, loc: Location) -> Result<Option<RegId>> {
    use Location::*;
    match loc {
      Temporary => Ok(Some(self.allocate_temporary()?)),
      Slot(r) => Ok(Some(r)),
      FreeVar(_) => Ok(None),
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
  ) -> Result<Option<RegId>> {
    use Location::*;

    match (loc, opr) {
      (Temporary, Value::Loc(Temporary)) => Ok(Some(self.peek_temporary())),
      (Slot(r), Value::Loc(Slot(r2))) if r == r2 => Ok(Some(r)),
      (FreeVar(i), Value::Loc(FreeVar(j))) if i == j => Ok(None),
      (FreeVar(_), _) => self.diagnostic.fatal("storing into a captured variable is not supported"),
      (_, _) => match self.is_register_destination(loc)? {
        Some(r) => {
          match opr {
            Value::Loc(Slot(r2)) => bc.push(Bytecode::mov(r.into(), r2.into())),
            Value::Loc(Temporary) => {
              let r2 = self.get_temporary();
              bc.push(Bytecode::mov(r.into(), r2.into()));
            }
            Value::Loc(FreeVar(i)) => self.reify_freevar(bc, r, i),
            Value::Unit => self.reify_raw_value(bc, r, val::Val::null()),
            Value::BoolLiteral(b) => self.reify_raw_value(bc, r, val::Val::from_bool(b)),
            Value::IntLiteral(i) => self.reify_int_literal(bc, r, i),
            Value::FloatLiteral(f) => self.reify_float_literal(bc, r, f.0),
            Value::StrLiteral(s) => self.reify_string_literal(bc, r, s),
            Value::Test(t) => self.reify_test(bc, r, t, fuse_br),
          }
          Ok(Some(r))
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
        EqualImm(r, imm) => bc.push(Bytecode::cmpeqdi(r.into(), imm.into())),
        NotEqImm(r, imm) => bc.push(Bytecode::cmpnedi(r.into(), imm.into())),
        EqualConst(r, c) => bc.push(Bytecode::cmpeqdc(r.into(), c.0.into())),
        NotEqConst(r, c) => bc.push(Bytecode::cmpnedc(r.into(), c.0.into())),
        LessConst(r, c) => bc.push(Bytecode::cmpltdc(r.into(), c.0.into())),
        LessOrEqualConst(r, c) => bc.push(Bytecode::cmpledc(r.into(), c.0.into())),
        GreaterConst(r, c) => bc.push(Bytecode::cmpgtdc(r.into(), c.0.into())),
        GreaterOrEqualConst(r, c) => bc.push(Bytecode::cmpgedc(r.into(), c.0.into())),
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
        EqualImm(r, imm) => bc.push(Bytecode::cmpnedi(r.into(), imm.into())),
        NotEqImm(r, imm) => bc.push(Bytecode::cmpeqdi(r.into(), imm.into())),
        EqualConst(r, c) => bc.push(Bytecode::cmpnedc(r.into(), c.0.into())),
        NotEqConst(r, c) => bc.push(Bytecode::cmpeqdc(r.into(), c.0.into())),
        LessConst(r, c) => bc.push(Bytecode::cmpgedc(r.into(), c.0.into())),
        LessOrEqualConst(r, c) => bc.push(Bytecode::cmpgtdc(r.into(), c.0.into())),
        GreaterConst(r, c) => bc.push(Bytecode::cmpltdc(r.into(), c.0.into())),
        GreaterOrEqualConst(r, c) => bc.push(Bytecode::cmpledc(r.into(), c.0.into())),
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
    matches!(
      test,
      Test::EqualImm(..)
        | Test::NotEqImm(..)
        | Test::EqualConst(..)
        | Test::NotEqConst(..)
        | Test::Equal(..)
        | Test::NotEq(..)
        | Test::NotF(..)
    )
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
  ) -> Result<()> {
    use self::Test::*;
    use Value::*;
    match opr {
      Value::Test(test) => {
        if let Some(loc) = dest {
          // not for Effect
          let mut fu = Fusion::enabled();
          let r = self.set_location(bc, loc, opr, &mut fu)?;
          self.emit_branch(bc, Some(r.expect("set_location: invariant")), test, fu, c1, c2, next);
        } else if matches!((c1, c2), (Control::Return, _) | (_, Control::Return)) {
          let mut fu = Fusion::enabled();
          let r = self.get_value(bc, opr, &mut fu)?;
          self.emit_branch(bc, Some(r), test, fu, c1, c2, next);
        } else {
          self.emit_branch(bc, None, test, Fusion::disabled(), c1, c2, next);
        }
      }
      Loc(_) | Unit | BoolLiteral(_) | IntLiteral(_) | FloatLiteral(_) | StrLiteral(_) => {
        let mut disable_fu = Fusion::disabled();
        if let Some(loc) = dest {
          // not for Effect
          let r = match self.set_location(bc, loc, opr, &mut disable_fu)? {
            Some(r) => r,
            None => {
              // this doesn't procedure duplicate code because of `set_location`'s invariant
              self.get_value(bc, opr, &mut disable_fu)?
            }
          };
          self.emit_branch(bc, None, NotF(r), disable_fu, c1, c2, next);
        } else {
          let r = self.get_value(bc, opr, &mut disable_fu)?;
          // pass r anyway to avoid the check of return
          self.emit_branch(bc, Some(r), NotF(r), disable_fu, c1, c2, next);
        };
      }
    }
    Ok(())
  }

  fn emit_store(
    &mut self,
    bc: &mut BytecodeCtx,
    opr: Value,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) -> Result<()> {
    use ControlDest::*;
    use DataDest::*;
    match (data, control) {
      (Effect, Uncond(c)) => {
        self.get_value_may_have_effect(bc, opr)?;
        if c != next {
          self.emit_jump(bc, c, None);
        }
      }
      (Effect, Branch(c1, c2)) => {
        self.emit_value_with_branch(bc, opr, None, c1, c2, next)?;
      }
      (Loc(loc), Uncond(c)) => {
        self.set_location(bc, loc, opr, &mut Fusion::disabled())?;
        if c != next {
          self.emit_jump(bc, c, None);
        }
      }
      (Loc(loc), Branch(c1, c2)) => {
        self.emit_value_with_branch(bc, opr, Some(loc), c1, c2, next)?;
      }
      (RetValue, Uncond(c)) => {
        if c != Control::Return {
          self.diagnostic.ice("data destination: return value; control destination: not return");
        }
        let r = self.get_value(bc, opr, &mut Fusion::disabled())?;
        self.emit_jump(bc, c, Some(r));
      }
      (RetValue, Branch(_l1, _l2)) => {
        self.diagnostic.ice("data destination: return value; control destination: branch");
      }
    }
    Ok(())
  }

  fn emit_binary_op_with_slots(
    &mut self,
    bc: &mut BytecodeCtx,
    op: &'a str,
    opr1: RegId,
    opr2: ArithOperand,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) -> Result<()> {
    let make_bc = |op_str: &str, dst: RegId, o1: RegId, o2: ArithOperand| -> Bytecode {
      let dst = dst.into();
      let o1 = o1.into();
      match (op_str, o2) {
        ("+", ArithOperand::Reg(r)) => Bytecode::adddd(dst, o1, r.into()),
        ("-", ArithOperand::Reg(r)) => Bytecode::subdd(dst, o1, r.into()),
        ("*", ArithOperand::Reg(r)) => Bytecode::muldd(dst, o1, r.into()),
        ("/", ArithOperand::Reg(r)) => Bytecode::divdd(dst, o1, r.into()),
        ("%", ArithOperand::Reg(r)) => Bytecode::remdd(dst, o1, r.into()),
        ("+", ArithOperand::Const(c)) => Bytecode::adddc(dst, o1, c.into()),
        ("-", ArithOperand::Const(c)) => Bytecode::subdc(dst, o1, c.into()),
        ("*", ArithOperand::Const(c)) => Bytecode::muldc(dst, o1, c.into()),
        ("/", ArithOperand::Const(c)) => Bytecode::divdc(dst, o1, c.into()),
        ("%", ArithOperand::Const(c)) => Bytecode::remdc(dst, o1, c.into()),
        _ => unreachable!("unknown binary operator: {}", op_str),
      }
    };

    self.emit_with_dest(bc, data, control, next, |_, bc, r| {
      bc.push(make_bc(op, r, opr1, opr2));
      Ok(())
    })
  }

  fn emit_unary_op_with_slots(
    &mut self,
    bc: &mut BytecodeCtx,
    op: &'a str,
    opr1: RegId,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) -> Result<()> {
    let make_bc = |op_str: &str, dst: RegId, o1: RegId| -> Bytecode {
      let dst = dst.into();
      let o1 = o1.into();
      match op_str {
        "-" => Bytecode::negd(dst, o1),
        _ => unreachable!("unknown unary operator: {}", op_str),
      }
    };

    self.emit_with_dest(bc, data, control, next, |_, bc, r| {
      bc.push(make_bc(op, r, opr1));
      Ok(())
    })
  }

  fn eval_any_loc_args(
    &mut self,
    bc: &mut BytecodeCtx,
    args: &[ExprRef<'a, InfoKey>],
  ) -> Result<(Box<[RegId]>, usize)> {
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
      )?;
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
      let r = self.allocate_temporary()?;
      self.set_location(bc, Location::Slot(r), v, &mut Fusion::disabled())?;
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

    Ok((res_regs.into_boxed_slice(), n_reified + n_complex))
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

  fn expr_as_pool_literal<'b>(expr: &'b Expr<'a, InfoKey>) -> Option<Value<'a>> {
    match expr {
      Expr::IntLiteral(i, _) => Some(Value::IntLiteral(*i)),
      Expr::FloatLiteral(f, _) => Some(Value::FloatLiteral(*f)),
      Expr::StrLiteral(s, _) => Some(Value::StrLiteral(s)),
      _ => None,
    }
  }

  fn value_to_cmp_operand(bc: &mut BytecodeCtx, v: &Value, is_eq_ne: bool) -> Option<CmpOperand> {
    match v {
      Value::IntLiteral(i) if is_eq_ne && fits_i16(*i) => Some(CmpOperand::Imm(*i as i16)),
      Value::IntLiteral(i) => Some(CmpOperand::Const(bc.add_int(*i))),
      Value::FloatLiteral(f) => Some(CmpOperand::Const(bc.add_float(f.0))),
      Value::StrLiteral(s) => Some(CmpOperand::Const(bc.add_str(s.to_string()))),
      _ => None,
    }
  }

  fn value_to_arith_operand(bc: &mut BytecodeCtx, v: &Value) -> Option<SmallConstantId> {
    match v {
      Value::IntLiteral(i) if fits_i32(*i) => bc.add_int(*i).try_small(),
      Value::IntLiteral(i) if fits_safe_integer(*i) => bc.add_float(*i as f64).try_small(),
      Value::FloatLiteral(f) => bc.add_float(f.0).try_small(),
      _ => None,
    }
  }

  fn emit_arith_args(
    &mut self,
    bc: &mut BytecodeCtx,
    _op: &str,
    args: ExprsRef<'a, InfoKey>,
  ) -> Result<(RegId, ArithOperand)> {
    let rhs_literal = Self::expr_as_pool_literal(&args[1]);

    if let Some(ref rhs_val) = rhs_literal {
      if let Some(sc) = Self::value_to_arith_operand(bc, rhs_val) {
        let (regs, n_temps) = self.eval_any_loc_args(bc, &args[..1])?;
        let r1 = regs[0];
        self.clean_any_loc_args(n_temps);
        return Ok((r1, ArithOperand::Const(sc)));
      }
    }

    let (regs, n_temps) = self.eval_any_loc_args(bc, args)?;
    let (r1, r2) = (regs[0], regs[1]);
    self.clean_any_loc_args(n_temps);
    Ok((r1, ArithOperand::Reg(r2)))
  }

  fn make_dc_test(op: &str, r: RegId, c: ConstantId) -> Test {
    match op {
      "==" => Test::EqualConst(r, c),
      "!=" => Test::NotEqConst(r, c),
      "<" => Test::LessConst(r, c),
      "<=" => Test::LessOrEqualConst(r, c),
      ">" => Test::GreaterConst(r, c),
      ">=" => Test::GreaterOrEqualConst(r, c),
      _ => unreachable!(),
    }
  }

  fn emit_cmp_operand(op: &str, r: RegId, operand: CmpOperand) -> Test {
    match operand {
      CmpOperand::Imm(imm) => {
        if op == "==" {
          Test::EqualImm(r, imm)
        } else {
          Test::NotEqImm(r, imm)
        }
      }
      CmpOperand::Const(cid) => Self::make_dc_test(op, r, cid),
    }
  }

  fn emit_cmp_args(
    &mut self,
    bc: &mut BytecodeCtx,
    op: &str,
    args: ExprsRef<'a, InfoKey>,
  ) -> Result<Test> {
    let is_eq_ne = matches!(op, "==" | "!=");
    let rhs_literal = Self::expr_as_pool_literal(&args[1]);
    let lhs_literal = if is_eq_ne { Self::expr_as_pool_literal(&args[0]) } else { None };

    // RHS is a literal — emit only the LHS, use DI or DC for the RHS.
    if let Some(ref rhs_val) = rhs_literal {
      if let Some(operand) = Self::value_to_cmp_operand(bc, rhs_val, is_eq_ne) {
        let (regs, n_temps) = self.eval_any_loc_args(bc, &args[..1])?;
        let r1 = regs[0];
        self.clean_any_loc_args(n_temps);
        return Ok(Self::emit_cmp_operand(op, r1, operand));
      }
    }

    // Commute eq/ne when LHS is a literal — emit only the RHS.
    if let Some(ref lhs_val) = lhs_literal {
      if let Some(operand) = Self::value_to_cmp_operand(bc, lhs_val, is_eq_ne) {
        let (regs, n_temps) = self.eval_any_loc_args(bc, &args[1..2])?;
        let r2 = regs[0];
        self.clean_any_loc_args(n_temps);
        return Ok(Self::emit_cmp_operand(op, r2, operand));
      }
    }

    // Fallback: materialize both into registers, emit DD.
    let (regs, n_temps) = self.eval_any_loc_args(bc, args)?;
    let (r1, r2) = (regs[0], regs[1]);
    self.clean_any_loc_args(n_temps);
    Ok(match op {
      "<" => Test::Less(r1, r2),
      ">" => Test::Greater(r1, r2),
      "<=" => Test::LessOrEqual(r1, r2),
      ">=" => Test::GreaterOrEqual(r1, r2),
      "==" => Test::Equal(r1, r2),
      "!=" => Test::NotEq(r1, r2),
      _ => unreachable!(),
    })
  }

  fn emit_source_builtin_apply(
    &mut self,
    bc: &mut BytecodeCtx,
    builtin: SourceBuiltin,
    args: ExprsRef<'a, InfoKey>,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) -> Result<()> {
    let expected = builtin.arity();
    if args.len() != expected {
      let plural = if expected == 1 { "" } else { "s" };
      return self
        .diagnostic
        .fatal(&format!("expected {expected} argument{plural} for {}", builtin.name()));
    }

    let (regs, n_temps) = self.eval_any_loc_args(bc, args)?;

    match builtin {
      SourceBuiltin::PrintRaw | SourceBuiltin::PrintRawHex => {
        let start = regs[0];
        let end =
          start.0.checked_add(1).ok_or_else(|| self.diagnostic.error("register range overflow"))?;
        self.clean_any_loc_args(n_temps);
        bc.push(Bytecode::trap(builtin.trap_id().into(), start.into(), end.into()));
        self.emit_store(bc, Value::Unit, data, control, next)?;
      }
      SourceBuiltin::AssertEq => {
        let lhs = regs[0];
        let rhs = regs[1];
        self.clean_any_loc_args(n_temps);
        bc.push(Bytecode::trap(builtin.trap_id().into(), lhs.into(), rhs.into()));
        self.emit_store(bc, Value::Unit, data, control, next)?;
      }
      SourceBuiltin::PrintObject => {
        let value = regs[0];
        self.clean_any_loc_args(n_temps);
        bc.push(Bytecode::trap(builtin.trap_id().into(), value.into(), 0u8.into()));
        self.emit_store(bc, Value::Unit, data, control, next)?;
      }
      SourceBuiltin::PrintHeapStat => {
        self.clean_any_loc_args(n_temps);
        bc.push(Bytecode::trap(builtin.trap_id().into(), 0u8.into(), 0u8.into()));
        self.emit_store(bc, Value::Unit, data, control, next)?;
      }
      SourceBuiltin::Open => {
        let path = regs[0];
        self.clean_any_loc_args(n_temps);
        self.emit_with_dest(bc, data, control, next, |_, bc, dst| {
          bc.push(Bytecode::trap(builtin.trap_id().into(), path.into(), dst.into()));
          Ok(())
        })?;
      }
      SourceBuiltin::Close => {
        let path = regs[0];
        self.clean_any_loc_args(n_temps);
        bc.push(Bytecode::trap(builtin.trap_id().into(), path.into(), 0u8.into()));
        self.emit_store(bc, Value::Unit, data, control, next)?;
      }
      SourceBuiltin::Edit => {
        let offset = regs[0];
        let byte = regs[1];
        self.clean_any_loc_args(n_temps);
        bc.push(Bytecode::trap(builtin.trap_id().into(), offset.into(), byte.into()));
        self.emit_store(bc, Value::Unit, data, control, next)?;
      }
    }

    Ok(())
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
  ) -> Result<()> {
    if let Expr::Op(op_str, _) = op {
      match op_str.0 {
        "+" | "-" | "*" | "/" | "%" => {
          if args.len() == 2 {
            let (r1, opr2) = self.emit_arith_args(bc, op_str.0, args)?;
            self.emit_binary_op_with_slots(bc, op_str.0, r1, opr2, data, control, next)?;
          } else if op_str.0 == "-" && args.len() == 1 {
            let (regs, n_temps) = self.eval_any_loc_args(bc, args)?;
            let r1 = regs[0];
            self.clean_any_loc_args(n_temps);
            self.emit_unary_op_with_slots(bc, op_str.0, r1, data, control, next)?;
          } else {
            return self.diagnostic.fatal("expected two arguments for binary arithmetic operators");
          }
        }
        "<" | "<=" | ">" | ">=" | "==" | "!=" => {
          if args.len() == 2 {
            let test = self.emit_cmp_args(bc, op_str.0, args)?;
            self.emit_store(bc, Value::Test(test), data, control, next)?;
          } else {
            return self.diagnostic.fatal("expected two arguments for comparison operator");
          }
        }
        _ => return self.diagnostic.fatal(&format!("unknown operator: {}", op_str.0)),
      }
    } else {
      return self.diagnostic.fatal("expected operator");
    }
    Ok(())
  }

  fn emit_expr_maybe_value<'b>(
    &'b mut self,
    bc: &mut BytecodeCtx,
    expr: ExprRef<'a, InfoKey>,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) -> Result<Option<Value<'a>>> {
    use Expr::*;
    let v = match expr {
      Unit(_) => Some(Value::Unit),
      BoolLiteral(b, _) => Some(Value::BoolLiteral(*b)),
      IntLiteral(i, _) => Some(Value::IntLiteral(*i)),
      FloatLiteral(f, _) => Some(Value::FloatLiteral(*f)),
      StrLiteral(s, _) => Some(Value::StrLiteral(s)),
      Ident(token_str, _) => match self.scope.get_bound(token_str) {
        Some(Binding::Var(loc) | Binding::Type(loc, _)) => Some(Value::Loc(loc)),
        Some(Binding::Method) => {
          self.emit_expr(bc, expr, data, control, next)?;
          None
        }
        None => return self.diagnostic.fail(format!("undeclared identifier: {}", token_str.0)),
      },
      _ => {
        self.emit_expr(bc, expr, data, control, next)?;
        None
      }
    };
    Ok(v)
  }

  fn emit_expr(
    &mut self,
    bc: &mut BytecodeCtx,
    expr: ExprRef<'a, InfoKey>,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) -> Result<()> {
    use Expr::*;
    match expr {
      Unit(_) => {
        self.emit_store(bc, Value::Unit, data, control, next)?;
      }
      EmptyArray(_) => {
        self.emit_empty_object(bc, Tag::ARRAY, data, control, next)?;
      }
      EmptyMap(_) => {
        self.emit_empty_object(bc, Tag::MAP, data, control, next)?;
      }
      BoolLiteral(b, _) => {
        self.emit_store(bc, Value::BoolLiteral(*b), data, control, next)?;
      }
      IntLiteral(i, _) => {
        self.emit_store(bc, Value::IntLiteral(*i), data, control, next)?;
      }
      FloatLiteral(f, _) => {
        self.emit_store(bc, Value::FloatLiteral(*f), data, control, next)?;
      }
      StrLiteral(s, _) => {
        self.emit_store(bc, Value::StrLiteral(s), data, control, next)?;
      }
      Ident(token_str, _) => match self.scope.get_bound(token_str) {
        Some(Binding::Var(loc) | Binding::Type(loc, _)) => {
          self.emit_store(bc, Value::Loc(loc), data, control, next)?;
        }
        Some(Binding::Method) => {
          self.emit_member(bc, self.self_expr, token_str, data, control, next)?;
        }
        None => return self.diagnostic.fail(format!("undeclared identifier: {}", token_str.0)),
      },
      Op(op_str, _) => {
        return self.diagnostic.fatal(&format!(
          "use operator `{}` as a first-class value is not supported yet",
          op_str.0
        ));
      }
      OpApply { op, pair, args, info: _ } => {
        self.emit_op(bc, op, *pair, args, data, control, next)?
      }
      Apply { func, pair: _, args, info: _ } => {
        if let Ident(token_str, _) = func {
          let bound = self.scope.get_bound(token_str);
          if bound == Some(Binding::Method) {
            let recv = self.self_expr;
            return self.emit_member_apply(bc, recv, token_str, args, data, control, next);
          }
          if bound.is_none()
            && let Some(builtin) = SourceBuiltin::from_name(token_str.0)
          {
            return self.emit_source_builtin_apply(bc, builtin, args, data, control, next);
          }
        }

        let func_reg = self.allocate_temporary()?;
        self.emit_expr_to(bc, func, func_reg)?;
        let _frame_ra = self.allocate_temporary()?;
        // don't explicity set the value of frame return address
        let mut args_regs = Vec::with_capacity(args.len());
        for _ in 0..args.len() {
          args_regs.push(self.allocate_temporary()?);
        }
        for (elem, r) in (*args).iter().zip(args_regs.into_iter()) {
          self.emit_expr_to(bc, elem, r)?;
        }
        let args_len: u16 =
          args.len().try_into().map_err(|_| self.diagnostic.error("argument length overflow"))?;
        for _ in 0..args_len {
          reg_pop!(self);
        }
        reg_pop!(self);
        bc.push(Bytecode::apply(func_reg.into(), args_len.into()));
        self.emit_store(bc, Value::Loc(Location::Temporary), data, control, next)?;
      }
      Bind { rec, name, expr, info: _ } => {
        let r = self.allocate_named(name.0)?;
        if *rec {
          self.update_symbols(name, r);
        }
        self.emit_expr_to(bc, expr, r)?;
        if !*rec {
          self.update_symbols(name, r);
        }
        self.emit_store(bc, Value::Unit, data, control, next)?;
      }
      Fn { name, params, body, info } => {
        let id = self.emit_function(bc, name, params, body, *info)?;
        self.emit_with_dest(bc, data, control, next, |s, bc, r| s.reify_closure(bc, r, id))?;
      }
      Block(exprs, _) => match exprs {
        [] => self.emit_store(bc, Value::Unit, data, control, next)?,
        [expr] => self.emit_expr(bc, expr, data, control, next)?,
        [exprs @ .., last_expr] => {
          for expr in exprs {
            let l = bc.fresh_label();
            let c = Control::Pos(l);
            self.emit_expr(bc, expr, DataDest::Effect, ControlDest::Uncond(c), c)?;
            bc.push_label(l);
          }
          self.emit_expr(bc, last_expr, data, control, next)?;
        }
      },
      If(c, t, f, _) => {
        let l1 = bc.fresh_label();
        let l2 = bc.fresh_label();
        let c1 = Control::Pos(l1);
        let c2 = Control::Pos(l2);
        // Pin a register when the destination is a temporary so that both
        // branches write to the same slot.
        let data = if data == DataDest::Loc(Location::Temporary) {
          let r = self.allocate_temporary()?;
          DataDest::Loc(Location::Slot(r))
        } else {
          data
        };
        self.emit_expr(bc, c, DataDest::Effect, ControlDest::Branch(c1, c2), c1)?;
        bc.push_label(l1);
        self.emit_expr(bc, t, data, control, c2)?;
        bc.push_label(l2);
        self.emit_expr(bc, f, data, control, next)?;
      }
      Tuple(exprs, _) => {
        let len = exprs.len();
        let mut elems_regs = Vec::with_capacity(len);
        for _ in 0..exprs.len() {
          elems_regs.push(self.allocate_temporary()?);
        }
        if let Some(&tuple_reg) = elems_regs.first() {
          for (elem, r) in (*exprs).iter().zip(elems_regs.into_iter()) {
            self.emit_expr_to(bc, elem, r)?;
          }
          for _ in 1..len {
            reg_pop!(self);
          }
          match data {
            DataDest::Loc(slot @ Location::Slot(r)) if r == tuple_reg => {
              self.wrap_object(bc, Tag::TUPLE, tuple_reg, len)?;
              self.emit_store(bc, Value::Loc(slot), data, control, next)?;
            }
            DataDest::Effect
            | DataDest::Loc(Location::Slot(_))
            | DataDest::Loc(Location::Temporary)
            | DataDest::Loc(Location::FreeVar(_))
            | DataDest::RetValue => {
              self.wrap_object(bc, Tag::TUPLE, tuple_reg, len)?;
              self.emit_store(bc, Value::Loc(Location::Temporary), data, control, next)?;
            }
          }
        } else {
          self.emit_store(bc, Value::Unit, data, control, next)?;
        };
      }
      StructDecl { name, fields, methods, info: _ } => {
        self.emit_struct_decl(bc, name, fields, methods, data, control, next)?;
      }
      Construct { ty, inits, info: _ } => {
        self.emit_construct(bc, ty, inits, data, control, next)?;
      }
      Member { receiver, member, info: _ } => {
        self.emit_member(bc, receiver, member, data, control, next)?;
      }
      MemberApply { receiver, member, args, info: _ } => {
        self.emit_member_apply(bc, receiver, member, args, data, control, next)?;
      }
      Index { receiver, index, info: _ } => {
        self.emit_index(bc, receiver, *index, data, control, next)?;
      }
      Assign { target, value, info: _ } => {
        self.emit_assign(bc, target, value, data, control, next)?;
      }
    }
    Ok(())
  }

  /// Compiles a function body as a new thunk and returns its index.
  fn emit_function(
    &mut self,
    bc: &mut BytecodeCtx,
    name: &Option<TokenStr<'a>>,
    params: &[TokenStr<'a>],
    body: ExprRef<'a, InfoKey>,
    info: InfoKey,
  ) -> Result<usize> {
    let freevars = &self.information.get(info).unwrap().freevars;
    let mut captured = vec![];
    let mut through_self = false;
    for fv in freevars {
      match self.scope.get_bound(fv) {
        Some(Binding::Method) => through_self = true,
        Some(binding) => captured.push((*fv, binding)),
        None if SourceBuiltin::from_name(fv.0).is_some() => {}
        None => {
          return self.diagnostic.fail(format!("unable to find captured variable: {}", fv.0));
        }
      }
    }
    // A method named inside a nested closure needs the enclosing `self`.
    let self_name = TokenStr::new("self");
    if through_self && !captured.iter().any(|(n, _)| *n == self_name) {
      let binding = self
        .scope
        .get_bound(&self_name)
        .ok_or_else(|| self.diagnostic.error("no `self` to reach a method through"))?;
      captured.push((self_name, binding));
    }
    let fvlocs: Vec<Location> =
      captured.iter().map(|(_, b)| b.loc().expect("captured binding has a location")).collect();
    debug_assert!(!fvlocs.contains(&Location::Temporary));

    self.scope.enter_function(name, &captured, &self.rec_group)?;
    self.enter_new_frame();
    for param in params {
      let reg = self.allocate_temporary()?;
      self.update_symbols(param, reg);
    }
    bc.push_thunk("fn", fvlocs.into_boxed_slice(), params.len() as u8);
    self.emit_expr(
      bc,
      body,
      DataDest::RetValue,
      ControlDest::Uncond(Control::Return),
      Control::End,
    )?;
    bc.set_nregs(frame_top!(self).max_regs as u8);
    let id = bc.pop_thunk();
    self.leave_frame();
    self.scope.leave();
    Ok(id)
  }

  fn emit_struct_decl(
    &mut self,
    bc: &mut BytecodeCtx,
    name: &TokenStr<'a>,
    fields: &[TokenStr<'a>],
    methods: ExprsRef<'a, InfoKey>,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) -> Result<()> {
    let mut decls = Vec::with_capacity(methods.len());
    for method in methods {
      match method {
        Expr::Fn { name: Some(mname), params, body, info } => {
          decls.push((*mname, *params, *body, *info))
        }
        _ => self.diagnostic.ice("struct method is not a named function"),
      }
    }
    let field_names: Vec<&str> = fields.iter().map(|f| f.0).collect();
    let method_names: Vec<&str> = decls.iter().map(|d| d.0.0).collect();
    let desc =
      TypeDesc::new(name.0, &field_names, &method_names).map_err(|e| self.diagnostic.error(e))?;
    let id =
      bc.add_type(desc).ok_or_else(|| self.diagnostic.error("too many type declarations"))?;

    // The runtime type value is built in place: the description handle, then
    // one closure per method, wrapped together.
    let r = self.allocate_named(name.0)?;
    bc.push(Bytecode::loadtype(r.into(), id.into()));
    let outer = std::mem::replace(&mut self.rec_group, decls.iter().map(|d| d.0).collect());
    for (_, params, body, info) in decls.iter() {
      let reg = self.allocate_temporary()?;
      let fid = self.emit_function(bc, &None, params, body, *info)?;
      self.reify_closure(bc, reg, fid)?;
    }
    self.rec_group = outer;
    for _ in decls.iter() {
      reg_pop!(self);
    }
    self.wrap_object(bc, Tag::TYPE, r, 1 + decls.len())?;
    self.scope.insert(name, Binding::Type(Location::Slot(r), id));
    self.emit_store(bc, Value::Unit, data, control, next)
  }

  // Labels select their field; positional initializers take the first field
  // still unassigned. The result keeps source order for evaluation.
  fn resolve_inits(
    &self,
    tname: &str,
    desc: &TypeDesc,
    inits: &[Init<'a, InfoKey>],
  ) -> Result<Vec<(usize, ExprRef<'a, InfoKey>)>> {
    let nfields = usize::from(desc.nfields);
    let mut assigned = vec![false; nfields];
    let mut order = Vec::with_capacity(inits.len());
    for init in inits {
      let slot = match init.label {
        Some(label) => match desc.slot(label.0) {
          Some(slot) if usize::from(slot) < nfields => usize::from(slot),
          _ => return self.diagnostic.fail(format!("`{tname}` has no field `{}`", label.0)),
        },
        None => match assigned.iter().position(|a| !a) {
          Some(slot) => slot,
          None => return self.diagnostic.fail(format!("too many initializers for `{tname}`")),
        },
      };
      if std::mem::replace(&mut assigned[slot], true) {
        let field = desc.slot_name(slot as u16);
        return self.diagnostic.fail(format!("field `{field}` of `{tname}` is initialized twice"));
      }
      order.push((slot, init.expr));
    }
    if let Some(slot) = assigned.iter().position(|a| !a) {
      let field = desc.slot_name(slot as u16);
      return self.diagnostic.fail(format!("missing field `{field}` of `{tname}`"));
    }
    Ok(order)
  }

  fn emit_construct(
    &mut self,
    bc: &mut BytecodeCtx,
    ty: ExprRef<'a, InfoKey>,
    inits: &[Init<'a, InfoKey>],
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) -> Result<()> {
    let Expr::Ident(tname, _) = ty else {
      return self.diagnostic.fail("a constructor needs a type name");
    };
    let Some(Binding::Type(tloc, id)) = self.scope.get_bound(tname) else {
      return self.diagnostic.fail(format!("`{}` is not a struct type", tname.0));
    };
    let desc = bc.type_desc(id);
    let nfields = usize::from(desc.nfields);
    let order = self.resolve_inits(tname.0, desc, inits)?;

    // The instance is built in place when the destination is the register on
    // top of the frame, otherwise in fresh temporaries.
    let base = match data {
      DataDest::Loc(Location::Slot(r)) if free_reg!(self) == usize::from(r.0) + 1 => r,
      _ => self.allocate_temporary()?,
    };
    let mut regs = Vec::with_capacity(nfields);
    for _ in 0..nfields {
      regs.push(self.allocate_temporary()?);
    }
    self.set_location(bc, Location::Slot(base), Value::Loc(tloc), &mut Fusion::disabled())?;
    for (slot, expr) in order {
      self.emit_expr_to(bc, expr, regs[slot])?;
    }
    for _ in 0..nfields {
      reg_pop!(self);
    }
    self.wrap_object(bc, Tag::STRUCT, base, 1 + nfields)?;
    let value = if data == DataDest::Loc(Location::Slot(base)) {
      Value::Loc(Location::Slot(base))
    } else {
      Value::Loc(Location::Temporary)
    };
    self.emit_store(bc, value, data, control, next)
  }

  /// Member operands are 8-bit indices into the thunk's constant table.
  fn member_constant(&self, bc: &mut BytecodeCtx, member: &TokenStr<'a>) -> Result<Op8> {
    self.small_constant(bc.add_str(member.0.to_string()), member.0)
  }

  /// A position is an int constant; the VM counts fields from 0.
  fn position_constant(&self, bc: &mut BytecodeCtx, index: u32) -> Result<Op8> {
    self.small_constant(bc.add_int(i64::from(index) - 1), &index.to_string())
  }

  fn small_constant(&self, id: ConstantId, member: &str) -> Result<Op8> {
    let small = id.try_small().ok_or_else(|| {
      self.diagnostic.error(format!("too many constants to address member `{member}`"))
    })?;
    Ok(small.into())
  }

  fn emit_member(
    &mut self,
    bc: &mut BytecodeCtx,
    receiver: ExprRef<'a, InfoKey>,
    member: &TokenStr<'a>,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) -> Result<()> {
    let m = self.member_constant(bc, member)?;
    self.emit_load_field(bc, receiver, m, data, control, next)
  }

  fn emit_index(
    &mut self,
    bc: &mut BytecodeCtx,
    receiver: ExprRef<'a, InfoKey>,
    index: u32,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) -> Result<()> {
    let m = self.position_constant(bc, index)?;
    self.emit_load_field(bc, receiver, m, data, control, next)
  }

  // The receiver is evaluated before the value; the assignment itself is unit.
  fn emit_assign(
    &mut self,
    bc: &mut BytecodeCtx,
    target: ExprRef<'a, InfoKey>,
    value: ExprRef<'a, InfoKey>,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) -> Result<()> {
    let (receiver, m) = match target {
      Expr::Member { receiver, member, info: _ } => (*receiver, self.member_constant(bc, member)?),
      Expr::Index { receiver, index, info: _ } => (*receiver, self.position_constant(bc, *index)?),
      _ => self.diagnostic.ice("assignment target is not a field"),
    };
    let (regs, n_temps) = self.eval_any_loc_args(bc, &[receiver, value])?;
    bc.push(Bytecode::setfield(regs[1].into(), regs[0].into(), m));
    self.clean_any_loc_args(n_temps);
    self.emit_store(bc, Value::Unit, data, control, next)
  }

  fn emit_load_field(
    &mut self,
    bc: &mut BytecodeCtx,
    receiver: ExprRef<'a, InfoKey>,
    m: Op8,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) -> Result<()> {
    let (regs, n_temps) = self.eval_any_loc_args(bc, std::slice::from_ref(&receiver))?;
    let recv = regs[0];
    self.clean_any_loc_args(n_temps);
    self.emit_with_dest(bc, data, control, next, |_, bc, r| {
      bc.push(Bytecode::loadfield(r.into(), recv.into(), m));
      Ok(())
    })
  }

  // The call region is laid out like an ordinary application: the closure
  // slot, the return address, then the receiver as the first argument.
  fn emit_member_apply(
    &mut self,
    bc: &mut BytecodeCtx,
    receiver: ExprRef<'a, InfoKey>,
    member: &TokenStr<'a>,
    args: ExprsRef<'a, InfoKey>,
    data: DataDest,
    control: ControlDest,
    next: Control,
  ) -> Result<()> {
    let dst = self.allocate_temporary()?;
    let _frame_ra = self.allocate_temporary()?;
    let recv = self.allocate_temporary()?;
    self.emit_expr_to(bc, receiver, recv)?;
    let mut args_regs = Vec::with_capacity(args.len());
    for _ in 0..args.len() {
      args_regs.push(self.allocate_temporary()?);
    }
    for (arg, r) in args.iter().zip(args_regs.into_iter()) {
      self.emit_expr_to(bc, arg, r)?;
    }
    for _ in 0..args.len() + 2 {
      reg_pop!(self);
    }
    let m = self.member_constant(bc, member)?;
    bc.push(Bytecode::invoke(dst.into(), recv.into(), m));
    self.emit_store(bc, Value::Loc(Location::Temporary), data, control, next)
  }

  pub fn emit_tree(&mut self, bc: &mut BytecodeCtx) -> Result<()> {
    self.scope.enter();
    self.emit_expr(
      bc,
      self.tree,
      DataDest::RetValue,
      ControlDest::Uncond(Control::Return),
      Control::End,
    )?;
    self.scope.leave();
    bc.set_nregs(frame_top!(self).max_regs as u8);
    Ok(())
  }
}
