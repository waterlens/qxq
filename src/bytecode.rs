use std::fmt::{self, Display};

use hashbrown::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Op8(u8);

impl From<u8> for Op8 {
  fn from(x: u8) -> Self {
    Self(x)
  }
}

impl TryFrom<usize> for Op8 {
  type Error = ();
  fn try_from(value: usize) -> Result<Self, Self::Error> {
    if value <= 0xff {
      Ok(Self(value as u8))
    } else {
      Err(())
    }
  }
}

impl Display for Op8 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.0)
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Op16(u16);

impl From<u16> for Op16 {
  fn from(x: u16) -> Self {
    Self(x)
  }
}

impl Display for Op16 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.0)
  }
}

impl TryFrom<usize> for Op16 {
  type Error = ();
  fn try_from(value: usize) -> Result<Self, Self::Error> {
    if value <= u16::MAX as usize {
      Ok(Self(value as u16))
    } else {
      Err(())
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpS16(i16);

impl From<i16> for OpS16 {
  fn from(x: i16) -> Self {
    Self(x)
  }
}

impl TryFrom<isize> for OpS16 {
  type Error = ();
  fn try_from(value: isize) -> Result<Self, Self::Error> {
    if value >= i16::MIN as isize && value <= i16::MAX as isize {
      Ok(Self(value as i16))
    } else {
      Err(())
    }
  }
}

impl Display for OpS16 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.0)
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Op24(u32);

impl TryFrom<usize> for Op24 {
  type Error = ();
  fn try_from(value: usize) -> Result<Self, Self::Error> {
    if value <= 0x00ffffff {
      Ok(Self(value as u32))
    } else {
      Err(())
    }
  }
}

impl From<u16> for Op24 {
  fn from(x: u16) -> Self {
    Self(x as u32)
  }
}

impl Display for Op24 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let val = self.0;
    write!(f, "{}", val)
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpS24(i32);

impl From<i16> for OpS24 {
  fn from(x: i16) -> Self {
    Self(x as i32)
  }
}

impl TryFrom<isize> for OpS24 {
  type Error = ();
  fn try_from(value: isize) -> Result<Self, Self::Error> {
    if (!0x00ffffff..=0x00ffffff_isize).contains(&value) {
      Ok(Self(value as i32))
    } else {
      Err(())
    }
  }
}

impl Display for OpS24 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let val = self.0;
    write!(f, "{}", val)
  }
}

#[derive(Debug, Copy, Clone)]
pub struct OpXYZ {
  dst: Op8,
  o1: Op8,
  o2: Op8,
}

#[derive(Debug, Copy, Clone)]
pub struct OpABC {
  dst: Op8,
  o1: Op8,
  o2: Op8,
}

#[derive(Debug, Copy, Clone)]
pub struct OpAB {
  dst: Op8,
  o1: Op16,
}

#[derive(Debug, Copy, Clone)]
pub struct OpABS {
  dst: Op8,
  o1: OpS16,
}

#[derive(Debug, Copy, Clone)]
pub struct OpA {
  o1: Op24,
}

#[derive(Debug, Copy, Clone)]
pub struct OpAS {
  dst: OpS24,
}

pub type OpCond = OpAB;
pub type OpCondS = OpABS;

#[derive(Debug, Clone, Copy)]
pub enum OpKind {
  Dyn,
  CInt,
  IInt,
}

#[derive(Debug, Clone, Copy)]
pub enum Operands {
  N,
  ABC(OpABC),
  XYZ(OpXYZ),
  AB(OpAB),
  ABS(OpABS),
  A(OpA),
  AS(OpAS),
  Cond(OpCond),
  CondS(OpCondS),
}

impl Operands {
  pub fn xyz(dst: Op8, o1: Op8, o2: Op8) -> OpXYZ {
    OpXYZ { dst, o1, o2 }
  }

  pub fn abc(dst: Op8, o1: Op8, o2: Op8) -> OpABC {
    OpABC { dst, o1, o2 }
  }

  pub fn ab(dst: Op8, o1: Op16) -> OpAB {
    OpAB { dst, o1 }
  }

  pub fn ab_signed(dst: Op8, o1: OpS16) -> OpABS {
    OpABS { dst, o1 }
  }

  pub fn a(o1: Op24) -> OpA {
    OpA { o1 }
  }

  pub fn a_signed(dst: OpS24) -> OpAS {
    OpAS { dst }
  }

  pub fn cond(dst: Op8, o1: Op16) -> OpCond {
    OpCond { dst, o1 }
  }

  pub fn cond_signed(dst: Op8, o1: OpS16) -> OpCondS {
    OpCondS { dst, o1 }
  }
}

macro_rules! define_operators {
  ( $( $variant:ident $op_info:tt fn $fn_name:ident $params:tt $construct:tt => $display:tt ),* $(,)? ) => {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Operator {
      $( $variant, )*
    }
  }
}

macro_rules! define_constructors {
  (@step $variant:ident ( $op_enum:ident ) fn $fn_name:ident () {} ) => {
    pub fn $fn_name() -> Self {
      Self(Operator::$variant, Operands::$op_enum)
    }
  };

  (@step $variant:ident ( $op_enum:ident, $op_struct:ident, $op_var:ident ) fn $fn_name:ident ( $($arg:ident : $arg_ty:ty),* ) { $($field:ident $(: $val:expr)?),* } ) => {
    pub fn $fn_name($($arg : $arg_ty),*) -> Self {
      Self(
        Operator::$variant,
        Operands::$op_enum($op_struct {
          $($field $(: $val)?),*
        })
      )
    }
  };

  ( $( $variant:ident $op_info:tt fn $fn_name:ident $params:tt $construct:tt => $display:tt ),* $(,)? ) => {
    impl Bytecode {
      $(
        define_constructors!(@step $variant $op_info fn $fn_name $params $construct);
      )*
    }
  }
}

macro_rules! define_display {
  // Entry point
  ( $( $variant:ident $op_info:tt fn $fn_name:ident $params:tt $construct:tt => $display:tt ),* $(,)? ) => {
    impl Display for Bytecode {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
          $(
            Operator::$variant => {
              define_display!(@fmt_inner self.1, f, $op_info, $display)
            }
          )*
        }
        Ok(())
      }
    }
  };

  // Inner helpers
  (@fmt_inner $val:expr, $f:ident, ($op_enum:ident), ($($fmt:tt)+)) => {
    if let Operands::$op_enum = $val {
      write!($f, $($fmt)+)?;
    } else {
      unreachable!("mismatched operands")
    }
  };

  (@fmt_inner $val:expr, $f:ident, ($op_enum:ident, $op_struct:ident, $op_var:ident), ($($fmt:tt)+)) => {
    if let Operands::$op_enum($op_var) = $val {
      write!($f, $($fmt)+)?;
    } else {
      unreachable!("mismatched operands")
    }
  };
}

macro_rules! define_bytecode {
  ( $($all:tt)* ) => {
    define_operators!($($all)*);
    define_constructors!($($all)*);
    define_display!($($all)*);
  }
}

#[derive(Debug, Copy, Clone)]
pub struct Bytecode(pub Operator, pub Operands);

define_bytecode! {
  Trap   (ABC, OpABC, op)   fn trap(dst: Op8, o1: Op8, o2: Op8)   { dst, o1, o2 } => ("{:<10} #{}, r{}, r{}", "trap", op.dst, op.o1, op.o2),
  Nop    (N)                fn nop()                              {}              => ("nop"),
  Exta   (A, OpA, op)       fn exta(o1: Op24)                     { o1 }          => ("{:<10} #{}", "exta", op.o1),
  LoadI  (AB, OpAB, op)     fn loadi(dst: Op8, o1: Op16)          { dst, o1 }     => ("{:<10} r{}, #{}", "loadi", op.dst, op.o1),
  LoaduI (AB, OpAB, op)     fn loadui(dst: Op8, o1: Op16)         { dst, o1 }     => ("{:<10} r{}, #{}", "loadui", op.dst, op.o1),
  LoadC  (AB, OpAB, op)     fn loadc(dst: Op8, o1: Op16)          { dst, o1 }     => ("{:<10} r{}, @{}", "loadc", op.dst, op.o1),
  Move   (ABC, OpABC, op)   fn mov(dst: Op8, o1: Op8)             { dst, o1, o2: 0.into() } => ("{:<10} r{}, r{}", "move", op.dst, op.o1),
  Apply  (ABS, OpABS, op)   fn apply(dst: Op8, o1: OpS16)         { dst, o1 }     => ("{:<10} r{}", "apply", op.dst),
  Call   (ABS, OpABS, op)   fn call(dst: Op8, o1: OpS16)          { dst, o1 }     => ("{:<10} r{}, f{}", "call", op.dst, op.o1),
  Retu   (N)                fn retu()                             {}              => ("retu"),
  Ret    (AS, OpAS, op)     fn ret(dst: OpS24)                    { dst }         => ("{:<10} r{}", "ret", op.dst),
  Retn   (ABS, OpABS, op)   fn retn(dst: Op8, o1: OpS16)          { dst, o1 }     => ("{:<10} r{}", "retn", op.dst),
  Jmp    (AS, OpAS, op)     fn jmp(dst: OpS24)                    { dst }         => ("{:<10} #{}", "jmp", op.dst),
  Goto   (AS, OpAS, op)     fn goto(dst: OpS24)                   { dst }         => ("{:<10} #{}", "goto", op.dst),

  AddDI  (XYZ, OpXYZ, op)   fn adddi(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<10} r{}, r{}, #{}", "add.di", op.dst, op.o1, op.o2),
  SubDI  (XYZ, OpXYZ, op)   fn subdi(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<10} r{}, r{}, #{}", "sub.di", op.dst, op.o1, op.o2),
  MulDI  (XYZ, OpXYZ, op)   fn muldi(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<10} r{}, r{}, #{}", "mul.di", op.dst, op.o1, op.o2),
  DivDI  (XYZ, OpXYZ, op)   fn divdi(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<10} r{}, r{}, #{}", "div.di", op.dst, op.o1, op.o2),
  ModDI  (XYZ, OpXYZ, op)   fn moddi(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<10} r{}, r{}, #{}", "mod.di", op.dst, op.o1, op.o2),

  AddDD  (XYZ, OpXYZ, op)   fn adddd(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<10} r{}, r{}, r{}", "add.dd", op.dst, op.o1, op.o2),
  SubDD  (XYZ, OpXYZ, op)   fn subdd(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<10} r{}, r{}, r{}", "sub.dd", op.dst, op.o1, op.o2),
  MulDD  (XYZ, OpXYZ, op)   fn muldd(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<10} r{}, r{}, r{}", "mul.dd", op.dst, op.o1, op.o2),
  DivDD  (XYZ, OpXYZ, op)   fn divdd(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<10} r{}, r{}, r{}", "div.dd", op.dst, op.o1, op.o2),
  ModDD  (XYZ, OpXYZ, op)   fn moddd(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<10} r{}, r{}, r{}", "mod.dd", op.dst, op.o1, op.o2),

  CmpEqDI (Cond, OpCond, op) fn cmpeqdi(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<10} r{}, #{}", "cmp.eq.di", op.dst, op.o1),
  CmpNeDI (Cond, OpCond, op) fn cmpnedi(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<10} r{}, #{}", "cmp.ne.di", op.dst, op.o1),
  CmpLtDI (Cond, OpCond, op) fn cmpltdi(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<10} r{}, #{}", "cmp.lt.di", op.dst, op.o1),
  CmpLeDI (Cond, OpCond, op) fn cmpledi(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<10} r{}, #{}", "cmp.le.di", op.dst, op.o1),
  CmpGtDI (Cond, OpCond, op) fn cmpgtdi(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<10} r{}, #{}", "cmp.gt.di", op.dst, op.o1),
  CmpGeDI (Cond, OpCond, op) fn cmpgedi(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<10} r{}, #{}", "cmp.ge.di", op.dst, op.o1),

  CmpEqDC (Cond, OpCond, op) fn cmpeqdc(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<10} r{}, @{}", "cmp.eq.dc", op.dst, op.o1),
  CmpNeDC (Cond, OpCond, op) fn cmpnedc(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<10} r{}, @{}", "cmp.ne.dc", op.dst, op.o1),
  CmpLtDC (Cond, OpCond, op) fn cmpltdc(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<10} r{}, @{}", "cmp.lt.dc", op.dst, op.o1),
  CmpLeDC (Cond, OpCond, op) fn cmpledc(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<10} r{}, @{}", "cmp.le.dc", op.dst, op.o1),
  CmpGtDC (Cond, OpCond, op) fn cmpgtdc(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<10} r{}, @{}", "cmp.gt.dc", op.dst, op.o1),
  CmpGeDC (Cond, OpCond, op) fn cmpgedc(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<10} r{}, @{}", "cmp.ge.dc", op.dst, op.o1),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Label(i32);

impl Default for Label {
  fn default() -> Self {
    Self::new()
  }
}

impl Label {
  pub fn new() -> Self {
    Self(-1) // invalid label
  }
  pub fn is_valid(&self) -> bool {
    self.0 >= 0
  }
}

pub struct BytecodeCtx {
  code: Vec<Bytecode>,
  relocate: Vec<(u32, Label)>, // (pc to be relocated, label)
  labels: HashMap<Label, u32>, // (label, the pc of the label)
  fresh: i32,
}

impl Default for BytecodeCtx {
  fn default() -> Self {
    Self::new()
  }
}

impl BytecodeCtx {
  pub fn new() -> Self {
    Self { code: Vec::new(), relocate: Vec::new(), labels: HashMap::new(), fresh: 0 }
  }

  pub fn push(&mut self, code: Bytecode) {
    self.code.push(code);
  }

  pub fn pc(&self) -> u32 {
    self.code.len() as u32
  }

  pub fn fresh_label(&mut self) -> Label {
    let label = Label(self.fresh);
    self.fresh += 1;
    label
  }

  pub fn push_label(&mut self, label: Label) {
    self.labels.insert(label, self.pc());
  }

  pub fn push_relocate(&mut self, label: Label) {
    self.relocate.push((self.pc(), label));
  }

  pub fn edit(&mut self, pc: u32, code: Bytecode) {
    self.code[pc as usize] = code;
  }

  pub fn relocate_all(&mut self) {
    let mut edit_list = vec![];
    for (pc, label) in self.relocate.iter() {
      debug_assert!(label.is_valid());
      let target_pc = self.labels[label];
      match self.code[*pc as usize] {
        Bytecode(Operator::Jmp, Operands::AS(_)) => {
          edit_list.push((
            *pc,
            Bytecode::jmp(
              (target_pc as isize - *pc as isize - 1)
                .try_into()
                .unwrap_or_else(|_| panic!("jump target is too far: {} - {}", target_pc, *pc)),
            ),
          ));
        }
        bc => panic!("Invalid bytecode for relocation: {}", bc),
      }
    }
    edit_list.into_iter().for_each(|(pc, code)| self.edit(pc, code));
  }
}

impl Display for BytecodeCtx {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    for code in self.code.iter() {
      writeln!(f, "{code}")?;
    }
    Ok(())
  }
}
