use std::fmt::{self, Display};

use hashbrown::HashMap;

#[derive(Copy, Clone)]
pub struct Op8(u8);

#[derive(Copy, Clone)]
pub struct Op16(u16);

#[derive(Copy, Clone)]
#[repr(packed)]
pub struct Op24 {
  high: u16,
  low: u8,
}

#[derive(Copy, Clone)]
#[repr(packed)]
pub struct OpXYZ {
  dst: Op8,
  o1: Op8,
  o2: Op8,
}

#[derive(Copy, Clone)]
#[repr(packed)]
pub struct OpABC {
  dst: Op8,
  o1: Op8,
  o2: Op8,
}

#[derive(Copy, Clone)]
#[repr(packed)]
pub struct OpAB {
  dst: Op8,
  o1: Op16,
}

#[derive(Copy, Clone)]
#[repr(packed)]
pub struct OpABS {
  dst: Op8,
  o1: Op8,
}

#[derive(Copy, Clone)]
#[repr(packed)]
pub struct OpA {
  o1: Op24,
}

#[derive(Copy, Clone)]
#[repr(packed)]
pub struct OpAS {
  dst: Op24,
}

pub type OpCond = OpAB;
pub type OpCondS = OpABS;

pub struct Opcode;

impl Opcode {
  pub fn xyz(dst: u8, o1: u8, o2: u8) -> OpXYZ {
    OpXYZ { dst: dst.into(), o1: o1.into(), o2: o2.into() }
  }

  pub fn abc(dst: u8, o1: u8, o2: u8) -> OpABC {
    OpABC { dst: dst.into(), o1: o1.into(), o2: o2.into() }
  }

  pub fn ab(dst: u8, o1: u16) -> OpAB {
    OpAB { dst: dst.into(), o1: o1.into() }
  }

  pub fn abs(dst: u8, o1: u8) -> OpABS {
    OpABS { dst: dst.into(), o1: o1.into() }
  }

  pub fn a(o1: u32) -> OpA {
    OpA { o1: o1.into() }
  }

  pub fn r#as(dst: u32) -> OpAS {
    OpAS { dst: dst.into() }
  }

  pub fn cond(dst: u8, o1: u16) -> OpCond {
    OpCond { dst: dst.into(), o1: o1.into() }
  }

  pub fn conds(dst: u8, o1: u8) -> OpCondS {
    OpCondS { dst: dst.into(), o1: o1.into() }
  }
}

impl From<u8> for Op8 {
  fn from(x: u8) -> Self {
    Self(x)
  }
}

impl From<u16> for Op16 {
  fn from(x: u16) -> Self {
    Self(x)
  }
}

impl From<u32> for Op24 {
  fn from(x: u32) -> Self {
    Self { high: ((x & 0xffff00) >> 8) as u16, low: (x & 0xff) as u8 }
  }
}

impl From<u16> for Op24 {
  fn from(x: u16) -> Self {
    Self { high: (x & 0xff00) >> 8, low: (x & 0xff) as u8 }
  }
}

impl From<u8> for Op24 {
  fn from(x: u8) -> Self {
    Self { high: 0, low: x }
  }
}

pub enum OpKind {
  Dyn,
  CInt,
  IInt,
}

/*
trap                    {trap number}, {start register}, {end register}
nop
extra-argument          {payload}
load-immediate          {value}
load-unsigned-immediate {value}
load-constant           {value}
apply                   {callable base (closure)}
call                    {callable base (function)} {function number}
return
return                  {return value register}
return-n                {return value register}
jump                    {pc offset}
*/

pub enum Bytecode {
  Trap(OpABC),
  Nop,
  Exta(Op24),
  LoadI(OpAB),
  LoaduI(OpAB),
  LoadC(OpAB),
  Move(OpABS),
  Apply(OpAS),
  Call(OpABS),
  Retu,
  Ret(OpAS),
  Retn(OpABS),
  Jmp(OpA),
  Goto(OpAS),

  AddDI(OpXYZ),
  SubDI(OpXYZ),
  MulDI(OpXYZ),
  DivDI(OpXYZ),
  ModDI(OpXYZ),

  AddDD(OpXYZ),
  SubDD(OpXYZ),
  MulDD(OpXYZ),
  DivDD(OpXYZ),
  ModDD(OpXYZ),

  CmpEqDI(OpCond),
  CmpNeDI(OpCond),
  CmpLtDI(OpCond),
  CmpLeDI(OpCond),
  CmpGtDI(OpCond),
  CmpGeDI(OpCond),

  CmpEqDC(OpCond),
  CmpNeDC(OpCond),
  CmpLtDC(OpCond),
  CmpLeDC(OpCond),
  CmpGtDC(OpCond),
  CmpGeDC(OpCond),
}

#[macro_export]
macro_rules! commutative {
  (Bytecode::AddDD) => {
    true
  };
  (Bytecode::MulDD) => {
    true
  };
  ($($op:tt)*) => {
    false
  };
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Label(i32);

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
}

impl Display for BytecodeCtx {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    for code in self.code.iter() {
      writeln!(f, "{code}")?;
    }
    Ok(())
  }
}

impl Display for Op8 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.0)
  }
}

impl Display for Op16 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.0)
  }
}

impl Display for Op24 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let val = (self.high as u32) << 8 | self.low as u32;
    write!(f, "{}", val)
  }
}

impl Display for Bytecode {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    use Bytecode::*;
    let padding = 10;
    match self {
      Trap(op) => write!(f, "{:<padding$} #{}, r{}, r{}", "trap", op.dst, op.o1, op.o2),
      Nop => write!(f, "nop"),
      Exta(op) => write!(f, "{:<padding$} #{}", "exta", op),
      LoadI(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, #{}", "loadi", op.dst, o1)
      }
      LoaduI(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, #{}", "loadui", op.dst, o1)
      }
      LoadC(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, @{}", "loadc", op.dst, o1)
      }
      Move(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, r{}", "move", op.dst, o1)
      }
      Apply(op) => write!(f, "{:<padding$} r{}", "apply", op.dst),
      Call(op) => write!(f, "{:<padding$} r{}, f{}", "call", op.dst, op.o1),
      Retu => write!(f, "retu"),
      Ret(op) => write!(f, "{:<padding$} r{}", "ret", op.dst),
      Retn(op) => write!(f, "{:<padding$} r{}", "retn", op.dst),
      Jmp(op) => write!(f, "{:<padding$} #{}", "jmp", op.o1),
      Goto(op) => write!(f, "{:<padding$} #{}", "goto", op.dst),

      AddDI(op) => write!(f, "{:<padding$} r{}, r{}, #{}", "add.di", op.dst, op.o1, op.o2),
      SubDI(op) => write!(f, "{:<padding$} r{}, r{}, #{}", "sub.di", op.dst, op.o1, op.o2),
      MulDI(op) => write!(f, "{:<padding$} r{}, r{}, #{}", "mul.di", op.dst, op.o1, op.o2),
      DivDI(op) => write!(f, "{:<padding$} r{}, r{}, #{}", "div.di", op.dst, op.o1, op.o2),

      ModDI(op) => write!(f, "{:<padding$} r{}, r{}, #{}", "mod.di", op.dst, op.o1, op.o2),
      AddDD(op) => write!(f, "{:<padding$} r{}, r{}, r{}", "add.dd", op.dst, op.o1, op.o2),
      SubDD(op) => write!(f, "{:<padding$} r{}, r{}, r{}", "sub.dd", op.dst, op.o1, op.o2),
      MulDD(op) => write!(f, "{:<padding$} r{}, r{}, r{}", "mul.dd", op.dst, op.o1, op.o2),
      DivDD(op) => write!(f, "{:<padding$} r{}, r{}, r{}", "div.dd", op.dst, op.o1, op.o2),
      ModDD(op) => write!(f, "{:<padding$} r{}, r{}, r{}", "mod.dd", op.dst, op.o1, op.o2),

      CmpEqDI(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, #{}", "cmp.eq.di", op.dst, o1)
      }
      CmpNeDI(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, #{}", "cmp.ne.di", op.dst, o1)
      }
      CmpLtDI(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, #{}", "cmp.lt.di", op.dst, o1)
      }
      CmpLeDI(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, #{}", "cmp.le.di", op.dst, o1)
      }
      CmpGtDI(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, #{}", "cmp.gt.di", op.dst, o1)
      }
      CmpGeDI(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, #{}", "cmp.ge.di", op.dst, o1)
      }

      CmpEqDC(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, @{}", "cmp.eq.dc", op.dst, o1)
      }
      CmpNeDC(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, @{}", "cmp.ne.dc", op.dst, o1)
      }
      CmpLtDC(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, @{}", "cmp.lt.dc", op.dst, o1)
      }
      CmpLeDC(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, @{}", "cmp.le.dc", op.dst, o1)
      }
      CmpGtDC(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, @{}", "cmp.gt.dc", op.dst, o1)
      }
      CmpGeDC(op) => {
        let o1 = op.o1;
        write!(f, "{:<padding$} r{}, @{}", "cmp.ge.dc", op.dst, o1)
      }
    }
  }
}
