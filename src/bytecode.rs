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

#[derive(Debug, Copy, Clone)]
pub struct Bytecode(Operator, Operands);

impl Bytecode {
  pub fn trap(dst: Op8, o1: Op8, o2: Op8) -> Self {
    UnfinishedBytecode(Operator::Trap).fill_operands(Operands::ABC(Operands::abc(dst, o1, o2)))
  }

  pub fn nop() -> Self {
    UnfinishedBytecode(Operator::Nop).fill_operands(Operands::N)
  }

  pub fn exta(o1: Op24) -> Self {
    UnfinishedBytecode(Operator::Exta).fill_operands(Operands::A(Operands::a(o1)))
  }

  pub fn loadi(dst: Op8, o1: Op16) -> Self {
    UnfinishedBytecode(Operator::LoadI).fill_operands(Operands::AB(Operands::ab(dst, o1)))
  }

  pub fn loadui(dst: Op8, o1: Op16) -> Self {
    UnfinishedBytecode(Operator::LoaduI).fill_operands(Operands::AB(Operands::ab(dst, o1)))
  }

  pub fn loadc(dst: Op8, o1: Op16) -> Self {
    UnfinishedBytecode(Operator::LoadC).fill_operands(Operands::AB(Operands::ab(dst, o1)))
  }

  pub fn mov(dst: Op8, o1: Op8) -> Self {
    let o2 = 0.into();
    UnfinishedBytecode(Operator::Move).fill_operands(Operands::ABC(Operands::abc(dst, o1, o2)))
  }

  pub fn apply(dst: Op8, o1: OpS16) -> Self {
    UnfinishedBytecode(Operator::Apply).fill_operands(Operands::ABS(Operands::ab_signed(dst, o1)))
  }

  pub fn call(dst: Op8, o1: OpS16) -> Self {
    UnfinishedBytecode(Operator::Call).fill_operands(Operands::ABS(Operands::ab_signed(dst, o1)))
  }

  pub fn retu() -> Self {
    UnfinishedBytecode(Operator::Retu).fill_operands(Operands::N)
  }

  pub fn ret(dst: OpS24) -> Self {
    UnfinishedBytecode(Operator::Ret).fill_operands(Operands::AS(Operands::a_signed(dst)))
  }

  pub fn retn(dst: Op8, o1: OpS16) -> Self {
    UnfinishedBytecode(Operator::Retn).fill_operands(Operands::ABS(Operands::ab_signed(dst, o1)))
  }

  pub fn jmp(o1: OpS24) -> Self {
    UnfinishedBytecode(Operator::Jmp).fill_operands(Operands::AS(Operands::a_signed(o1)))
  }

  pub fn goto(dst: OpS24) -> Self {
    UnfinishedBytecode(Operator::Goto).fill_operands(Operands::AS(Operands::a_signed(dst)))
  }

  pub fn adddi(dst: Op8, o1: Op8, o2: Op8) -> Self {
    UnfinishedBytecode(Operator::AddDI).fill_operands(Operands::XYZ(Operands::xyz(dst, o1, o2)))
  }

  pub fn subdi(dst: Op8, o1: Op8, o2: Op8) -> Self {
    UnfinishedBytecode(Operator::SubDI).fill_operands(Operands::XYZ(Operands::xyz(dst, o1, o2)))
  }

  pub fn muldi(dst: Op8, o1: Op8, o2: Op8) -> Self {
    UnfinishedBytecode(Operator::MulDI).fill_operands(Operands::XYZ(Operands::xyz(dst, o1, o2)))
  }

  pub fn divdi(dst: Op8, o1: Op8, o2: Op8) -> Self {
    UnfinishedBytecode(Operator::DivDI).fill_operands(Operands::XYZ(Operands::xyz(dst, o1, o2)))
  }

  pub fn moddi(dst: Op8, o1: Op8, o2: Op8) -> Self {
    UnfinishedBytecode(Operator::ModDI).fill_operands(Operands::XYZ(Operands::xyz(dst, o1, o2)))
  }

  pub fn adddd(dst: Op8, o1: Op8, o2: Op8) -> Self {
    UnfinishedBytecode(Operator::AddDD).fill_operands(Operands::XYZ(Operands::xyz(dst, o1, o2)))
  }

  pub fn subdd(dst: Op8, o1: Op8, o2: Op8) -> Self {
    UnfinishedBytecode(Operator::SubDD).fill_operands(Operands::XYZ(Operands::xyz(dst, o1, o2)))
  }

  pub fn muldd(dst: Op8, o1: Op8, o2: Op8) -> Self {
    UnfinishedBytecode(Operator::MulDD).fill_operands(Operands::XYZ(Operands::xyz(dst, o1, o2)))
  }

  pub fn divdd(dst: Op8, o1: Op8, o2: Op8) -> Self {
    UnfinishedBytecode(Operator::DivDD).fill_operands(Operands::XYZ(Operands::xyz(dst, o1, o2)))
  }

  pub fn moddd(dst: Op8, o1: Op8, o2: Op8) -> Self {
    UnfinishedBytecode(Operator::ModDD).fill_operands(Operands::XYZ(Operands::xyz(dst, o1, o2)))
  }

  pub fn cmpeqdi(dst: Op8, o1: Op16) -> Self {
    UnfinishedBytecode(Operator::CmpEqDI).fill_operands(Operands::Cond(Operands::cond(dst, o1)))
  }

  pub fn cmpnedi(dst: Op8, o1: Op16) -> Self {
    UnfinishedBytecode(Operator::CmpNeDI).fill_operands(Operands::Cond(Operands::cond(dst, o1)))
  }

  pub fn cmpltdi(dst: Op8, o1: Op16) -> Self {
    UnfinishedBytecode(Operator::CmpLtDI).fill_operands(Operands::Cond(Operands::cond(dst, o1)))
  }

  pub fn cmpledi(dst: Op8, o1: Op16) -> Self {
    UnfinishedBytecode(Operator::CmpLeDI).fill_operands(Operands::Cond(Operands::cond(dst, o1)))
  }

  pub fn cmpgtdi(dst: Op8, o1: Op16) -> Self {
    UnfinishedBytecode(Operator::CmpGtDI).fill_operands(Operands::Cond(Operands::cond(dst, o1)))
  }

  pub fn cmpgedi(dst: Op8, o1: Op16) -> Self {
    UnfinishedBytecode(Operator::CmpGeDI).fill_operands(Operands::Cond(Operands::cond(dst, o1)))
  }

  pub fn cmpeqdc(dst: Op8, o1: Op16) -> Self {
    UnfinishedBytecode(Operator::CmpEqDC).fill_operands(Operands::Cond(Operands::cond(dst, o1)))
  }

  pub fn cmpnedc(dst: Op8, o1: Op16) -> Self {
    UnfinishedBytecode(Operator::CmpNeDC).fill_operands(Operands::Cond(Operands::cond(dst, o1)))
  }

  pub fn cmpltdc(dst: Op8, o1: Op16) -> Self {
    UnfinishedBytecode(Operator::CmpLtDC).fill_operands(Operands::Cond(Operands::cond(dst, o1)))
  }

  pub fn cmpledc(dst: Op8, o1: Op16) -> Self {
    UnfinishedBytecode(Operator::CmpLeDC).fill_operands(Operands::Cond(Operands::cond(dst, o1)))
  }

  pub fn cmpgtdc(dst: Op8, o1: Op16) -> Self {
    UnfinishedBytecode(Operator::CmpGtDC).fill_operands(Operands::Cond(Operands::cond(dst, o1)))
  }

  pub fn cmpgedc(dst: Op8, o1: Op16) -> Self {
    UnfinishedBytecode(Operator::CmpGeDC).fill_operands(Operands::Cond(Operands::cond(dst, o1)))
  }
}

#[derive(Debug, Clone, Copy)]
pub enum Operator {
  Trap,
  Nop,
  Exta,
  LoadI,
  LoaduI,
  LoadC,
  Move,
  Apply,
  Call,
  Retu,
  Ret,
  Retn,
  Jmp,
  Goto,

  AddDI,
  SubDI,
  MulDI,
  DivDI,
  ModDI,

  AddDD,
  SubDD,
  MulDD,
  DivDD,
  ModDD,

  CmpEqDI,
  CmpNeDI,
  CmpLtDI,
  CmpLeDI,
  CmpGtDI,
  CmpGeDI,

  CmpEqDC,
  CmpNeDC,
  CmpLtDC,
  CmpLeDC,
  CmpGtDC,
  CmpGeDC,
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

pub struct UnfinishedBytecode(Operator);

impl UnfinishedBytecode {
  pub fn new(operator: Operator) -> Self {
    Self(operator)
  }
  pub fn fill_operands(self, operands: Operands) -> Bytecode {
    use Operands::*;
    use Operator::*;
    match (self.0, operands) {
      (Trap, ABC(_)) => Bytecode(self.0, operands),
      (Nop, N) => Bytecode(self.0, operands),
      (Exta, A(_)) => Bytecode(self.0, operands),
      (LoadI, AB(_)) => Bytecode(self.0, operands),
      (LoaduI, AB(_)) => Bytecode(self.0, operands),
      (LoadC, AB(_)) => Bytecode(self.0, operands),
      (Move, ABC(_)) => Bytecode(self.0, operands),
      (Apply, ABS(_)) => Bytecode(self.0, operands),
      (Call, ABS(_)) => Bytecode(self.0, operands),
      (Retu, N) => Bytecode(self.0, operands),
      (Ret, AS(_)) => Bytecode(self.0, operands),
      (Retn, ABS(_)) => Bytecode(self.0, operands),
      (Jmp, AS(_)) => Bytecode(self.0, operands),
      (Goto, AS(_)) => Bytecode(self.0, operands),
      (AddDI, XYZ(_)) => Bytecode(self.0, operands),
      (SubDI, XYZ(_)) => Bytecode(self.0, operands),
      (MulDI, XYZ(_)) => Bytecode(self.0, operands),
      (DivDI, XYZ(_)) => Bytecode(self.0, operands),
      (ModDI, XYZ(_)) => Bytecode(self.0, operands),
      (AddDD, XYZ(_)) => Bytecode(self.0, operands),
      (SubDD, XYZ(_)) => Bytecode(self.0, operands),
      (MulDD, XYZ(_)) => Bytecode(self.0, operands),
      (DivDD, XYZ(_)) => Bytecode(self.0, operands),
      (ModDD, XYZ(_)) => Bytecode(self.0, operands),
      (CmpEqDI, Cond(_)) => Bytecode(self.0, operands),
      (CmpNeDI, Cond(_)) => Bytecode(self.0, operands),
      (CmpLtDI, Cond(_)) => Bytecode(self.0, operands),
      (CmpLeDI, Cond(_)) => Bytecode(self.0, operands),
      (CmpGtDI, Cond(_)) => Bytecode(self.0, operands),
      (CmpGeDI, Cond(_)) => Bytecode(self.0, operands),
      (CmpEqDC, Cond(_)) => Bytecode(self.0, operands),
      (CmpNeDC, Cond(_)) => Bytecode(self.0, operands),
      (CmpLtDC, Cond(_)) => Bytecode(self.0, operands),
      (CmpLeDC, Cond(_)) => Bytecode(self.0, operands),
      (CmpGtDC, Cond(_)) => Bytecode(self.0, operands),
      (CmpGeDC, Cond(_)) => Bytecode(self.0, operands),
      _ => panic!("Invalid operands for operator: {:?} {:?}", self.0, operands),
    }
  }
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

impl Display for Bytecode {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    use Operands::*;
    use Operator::*;
    let padding = 10;
    match (self.0, self.1) {
      (Trap, ABC(op)) => write!(f, "{:<padding$} #{}, r{}, r{}", "trap", op.dst, op.o1, op.o2),
      (Nop, N) => write!(f, "nop"),
      (Exta, A(op)) => write!(f, "{:<padding$} #{}", "exta", op.o1),
      (LoadI, AB(op)) => write!(f, "{:<padding$} r{}, #{}", "loadi", op.dst, op.o1),
      (LoaduI, AB(op)) => write!(f, "{:<padding$} r{}, #{}", "loadui", op.dst, op.o1),
      (LoadC, AB(op)) => write!(f, "{:<padding$} r{}, @{}", "loadc", op.dst, op.o1),
      (Move, ABC(op)) => write!(f, "{:<padding$} r{}, r{}", "move", op.dst, op.o1),
      (Apply, AS(op)) => write!(f, "{:<padding$} r{}", "apply", op.dst),
      (Call, ABS(op)) => write!(f, "{:<padding$} r{}, f{}", "call", op.dst, op.o1),
      (Retu, N) => write!(f, "retu"),
      (Ret, AS(op)) => write!(f, "{:<padding$} r{}", "ret", op.dst),
      (Retn, ABS(op)) => write!(f, "{:<padding$} r{}", "retn", op.dst),
      (Jmp, AS(op)) => write!(f, "{:<padding$} #{}", "jmp", op.dst),
      (Goto, AS(op)) => write!(f, "{:<padding$} #{}", "goto", op.dst),
      (AddDI, XYZ(op)) => write!(f, "{:<padding$} r{}, r{}, #{}", "add.di", op.dst, op.o1, op.o2),
      (SubDI, XYZ(op)) => write!(f, "{:<padding$} r{}, r{}, #{}", "sub.di", op.dst, op.o1, op.o2),
      (MulDI, XYZ(op)) => write!(f, "{:<padding$} r{}, r{}, #{}", "mul.di", op.dst, op.o1, op.o2),
      (DivDI, XYZ(op)) => write!(f, "{:<padding$} r{}, r{}, #{}", "div.di", op.dst, op.o1, op.o2),
      (ModDI, XYZ(op)) => write!(f, "{:<padding$} r{}, r{}, #{}", "mod.di", op.dst, op.o1, op.o2),
      (AddDD, XYZ(op)) => write!(f, "{:<padding$} r{}, r{}, r{}", "add.dd", op.dst, op.o1, op.o2),
      (SubDD, XYZ(op)) => write!(f, "{:<padding$} r{}, r{}, r{}", "sub.dd", op.dst, op.o1, op.o2),
      (MulDD, XYZ(op)) => write!(f, "{:<padding$} r{}, r{}, r{}", "mul.dd", op.dst, op.o1, op.o2),
      (DivDD, XYZ(op)) => write!(f, "{:<padding$} r{}, r{}, r{}", "div.dd", op.dst, op.o1, op.o2),
      (ModDD, XYZ(op)) => write!(f, "{:<padding$} r{}, r{}, r{}", "mod.dd", op.dst, op.o1, op.o2),
      (CmpEqDI, Cond(op)) => write!(f, "{:<padding$} r{}, #{}", "cmp.eq.di", op.dst, op.o1),
      (CmpNeDI, Cond(op)) => write!(f, "{:<padding$} r{}, #{}", "cmp.ne.di", op.dst, op.o1),
      (CmpLtDI, Cond(op)) => write!(f, "{:<padding$} r{}, #{}", "cmp.lt.di", op.dst, op.o1),
      (CmpLeDI, Cond(op)) => write!(f, "{:<padding$} r{}, #{}", "cmp.le.di", op.dst, op.o1),
      (CmpGtDI, Cond(op)) => write!(f, "{:<padding$} r{}, #{}", "cmp.gt.di", op.dst, op.o1),
      (CmpGeDI, Cond(op)) => write!(f, "{:<padding$} r{}, #{}", "cmp.ge.di", op.dst, op.o1),
      (CmpEqDC, Cond(op)) => write!(f, "{:<padding$} r{}, @{}", "cmp.eq.dc", op.dst, op.o1),
      (CmpNeDC, Cond(op)) => write!(f, "{:<padding$} r{}, @{}", "cmp.ne.dc", op.dst, op.o1),
      (CmpLtDC, Cond(op)) => write!(f, "{:<padding$} r{}, @{}", "cmp.lt.dc", op.dst, op.o1),
      (CmpLeDC, Cond(op)) => write!(f, "{:<padding$} r{}, @{}", "cmp.le.dc", op.dst, op.o1),
      (CmpGtDC, Cond(op)) => write!(f, "{:<padding$} r{}, @{}", "cmp.gt.dc", op.dst, op.o1),
      (CmpGeDC, Cond(op)) => write!(f, "{:<padding$} r{}, @{}", "cmp.ge.dc", op.dst, op.o1),
      _ => panic!("Invalid operands for operator: {:?} {:?}", self.0, self.1),
    }
  }
}
