use boxing::nan::NanBox;

pub struct Op8(u8);
pub struct Op16(u16);

#[repr(packed)]
pub struct Op24 {
  high: u16,
  low: u8,
}

#[repr(packed)]
pub struct OpXYZ {
  dst: Op8,
  o1: Op8,
  o2: Op8,
}

#[repr(packed)]
pub struct OpABC {
  dst: Op8,
  o1: Op8,
  o2: Op8,
}

#[repr(packed)]
pub struct OpAB {
  dst: Op8,
  o1: Op16,
}

#[repr(packed)]
pub struct OpABS {
  dst: Op8,
  o1: Op8,
}

#[repr(packed)]
pub struct OpA {
  o1: Op24,
}

#[repr(packed)]
pub struct OpAS {
  dst: Op24,
}

pub type OpCond = OpAB;
pub type OpCondS = OpABS;

impl From<u8> for Op8 {
  fn from(x: u8) -> Self { Self(x) }
}

impl From<u16> for Op16 {
  fn from(x: u16) -> Self { Self(x) }
}

impl From<u32> for Op24 {
  fn from(x: u32) -> Self { Self { high: ((x & 0xffff00) >> 8) as u16, low: (x & 0xff) as u8 } }
}

impl From<u16> for Op24 {
  fn from(x: u16) -> Self { Self { high: (x & 0xff00) >> 8, low: (x & 0xff) as u8 } }
}

impl From<u8> for Op24 {
  fn from(x: u8) -> Self { Self { high: 0, low: x } }
}

pub enum OpKind {
  Dyn,
  CInt,
  IInt,
}

pub enum Bytecode {
  Trap(OpABC),
  Nop,
  Exta(Op24),
  LoadI(OpAB),
  LoaduI(OpAB),
  LoadC(OpABS),
  Apply(OpAS),
  Call(OpAS),
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

  AddDC(OpXYZ),
  SubDC(OpXYZ),
  MulDC(OpXYZ),
  DivDC(OpXYZ),
  ModDC(OpXYZ),

  AddCD(OpXYZ),
  SubCD(OpXYZ),
  MulCD(OpXYZ),
  DivCD(OpXYZ),
  ModCD(OpXYZ),

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

  CmpEQDD(OpCondS),
  CmpNeDD(OpCondS),
  CmpLtDD(OpCondS),
  CmpLeDD(OpCondS),
  CmpGtDD(OpCondS),
  CmpGeDD(OpCondS),
}

pub struct Function<'a> {
  pub strtab: Box<Box<[u8]>>,
  pub numtab: Box<[NanBox<'a, ()>]>,
  pub instrs: Box<[Bytecode]>,
}

pub struct Module<'a> {
  pub functions: Box<[Function<'a>]>,
}
