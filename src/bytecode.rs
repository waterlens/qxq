use crate::diagnostic::Result as DResult;
use crate::runtime::OwnedHeap;
use crate::val::Val;
use hashbrown::HashMap;
use indexmap::IndexMap;
use qxq_macros::define_bytecode;
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};

/// Object tags, matching `enum tag` in vm/src/object.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tag(u8);

impl Tag {
  pub const TUPLE: Self = Tag(0);
  pub const ARRAY: Self = Tag(1);
  pub const MAP: Self = Tag(2);
  pub const TYPE: Self = Tag(3);
  pub const STRUCT: Self = Tag(4);
  pub const THUNK: Self = Tag(5);
  pub const STR: Self = Tag(6);

  /// Words objects, whose every slot is a value, precede the exotic layouts.
  pub fn is_words(self) -> bool {
    self < Self::THUNK
  }
}

impl From<u8> for Tag {
  fn from(x: u8) -> Self {
    Self(x)
  }
}

impl TryFrom<usize> for Tag {
  type Error = ();
  fn try_from(value: usize) -> Result<Self, Self::Error> {
    if value <= 0xff { Ok(Self(value as u8)) } else { Err(()) }
  }
}

impl From<Tag> for u64 {
  fn from(x: Tag) -> Self {
    x.0 as u64
  }
}

impl From<Tag> for u8 {
  fn from(x: Tag) -> Self {
    x.0
  }
}

impl Display for Tag {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let name = match *self {
      Self::TUPLE => "tuple",
      Self::ARRAY => "array",
      Self::MAP => "map",
      Self::TYPE => "type",
      Self::STRUCT => "struct",
      Self::THUNK => "thunk",
      Self::STR => "str",
      _ => return write!(f, "tag::{}", self.0),
    };
    write!(f, "tag::{name}")
  }
}

/// Constant-table entry kinds in the image, see doc/bytecode-file-format.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstKind(u8);

impl ConstKind {
  pub const INT: Self = ConstKind(0);
  pub const FLOAT: Self = ConstKind(1);
  pub const STR: Self = ConstKind(2);
}

impl From<u8> for ConstKind {
  fn from(x: u8) -> Self {
    Self(x)
  }
}

impl From<ConstKind> for u64 {
  fn from(x: ConstKind) -> Self {
    x.0 as u64
  }
}

/// A raw immediate as `loadr` shows it: trivial values by name, the rest in hex.
struct RawImm(u16);

impl Display for RawImm {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let v = Val::from_raw(u64::from(self.0));
    if v.is_empty() {
      f.write_str("empty")
    } else if v.is_null() {
      f.write_str("unit")
    } else if v.is_bool() {
      write!(f, "{}", v.as_bool())
    } else {
      write!(f, "#{:#x}", self.0)
    }
  }
}

#[derive(Debug, Clone, Copy)]
pub struct FloatBits(pub f64);

impl PartialEq for FloatBits {
  fn eq(&self, other: &Self) -> bool {
    self.0.to_bits() == other.0.to_bits()
  }
}

impl Eq for FloatBits {}

impl Hash for FloatBits {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.0.to_bits().hash(state);
  }
}

impl Display for FloatBits {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    // Use Debug format for f64 to ensure `.0` suffix on whole numbers
    write!(f, "{:?}", self.0)
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstantId(pub u16);

impl Display for ConstantId {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "@{}", self.0)
  }
}

impl ConstantId {
  pub fn try_small(&self) -> Option<SmallConstantId> {
    u8::try_from(self.0).ok().map(SmallConstantId)
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SmallConstantId(pub u8);

impl From<SmallConstantId> for Op8 {
  fn from(id: SmallConstantId) -> Self {
    id.0.into()
  }
}

pub struct ConstantPool {
  pub ipool: IndexMap<i64, ConstantId>,
  pub fpool: IndexMap<u64, ConstantId>,
  pub spool: IndexMap<String, ConstantId>,
}

impl ConstantPool {
  pub fn new() -> Self {
    Self { ipool: IndexMap::new(), fpool: IndexMap::new(), spool: IndexMap::new() }
  }

  fn next_id(&self) -> ConstantId {
    ConstantId((self.ipool.len() + self.fpool.len() + self.spool.len()) as u16)
  }

  pub fn add_int(&mut self, n: i64) -> ConstantId {
    let next_id = self.next_id();
    *self.ipool.entry(n).or_insert(next_id)
  }

  pub fn add_float(&mut self, f: f64) -> ConstantId {
    let next_id = self.next_id();
    *self.fpool.entry(f.to_bits()).or_insert(next_id)
  }

  pub fn add_str(&mut self, s: String) -> ConstantId {
    let next_id = self.next_id();
    *self.spool.entry(s).or_insert(next_id)
  }

  pub fn to_vec(&self, heap: &mut OwnedHeap) -> Box<[Val]> {
    let total = self.ipool.len() + self.fpool.len() + self.spool.len();
    let mut v = vec![Val::empty(); total];
    for (val, idx) in self.ipool.iter() {
      debug_assert!(
        *val >= i32::MIN as i64 && *val <= i32::MAX as i64,
        "integer constant {} out of i32 range; should have been caught by frontend",
        val
      );
      v[idx.0 as usize] = Val::from_i32(*val as i32);
    }
    for (bits, idx) in self.fpool.iter() {
      v[idx.0 as usize] = Val::from_f64(f64::from_bits(*bits));
    }
    for (s, idx) in self.spool.iter() {
      v[idx.0 as usize] = heap.alloc_str(s).expect("failed to allocate string constant");
    }
    v.into_boxed_slice()
  }
}

impl Default for ConstantPool {
  fn default() -> Self {
    Self::new()
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegId(pub u8);

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

impl From<RegId> for u8 {
  fn from(id: RegId) -> Self {
    id.0
  }
}

impl From<RegId> for Op8 {
  fn from(id: RegId) -> Self {
    id.0.into()
  }
}

impl From<RegId> for Op16 {
  fn from(id: RegId) -> Self {
    (id.0 as u16).into()
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FreeVarId(pub u16);

impl FreeVarId {
  pub fn new(id: u16) -> Self {
    FreeVarId(id)
  }
}

impl From<u16> for FreeVarId {
  fn from(id: u16) -> Self {
    FreeVarId(id)
  }
}

impl From<FreeVarId> for u16 {
  fn from(id: FreeVarId) -> Self {
    id.0
  }
}

impl From<FreeVarId> for Op16 {
  fn from(id: FreeVarId) -> Self {
    id.0.into()
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Location {
  Temporary,
  Slot(RegId),
  FreeVar(FreeVarId),
}

impl Display for Location {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    use Location::*;
    match self {
      Temporary => write!(f, "?t"),
      Slot(r) => write!(f, "r{}", r.0),
      FreeVar(fv) => write!(f, "^{}", fv.0),
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Op8(u8);

impl From<u8> for Op8 {
  fn from(x: u8) -> Self {
    Self(x)
  }
}

impl From<Op8> for u8 {
  fn from(x: Op8) -> Self {
    x.0
  }
}

impl From<Op8> for usize {
  fn from(x: Op8) -> Self {
    x.0.into()
  }
}

impl From<Tag> for Op8 {
  fn from(x: Tag) -> Self {
    x.0.into()
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TrapId(u8);

impl TrapId {
  pub const HALT: Self = TrapId(2);
  pub const PRINT_REGS: Self = TrapId(4);
  pub const PRINT_REGS_HEX: Self = TrapId(5);
  pub const ASSERT_EQ: Self = TrapId(6);
  pub const PRINT_OBJ: Self = TrapId(7);
  pub const HEAP_STAT: Self = TrapId(8);
  pub const FILE_OPEN: Self = TrapId(9);
  pub const FILE_CLOSE: Self = TrapId(10);
  pub const FILE_EDIT: Self = TrapId(11);
}

impl From<TrapId> for Op8 {
  fn from(x: TrapId) -> Self {
    x.0.into()
  }
}

impl TryFrom<usize> for Op8 {
  type Error = ();
  fn try_from(value: usize) -> Result<Self, Self::Error> {
    if value <= 0xff { Ok(Self(value as u8)) } else { Err(()) }
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

impl From<Op16> for u16 {
  fn from(x: Op16) -> Self {
    x.0
  }
}

impl From<Op16> for usize {
  fn from(x: Op16) -> Self {
    x.0.into()
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
    if value <= u16::MAX as usize { Ok(Self(value as u16)) } else { Err(()) }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpS16(i16);

impl From<i16> for OpS16 {
  fn from(x: i16) -> Self {
    Self(x)
  }
}

impl From<OpS16> for i16 {
  fn from(x: OpS16) -> Self {
    x.0
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

impl TryFrom<u32> for Op24 {
  type Error = ();
  fn try_from(value: u32) -> Result<Self, Self::Error> {
    if value <= 0x00ffffff { Ok(Self(value)) } else { Err(()) }
  }
}

impl TryFrom<usize> for Op24 {
  type Error = ();
  fn try_from(value: usize) -> Result<Self, Self::Error> {
    if value <= 0x00ffffff { Ok(Self(value as u32)) } else { Err(()) }
  }
}

impl From<u16> for Op24 {
  fn from(x: u16) -> Self {
    Self(x as u32)
  }
}

impl From<Op24> for u32 {
  fn from(x: Op24) -> Self {
    x.0
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

impl From<OpS24> for i32 {
  fn from(x: OpS24) -> Self {
    x.0
  }
}

impl TryFrom<i32> for OpS24 {
  type Error = ();
  fn try_from(value: i32) -> Result<Self, Self::Error> {
    if (-0x007fffff..=0x007fffff).contains(&value) { Ok(Self(value)) } else { Err(()) }
  }
}

impl TryFrom<isize> for OpS24 {
  type Error = ();
  fn try_from(value: isize) -> Result<Self, Self::Error> {
    if (-0x007fffff..=0x007fffff_isize).contains(&value) { Ok(Self(value as i32)) } else { Err(()) }
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
  pub dst: Op8,
  pub o1: Op8,
  pub o2: Op8,
}

#[derive(Debug, Copy, Clone)]
pub struct OpABC {
  pub dst: Op8,
  pub o1: Op8,
  pub o2: Op8,
}

#[derive(Debug, Copy, Clone)]
pub struct OpAB {
  pub dst: Op8,
  pub o1: Op16,
}

#[derive(Debug, Copy, Clone)]
pub struct OpABS {
  pub dst: Op8,
  pub o1: OpS16,
}

#[derive(Debug, Copy, Clone)]
pub struct OpA {
  pub o1: Op24,
}

#[derive(Debug, Copy, Clone)]
pub struct OpAS {
  pub dst: OpS24,
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

pub trait BinaryRepr: Sized {
  fn dump(&self, buf: &mut [u8]);
  fn load(buf: &[u8]) -> DResult<Self>;
}

impl BinaryRepr for OpABC {
  fn dump(&self, buf: &mut [u8]) {
    buf[0] = self.dst.into();
    buf[1] = self.o1.into();
    buf[2] = self.o2.into();
  }

  fn load(buf: &[u8]) -> DResult<Self> {
    Ok(Self { dst: buf[0].into(), o1: buf[1].into(), o2: buf[2].into() })
  }
}

impl BinaryRepr for OpXYZ {
  fn dump(&self, buf: &mut [u8]) {
    buf[0] = self.o1.into();
    buf[1] = self.o2.into();
    buf[2] = self.dst.into();
  }

  fn load(buf: &[u8]) -> DResult<Self> {
    Ok(Self { dst: buf[2].into(), o1: buf[0].into(), o2: buf[1].into() })
  }
}

impl BinaryRepr for OpAB {
  fn dump(&self, buf: &mut [u8]) {
    buf[0] = self.dst.into();
    let o1: u16 = self.o1.into();
    buf[1..3].copy_from_slice(&o1.to_le_bytes());
  }

  fn load(buf: &[u8]) -> DResult<Self> {
    Ok(Self { dst: buf[0].into(), o1: u16::from_le_bytes([buf[1], buf[2]]).into() })
  }
}

impl BinaryRepr for OpABS {
  fn dump(&self, buf: &mut [u8]) {
    buf[0] = self.dst.into();
    let o1: i16 = self.o1.into();
    buf[1..3].copy_from_slice(&o1.to_le_bytes());
  }

  fn load(buf: &[u8]) -> DResult<Self> {
    Ok(Self { dst: buf[0].into(), o1: i16::from_le_bytes([buf[1], buf[2]]).into() })
  }
}

impl BinaryRepr for OpA {
  fn dump(&self, buf: &mut [u8]) {
    let o1: u32 = self.o1.into();
    buf[0..3].copy_from_slice(&o1.to_le_bytes()[0..3]);
  }

  fn load(buf: &[u8]) -> DResult<Self> {
    let mut tmp = [0u8; 4];
    tmp[0..3].copy_from_slice(&buf[0..3]);
    Ok(Self { o1: u32::from_le_bytes(tmp).try_into().unwrap() })
  }
}

impl BinaryRepr for OpAS {
  fn dump(&self, buf: &mut [u8]) {
    let dst: i32 = self.dst.into();
    buf[0..3].copy_from_slice(&dst.to_le_bytes()[0..3]);
  }

  fn load(buf: &[u8]) -> DResult<Self> {
    let mut tmp = [0u8; 4];
    tmp[0..3].copy_from_slice(&buf[0..3]);
    // Sign extend 24-bit to 32-bit
    if tmp[2] & 0x80 != 0 {
      tmp[3] = 0xff;
    }
    Ok(Self { dst: i32::from_le_bytes(tmp).try_into().unwrap() })
  }
}

impl Operands {
  pub fn dump(&self, buf: &mut [u8]) {
    match self {
      Operands::N => {}
      Operands::ABC(op) => op.dump(buf),
      Operands::XYZ(op) => op.dump(buf),
      Operands::AB(op) => op.dump(buf),
      Operands::ABS(op) => op.dump(buf),
      Operands::A(op) => op.dump(buf),
      Operands::AS(op) => op.dump(buf),
      Operands::Cond(op) => op.dump(buf),
      Operands::CondS(op) => op.dump(buf),
    }
  }
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

#[derive(Debug, Copy, Clone)]
pub struct Bytecode(pub Operator, pub Operands);

define_bytecode! {
  Trap   (ABC, OpABC, op)   fn trap(dst: Op8, o1: Op8, o2: Op8)   { dst, o1, o2 } => ("{:<12} #{}, r{}, r{}", "trap", op.dst, op.o1, op.o2),
  Nop    (N)                fn nop()                              {}              => ("{:<12}", "nop"),
  Exta   (A, OpA, op)       fn exta(o1: Op24)                     { o1 }          => ("{:<12} #{}", "exta", op.o1),
  LoadI  (ABS, OpABS, op)   fn loadi(dst: Op8, o1: OpS16)         { dst, o1 }     => ("{:<12} r{}, #{}", "loadi", op.dst, op.o1),
  LoaduI (AB, OpAB, op)     fn loadui(dst: Op8, o1: Op16)         { dst, o1 }     => ("{:<12} r{}, #{}", "loadui", op.dst, op.o1),
  LoadR  (AB, OpAB, op)     fn loadr(dst: Op8, raw: Op16)         { dst, o1: raw } => ("{:<12} r{}, {}", "loadr", op.dst, RawImm(u16::from(op.o1))),
  LoadC  (AB, OpAB, op)     fn loadc(dst: Op8, o1: Op16)          { dst, o1 }     => ("{:<12} r{}, @{}", "loadc", op.dst, op.o1),
  LoadType (AB, OpAB, op)   fn loadtype(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<12} r{}, @{}", "loadtype", op.dst, op.o1),
  LoadFree (AB, OpAB, op)   fn loadfree(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<12} r{}, ^{}", "loadfree", op.dst, op.o1),
  LoadField (ABC, OpABC, op) fn loadfield(dst: Op8, o1: Op8, o2: Op8) { dst, o1, o2 } => ("{:<12} r{}, r{}, @{}", "load.field", op.dst, op.o1, op.o2),
  SetField (ABC, OpABC, op) fn setfield(src: Op8, o1: Op8, o2: Op8) { dst: src, o1, o2 } => ("{:<12} r{}, r{}, @{}", "set.field", op.dst, op.o1, op.o2),
  Move   (ABC, OpABC, op)   fn mov(dst: Op8, o1: Op8)             { dst, o1, o2: 0.into() } => ("{:<12} r{}, r{}", "move", op.dst, op.o1),
  Apply  (AB, OpAB, op)     fn apply(dst: Op8, o1: Op16)          { dst, o1 }     => ("{:<12} r{}, #{}", "apply", op.dst, op.o1),
  Invoke (ABC, OpABC, op)   fn invoke(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<12} r{}, r{}, @{}", "invoke", op.dst, op.o1, op.o2),
  Call   (AB, OpAB, op)     fn call(dst: Op8, o1: Op16)           { dst, o1 }     => ("{:<12} r{}, fn{}", "call", op.dst, op.o1),
  Retu   (N)                fn retu()                             {}              => ("{:<12}", "retu"),
  Ret    (ABC, OpABC, op)   fn ret(src: Op8)                      { dst: src, o1: 0.into(), o2: 0.into() } => ("{:<12} r{}", "ret", op.dst),
  Retn   (AB, OpAB, op)     fn retn(dst: Op8, o1: Op16)           { dst, o1 }     => ("{:<12} r{}, #{}", "retn", op.dst, op.o1),
  Clos   (AB, OpAB, op)     fn clos(dst: Op8, id: Op16)           { dst, o1: id } => ("{:<12} r{}, fn{}", "clos", op.dst, op.o1),
  WObj   (ABC, OpABC, op)   fn wobj(dst: Op8, tag: Op8, n: Op8)   { dst, o1: tag, o2: n } => ("{:<12} r{}, {}, #{}", "wrap", op.dst, Tag::from(u8::from(op.o1)), op.o2),
  Jmp    (AS, OpAS, op)     fn jmp(dst: OpS24)                    { dst }         => ("{:<12} #{}", "jmp", op.dst),
  Goto   (AB, OpAB, op)     fn goto(addr: Op8)                    { dst: addr, o1: 0.into() } => ("{:<12} #{}", "goto", op.dst),

  AddDC  (XYZ, OpXYZ, op)   fn adddc(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<12} r{}, r{}, @{}", "add.dc", op.dst, op.o1, op.o2),
  SubDC  (XYZ, OpXYZ, op)   fn subdc(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<12} r{}, r{}, @{}", "sub.dc", op.dst, op.o1, op.o2),
  MulDC  (XYZ, OpXYZ, op)   fn muldc(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<12} r{}, r{}, @{}", "mul.dc", op.dst, op.o1, op.o2),
  DivDC  (XYZ, OpXYZ, op)   fn divdc(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<12} r{}, r{}, @{}", "div.dc", op.dst, op.o1, op.o2),
  RemDC  (XYZ, OpXYZ, op)   fn remdc(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<12} r{}, r{}, @{}", "rem.dc", op.dst, op.o1, op.o2),

  AddDD  (XYZ, OpXYZ, op)   fn adddd(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<12} r{}, r{}, r{}", "add.dd", op.dst, op.o1, op.o2),
  SubDD  (XYZ, OpXYZ, op)   fn subdd(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<12} r{}, r{}, r{}", "sub.dd", op.dst, op.o1, op.o2),
  MulDD  (XYZ, OpXYZ, op)   fn muldd(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<12} r{}, r{}, r{}", "mul.dd", op.dst, op.o1, op.o2),
  DivDD  (XYZ, OpXYZ, op)   fn divdd(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<12} r{}, r{}, r{}", "div.dd", op.dst, op.o1, op.o2),
  RemDD  (XYZ, OpXYZ, op)   fn remdd(dst: Op8, o1: Op8, o2: Op8)  { dst, o1, o2 } => ("{:<12} r{}, r{}, r{}", "rem.dd", op.dst, op.o1, op.o2),
  NegD   (XYZ, OpXYZ, op)   fn negd(dst: Op8, o1: Op8)            { dst, o1, o2: 0.into() } => ("{:<12} r{}, r{}", "neg.d", op.dst, op.o1),

  SetCond  (ABS, OpABS, op) fn setcond(dst: Op8, o1: OpS16)       { dst, o1 }     => ("{:<12} r{}, #{}", "setc", op.dst, op.o1),
  SetCondJ (ABS, OpABS, op) fn setcondj(dst: Op8, o1: OpS16)      { dst, o1 }     => ("{:<12} r{}, #{}", "setcj", op.dst, op.o1),

  CmpNotF (Cond, OpCond, op) fn cmpnotf(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<12} r{}, #{}", "cmp.not.f", op.dst, op.o1),
  CmpEqDI (CondS, OpCondS, op) fn cmpeqdi(dst: Op8, o1: OpS16)    { dst, o1 }     => ("{:<12} r{}, #{}", "cmp.eq.di", op.dst, op.o1),
  CmpNeDI (CondS, OpCondS, op) fn cmpnedi(dst: Op8, o1: OpS16)    { dst, o1 }     => ("{:<12} r{}, #{}", "cmp.ne.di", op.dst, op.o1),

  CmpEqDC (Cond, OpCond, op) fn cmpeqdc(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<12} r{}, @{}", "cmp.eq.dc", op.dst, op.o1),
  CmpNeDC (Cond, OpCond, op) fn cmpnedc(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<12} r{}, @{}", "cmp.ne.dc", op.dst, op.o1),
  CmpLtDC (Cond, OpCond, op) fn cmpltdc(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<12} r{}, @{}", "cmp.lt.dc", op.dst, op.o1),
  CmpLeDC (Cond, OpCond, op) fn cmpledc(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<12} r{}, @{}", "cmp.le.dc", op.dst, op.o1),
  CmpGtDC (Cond, OpCond, op) fn cmpgtdc(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<12} r{}, @{}", "cmp.gt.dc", op.dst, op.o1),
  CmpGeDC (Cond, OpCond, op) fn cmpgedc(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<12} r{}, @{}", "cmp.ge.dc", op.dst, op.o1),

  CmpEqDD (Cond, OpCond, op) fn cmpeqdd(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<12} r{}, r{}", "cmp.eq.dd", op.dst, op.o1),
  CmpNeDD (Cond, OpCond, op) fn cmpnedd(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<12} r{}, r{}", "cmp.ne.dd", op.dst, op.o1),
  CmpLtDD (Cond, OpCond, op) fn cmpltdd(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<12} r{}, r{}", "cmp.lt.dd", op.dst, op.o1),
  CmpLeDD (Cond, OpCond, op) fn cmpledd(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<12} r{}, r{}", "cmp.le.dd", op.dst, op.o1),
  // These are necessary: a < b is not equivalent to !(a >= b) under IEEE754
  CmpGtDD (Cond, OpCond, op) fn cmpgtdd(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<12} r{}, r{}", "cmp.gt.dd", op.dst, op.o1),
  CmpGeDD (Cond, OpCond, op) fn cmpgedd(dst: Op8, o1: Op16)       { dst, o1 }     => ("{:<12} r{}, r{}", "cmp.ge.dd", op.dst, op.o1),
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

pub struct ThunkCtx {
  name: String,
  code: Vec<Bytecode>,
  relocate: Vec<(u32, Label)>, // (pc to be relocated, label)
  labels: HashMap<Label, u32>, // (label, the pc of the label)
  fresh: i32,
  fvlocs: Box<[Location]>,
  nparams: u8,
  nregs: u8,
  constants: ConstantPool,
}

pub struct Thunk {
  pub name: String,
  pub code: Box<[Bytecode]>,
  pub fvlocs: Box<[Location]>,
  pub nparams: u8,
  pub nregs: u8,
  pub constants: Box<[Val]>,
}

/// Image-owned description of a struct type. Member names map to slots of one
/// array: the declared fields first, then the methods in declaration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDesc {
  pub name: String,
  pub nfields: u16,
  pub nslots: u16,
  pub members: Box<[(String, u16)]>,
}

impl TypeDesc {
  pub fn new(name: &str, fields: &[&str], methods: &[&str]) -> Result<Self, String> {
    let mut members: IndexMap<String, u16> = IndexMap::new();
    for (slot, name) in fields.iter().chain(methods).enumerate() {
      if members.insert(name.to_string(), slot as u16).is_some() {
        return Err(format!("duplicate member `{name}`"));
      }
    }
    Ok(Self {
      name: name.to_string(),
      nfields: fields.len() as u16,
      nslots: (fields.len() + methods.len()) as u16,
      members: members.into_iter().collect(),
    })
  }

  pub fn slot(&self, name: &str) -> Option<u16> {
    self.members.iter().find(|m| m.0 == name).map(|m| m.1)
  }

  /// The declared name of a slot.
  pub fn slot_name(&self, slot: u16) -> &str {
    self.members.iter().find(|m| m.1 == slot).map_or("?", |m| m.0.as_str())
  }
}

impl ThunkCtx {
  pub fn new(name: &str, fvlocs: Box<[Location]>, nparams: u8) -> Self {
    Self {
      name: name.to_string(),
      code: Vec::new(),
      relocate: Vec::new(),
      labels: HashMap::new(),
      fresh: 0,
      fvlocs,
      nparams,
      nregs: 0,
      constants: ConstantPool::new(),
    }
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

  pub fn fetch(&mut self, pc: u32) -> Bytecode {
    self.code[pc as usize]
  }

  pub fn reverse_setcond(&mut self, pc: u32) -> bool {
    let new_opc = match self.fetch(pc) {
      Bytecode(Operator::SetCond, Operands::ABS(OpABS { dst, o1 })) => {
        Bytecode::setcond(dst, (if o1.0 == 0 { 1 } else { 0 }).into())
      }
      Bytecode(Operator::SetCondJ, Operands::ABS(OpABS { dst, o1 })) => {
        Bytecode::setcondj(dst, (if o1.0 == 0 { 1 } else { 0 }).into())
      }
      _ => return false,
    };
    self.edit(pc, new_opc);
    true
  }

  pub fn relocate_all(mut self, heap: &mut OwnedHeap) -> Thunk {
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
    Thunk {
      name: self.name,
      code: self.code.into_boxed_slice(),
      fvlocs: self.fvlocs,
      nparams: self.nparams,
      nregs: self.nregs,
      constants: self.constants.to_vec(heap),
    }
  }
}

impl Display for Thunk {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "thunk::{} params::{} regs::{} captured::[", self.name, self.nparams, self.nregs)?;
    let mut fvlocs = self.fvlocs.iter();
    if let Some(fvloc) = fvlocs.next() {
      write!(f, "{fvloc} as ^1")?;

      for (i, fvloc) in fvlocs.enumerate() {
        let i = i + 2;
        write!(f, ", {fvloc} as ^{i}")?;
      }
    }
    writeln!(f, "]")?;

    if !self.constants.is_empty() {
      writeln!(f, "constants::[")?;
      for (i, constant) in self.constants.iter().enumerate() {
        writeln!(f, "  @{}: {}", i, constant)?;
      }
      writeln!(f, "]")?;
    }

    for code in self.code.iter() {
      writeln!(f, "{code}")?;
    }
    Ok(())
  }
}

pub struct BytecodeCtx {
  buffer: Vec<ThunkCtx>,
  current: ThunkCtx,
  finished: Vec<Thunk>,
  types: Vec<TypeDesc>,
  heap: OwnedHeap,
}

/// Bytecode together with the exclusive VM heap backing its pointer-valued constants.
pub struct BytecodeImage {
  thunks: Vec<Thunk>,
  types: Vec<TypeDesc>,
  heap: OwnedHeap,
}

impl BytecodeImage {
  pub(crate) fn new(thunks: Vec<Thunk>, types: Vec<TypeDesc>, heap: OwnedHeap) -> Self {
    Self { thunks, types, heap }
  }

  pub fn thunks(&self) -> &[Thunk] {
    &self.thunks
  }

  pub fn types(&self) -> &[TypeDesc] {
    &self.types
  }

  pub(crate) fn into_parts(self) -> (Vec<Thunk>, Vec<TypeDesc>, OwnedHeap) {
    (self.thunks, self.types, self.heap)
  }
}

impl BytecodeCtx {
  pub fn new(heap: OwnedHeap) -> Self {
    Self {
      buffer: Vec::new(),
      current: ThunkCtx::new("__top_thunk__", Box::new([]), 0),
      finished: Vec::new(),
      types: Vec::new(),
      heap,
    }
  }

  pub fn add_type(&mut self, desc: TypeDesc) -> Option<u16> {
    let id = u16::try_from(self.types.len()).ok()?;
    self.types.push(desc);
    Some(id)
  }

  pub fn type_desc(&self, id: u16) -> &TypeDesc {
    &self.types[id as usize]
  }

  pub fn push_thunk(&mut self, name: &str, fvlocs: Box<[Location]>, nparams: u8) {
    self.buffer.push(std::mem::replace(&mut self.current, ThunkCtx::new(name, fvlocs, nparams)))
  }

  pub fn pop_thunk(&mut self) -> usize {
    let len = self.finished.len();
    let thunk = std::mem::replace(&mut self.current, self.buffer.pop().unwrap());
    self.finished.push(thunk.relocate_all(&mut self.heap));
    len
  }

  pub fn push(&mut self, code: Bytecode) {
    self.current.push(code);
  }

  pub fn pc(&self) -> u32 {
    self.current.pc()
  }

  pub fn set_nregs(&mut self, nregs: u8) {
    self.current.nregs = nregs;
  }

  pub fn add_int(&mut self, n: i64) -> ConstantId {
    self.current.constants.add_int(n)
  }

  pub fn add_float(&mut self, f: f64) -> ConstantId {
    self.current.constants.add_float(f)
  }

  pub fn add_str(&mut self, s: String) -> ConstantId {
    self.current.constants.add_str(s)
  }

  pub fn fresh_label(&mut self) -> Label {
    self.current.fresh_label()
  }

  pub fn push_label(&mut self, label: Label) {
    self.current.push_label(label)
  }

  pub fn push_relocate(&mut self, label: Label) {
    self.current.push_relocate(label)
  }

  pub fn edit(&mut self, pc: u32, code: Bytecode) {
    self.current.edit(pc, code)
  }

  pub fn fetch(&mut self, pc: u32) -> Bytecode {
    self.current.fetch(pc)
  }

  pub fn reverse_setcond(&mut self, pc: u32) -> bool {
    self.current.reverse_setcond(pc)
  }

  pub fn finalize(self) -> BytecodeImage {
    let Self { buffer, current, mut finished, types, mut heap } = self;
    assert!(buffer.is_empty());
    finished.push(current.relocate_all(&mut heap));
    BytecodeImage::new(finished, types, heap)
  }
}
