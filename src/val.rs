use std::fmt::{self, Display};

unsafe extern "C" {
  fn vm_alloc_str(data: *const u8, len: u32) -> *mut u8;
}

const VAL_NUN_BIAS: u64 = 0x0002_0000_0000_0000;
const VAL_FLOAT_TAG: u64 = 0xfffe_0000_0000_0000;
const VAL_OTHER_TAG: u64 = 0x2;
const VAL_BOOL_TAG: u64 = 0x4;
const VAL_BOOL_VAL_BIT: u64 = 0x1;
const VAL_NOT_CELL_MASK: u64 = VAL_FLOAT_TAG | VAL_OTHER_TAG;

const VAL_EMPTY: u64 = 0x0;
const VAL_NULL: u64 = VAL_OTHER_TAG;
const VAL_FALSE: u64 = VAL_OTHER_TAG | VAL_BOOL_TAG;
const VAL_TRUE: u64 = VAL_OTHER_TAG | VAL_BOOL_TAG | VAL_BOOL_VAL_BIT;

// Object kind, matching `enum obj_kind` in vm/src/object.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectKind(u8);

impl ObjectKind {
  pub const WORDS: Self = ObjectKind(0);
  pub const NOSCAN: Self = ObjectKind(1);
}

impl From<u8> for ObjectKind {
  fn from(x: u8) -> Self {
    Self(x)
  }
}

impl TryFrom<usize> for ObjectKind {
  type Error = ();
  fn try_from(value: usize) -> Result<Self, Self::Error> {
    if value <= 0xff { Ok(Self(value as u8)) } else { Err(()) }
  }
}

impl From<ObjectKind> for u64 {
  fn from(x: ObjectKind) -> Self {
    x.0 as u64
  }
}

impl From<ObjectKind> for u8 {
  fn from(x: ObjectKind) -> Self {
    x.0
  }
}

/// Mirrors `struct str` in vm/src/object.h.
/// The metainfo size field stores `sizeof(StrObj) + len + 1` (the unpadded object size).
#[repr(C)]
struct StrObj {
  hd: u64,
  // followed by `len` bytes of data + '\0' sentinel
}

/// Maximum integer exactly representable as f64 (2^53).
pub const MAX_SAFE_INTEGER: i64 = 1_i64 << 53;
/// Minimum integer exactly representable as f64 (-(2^53)).
pub const MIN_SAFE_INTEGER: i64 = -(1_i64 << 53);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Val(u64);

impl Val {
  #[inline]
  pub const fn empty() -> Self {
    Val(VAL_EMPTY)
  }

  #[inline]
  pub const fn null() -> Self {
    Val(VAL_NULL)
  }

  #[inline]
  pub const fn from_bool(v: bool) -> Self {
    Val(if v { VAL_TRUE } else { VAL_FALSE })
  }

  #[inline]
  pub const fn from_i32(n: i32) -> Self {
    Val(VAL_FLOAT_TAG | (n as u32 as u64))
  }

  #[inline]
  pub fn from_f64(n: f64) -> Self {
    Val(n.to_bits() + VAL_NUN_BIAS)
  }

  /// Allocates an immutable string constant via the C VM allocator.
  pub fn from_rust_str(s: &str) -> Self {
    unsafe {
      let ptr = vm_alloc_str(s.as_ptr(), s.len() as u32);
      assert!(!ptr.is_null(), "vm_alloc_str failed");
      Val(ptr as u64)
    }
  }

  #[inline]
  pub fn is_empty(self) -> bool {
    self.0 == VAL_EMPTY
  }

  #[inline]
  pub fn is_null(self) -> bool {
    self.0 == VAL_NULL
  }

  #[inline]
  pub fn is_bool(self) -> bool {
    (self.0 & !VAL_BOOL_VAL_BIT) == VAL_FALSE
  }

  #[inline]
  pub fn is_int(self) -> bool {
    (self.0 & VAL_FLOAT_TAG) == VAL_FLOAT_TAG
  }

  #[inline]
  pub fn is_float(self) -> bool {
    (self.0 & VAL_FLOAT_TAG) != 0 && !self.is_int()
  }

  #[inline]
  pub fn is_cell(self) -> bool {
    (self.0 & VAL_NOT_CELL_MASK) == 0
  }

  /// Returns true if this is a non-empty cell pointer (e.g. a string object).
  #[inline]
  pub fn is_ptr(self) -> bool {
    self.is_cell() && !self.is_empty()
  }

  #[inline]
  pub fn as_i32(self) -> i32 {
    debug_assert!(self.is_int());
    (self.0 as u32) as i32
  }

  #[inline]
  pub fn as_f64(self) -> f64 {
    debug_assert!(self.is_float());
    f64::from_bits(self.0 - VAL_NUN_BIAS)
  }

  #[inline]
  pub fn as_bool(self) -> bool {
    debug_assert!(self.is_bool());
    self.0 == VAL_TRUE
  }

  /// Returns the string data as a byte slice. Only valid for string cell values.
  pub fn str_as_bytes(&self) -> &[u8] {
    assert!(self.is_ptr());
    unsafe {
      let obj = self.0 as *const StrObj;
      let hd = (*obj).hd;
      let kind = ((hd >> 48) & 0xff) as u8;
      assert!(kind == u8::from(ObjectKind::NOSCAN), "str_as_bytes: not a string object");
      let obj_size = (hd as u32) as usize;
      let len = obj_size - std::mem::size_of::<StrObj>() - 1;
      let bytes_ptr = (self.0 as *const u8).add(std::mem::size_of::<StrObj>());
      std::slice::from_raw_parts(bytes_ptr, len)
    }
  }

  /// Returns the string data as a &str. Only valid for string cell values
  /// containing valid UTF-8.
  pub fn str_as_str(&self) -> &str {
    std::str::from_utf8(self.str_as_bytes()).expect("invalid utf8 in string constant")
  }

  #[inline]
  pub fn raw(self) -> u64 {
    self.0
  }
}

impl From<Val> for u64 {
  fn from(v: Val) -> u64 {
    v.0
  }
}

impl Display for Val {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if self.is_int() {
      write!(f, "{}", self.as_i32())
    } else if self.is_float() {
      // Use Debug format for f64 to ensure `.0` suffix on whole numbers
      write!(f, "{:?}", self.as_f64())
    } else if self.is_null() {
      write!(f, "()")
    } else if self.is_bool() {
      write!(f, "{}", self.as_bool())
    } else if self.is_ptr() {
      write!(f, "\"{}\"", self.str_as_str().escape_default())
    } else {
      write!(f, "<val:{:#x}>", self.0)
    }
  }
}

impl fmt::Debug for Val {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    Display::fmt(self, f)
  }
}
