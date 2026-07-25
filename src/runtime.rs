use std::{ffi::CStr, ptr::NonNull, rc::Rc};

use crate::{
  bytecode::{
    BinaryRepr, Bytecode, BytecodeImage, Location, Operands, Operator, Tag, Thunk, TrapId,
  },
  diagnostic::{Diagnostic, Result},
  val::Val,
  vm,
};

const STACK_HEADROOM_SLOTS: usize = 256;
const RESULT_BUF_LEN: usize = 64;

pub struct OwnedHeap {
  ptr: NonNull<vm::heap>,
}

impl OwnedHeap {
  pub fn new(diag: &Diagnostic) -> Result<Self> {
    Self::with_args(
      vm::runtime_args {
        trace_level: 0,
        base_size: 1024 * 1024,
        align: 8,
        descspace_size: 4 * 4 * 4096,
      },
      diag,
    )
  }

  fn with_args(args: vm::runtime_args, diag: &Diagnostic) -> Result<Self> {
    let ptr = unsafe { vm::vm_heap_alloc(&args) };
    let ptr = NonNull::new(ptr).ok_or_else(|| diag.error("failed to initialize vm heap"))?;
    Ok(Self { ptr })
  }

  pub(crate) fn alloc_str(&mut self, value: &str) -> Option<Val> {
    let len = u32::try_from(value.len()).ok()?;
    let mut out = 0;
    if unsafe { vm::vm_const_from_str(value.as_ptr().cast(), len, self.ptr.as_ptr(), &mut out) } {
      Some(Val::from_raw(out))
    } else {
      None
    }
  }

  fn as_ptr(&self) -> *mut vm::heap {
    self.ptr.as_ptr()
  }
}

impl Drop for OwnedHeap {
  fn drop(&mut self) {
    unsafe {
      vm::vm_heap_free(self.ptr.as_ptr());
    }
  }
}

/// Executes `image`, consuming both its bytecode and its associated heap.
pub fn execute(image: BytecodeImage, diag: Rc<Diagnostic>) -> Result<String> {
  let validator = ImageValidator::new(Rc::clone(&diag));
  validator.validate(image.thunks())?;

  let max_nregs = image.thunks().iter().map(|t| t.nregs as usize).max().unwrap_or(0);
  let numobject = compute_numobject(image.thunks());
  let (mut thunks, heap) = image.into_parts();
  append_entry_thunk(&mut thunks, &diag)?;

  let mut native_thunks = NativeThunkSet::new(&thunks, &diag)?;
  let mut result = 0;
  let stack_slots = STACK_HEADROOM_SLOTS + max_nregs;
  let status = unsafe {
    vm::vm_exec_with(
      heap.as_ptr(),
      native_thunks.entry_ptr(),
      native_thunks.thunk_ptrs_mut(),
      thunks.len(),
      numobject,
      stack_slots,
      &mut result,
      std::ptr::null_mut(),
    )
  };

  if status != vm::status_t_S_OK {
    return diag.fail(format!("vm execution failed with status {}", status_name(status)));
  }

  format_result(result, &diag)
}

fn append_entry_thunk(thunks: &mut Vec<Thunk>, diag: &Diagnostic) -> Result<()> {
  let top_idx: u16 =
    (thunks.len() - 1).try_into().map_err(|_| diag.error("too many functions to execute"))?;
  thunks.push(Thunk {
    name: "__entry__".to_string(),
    code: vec![
      Bytecode::call(0u8.into(), top_idx.into()),
      Bytecode::trap(TrapId::HALT.into(), 0u8.into(), 0u8.into()),
    ]
    .into(),
    fvlocs: Box::new([]),
    nparams: 0,
    nregs: 0,
    constants: Box::new([]),
  });
  Ok(())
}

fn compute_numobject(thunks: &[Thunk]) -> usize {
  let mut max_tag: usize = 0;
  for thunk in thunks {
    for bc in thunk.code.iter() {
      match *bc {
        Bytecode(Operator::WObj, Operands::ABC(op)) => {
          let tag = u8::from(op.o1) as usize;
          if tag > max_tag {
            max_tag = tag;
          }
        }
        _ => {}
      }
    }
  }
  max_tag
}

struct ImageValidator {
  diag: Rc<Diagnostic>,
}

impl ImageValidator {
  fn new(diag: Rc<Diagnostic>) -> Self {
    Self { diag }
  }

  fn validate(&self, thunks: &[Thunk]) -> Result<()> {
    if thunks.is_empty() {
      return self.diag.fail("cannot execute an empty bytecode image");
    }

    for thunk in thunks {
      for loc in thunk.fvlocs.iter() {
        if matches!(loc, Location::Temporary) {
          return self.diag.fail("temporary location in thunk capture list");
        }
      }
      for bc in thunk.code.iter() {
        self.validate_bytecode(*bc, thunk.fvlocs.len())?;
      }
    }

    Ok(())
  }

  fn validate_bytecode(&self, bytecode: Bytecode, nfree: usize) -> Result<()> {
    use Operator::*;
    match bytecode {
      Bytecode(LoadF, Operands::AB(op)) if usize::from(u16::from(op.o1)) <= nfree => Ok(()),
      Bytecode(LoadF | SetF, _) => self.diag.fail(format!("illegal instruction: {}", bytecode)),
      Bytecode(MObj, Operands::ABC(op)) if u8::from(op.o1) >= u8::from(Tag::INT) => {
        self.diag.fail(format!("illegal instruction: heap-backed mobj: {}", bytecode))
      }
      Bytecode(LoadR, Operands::AB(op)) => {
        let value = Val::from_raw(u64::from(u16::from(op.o1)));
        if value.is_empty() || value.is_null() || value.is_bool() {
          Ok(())
        } else {
          self.diag.fail(format!("illegal instruction: noncanonical raw value: {}", bytecode))
        }
      }
      _ => Ok(()),
    }
  }
}

struct NativeThunkSet {
  thunks: Vec<OwnedThunk>,
}

impl NativeThunkSet {
  fn new(thunks: &[Thunk], diag: &Diagnostic) -> Result<Self> {
    let mut native_thunks = Vec::with_capacity(thunks.len());

    for thunk in thunks {
      native_thunks.push(OwnedThunk::from_thunk(thunk, diag)?);
    }

    Ok(Self { thunks: native_thunks })
  }

  fn thunk_ptrs_mut(&mut self) -> *mut *mut vm::thunk {
    // SAFETY: OwnedThunk is #[repr(transparent)] over NonNull<vm::thunk>,
    // which has the same layout as *mut vm::thunk.
    self.thunks.as_mut_ptr().cast()
  }

  fn entry_ptr(&self) -> *mut vm::thunk {
    self.thunks.last().unwrap().as_ptr()
  }
}

/// SAFETY: This type is `#[repr(transparent)]` over `NonNull<vm::thunk>`,
/// which has the same layout as `*mut vm::thunk`. This allows
/// `&mut [OwnedThunk]` to be cast to `*mut *mut vm::thunk` for FFI
/// without copying.
#[repr(transparent)]
struct OwnedThunk {
  ptr: NonNull<vm::thunk>,
}

impl OwnedThunk {
  fn from_thunk(thunk: &Thunk, diag: &Diagnostic) -> Result<Self> {
    let ops = thunk.code.iter().map(|bc| encode_bytecode(*bc)).collect::<Vec<_>>();
    let fvlocs = encode_capture_locations(&thunk.fvlocs, diag)?;
    let ptr = unsafe {
      vm::vm_thunk_alloc(
        ops.as_ptr(),
        ops.len(),
        thunk.constants.as_ptr().cast::<vm::val_t>(),
        thunk.constants.len(),
        thunk.nregs,
        fvlocs.as_ptr(),
        thunk.fvlocs.len(),
      )
    };
    let ptr = NonNull::new(ptr).ok_or_else(|| diag.error("failed to allocate vm thunk"))?;
    Ok(Self { ptr })
  }

  fn as_ptr(&self) -> *mut vm::thunk {
    self.ptr.as_ptr()
  }
}

fn encode_capture_locations(
  fvlocs: &[Location],
  diag: &Diagnostic,
) -> Result<Vec<vm::capture_loc>> {
  fvlocs
    .iter()
    .map(|loc| match loc {
      Location::Slot(reg) => Ok(vm::capture_loc { kind: 0, index: reg.0.into() }),
      Location::FreeVar(fv) => Ok(vm::capture_loc { kind: 1, index: fv.0 }),
      Location::Temporary => diag.fail("temporary location in thunk capture list"),
    })
    .collect()
}

impl Drop for OwnedThunk {
  fn drop(&mut self) {
    unsafe {
      vm::vm_thunk_free(self.ptr.as_ptr());
    }
  }
}

fn encode_bytecode(bytecode: Bytecode) -> vm::bc_t {
  let mut buf = [0u8; 4];
  bytecode.dump(&mut buf);
  u32::from_le_bytes(buf)
}

fn status_name(status: vm::status_t) -> String {
  unsafe { CStr::from_ptr(vm::vm_status_name(status)).to_string_lossy().into_owned() }
}

fn format_result(value: vm::val_t, diag: &Diagnostic) -> Result<String> {
  let mut buf = [0u8; RESULT_BUF_LEN];
  let ok = unsafe { vm::vm_format_result(value, buf.as_mut_ptr().cast(), buf.len()) };
  if ok {
    let cstr = unsafe { CStr::from_ptr(buf.as_ptr().cast()) };
    Ok(cstr.to_string_lossy().into_owned())
  } else {
    diag.fail("vm produced an unsupported result")
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bytecode::BytecodeCtx;
  use crate::codegen::CodeGenCtx;
  use crate::parser::Parser;
  use bumpalo::Bump;
  use std::sync::Mutex;

  static VM_LOCK: Mutex<()> = Mutex::new(());

  fn compile_and_run(source: &str) -> Result<String> {
    let _guard = VM_LOCK.lock().unwrap();
    let arena = Bump::new();
    let diag = Rc::new(Diagnostic::new());
    let parser = Parser::new(&arena, Rc::clone(&diag), source);
    let tree = parser.parse()?;
    let diag = Rc::new(Diagnostic::new());
    let mut codegen = CodeGenCtx::new(&arena, Rc::clone(&diag), tree);
    let heap = OwnedHeap::new(&diag)?;
    let mut bc = BytecodeCtx::new(heap);
    codegen.emit_tree(&mut bc)?;
    let image = bc.finalize();
    execute(image, Rc::clone(&diag))
  }

  fn execute_with_stats(
    image: BytecodeImage,
    diag: Rc<Diagnostic>,
  ) -> Result<(String, vm::gc_stats)> {
    let _guard = VM_LOCK.lock().unwrap();
    let validator = ImageValidator::new(Rc::clone(&diag));
    validator.validate(image.thunks())?;
    let max_nregs = image.thunks().iter().map(|t| t.nregs as usize).max().unwrap_or(0);
    let numobject = compute_numobject(image.thunks());
    let (mut thunks, heap) = image.into_parts();
    append_entry_thunk(&mut thunks, &diag)?;
    let mut native_thunks = NativeThunkSet::new(&thunks, &diag)?;
    let mut result = 0;
    let stack_slots = STACK_HEADROOM_SLOTS + max_nregs;
    let mut stats: vm::gc_stats = unsafe { std::mem::zeroed() };
    let status = unsafe {
      vm::vm_exec_with(
        heap.as_ptr(),
        native_thunks.entry_ptr(),
        native_thunks.thunk_ptrs_mut(),
        thunks.len(),
        numobject,
        stack_slots,
        &mut result,
        &mut stats,
      )
    };
    if status != vm::status_t_S_OK {
      return diag.fail(format!("vm execution failed with status {}", status_name(status)));
    }
    Ok((format_result(result, &diag)?, stats))
  }

  fn obj_size(nfields: usize) -> usize {
    unsafe { vm::vm_object_size_for_fields(nfields) }
  }

  #[test]
  fn execution_basic_arithmetic() {
    let result = compile_and_run("1 + 2 + 3").unwrap();
    assert_eq!(result, "6");
  }

  #[test]
  fn execution_conditional() {
    let result = compile_and_run("if 10 > 5 then 42 else 0 end").unwrap();
    assert_eq!(result, "42");
  }

  #[test]
  fn execution_boolean_literals() {
    assert_eq!(compile_and_run("false").unwrap(), "false");
    assert_eq!(compile_and_run("true").unwrap(), "true");
  }

  #[test]
  fn execution_unit_literal() {
    let result = compile_and_run("()").unwrap();
    assert_eq!(result, "()");
  }

  #[test]
  fn execution_unit_binding() {
    let result = compile_and_run("let x = (); x").unwrap();
    assert_eq!(result, "()");
  }

  #[test]
  fn execution_unit_is_falsy() {
    let result = compile_and_run("if () then 1 else 2 end").unwrap();
    assert_eq!(result, "2");
  }

  #[test]
  fn execution_empty_array_literal() {
    let result = compile_and_run("[]").unwrap();
    assert_eq!(result, "[]");
  }

  #[test]
  fn execution_empty_map_literal() {
    let result = compile_and_run("{}").unwrap();
    assert_eq!(result, "{}");
  }

  #[test]
  fn execution_string_result_uses_escaped_display() {
    let result = compile_and_run("\"hello\"").unwrap();
    assert_eq!(result, "\"hello\"");

    let result = compile_and_run("\"\\n\\t\\\\\\\"\"").unwrap();
    assert_eq!(result, "\"\\n\\t\\\\\\\"\"");

    let result = compile_and_run(concat!("\"", "\u{e9}", "\"")).unwrap();
    assert_eq!(result, concat!("\"", "\u{e9}", "\""));

    let result = compile_and_run(concat!("\"", "\u{1}", "\"")).unwrap();
    assert_eq!(result, "\"\\x01\"");
  }

  #[test]
  fn validator_allows_captures_and_valid_loadf() {
    use crate::bytecode::Location;
    let diag = Rc::new(Diagnostic::new());
    let validator = ImageValidator::new(Rc::clone(&diag));
    let thunk = Thunk {
      name: String::new(),
      code: vec![Bytecode::loadf(0.into(), 0.into()), Bytecode::loadf(0.into(), 1.into())].into(),
      fvlocs: vec![Location::Slot(0.into())].into(),
      nparams: 0,
      nregs: 1,
      constants: vec![].into(),
    };
    assert!(validator.validate(&[thunk]).is_ok());
  }

  #[test]
  fn validator_rejects_out_of_range_loadf() {
    use crate::bytecode::Location;
    let diag = Rc::new(Diagnostic::new());
    let validator = ImageValidator::new(Rc::clone(&diag));
    let thunk = Thunk {
      name: String::new(),
      code: vec![Bytecode::loadf(0.into(), 2.into())].into(),
      fvlocs: vec![Location::Slot(0.into())].into(),
      nparams: 0,
      nregs: 1,
      constants: vec![].into(),
    };
    assert!(validator.validate(&[thunk]).is_err());
  }

  #[test]
  fn validator_rejects_setf() {
    let diag = Rc::new(Diagnostic::new());
    let validator = ImageValidator::new(Rc::clone(&diag));
    let thunk = Thunk {
      name: String::new(),
      code: vec![Bytecode::setf(0.into(), 0.into())].into(),
      fvlocs: vec![].into(),
      nparams: 0,
      nregs: 1,
      constants: vec![].into(),
    };
    assert!(validator.validate(&[thunk]).is_err());
  }

  #[test]
  fn validator_rejects_noncanonical_loadr() {
    let diag = Rc::new(Diagnostic::new());
    let validator = ImageValidator::new(Rc::clone(&diag));
    let thunk = Thunk {
      name: String::new(),
      code: vec![Bytecode::loadr(0.into(), 1.into())].into(),
      fvlocs: vec![].into(),
      nparams: 0,
      nregs: 1,
      constants: vec![].into(),
    };
    assert!(validator.validate(&[thunk]).is_err());
  }

  #[test]
  fn validator_rejects_heap_mobj() {
    let diag = Rc::new(Diagnostic::new());
    let validator = ImageValidator::new(Rc::clone(&diag));
    let thunk = Thunk {
      name: String::new(),
      code: vec![Bytecode::mobj(0.into(), Tag::INT.into(), 0.into())].into(),
      fvlocs: vec![].into(),
      nparams: 0,
      nregs: 1,
      constants: vec![].into(),
    };
    assert!(validator.validate(&[thunk]).is_err());
  }

  #[test]
  fn validator_allows_trivial_mobj() {
    let diag = Rc::new(Diagnostic::new());
    let validator = ImageValidator::new(Rc::clone(&diag));
    let thunk = Thunk {
      name: String::new(),
      code: vec![Bytecode::mobj(0.into(), Tag::UNIT.into(), 0.into())].into(),
      fvlocs: vec![].into(),
      nparams: 0,
      nregs: 1,
      constants: vec![].into(),
    };
    assert!(validator.validate(&[thunk]).is_ok());
  }

  #[test]
  fn validator_allows_canonical_loadr() {
    let diag = Rc::new(Diagnostic::new());
    let validator = ImageValidator::new(Rc::clone(&diag));
    let thunk = Thunk {
      name: String::new(),
      code: vec![
        Bytecode::loadr(0.into(), u16::try_from(Val::empty().raw()).unwrap().into()),
        Bytecode::loadr(0.into(), u16::try_from(Val::null().raw()).unwrap().into()),
        Bytecode::loadr(0.into(), u16::try_from(Val::from_bool(false).raw()).unwrap().into()),
        Bytecode::loadr(0.into(), u16::try_from(Val::from_bool(true).raw()).unwrap().into()),
      ]
      .into(),
      fvlocs: vec![].into(),
      nparams: 0,
      nregs: 1,
      constants: vec![].into(),
    };
    assert!(validator.validate(&[thunk]).is_ok());
  }

  #[test]
  fn validator_rejects_temporary_captures() {
    use crate::bytecode::Location;
    let diag = Rc::new(Diagnostic::new());
    let validator = ImageValidator::new(Rc::clone(&diag));
    let thunk = Thunk {
      name: String::new(),
      code: vec![].into(),
      fvlocs: vec![Location::Temporary].into(),
      nparams: 0,
      nregs: 0,
      constants: vec![].into(),
    };
    assert!(validator.validate(&[thunk]).is_err());
  }

  #[test]
  fn execution_nested_let() {
    let result = compile_and_run("let x = 10; let y = x * 2; y + 1").unwrap();
    assert_eq!(result, "21");
  }

  #[test]
  fn gc_sweep_accounting_regression() {
    let s1 = obj_size(1);
    let heap_size: usize = 16384;
    let n_dead = heap_size / s1 + 200;
    let mut code = Vec::new();
    for i in 1u8..=5 {
      code.push(Bytecode::loadi(i.into(), (i as i16).into()));
      code.push(Bytecode::wobj(i.into(), 0u8.into(), 1u8.into()));
    }
    for j in 0..n_dead {
      let v = (j & 0x7fff) as i16;
      code.push(Bytecode::loadi(6u8.into(), v.into()));
      code.push(Bytecode::wobj(6u8.into(), 0u8.into(), 1u8.into()));
    }
    code.push(Bytecode::loadi(0u8.into(), 42i16.into()));
    code.push(Bytecode::ret(0u8.into()));
    let thunk = Thunk {
      name: String::new(),
      code: code.into(),
      fvlocs: vec![].into(),
      nparams: 0,
      nregs: 7,
      constants: vec![].into(),
    };
    let diag = Rc::new(Diagnostic::new());
    let heap = OwnedHeap::with_args(
      vm::runtime_args { trace_level: 0, base_size: heap_size, align: 8, descspace_size: 4096 },
      &diag,
    )
    .unwrap();
    let image = BytecodeImage::new(vec![thunk], heap);
    let (result, stats) = execute_with_stats(image, diag).unwrap();
    assert_eq!(result, "42");
    assert_eq!(stats.mark_to_sweep_transitions, 1);
    assert!(
      stats.last_completed_trigger_bytes > 4096,
      "trigger {} must exceed the 4096 floor (swept-prefix allocs counted)",
      stats.last_completed_trigger_bytes,
    );
  }

  #[test]
  fn gc_coalescing_regression() {
    let s1 = obj_size(1);
    let s3 = obj_size(3);
    assert!(s3 > s1, "3-field object must be larger than 1-field object");
    assert!(s3 <= 2 * s1, "3-field object must fit in two coalesced 1-field blocks");
    let heap_size = 1024;
    let n_fill = heap_size / s1;
    let mut code = Vec::new();
    code.push(Bytecode::loadi(1u8.into(), 1i16.into()));
    code.push(Bytecode::wobj(1u8.into(), 0u8.into(), 1u8.into()));
    code.push(Bytecode::loadi(2u8.into(), 2i16.into()));
    code.push(Bytecode::wobj(2u8.into(), 0u8.into(), 1u8.into()));
    code.push(Bytecode::loadi(3u8.into(), 3i16.into()));
    code.push(Bytecode::wobj(3u8.into(), 0u8.into(), 1u8.into()));
    code.push(Bytecode::loadi(4u8.into(), 4i16.into()));
    code.push(Bytecode::wobj(4u8.into(), 0u8.into(), 1u8.into()));
    for _ in 0..n_fill {
      code.push(Bytecode::loadi(5u8.into(), 0i16.into()));
      code.push(Bytecode::wobj(5u8.into(), 0u8.into(), 1u8.into()));
    }
    code.push(Bytecode::loadi(1u8.into(), 0i16.into()));
    code.push(Bytecode::loadi(2u8.into(), 0i16.into()));
    code.push(Bytecode::loadi(4u8.into(), 0i16.into()));
    code.push(Bytecode::loadi(5u8.into(), 10i16.into()));
    code.push(Bytecode::loadi(6u8.into(), 20i16.into()));
    code.push(Bytecode::loadi(7u8.into(), 30i16.into()));
    code.push(Bytecode::wobj(5u8.into(), 0u8.into(), 3u8.into()));
    code.push(Bytecode::loadi(0u8.into(), 99i16.into()));
    code.push(Bytecode::ret(0u8.into()));
    let thunk = Thunk {
      name: String::new(),
      code: code.into(),
      fvlocs: vec![].into(),
      nparams: 0,
      nregs: 8,
      constants: vec![].into(),
    };
    let diag = Rc::new(Diagnostic::new());
    let heap = OwnedHeap::with_args(
      vm::runtime_args { trace_level: 0, base_size: heap_size, align: 8, descspace_size: 4096 },
      &diag,
    )
    .unwrap();
    let image = BytecodeImage::new(vec![thunk], heap);
    let (result, _stats) = execute_with_stats(image, diag).unwrap();
    assert_eq!(result, "99");
  }
}
