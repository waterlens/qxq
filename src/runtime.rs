use std::{ffi::CStr, ptr::NonNull, rc::Rc};

use crate::{
  bytecode::{
    BinaryRepr, Bytecode, BytecodeImage, Location, Operands, Operator, Tag, Thunk, TrapId, TypeDesc,
  },
  diagnostic::{Diagnostic, Result},
  val::Val,
  vm,
};

const STACK_HEADROOM_SLOTS: usize = 256;
/// Return-value and return-address slots between frames, as in vm.h.
const FRAME_HEADER_SIZE: usize = 2;

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
  run(image, diag, None)
}

fn run(
  image: BytecodeImage,
  diag: Rc<Diagnostic>,
  stats: Option<&mut vm::gc_stats>,
) -> Result<String> {
  let validator = ImageValidator::new(Rc::clone(&diag));
  validator.validate(image.thunks(), image.types())?;

  let max_nregs = image.thunks().iter().map(|t| t.nregs as usize).max().unwrap_or(0);
  let (mut thunks, types, heap) = image.into_parts();
  append_entry_thunk(&mut thunks, &diag)?;

  let mut native_thunks = NativeThunkSet::new(&thunks, &diag)?;
  let mut native_types = OwnedType::from_descs(&types, &diag)?;
  let mut result = 0;
  let stack_slots = STACK_HEADROOM_SLOTS + max_nregs;
  let status = unsafe {
    vm::vm_exec_with(
      heap.as_ptr(),
      native_thunks.entry_ptr(),
      native_thunks.thunk_ptrs_mut(),
      native_types.as_mut_ptr().cast(),
      thunks.len(),
      native_types.len(),
      stack_slots,
      &mut result,
      stats.map_or(std::ptr::null_mut(), |s| s),
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

struct ImageValidator {
  diag: Rc<Diagnostic>,
}

impl ImageValidator {
  fn new(diag: Rc<Diagnostic>) -> Self {
    Self { diag }
  }

  fn validate(&self, thunks: &[Thunk], types: &[TypeDesc]) -> Result<()> {
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
        self.validate_bytecode(*bc, thunk, types)?;
      }
    }

    Ok(())
  }

  fn validate_bytecode(&self, bytecode: Bytecode, thunk: &Thunk, types: &[TypeDesc]) -> Result<()> {
    use Operator::*;
    // Operands as the VM decodes them: A, B and C in encoding order.
    let Bytecode(operator, operands) = bytecode;
    let (a, b, c) = match operands {
      Operands::AB(op) => (usize::from(op.dst), usize::from(op.o1), 0),
      Operands::ABC(op) => (usize::from(op.dst), usize::from(op.o1), usize::from(op.o2)),
      _ => (0, 0, 0),
    };
    let string = |i: usize| thunk.constants.get(i).is_some_and(|v| v.is_ptr());
    let position = |i: usize| thunk.constants.get(i).is_some_and(|v| v.is_int());
    let illegal = |what: &str| self.diag.fail(format!("illegal instruction: {what}: {bytecode}"));
    match operator {
      LoadFree if b > thunk.fvlocs.len() => illegal("free variable out of range"),
      LoadType if b >= types.len() => illegal("type out of range"),
      LoadField | SetField if !string(c) && !position(c) => {
        illegal("member is not a string or int")
      }
      Invoke if !string(c) => illegal("member is not a string constant"),
      Invoke if b != a + FRAME_HEADER_SIZE => illegal("call region not after destination"),
      WObj if !Tag::from(b as u8).is_words() => illegal("wrap tag is not a words object"),
      LoadR if !Val::from_raw(b as u64).is_trivial() => illegal("nontrivial raw value"),
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

/// SAFETY: `#[repr(transparent)]` over `NonNull<vm::type_desc>`, so a
/// `Vec<OwnedType>` is handed to the VM as `*mut *mut vm::type_desc`.
#[repr(transparent)]
struct OwnedType {
  ptr: NonNull<vm::type_desc>,
}

impl OwnedType {
  fn from_descs(descs: &[TypeDesc], diag: &Diagnostic) -> Result<Vec<Self>> {
    descs.iter().map(|desc| Self::from_desc(desc, diag)).collect()
  }

  fn from_desc(desc: &TypeDesc, diag: &Diagnostic) -> Result<Self> {
    let members: Vec<vm::member_desc> = desc
      .members
      .iter()
      .map(|(name, slot)| vm::member_desc {
        name: name.as_ptr().cast(),
        len: name.len() as u32,
        slot: u32::from(*slot),
      })
      .collect();
    let ptr = unsafe {
      vm::vm_type_alloc(
        desc.name.as_ptr().cast(),
        desc.name.len() as u32,
        desc.nfields.into(),
        desc.nslots.into(),
        members.as_ptr(),
        members.len(),
      )
    };
    let ptr = NonNull::new(ptr).ok_or_else(|| diag.error("failed to allocate vm type"))?;
    Ok(Self { ptr })
  }
}

impl Drop for OwnedType {
  fn drop(&mut self) {
    unsafe {
      vm::vm_type_free(self.ptr.as_ptr());
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
  let text = unsafe { vm::vm_format_result(value) };
  if text.is_null() {
    return diag.fail("out of memory while formatting the result");
  }
  let result = unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned();
  unsafe { vm::vm_result_free(text) };
  Ok(result)
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
    let mut stats: vm::gc_stats = unsafe { std::mem::zeroed() };
    let out = run(image, diag, Some(&mut stats))?;
    Ok((out, stats))
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
  fn execution_struct_result_names_its_fields() {
    let source = "type P = struct {x, y} with fn sum(self) self.x + self.y end end; P{1, 2}";
    assert_eq!(compile_and_run(source).unwrap(), "P{x = 1, y = 2}");
  }

  #[test]
  fn execution_shared_result_prints_twice() {
    let result = compile_and_run("let p = (1, 2); (p, p)").unwrap();
    assert_eq!(result, "((1, 2), (1, 2))");
  }

  #[test]
  fn execution_cyclic_result_marks_the_cycle() {
    let node = "type N = struct {next} end; let n = N{()}; n.next <- n;";
    assert_eq!(compile_and_run(&format!("{node} n")).unwrap(), "N{next = <cycle>}");
    // only a reference back to an enclosing value is a cycle
    let pair = compile_and_run(&format!("{node} (n, n)")).unwrap();
    assert_eq!(pair, "(N{next = <cycle>}, N{next = <cycle>})");
  }

  #[test]
  fn validator_allows_captures_and_valid_loadf() {
    use crate::bytecode::Location;
    let diag = Rc::new(Diagnostic::new());
    let validator = ImageValidator::new(Rc::clone(&diag));
    let thunk = Thunk {
      name: String::new(),
      code: vec![Bytecode::loadfree(0.into(), 0.into()), Bytecode::loadfree(0.into(), 1.into())]
        .into(),
      fvlocs: vec![Location::Slot(0.into())].into(),
      nparams: 0,
      nregs: 1,
      constants: vec![].into(),
    };
    assert!(validator.validate(&[thunk], &[]).is_ok());
  }

  #[test]
  fn validator_rejects_out_of_range_loadf() {
    use crate::bytecode::Location;
    let diag = Rc::new(Diagnostic::new());
    let validator = ImageValidator::new(Rc::clone(&diag));
    let thunk = Thunk {
      name: String::new(),
      code: vec![Bytecode::loadfree(0.into(), 2.into())].into(),
      fvlocs: vec![Location::Slot(0.into())].into(),
      nparams: 0,
      nregs: 1,
      constants: vec![].into(),
    };
    assert!(validator.validate(&[thunk], &[]).is_err());
  }

  #[test]
  fn validator_rejects_bad_member_operands() {
    let diag = Rc::new(Diagnostic::new());
    let validator = ImageValidator::new(Rc::clone(&diag));
    let thunk = |code: Vec<Bytecode>, constants: Vec<Val>| Thunk {
      name: String::new(),
      code: code.into(),
      fvlocs: vec![].into(),
      nparams: 0,
      nregs: 4,
      constants: constants.into(),
    };
    let string = OwnedHeap::new(&diag).unwrap().alloc_str("x").unwrap();
    let setfield = Bytecode::setfield(0.into(), 1.into(), 0.into());
    assert!(validator.validate(&[thunk(vec![setfield], vec![])], &[]).is_err());
    assert!(validator.validate(&[thunk(vec![setfield], vec![Val::from_f64(1.0)])], &[]).is_err());
    assert!(validator.validate(&[thunk(vec![setfield], vec![Val::from_i32(1)])], &[]).is_ok());
    assert!(validator.validate(&[thunk(vec![setfield], vec![string])], &[]).is_ok());
    let invoke = Bytecode::invoke(0.into(), 2.into(), 0.into());
    assert!(validator.validate(&[thunk(vec![invoke], vec![string])], &[]).is_ok());
    assert!(validator.validate(&[thunk(vec![invoke], vec![Val::from_i32(1)])], &[]).is_err());
    let invoke = Bytecode::invoke(0.into(), 1.into(), 0.into());
    assert!(validator.validate(&[thunk(vec![invoke], vec![string])], &[]).is_err());
    let loadtype = Bytecode::loadtype(0.into(), 0.into());
    assert!(validator.validate(&[thunk(vec![loadtype], vec![])], &[]).is_err());
    assert!(
      validator
        .validate(&[thunk(vec![loadtype], vec![])], &[TypeDesc::new("T", &[], &[]).unwrap()])
        .is_ok()
    );
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
    assert!(validator.validate(&[thunk], &[]).is_err());
  }

  #[test]
  fn validator_rejects_exotic_wobj() {
    let diag = Rc::new(Diagnostic::new());
    let validator = ImageValidator::new(Rc::clone(&diag));
    let thunk = Thunk {
      name: String::new(),
      code: vec![Bytecode::wobj(0.into(), Tag::THUNK.into(), 0.into())].into(),
      fvlocs: vec![].into(),
      nparams: 0,
      nregs: 1,
      constants: vec![].into(),
    };
    assert!(validator.validate(&[thunk], &[]).is_err());
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
    assert!(validator.validate(&[thunk], &[]).is_ok());
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
    assert!(validator.validate(&[thunk], &[]).is_err());
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
    let image = BytecodeImage::new(vec![thunk], vec![], heap);
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
    let image = BytecodeImage::new(vec![thunk], vec![], heap);
    let (result, _stats) = execute_with_stats(image, diag).unwrap();
    assert_eq!(result, "99");
  }
}
