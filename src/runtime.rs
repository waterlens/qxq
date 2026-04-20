use std::{ffi::CStr, ptr::NonNull, rc::Rc};

use crate::{
  bytecode::{BinaryRepr, Bytecode, BytecodeImage, Operands, Operator, Tag, Thunk},
  diagnostic::{Diagnostic, Result},
  vm,
};

const STACK_HEADROOM_SLOTS: usize = 32;
const RESULT_BUF_LEN: usize = 64;

pub fn execute(image: BytecodeImage, diag: Rc<Diagnostic>) -> Result<String> {
  let validator = ImageValidator::new(Rc::clone(&diag));
  validator.validate(&image)?;

  let max_nregs = image.thunks.iter().map(|t| t.nregs as usize).max().unwrap_or(0);
  let numobject = compute_numobject(&image);

  let mut native_functions = NativeFunctionSet::new(&image.thunks, &diag)?;
  let wrapper = OwnedFunction::wrapper(image.thunks.len() - 1, &diag)?;
  let mut result = 0u64;
  let stack_slots = STACK_HEADROOM_SLOTS + max_nregs;
  let status = unsafe {
    vm::vm_exec(
      wrapper.as_ptr(),
      native_functions.function_ptrs_mut(),
      image.thunks.len(),
      numobject,
      stack_slots,
      &mut result,
    )
  };

  if status != vm::status_t_S_OK {
    return diag.fail(format!("vm execution failed with status {}", status_name(status)));
  }

  format_result(result, &diag)
}

fn compute_numobject(image: &BytecodeImage) -> usize {
  let mut max_tag: usize = 0;
  for thunk in &image.thunks {
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

  fn validate(&self, image: &BytecodeImage) -> Result<()> {
    if image.thunks.is_empty() {
      return self.diag.fail("cannot execute an empty bytecode image");
    }

    for thunk in &image.thunks {
      if !thunk.fvlocs.is_empty() {
        return self.diag.fail("vm execution does not support captured variables (LoadF/SetF)");
      }
      for bc in thunk.code.iter() {
        self.validate_bytecode(*bc)?;
      }
    }

    Ok(())
  }

  fn validate_bytecode(&self, bytecode: Bytecode) -> Result<()> {
    use Operator::*;
    match bytecode {
      Bytecode(LoadF | SetF, _) => {
        self.diag.fail(format!("illegal instruction: {}", bytecode))
      }
      Bytecode(MObj, Operands::ABC(op)) if u8::from(op.o1) >= u8::from(Tag::INT) => {
        self.diag.fail(format!("illegal instruction: heap-backed mobj: {}", bytecode))
      }
      _ => Ok(()),
    }
  }
}

struct NativeFunctionSet {
  functions: Vec<OwnedFunction>,
}

impl NativeFunctionSet {
  fn new(thunks: &[Thunk], diag: &Diagnostic) -> Result<Self> {
    let mut functions = Vec::with_capacity(thunks.len());

    for thunk in thunks {
      functions.push(OwnedFunction::from_thunk(thunk, diag)?);
    }

    Ok(Self { functions })
  }

  fn function_ptrs_mut(&mut self) -> *mut *mut vm::function {
    // SAFETY: OwnedFunction is #[repr(transparent)] over NonNull<vm::function>,
    // which has the same layout as *mut vm::function.
    self.functions.as_mut_ptr().cast()
  }
}

/// SAFETY: This type is `#[repr(transparent)]` over `NonNull<vm::function>`,
/// which has the same layout as `*mut vm::function`. This allows
/// `&mut [OwnedFunction]` to be cast to `*mut *mut vm::function` for FFI
/// without copying.
#[repr(transparent)]
struct OwnedFunction {
  ptr: NonNull<vm::function>,
}

impl OwnedFunction {
  fn from_thunk(thunk: &Thunk, diag: &Diagnostic) -> Result<Self> {
    let ops = thunk.code.iter().map(|bc| encode_bytecode(*bc)).collect::<Vec<_>>();
    let ptr = unsafe {
      vm::vm_alloc_function(
        ops.as_ptr(),
        ops.len(),
        thunk.constants.as_ptr().cast::<vm::val_t>(),
        thunk.constants.len(),
        thunk.nregs,
      )
    };
    let ptr = NonNull::new(ptr).ok_or_else(|| diag.error("failed to allocate vm function"))?;
    Ok(Self { ptr })
  }

  fn wrapper(top_idx: usize, diag: &Diagnostic) -> Result<Self> {
    let ptr = unsafe { vm::vm_make_wrapper(top_idx) };
    let ptr = NonNull::new(ptr).ok_or_else(|| diag.error("failed to allocate vm wrapper"))?;
    Ok(Self { ptr })
  }

  fn as_ptr(&self) -> *mut vm::function {
    self.ptr.as_ptr()
  }
}

impl Drop for OwnedFunction {
  fn drop(&mut self) {
    unsafe {
      vm::vm_free_function(self.ptr.as_ptr());
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
    let mut bc = BytecodeCtx::new();
    codegen.emit_tree(&mut bc);
    let image = bc.finalize();
    execute(image, Rc::clone(&diag))
  }

  fn execute_with_heap(
    image: BytecodeImage,
    diag: Rc<Diagnostic>,
    heap_size: usize,
  ) -> Result<(String, vm::gc_stats)> {
    let _guard = VM_LOCK.lock().unwrap();
    let validator = ImageValidator::new(Rc::clone(&diag));
    validator.validate(&image)?;
    let max_nregs = image.thunks.iter().map(|t| t.nregs as usize).max().unwrap_or(0);
    let numobject = compute_numobject(&image);
    let mut native_functions = NativeFunctionSet::new(&image.thunks, &diag)?;
    let wrapper = OwnedFunction::wrapper(image.thunks.len() - 1, &diag)?;
    let mut result = 0u64;
    let stack_slots = STACK_HEADROOM_SLOTS + max_nregs;
    let mut rargs = vm::runtime_args {
      trace_level: 0,
      base_size: heap_size,
      align: 8,
      descspace_size: 4096,
    };
    let mut stats: vm::gc_stats = unsafe { std::mem::zeroed() };
    let status = unsafe {
      vm::vm_exec_with_args(
        wrapper.as_ptr(),
        native_functions.function_ptrs_mut(),
        image.thunks.len(),
        numobject,
        stack_slots,
        &mut result,
        &mut rargs,
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
  fn validator_rejects_loadf() {
    let diag = Rc::new(Diagnostic::new());
    let validator = ImageValidator::new(Rc::clone(&diag));
    let thunk = Thunk {
      name: String::new(),
      code: vec![Bytecode::loadf(0.into(), 0.into())].into(),
      fvlocs: vec![].into(),
      nparams: 0,
      nregs: 1,
      constants: vec![].into(),
    };
    let image = BytecodeImage { thunks: vec![thunk] };
    assert!(validator.validate(&image).is_err());
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
    let image = BytecodeImage { thunks: vec![thunk] };
    assert!(validator.validate(&image).is_err());
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
    let image = BytecodeImage { thunks: vec![thunk] };
    assert!(validator.validate(&image).is_err());
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
    let image = BytecodeImage { thunks: vec![thunk] };
    assert!(validator.validate(&image).is_ok());
  }

  #[test]
  fn validator_rejects_captures() {
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
    let image = BytecodeImage { thunks: vec![thunk] };
    assert!(validator.validate(&image).is_err());
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
      code.push(Bytecode::loadi(i.into(), (i as u16).into()));
      code.push(Bytecode::wobj(i.into(), 0u8.into(), 1u8.into()));
    }
    for j in 0..n_dead {
      let v = (j & 0x7fff) as u16;
      code.push(Bytecode::loadi(6u8.into(), v.into()));
      code.push(Bytecode::wobj(6u8.into(), 0u8.into(), 1u8.into()));
    }
    code.push(Bytecode::loadi(0u8.into(), 42u16.into()));
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
    let image = BytecodeImage { thunks: vec![thunk] };
    let (result, stats) = execute_with_heap(image, diag, heap_size).unwrap();
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
    code.push(Bytecode::loadi(1u8.into(), 1u16.into()));
    code.push(Bytecode::wobj(1u8.into(), 0u8.into(), 1u8.into()));
    code.push(Bytecode::loadi(2u8.into(), 2u16.into()));
    code.push(Bytecode::wobj(2u8.into(), 0u8.into(), 1u8.into()));
    code.push(Bytecode::loadi(3u8.into(), 3u16.into()));
    code.push(Bytecode::wobj(3u8.into(), 0u8.into(), 1u8.into()));
    code.push(Bytecode::loadi(4u8.into(), 4u16.into()));
    code.push(Bytecode::wobj(4u8.into(), 0u8.into(), 1u8.into()));
    for _ in 0..n_fill {
      code.push(Bytecode::loadi(5u8.into(), 0u16.into()));
      code.push(Bytecode::wobj(5u8.into(), 0u8.into(), 1u8.into()));
    }
    code.push(Bytecode::loadi(1u8.into(), 0u16.into()));
    code.push(Bytecode::loadi(2u8.into(), 0u16.into()));
    code.push(Bytecode::loadi(4u8.into(), 0u16.into()));
    code.push(Bytecode::loadi(5u8.into(), 10u16.into()));
    code.push(Bytecode::loadi(6u8.into(), 20u16.into()));
    code.push(Bytecode::loadi(7u8.into(), 30u16.into()));
    code.push(Bytecode::wobj(5u8.into(), 0u8.into(), 3u8.into()));
    code.push(Bytecode::loadi(0u8.into(), 99u16.into()));
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
    let image = BytecodeImage { thunks: vec![thunk] };
    let (result, _stats) = execute_with_heap(image, diag, heap_size).unwrap();
    assert_eq!(result, "99");
  }

}
