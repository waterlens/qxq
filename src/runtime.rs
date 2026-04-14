use std::{ffi::CStr, ptr::NonNull, rc::Rc};

use crate::{
  bytecode::{BinaryRepr, Bytecode, BytecodeImage, Operator, Thunk},
  diagnostic::{Diagnostic, Result},
  vm,
};

const STACK_HEADROOM_SLOTS: usize = 32;
const RESULT_BUF_LEN: usize = 64;

pub fn execute(image: BytecodeImage, diag: Rc<Diagnostic>) -> Result<String> {
  let validator = ImageValidator::new(Rc::clone(&diag));
  validator.validate(&image)?;

  let top =
    image.thunks.last().ok_or_else(|| diag.error("cannot execute an empty bytecode image"))?;

  let mut native_functions = NativeFunctionSet::new(&image.thunks, &diag)?;
  let wrapper = OwnedFunction::wrapper(image.thunks.len() - 1, &diag)?;
  let mut result = 0u64;
  let stack_slots = STACK_HEADROOM_SLOTS + top.nregs as usize;
  let status = unsafe {
    vm::vm_exec(
      wrapper.as_ptr(),
      native_functions.function_ptrs_mut(),
      image.thunks.len(),
      0,
      stack_slots,
      &mut result,
    )
  };

  if status != vm::status_t_S_OK {
    return diag.fail(format!("vm execution failed with status {}", status_name(status)));
  }

  format_result(result, &diag)
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
    if image.thunks.len() != 1 {
      return self.diag.fail(
        "vm cli execution currently supports only scalar top-level programs without functions",
      );
    }

    let thunk = &image.thunks[0];
    if !thunk.fvlocs.is_empty() {
      return self.diag.fail("vm cli execution does not support captured variables");
    }

    for bc in thunk.code.iter() {
      self.validate_bytecode(*bc)?;
    }

    Ok(())
  }

  fn validate_bytecode(&self, bytecode: Bytecode) -> Result<()> {
    use Operator::*;
    if matches!(
      bytecode.0,
      MObj | LoadF | SetF | Apply | Call | Retu | Retn | Clos | WObj | Goto | Trap
    ) {
      self.diag.fail(format!("illegal instruction: {}", bytecode))
    } else {
      Ok(())
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
