use crate::bytecode::{BinaryRepr, BytecodeImage, ConstKind, Location, TypeDesc};
use crate::checksum;
use crate::diagnostic::{Diagnostic, Result};
use crate::uleb8;
use std::rc::Rc;

pub struct Dumper {
  image: BytecodeImage,
  diag: Rc<Diagnostic>,
}

impl Dumper {
  const BYTECODE_IMAGE_HEADER: usize = 8;
  const BYTECODE_IMAGE_HEADER_SLICE_BEGIN: usize = 6;
  const BYTECODE_IMAGE_HEADER_SLICE_END: usize = 8;
  pub fn new(image: BytecodeImage, diag: Rc<Diagnostic>) -> Self {
    Self { image, diag }
  }

  pub fn dump(&self) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    let mut thunk_data = vec![0, 0, 0, 0];

    // Header (8 bytes)
    data.extend_from_slice(b"QXQ\x07"); // Magic (4)
    data.push(0x01); // Version (1)
    data.push(0x00); // Flags (1)
    data.extend_from_slice(&[0, 0]); // Checksum (2)
    debug_assert_eq!(data.len(), Self::BYTECODE_IMAGE_HEADER);

    // Thunk Table
    for thunk in self.image.thunks() {
      let ncaptured: u8 = thunk
        .fvlocs
        .len()
        .try_into()
        .map_err(|_| self.diag.error("too many captured variables to dump"))?;
      thunk_data.clear();
      thunk_data.extend_from_slice(&[0x00, thunk.nparams, thunk.nregs, ncaptured]);
      uleb8::encode_uleb128(thunk.constants.len() as u64, &mut thunk_data);
      uleb8::encode_uleb128(thunk.code.len() as u64, &mut thunk_data);

      let mut bc_buf = [0u8; 4];
      // Bytecode
      for bc in thunk.code.iter() {
        bc.dump(&mut bc_buf);
        thunk_data.extend_from_slice(&bc_buf);
      }

      // Captured variables
      for loc in thunk.fvlocs.iter() {
        match loc {
          Location::Slot(r) => {
            thunk_data.push(0);
            thunk_data.push(r.0);
          }
          Location::FreeVar(fv) => {
            let id: u8 = fv
              .0
              .try_into()
              .map_err(|_| self.diag.error("captured free variable index too large to dump"))?;
            thunk_data.push(1);
            thunk_data.push(id);
          }
          Location::Temporary => return self.diag.fail("temporary location in thunk"),
        }
      }

      // Constants
      for constant in thunk.constants.iter() {
        if constant.is_int() {
          uleb8::encode_uleb128(ConstKind::INT.into(), &mut thunk_data);
          let i = constant.as_i32() as i64;
          uleb8::encode_uleb128(u64::from_le_bytes(i.to_le_bytes()), &mut thunk_data);
        } else if constant.is_float() {
          uleb8::encode_uleb128(ConstKind::FLOAT.into(), &mut thunk_data);
          uleb8::encode_uleb128(constant.as_f64().to_bits(), &mut thunk_data);
        } else if constant.is_ptr() {
          let bytes = constant.str_as_bytes();
          uleb8::encode_uleb128(ConstKind::STR.into(), &mut thunk_data);
          uleb8::encode_uleb128(bytes.len() as u64, &mut thunk_data);
          thunk_data.extend_from_slice(bytes);
        }
      }

      // Write size and thunk data
      uleb8::encode_uleb128(thunk_data.len() as u64, &mut data);
      data.extend_from_slice(&thunk_data);
    }

    // End of thunk table
    uleb8::encode_uleb128(0, &mut data);

    // Type table
    let types = self.image.types();
    uleb8::encode_uleb128(types.len() as u64, &mut data);
    for ty in types {
      Self::dump_type(ty, &mut data);
    }

    // Calculate Checksum of the thunk table
    let checksum = checksum::crc16(&data[Self::BYTECODE_IMAGE_HEADER..]);
    data[Self::BYTECODE_IMAGE_HEADER_SLICE_BEGIN..Self::BYTECODE_IMAGE_HEADER_SLICE_END]
      .copy_from_slice(&checksum.to_le_bytes());

    Ok(data)
  }

  fn dump_type(ty: &TypeDesc, data: &mut Vec<u8>) {
    uleb8::encode_uleb128(ty.nfields.into(), data);
    uleb8::encode_uleb128(ty.nslots.into(), data);
    uleb8::encode_uleb128(ty.members.len() as u64, data);
    for (name, slot) in ty.members.iter() {
      uleb8::encode_uleb128((*slot).into(), data);
      uleb8::encode_uleb128(name.len() as u64, data);
      data.extend_from_slice(name.as_bytes());
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bytecode::{Bytecode, FreeVarId, RegId, Thunk};
  use crate::runtime::OwnedHeap;

  #[test]
  fn test_header() {
    let thunk = Thunk {
      name: "test".to_string(),
      code: Box::new([Bytecode::nop()]),
      fvlocs: Box::new([]),
      nparams: 0,
      nregs: 0,
      constants: Box::new([]),
    };
    let diag = Rc::new(Diagnostic::new());
    let image = BytecodeImage::new(vec![thunk], vec![], OwnedHeap::new(&diag).unwrap());
    let dumper = Dumper::new(image, diag);
    let data = dumper.dump().unwrap();
    assert_eq!(&data[0..4], b"QXQ\x07");
    assert_eq!(data[4], 0x01); // Version
  }

  #[test]
  fn rejects_capture_count_overflow() {
    let thunk = Thunk {
      name: "test".to_string(),
      code: Box::new([Bytecode::nop()]),
      fvlocs: vec![Location::Slot(RegId(0)); 256].into_boxed_slice(),
      nparams: 0,
      nregs: 1,
      constants: Box::new([]),
    };
    let diag = Rc::new(Diagnostic::new());
    let image = BytecodeImage::new(vec![thunk], vec![], OwnedHeap::new(&diag).unwrap());
    let dumper = Dumper::new(image, diag);
    assert!(dumper.dump().is_err());
  }

  #[test]
  fn rejects_capture_index_overflow() {
    let thunk = Thunk {
      name: "test".to_string(),
      code: Box::new([Bytecode::nop()]),
      fvlocs: Box::new([Location::FreeVar(FreeVarId(256))]),
      nparams: 0,
      nregs: 0,
      constants: Box::new([]),
    };
    let diag = Rc::new(Diagnostic::new());
    let image = BytecodeImage::new(vec![thunk], vec![], OwnedHeap::new(&diag).unwrap());
    let dumper = Dumper::new(image, diag);
    assert!(dumper.dump().is_err());
  }
}
