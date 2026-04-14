use crate::bytecode::{BinaryRepr, BytecodeImage, Constant, Location, Tag};
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
    for thunk in &self.image.thunks {
      thunk_data.clear();
      thunk_data.extend_from_slice(&[0x00, thunk.nparams, thunk.nregs, thunk.fvlocs.len() as u8]);
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
            thunk_data.push(1);
            thunk_data.push(fv.0 as u8);
          }
          Location::Temporary => return self.diag.fail("temporary location in thunk"),
        }
      }

      // Constants
      for constant in thunk.constants.iter() {
        match constant {
          Constant::Int(i) => {
            uleb8::encode_uleb128(Tag::INT.into(), &mut thunk_data);
            uleb8::encode_uleb128(u64::from_le_bytes(i.to_le_bytes()), &mut thunk_data);
          }
          Constant::Float(f) => {
            uleb8::encode_uleb128(Tag::FLOAT.into(), &mut thunk_data);
            uleb8::encode_uleb128(f.0.to_bits(), &mut thunk_data);
          }
          Constant::Str(s) => {
            uleb8::encode_uleb128(Tag::STR.into(), &mut thunk_data);
            uleb8::encode_uleb128(s.len() as u64, &mut thunk_data);
            thunk_data.extend_from_slice(s.as_bytes());
          }
        }
      }

      // Write size and thunk data
      uleb8::encode_uleb128(thunk_data.len() as u64, &mut data);
      data.extend_from_slice(&thunk_data);
    }

    // End of thunk table
    uleb8::encode_uleb128(0, &mut data);

    // Calculate Checksum of the thunk table
    let checksum = checksum::crc16(&data[Self::BYTECODE_IMAGE_HEADER..]);
    data[Self::BYTECODE_IMAGE_HEADER_SLICE_BEGIN..Self::BYTECODE_IMAGE_HEADER_SLICE_END]
      .copy_from_slice(&checksum.to_le_bytes());

    Ok(data)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bytecode::{Bytecode, Thunk};

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
    let dumper = Dumper::new(BytecodeImage { thunks: vec![thunk] }, Rc::new(Diagnostic::new()));
    let data = dumper.dump().unwrap();
    assert_eq!(&data[0..4], b"QXQ\x07");
    assert_eq!(data[4], 0x01); // Version
  }
}
