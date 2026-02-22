use crate::bytecode::{BytecodeImage, Constant, Location, Tag};

pub struct Serializer {
  image: BytecodeImage,
}

impl Serializer {
  pub fn new(image: BytecodeImage) -> Self {
    Self { image }
  }

  pub fn serialize(&self) -> Vec<u8> {
    let mut data = Vec::new();
    let mut thunk_table = Vec::new();

    // Thunk Table
    for thunk in &self.image.thunks {
      let mut thunk_data: Vec<u8> =
        vec![0x00, thunk.nparams, thunk.nregs, thunk.fvlocs.len() as u8];
      Self::encode_uleb128(thunk.constants.len() as u64, &mut thunk_data);
      Self::encode_uleb128(thunk.code.len() as u64, &mut thunk_data);

      // Bytecode
      for bc in thunk.code.iter() {
        thunk_data.extend_from_slice(&bc.serialize());
      }

      // Captured variables
      for loc in thunk.fvlocs.iter() {
        match loc {
          Location::Slot(r) => {
            thunk_data.push(r.0);
            thunk_data.push(0);
          }
          Location::FreeVar(fv) => {
            thunk_data.push(fv.0 as u8);
            thunk_data.push(1);
          }
          Location::Temporary => unreachable!(),
        }
      }

      // Constants
      for constant in thunk.constants.iter() {
        match constant {
          Constant::Int(i) => {
            Self::encode_uleb128(Tag::INT.into(), &mut thunk_data);
            Self::encode_uleb128(u64::from_le_bytes(i.to_le_bytes()), &mut thunk_data);
          }
          Constant::Str(s) => {
            Self::encode_uleb128(Tag::STR.into(), &mut thunk_data);
            Self::encode_uleb128(s.len() as u64, &mut thunk_data);
            thunk_data.extend_from_slice(s.as_bytes());
          }
        }
      }

      // Write size and thunk data
      Self::encode_uleb128(thunk_data.len() as u64, &mut thunk_table);
      thunk_table.extend_from_slice(&thunk_data);
    }

    // End of thunk table
    Self::encode_uleb128(0, &mut thunk_table);

    // Calculate Checksum of the thunk table
    let checksum = Self::crc16(&thunk_table);

    // Header (8 bytes)
    data.extend_from_slice(b"QXQ\x07"); // Magic (4)
    data.push(0x01); // Version (1)
    data.push(0x00); // Flags (1)
    data.extend_from_slice(&checksum.to_le_bytes()); // Checksum (2)

    data.extend_from_slice(&thunk_table);
    data
  }

  fn encode_uleb128(mut value: u64, buf: &mut Vec<u8>) {
    loop {
      let mut byte = (value & 0x7F) as u8;
      value >>= 7;
      if value != 0 {
        byte |= 0x80;
      }
      buf.push(byte);
      if value == 0 {
        break;
      }
    }
  }

  fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &byte in data {
      crc ^= byte as u16;
      for _ in 0..8 {
        if (crc & 0x0001) != 0 {
          crc = (crc >> 1) ^ 0xA001; // Polynomial 0x8005 reflected
        } else {
          crc >>= 1;
        }
      }
    }
    crc
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
    let serializer = Serializer::new(BytecodeImage { thunks: vec![thunk] });
    let data = serializer.serialize();
    assert_eq!(&data[0..4], b"QXQ\x07");
    assert_eq!(data[4], 0x01); // Version
  }
}
