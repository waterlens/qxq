use crate::bytecode::{Bytecode, BytecodeImage, Constant, FreeVarId, Location, RegId, Tag, Thunk};
use crate::uleb8;
use crate::checksum;
use std::io::{self, Read};

pub struct Loader<R: Read> {
  reader: R,
}

impl<R: Read> Loader<R> {
  pub fn new(reader: R) -> Self {
    Self { reader }
  }

  pub fn load(&mut self) -> io::Result<BytecodeImage> {
    let mut header = [0u8; 8];
    self.reader.read_exact(&mut header)?;

    if &header[0..4] != b"QXQ\x07" {
      return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid magic number"));
    }

    if header[4] != 0x01 {
      return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported version"));
    }

    // header[5] is flags (reserved)
    let expected_checksum = u16::from_le_bytes([header[6], header[7]]);

    let mut thunk_table_data = Vec::new();
    self.reader.read_to_end(&mut thunk_table_data)?;

    if checksum::crc16(&thunk_table_data) != expected_checksum {
      return Err(io::Error::new(io::ErrorKind::InvalidData, "checksum mismatch"));
    }

    let mut thunks = Vec::new();
    let mut cursor = 0;
    loop {
      let (tsize, n) = uleb8::decode_uleb128(&thunk_table_data[cursor..]);
      cursor += n;
      if tsize == 0 {
        break;
      }

      let thunk_data = &thunk_table_data[cursor..cursor + tsize as usize];
      cursor += tsize as usize;

      thunks.push(self.load_thunk(thunk_data)?);
    }

    Ok(BytecodeImage { thunks })
  }

  fn load_thunk(&self, data: &[u8]) -> io::Result<Thunk> {
    let mut cursor = 0;

    // Flags (reserved)
    let _tflags = data[cursor];
    cursor += 1;

    let nparams = data[cursor];
    cursor += 1;

    let nregs = data[cursor];
    cursor += 1;

    let ncaptured = data[cursor];
    cursor += 1;

    let (nconstants, n) = uleb8::decode_uleb128(&data[cursor..]);
    cursor += n;

    let (ninstrs, n) = uleb8::decode_uleb128(&data[cursor..]);
    cursor += n;

    let mut code = Vec::with_capacity(ninstrs as usize);
    for _ in 0..ninstrs {
      let mut bc_data = [0u8; 4];
      bc_data.copy_from_slice(&data[cursor..cursor + 4]);
      cursor += 4;
      code.push(Bytecode::load(bc_data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?);
    }

    let mut fvlocs = Vec::with_capacity(ncaptured as usize);
    for _ in 0..ncaptured {
      let kind = data[cursor];
      cursor += 1;
      let val = data[cursor];
      cursor += 1;

      match kind {
        0 => fvlocs.push(Location::Slot(RegId(val))),
        1 => fvlocs.push(Location::FreeVar(FreeVarId(val as u16))),
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid captured variable kind")),
      }
    }

    let mut constants = Vec::with_capacity(nconstants as usize);
    for _ in 0..nconstants {
      let (tag_val, n) = uleb8::decode_uleb128(&data[cursor..]);
      cursor += n;
      let tag = Tag::from(tag_val as u8);

      match tag {
        Tag::INT => {
          let (val, n) = uleb8::decode_uleb128(&data[cursor..]);
          cursor += n;
          constants.push(Constant::Int(i64::from_le_bytes(val.to_le_bytes())));
        }
        Tag::STR => {
          let (len, n) = uleb8::decode_uleb128(&data[cursor..]);
          cursor += n;
          let s = std::str::from_utf8(&data[cursor..cursor + len as usize])
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 string"))?;
          cursor += len as usize;
          constants.push(Constant::Str(s.to_string()));
        }
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported constant tag")),
      }
    }

    Ok(Thunk {
      name: "".to_string(), // Will be filled later
      code: code.into_boxed_slice(),
      fvlocs: fvlocs.into_boxed_slice(),
      nparams,
      nregs,
      constants: constants.into_boxed_slice(),
    })
  }
}

