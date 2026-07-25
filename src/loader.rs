use crate::bytecode::{BinaryRepr, BytecodeImage, FreeVarId, Location, RegId, Tag, Thunk};
use crate::checksum;
use crate::diagnostic::{Diagnostic, Result};
use crate::runtime::OwnedHeap;
use crate::uleb8;
use crate::val::Val;
use std::io::Read;
use std::rc::Rc;

pub struct Loader<R: Read> {
  reader: R,
  diag: Rc<Diagnostic>,
}

impl<R: Read> Loader<R> {
  pub fn new(reader: R, diag: Rc<Diagnostic>) -> Self {
    Self { reader, diag }
  }

  pub fn load(&mut self, mut heap: OwnedHeap) -> Result<BytecodeImage> {
    let mut header = [0u8; 8];
    self.diag.context(self.reader.read_exact(&mut header), "failed to read header")?;

    if &header[0..4] != b"QXQ\x07" {
      return self.diag.fail("invalid magic number");
    }

    if header[4] != 0x01 {
      return self.diag.fail("unsupported version");
    }

    // header[5] is flags (reserved)
    let expected_checksum = u16::from_le_bytes([header[6], header[7]]);

    let mut thunk_table_data = Vec::new();
    self
      .diag
      .context(self.reader.read_to_end(&mut thunk_table_data), "failed to read thunk table")?;

    if checksum::crc16(&thunk_table_data) != expected_checksum {
      return self.diag.fail("checksum mismatch");
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

      thunks.push(self.load_thunk(thunk_data, &mut heap)?);
    }

    Ok(BytecodeImage::new(thunks, heap))
  }

  fn load_thunk(&self, data: &[u8], heap: &mut OwnedHeap) -> Result<Thunk> {
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
      code.push(
        self
          .diag
          .context(BinaryRepr::load(&data[cursor..cursor + 4]), "failed to load bytecode")?,
      );
      cursor += 4;
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
        _ => return self.diag.fail("invalid captured variable kind"),
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
          let i = i64::from_le_bytes(val.to_le_bytes());
          constants.push(Val::from_i32(i as i32));
        }
        Tag::FLOAT => {
          let (val, n) = uleb8::decode_uleb128(&data[cursor..]);
          cursor += n;
          constants.push(Val::from_f64(f64::from_bits(val)));
        }
        Tag::STR => {
          let (len, n) = uleb8::decode_uleb128(&data[cursor..]);
          cursor += n;
          let s = std::str::from_utf8(&data[cursor..cursor + len as usize]);
          let s = self.diag.context(s, "invalid UTF-8 string")?;
          cursor += len as usize;
          let value = heap
            .alloc_str(s)
            .ok_or_else(|| self.diag.error("failed to allocate string constant"))?;
          constants.push(value);
        }
        _ => return self.diag.fail("unsupported constant tag"),
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
