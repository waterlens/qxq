use std::io::{self, Read};

pub fn encode_uleb128(mut value: u64, buf: &mut Vec<u8>) {
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

pub fn decode_uleb128(data: &[u8]) -> (u64, usize) {
  let mut result = 0u64;
  let mut shift = 0;
  let mut cursor = 0;
  loop {
    let b = data[cursor];
    cursor += 1;
    result |= ((b & 0x7F) as u64) << shift;
    if (b & 0x80) == 0 {
      break;
    }
    shift += 7;
  }
  (result, cursor)
}

pub fn read_uleb128<R: Read>(reader: &mut R) -> io::Result<u64> {
  let mut result = 0u64;
  let mut shift = 0;
  loop {
    let mut byte = [0u8; 1];
    reader.read_exact(&mut byte)?;
    let b = byte[0];
    result |= ((b & 0x7F) as u64) << shift;
    if (b & 0x80) == 0 {
      break;
    }
    shift += 7;
  }
  Ok(result)
}
