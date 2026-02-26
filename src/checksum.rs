pub fn crc16(data: &[u8]) -> u16 {
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
