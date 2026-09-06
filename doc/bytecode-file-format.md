# QxQ Bytecode File Format

## Information

- Extension Name: `.qxc`
- Endian: little-endian by default.

## File Header

The **File Header** is a 8-byte region that describes
the file type, version, enabled features, and the checksum of file.

| Name | Content | Length (Byte) | Field Name in Source Code | Comment |
| --------------- | --------------- | --------------- | --------------- | --------------- |
| Magic Number | QXQ\x07 | 4 | magic| |
| Version | \x01 | 1 | version | \x01 for qxq-0.1.2-compatible bytecode format |
| Flags | \x00 | 1 | gflags | Reserved |
| Checksum | | 2 | checksum | CRC-16 |

## Thunk Table

A **Thunk Table** contains multiple thunks where the last thunk is the
entry point. The format of thunk header is listed below:

| Name | Content | Length (Byte) | Field Name in Source Code | Comment |
| --------------- | --------------- | --------------- | --------------- | --------------- |
| Size | | | tsize | Thunk size encoded by ULEB128 |
| Flags | \x00 | 1 | tflags | Reserved |
| \#Params | | 1 | nparams | # of parameters |
| \#Frame | | 1 | nregs | Frame size |
| \#Captured | | 1 | ncaptured | # of captured variables |
| \#Constant | | | nconstants | # of constants encoded by ULEB128 |
| \#Bytecode | | | ninstrs | # of instructions encoded by ULEB128 |

After the header there is the bytecode region of a thunk.

| Name | Content | Length (Byte) | Field Name in Source Code | Comment |
| --------------- | --------------- | --------------- | --------------- | --------------- |
| Bytecode Instructions | | (4 * ninstrs) | | |

The captured variables region follows the bytecode region. For 2-byte
represented each captured variable, the 1st-byte can be either 0 for capturing
variable in caller's slot, or 1 for capturing caller's captured variables. The
2nd-byte corresponds to the index of frame slots (registers) or the index of
captured variables.

| Name | Content | Length (Byte) | Field Name in Source Code | Comment |
| --------------- | --------------- | --------------- | --------------- | --------------- |
| Captured Variables | | (2 * ncaptured) | | |

After the captured variables region, there is a constant region which stores
the string constant, number constant, or other kinds of constant in future.
Each constant starts with its kind encoded by ULEB128: 0 for an integer, 1 for
a float, 2 for a string. The raw data dump of the constant representation
follows the kind. For string, it contains a length together with a UTF-8
encoded string. For integer, the representation also uses ULEB128.

| Name | Content | Length (Byte) | Field Name in Source Code | Comment |
| --------------- | --------------- | --------------- | --------------- | --------------- |
| Constants | | (2 * nconstants) | | |

The thunk table always ends with a partial thunk header with size = 0.

## Type Table

The **Type Table** follows the thunk table and describes every struct type
declared in the image. `LoadType` addresses it by index.

| Name | Content | Length (Byte) | Field Name in Source Code | Comment |
| --------------- | --------------- | --------------- | --------------- | --------------- |
| \#Type | | | ntypes | # of type descriptions encoded by ULEB128 |

Each type description lists its declared name, then its member names and the
slots they resolve to. The declared fields occupy slots `0..nfields` and the
methods occupy `nfields..nslots`, both in declaration order.

| Name | Content | Length (Byte) | Field Name in Source Code | Comment |
| --------------- | --------------- | --------------- | --------------- | --------------- |
| Name | | | name | length encoded by ULEB128, then the UTF-8 name |
| \#Field | | | nfields | # of declared fields encoded by ULEB128 |
| \#Slot | | | nslots | # of member slots encoded by ULEB128 |
| \#Member | | | nmembers | # of member entries encoded by ULEB128 |
| Members | | | | (slot, length, UTF-8 name) with slot and length encoded by ULEB128 |
