//! Minimal RFC 4648 base64 decoder (standard alphabet).
//!
//! Only what buffer `data:` URIs need: extracting the payload after the
//! first comma and decoding it. A hand-rolled decoder keeps this crate at
//! `std` + `gltf` — pulling the `gltf` `import` feature for the same job
//! would drag the `image` crate into a geometry-only loader (textures are
//! an explicit non-goal, see the crate docs).

/// Symbols per base64 quantum.
const QUANTUM: usize = 4;
/// Decoded bytes per full quantum (before padding).
const BYTES_PER_QUANTUM: usize = 3;
/// Max padding `=` characters in a quantum.
const MAX_PADDING: usize = 2;
/// Bit shift for the first sextet in a 24-bit quantum.
const SEXTET0_SHIFT: u32 = 18;
/// Bit shift for the second sextet.
const SEXTET1_SHIFT: u32 = 12;
/// Bit shift for the third sextet.
const SEXTET2_SHIFT: u32 = 6;
/// Bit shift extracting the first output byte.
const BYTE0_SHIFT: u32 = 16;
/// Bit shift extracting the second output byte.
const BYTE1_SHIFT: u32 = 8;
/// Offset of lowercase letters in the standard alphabet.
const ALPHA_LOWER_OFFSET: u8 = 26;
/// Offset of digits in the standard alphabet.
const DIGIT_OFFSET: u8 = 52;
/// Alphabet value of `+`.
const PLUS_VALUE: u8 = 62;
/// Alphabet value of `/`.
const SLASH_VALUE: u8 = 63;

/// Decodes standard-alphabet base64, tolerating ASCII whitespace.
///
/// Padding (`=`) may only appear as the final one or two characters;
/// anything else (unknown symbols, short quanta) is an error.
pub(crate) fn decode(input: &str) -> Result<Vec<u8>, DecodeError> {
    let mut values = Vec::with_capacity(input.len());
    let mut padding = 0usize;
    for byte in input.bytes() {
        if byte.is_ascii_whitespace() {
            continue;
        }
        if byte == b'=' {
            padding += 1;
            values.push(0);
            continue;
        }
        if padding > 0 {
            return Err(DecodeError::DataAfterPadding);
        }
        values.push(decode_symbol(byte)?);
    }
    if values.len() % QUANTUM != 0 {
        return Err(DecodeError::BadLength);
    }
    if padding > MAX_PADDING {
        return Err(DecodeError::BadPadding);
    }
    let data_len = values.len() / QUANTUM * BYTES_PER_QUANTUM - padding;
    let mut out = Vec::with_capacity(data_len);
    for quad in values.chunks_exact(QUANTUM) {
        let n = (u32::from(quad[0]) << SEXTET0_SHIFT)
            | (u32::from(quad[1]) << SEXTET1_SHIFT)
            | (u32::from(quad[2]) << SEXTET2_SHIFT)
            | u32::from(quad[3]);
        out.push((n >> BYTE0_SHIFT) as u8);
        if out.len() < data_len {
            out.push((n >> BYTE1_SHIFT) as u8);
        }
        if out.len() < data_len {
            out.push(n as u8);
        }
    }
    Ok(out)
}

/// Base64 decode failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecodeError {
    /// Byte outside the standard alphabet.
    BadSymbol(u8),
    /// Total symbols not a multiple of 4.
    BadLength,
    /// More than two padding characters.
    BadPadding,
    /// Data after a padding character.
    DataAfterPadding,
}

/// Value of one standard-alphabet symbol.
fn decode_symbol(byte: u8) -> Result<u8, DecodeError> {
    match byte {
        b'A'..=b'Z' => Ok(byte - b'A'),
        b'a'..=b'z' => Ok(byte - b'a' + ALPHA_LOWER_OFFSET),
        b'0'..=b'9' => Ok(byte - b'0' + DIGIT_OFFSET),
        b'+' => Ok(PLUS_VALUE),
        b'/' => Ok(SLASH_VALUE),
        other => Err(DecodeError::BadSymbol(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_rfc_vectors() {
        assert_eq!(decode("").expect("empty"), b"");
        assert_eq!(decode("Zg==").expect("f"), b"f");
        assert_eq!(decode("Zm8=").expect("fo"), b"fo");
        assert_eq!(decode("Zm9v").expect("foo"), b"foo");
        assert_eq!(decode("Zm9vYmFy").expect("foobar"), b"foobar");
        assert_eq!(decode("TWFu").expect("Man"), b"Man");
    }

    #[test]
    fn tolerates_whitespace() {
        assert_eq!(decode("Zm9v\nYmFy").expect("wrapped"), b"foobar");
    }

    #[test]
    fn rejects_malformed_input() {
        assert_eq!(decode("Zm9").unwrap_err(), DecodeError::BadLength);
        assert_eq!(decode("====").unwrap_err(), DecodeError::BadPadding);
        assert_eq!(decode("Zm=9").unwrap_err(), DecodeError::DataAfterPadding);
        assert!(matches!(
            decode("Zm!9").unwrap_err(),
            DecodeError::BadSymbol(b'!')
        ));
    }
}
