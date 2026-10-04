//! sRGB-aware [`Color`] and illuminance [`Lux`] over the linear channel types.
//!
//! Scene files store light color as a linear `[f32; 3]` and intensity as a
//! bare `f32`. [`Color`] keeps [`LinearRgba`] internally and (de)serializes
//! as that same linear array, so swapping a scene field onto [`Color`] does
//! not change the RON shape. [`Lux`] is a transparent wrapper over the
//! existing intensity channel.

use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{LinearRgb, LinearRgba};

/// sRGB IEC 61966-2-1 linear/power junction (encoded channel).
const SRGB_LINEAR_THRESHOLD: f32 = 0.04045;
/// Reciprocal of the sRGB linear-segment slope.
const SRGB_LINEAR_SLOPE: f32 = 12.92;
/// sRGB power-segment offset.
const SRGB_POWER_OFFSET: f32 = 0.055;
/// sRGB power-segment scale.
const SRGB_POWER_SCALE: f32 = 1.055;
/// sRGB power-segment exponent.
const SRGB_POWER_GAMMA: f32 = 2.4;
/// Linear-channel junction of the inverse sRGB transfer (`0.04045 / 12.92`).
const SRGB_LINEAR_OUT_THRESHOLD: f32 = SRGB_LINEAR_THRESHOLD / SRGB_LINEAR_SLOPE;
/// 8-bit channel scale.
const U8_SCALE: f32 = 255.0;

/// Bit patterns of the IEC curve at `i / 255` for every byte `i`.
///
/// `f32::powf` is not a const fn on the pinned toolchain, so [`Color::srgb_u8`]
/// and [`Color::hex`] read this table. The unit test
/// `srgb_u8_table_matches_the_runtime_curve` checks each entry against
/// [`srgb_channel_to_linear`].
#[rustfmt::skip]
const SRGB_U8_TO_LINEAR_BITS: [u32; 256] = [
    0x0000_0000, 0x399F_22B4, 0x3A1F_22B4, 0x3A6E_B40E, 0x3A9F_22B4, 0x3AC6_EB61, 0x3AEE_B40E, 0x3B0B_3E5D,
    0x3B1F_22B4, 0x3B33_070B, 0x3B46_EB61, 0x3B5B_518D, 0x3B70_F18D, 0x3B83_E1C6, 0x3B8F_E616, 0x3B9C_87FD,
    0x3BA9_C9B7, 0x3BB7_AD6F, 0x3BC6_3549, 0x3BD5_6361, 0x3BE5_39C1, 0x3BF5_BA70, 0x3C03_73B5, 0x3C0C_6152,
    0x3C15_A703, 0x3C1F_45BE, 0x3C29_3E6B, 0x3C33_91F7, 0x3C3E_4149, 0x3C49_4D43, 0x3C54_B6C7, 0x3C60_7EB1,
    0x3C6C_A5DF, 0x3C79_2D22, 0x3C83_0AA8, 0x3C89_AF9F, 0x3C90_85DB, 0x3C97_8DC5, 0x3C9E_C7C2, 0x3CA6_3433,
    0x3CAD_D37D, 0x3CB5_A601, 0x3CBD_AC20, 0x3CC5_E639, 0x3CCE_54AB, 0x3CD6_F7D5, 0x3CDF_D010, 0x3CE8_DDB9,
    0x3CF2_212C, 0x3CFB_9AC1, 0x3D02_A569, 0x3D07_98DC, 0x3D0C_A7E6, 0x3D11_D2AF, 0x3D17_1963, 0x3D1C_7C2E,
    0x3D21_FB3C, 0x3D27_96B2, 0x3D2D_4EBB, 0x3D33_2380, 0x3D39_152B, 0x3D3F_23E3, 0x3D45_4FD1, 0x3D4B_991C,
    0x3D51_FFEF, 0x3D58_846A, 0x3D5F_26B7, 0x3D65_E6FE, 0x3D6C_C564, 0x3D73_C20F, 0x3D7A_DD29, 0x3D81_0B67,
    0x3D84_B795, 0x3D88_7330, 0x3D8C_3E4A, 0x3D90_18F6, 0x3D94_0345, 0x3D97_FD4A, 0x3D9C_0716, 0x3DA0_20BB,
    0x3DA4_4A4B, 0x3DA8_83D7, 0x3DAC_CD70, 0x3DB1_2728, 0x3DB5_9112, 0x3DBA_0B3B, 0x3DBE_95B5, 0x3DC3_3092,
    0x3DC7_DBE2, 0x3DCC_97B6, 0x3DD1_641F, 0x3DD6_412C, 0x3DDB_2EEF, 0x3DE0_2D77, 0x3DE5_3CD5, 0x3DEA_5D19,
    0x3DEF_8E52, 0x3DF4_D091, 0x3DFA_23E8, 0x3DFF_8861, 0x3E02_7F07, 0x3E05_4280, 0x3E08_0EA3, 0x3E0A_E378,
    0x3E0D_C105, 0x3E10_A754, 0x3E13_966B, 0x3E16_8E52, 0x3E19_8F10, 0x3E1C_98AD, 0x3E1F_AB30, 0x3E22_C6A3,
    0x3E25_EB09, 0x3E29_186C, 0x3E2C_4ED0, 0x3E2F_8E41, 0x3E32_D6C4, 0x3E36_2861, 0x3E39_831E, 0x3E3C_E703,
    0x3E40_5416, 0x3E43_CA5F, 0x3E47_49E4, 0x3E4A_D2AE, 0x3E4E_64C2, 0x3E52_0027, 0x3E55_A4E6, 0x3E59_5303,
    0x3E5D_0A8B, 0x3E60_CB7C, 0x3E64_95E0, 0x3E68_69BF, 0x3E6C_4720, 0x3E70_2E0C, 0x3E74_1E84, 0x3E78_1890,
    0x3E7C_1C38, 0x3E80_14C2, 0x3E82_203C, 0x3E84_308D, 0x3E86_45BA, 0x3E88_5FC5, 0x3E8A_7EB2, 0x3E8C_A283,
    0x3E8E_CB3D, 0x3E90_F8E1, 0x3E93_2B74, 0x3E95_62F8, 0x3E97_9F71, 0x3E99_E0E2, 0x3E9C_274E, 0x3E9E_72B7,
    0x3EA0_C322, 0x3EA3_1892, 0x3EA5_7308, 0x3EA7_D289, 0x3EAA_3718, 0x3EAC_A0B7, 0x3EAF_0F69, 0x3EB1_8333,
    0x3EB3_FC18, 0x3EB6_7A18, 0x3EB8_FD37, 0x3EBB_8579, 0x3EBE_12E1, 0x3EC0_A571, 0x3EC3_3D2D, 0x3EC5_DA17,
    0x3EC8_7C33, 0x3ECB_2383, 0x3ECD_D00B, 0x3ED0_81CD, 0x3ED3_38CC, 0x3ED5_F50B, 0x3ED8_B68D, 0x3EDB_7D54,
    0x3EDE_4965, 0x3EE1_1AC1, 0x3EE3_F16B, 0x3EE6_CD67, 0x3EE9_AEB6, 0x3EEC_955D, 0x3EEF_815D, 0x3EF2_72BA,
    0x3EF5_6976, 0x3EF8_6594, 0x3EFB_6717, 0x3EFE_6E02, 0x3F00_BD2D, 0x3F02_460E, 0x3F03_D1A7, 0x3F05_5FF9,
    0x3F06_F106, 0x3F08_84CF, 0x3F0A_1B56, 0x3F0B_B49B, 0x3F0D_50A0, 0x3F0E_EF67, 0x3F10_90F1, 0x3F12_353E,
    0x3F13_DC51, 0x3F15_862B, 0x3F17_32CD, 0x3F18_E239, 0x3F1A_946F, 0x3F1C_4971, 0x3F1E_0141, 0x3F1F_BBDF,
    0x3F21_794E, 0x3F23_398E, 0x3F24_FCA0, 0x3F26_C286, 0x3F28_8B41, 0x3F2A_56D3, 0x3F2C_253D, 0x3F2D_F680,
    0x3F2F_CA9E, 0x3F31_A197, 0x3F33_7B6C, 0x3F35_5820, 0x3F37_37B3, 0x3F39_1A26, 0x3F3A_FF7C, 0x3F3C_E7B5,
    0x3F3E_D2D2, 0x3F40_C0D4, 0x3F42_B1BE, 0x3F44_A590, 0x3F46_9C4B, 0x3F48_95F1, 0x3F4A_9282, 0x3F4C_9201,
    0x3F4E_946E, 0x3F50_99CB, 0x3F52_A218, 0x3F54_AD57, 0x3F56_BB8A, 0x3F58_CCB0, 0x3F5A_E0CD, 0x3F5C_F7E0,
    0x3F5F_11EC, 0x3F61_2EEE, 0x3F63_4EEF, 0x3F65_71E9, 0x3F67_97E3, 0x3F69_C0D6, 0x3F6B_ECCD, 0x3F6E_1BBF,
    0x3F70_4DB8, 0x3F72_82AF, 0x3F74_BAAE, 0x3F76_F5AE, 0x3F79_33B9, 0x3F7B_74C6, 0x3F7D_B8E0, 0x3F80_0000,
];

/// Rejection of [`Color::hex`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum HexColorError {
    /// The string does not start with `#`.
    #[error("hex color must start with '#'")]
    MissingHash,
    /// The digit run is neither 6 (`RRGGBB`) nor 8 (`RRGGBBAA`) characters.
    #[error("hex color must be #RRGGBB or #RRGGBBAA")]
    BadLength,
    /// A character is not a hexadecimal digit.
    #[error("hex color has a non-hex digit")]
    BadDigit,
}

/// Linear-light RGBA color.
///
/// Channels live in [`LinearRgba`]. [`Self::srgb`], [`Self::srgb_u8`] and
/// [`Self::hex`] decode sRGB into that space; [`Self::linear_rgb`] stores
/// the given channels unchanged (alpha `1`). Opaque values serialize as a
/// 3-float linear RGB array — the same shape and color space as the
/// `color` and `ambient` fields of a scene file. Values with any other
/// alpha serialize as four linear floats. Deserialization accepts both
/// lengths; a missing alpha becomes `1.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color(LinearRgba);

impl Color {
    /// Opaque linear white (`1, 1, 1, 1`).
    pub const WHITE: Self = Self::linear_rgb(1.0, 1.0, 1.0);
    /// Opaque linear black (`0, 0, 0, 1`).
    pub const BLACK: Self = Self::linear_rgb(0.0, 0.0, 0.0);

    /// Decodes sRGB channels in `0..=1` (the IEC curve, extended outside
    /// that range) into opaque linear light.
    pub fn srgb(r: f32, g: f32, b: f32) -> Self {
        Self::from_linear(
            srgb_channel_to_linear(r),
            srgb_channel_to_linear(g),
            srgb_channel_to_linear(b),
            1.0,
        )
    }

    /// Decodes 8-bit sRGB channels into opaque linear light.
    ///
    /// Const: the transfer is a 256-entry table because `f32::powf` is not
    /// const on this toolchain. Alpha is `1`.
    pub const fn srgb_u8(r: u8, g: u8, b: u8) -> Self {
        Self::from_linear(
            srgb_u8_channel(r),
            srgb_u8_channel(g),
            srgb_u8_channel(b),
            1.0,
        )
    }

    /// Parses `#RRGGBB` or `#RRGGBBAA` as sRGB (alpha is linear, as in CSS).
    ///
    /// Const, same as [`Self::srgb_u8`].
    ///
    /// # Errors
    ///
    /// [`HexColorError`] when `input` lacks a leading `#`, is not 6 or 8
    /// hex digits, or contains a non-hex digit.
    pub const fn hex(input: &str) -> Result<Self, HexColorError> {
        let bytes = input.as_bytes();
        let digits = match bytes {
            [b'#', rest @ ..] => rest,
            _ => return Err(HexColorError::MissingHash),
        };
        match digits {
            [r0, r1, g0, g1, b0, b1] => {
                let r = match hex_byte(*r0, *r1) {
                    Ok(value) => value,
                    Err(error) => return Err(error),
                };
                let g = match hex_byte(*g0, *g1) {
                    Ok(value) => value,
                    Err(error) => return Err(error),
                };
                let b = match hex_byte(*b0, *b1) {
                    Ok(value) => value,
                    Err(error) => return Err(error),
                };
                Ok(Self::srgb_u8(r, g, b))
            }
            [r0, r1, g0, g1, b0, b1, a0, a1] => {
                let r = match hex_byte(*r0, *r1) {
                    Ok(value) => value,
                    Err(error) => return Err(error),
                };
                let g = match hex_byte(*g0, *g1) {
                    Ok(value) => value,
                    Err(error) => return Err(error),
                };
                let b = match hex_byte(*b0, *b1) {
                    Ok(value) => value,
                    Err(error) => return Err(error),
                };
                let a = match hex_byte(*a0, *a1) {
                    Ok(value) => value,
                    Err(error) => return Err(error),
                };
                Ok(Self::srgb_u8(r, g, b).with_alpha(a as f32 / U8_SCALE))
            }
            _ => Err(HexColorError::BadLength),
        }
    }

    /// Opaque color from channels that are already linear.
    pub const fn linear_rgb(r: f32, g: f32, b: f32) -> Self {
        Self::from_linear(r, g, b, 1.0)
    }

    /// Same RGB channels with `alpha` (linear, not sRGB-encoded).
    pub const fn with_alpha(self, alpha: f32) -> Self {
        let [r, g, b, _] = self.to_linear_rgba().as_array();
        Self::from_linear(r, g, b, alpha)
    }

    /// Stored linear RGBA.
    pub const fn to_linear_rgba(self) -> LinearRgba {
        self.0
    }

    /// Stored linear RGB, dropping alpha.
    pub const fn to_linear_rgb(self) -> LinearRgb {
        self.0.rgb()
    }

    /// Linear alpha.
    pub const fn alpha(self) -> f32 {
        self.to_linear_rgba().as_array()[3]
    }

    /// Encodes the RGB channels back to sRGB (`0..=1` for in-range linear
    /// values). Alpha is not included.
    pub fn to_srgb(self) -> [f32; 3] {
        let [r, g, b] = self.to_linear_rgb().as_array();
        [
            linear_channel_to_srgb(r),
            linear_channel_to_srgb(g),
            linear_channel_to_srgb(b),
        ]
    }

    const fn from_linear(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self(LinearRgba::new([r, g, b, a]))
    }
}

impl Default for Color {
    /// Opaque black.
    fn default() -> Self {
        Self::BLACK
    }
}

impl From<LinearRgb> for Color {
    /// Opaque linear color.
    fn from(value: LinearRgb) -> Self {
        Self(value.with_alpha(1.0))
    }
}

impl From<LinearRgba> for Color {
    fn from(value: LinearRgba) -> Self {
        Self(value)
    }
}

impl From<Color> for LinearRgba {
    fn from(value: Color) -> Self {
        value.to_linear_rgba()
    }
}

impl From<Color> for LinearRgb {
    /// Drops alpha.
    fn from(value: Color) -> Self {
        value.to_linear_rgb()
    }
}

impl Serialize for Color {
    /// Linear RGB when alpha is exactly `1.0`, otherwise linear RGBA.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let [r, g, b, a] = self.to_linear_rgba().as_array();
        // `1.0` is the exact alpha written by the 3-channel decoder and by
        // `linear_rgb` / opaque hex, not a measured quantity.
        #[allow(clippy::float_cmp)]
        if a == 1.0 {
            [r, g, b].serialize(serializer)
        } else {
            [r, g, b, a].serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for Color {
    /// Accepts a linear RGB tuple or array (alpha `1.0`) or a linear RGBA
    /// tuple or array. Scene RON uses the 3-tuple form.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // `deserialize_any` so RON tuples `(r, g, b)` and JSON arrays share
        // one visitor. `deserialize_seq` alone rejects the tuple form scene
        // files already use.
        deserializer.deserialize_any(ColorVisitor)
    }
}

/// Illuminance multiplier: the scene light `intensity` channel.
///
/// A newtype over that `f32`, not a conversion into SI lux. `Lux(0.6)` is
/// the number the renderer already uploads as directional intensity. The
/// serde form is the bare `f32` (transparent).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Lux(pub f32);

impl Lux {
    /// No contribution.
    pub const ZERO: Self = Self(0.0);

    /// Wraps a raw intensity, including non-finite values, so the wire form
    /// stays a plain `f32`.
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    /// Raw intensity.
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl From<f32> for Lux {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<Lux> for f32 {
    fn from(value: Lux) -> Self {
        value.get()
    }
}

impl Serialize for Lux {
    /// Wire form is the raw `f32`.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f32(self.0)
    }
}

impl<'de> Deserialize<'de> for Lux {
    /// Reads the raw `f32` wire form.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self(f32::deserialize(deserializer)?))
    }
}

struct ColorVisitor;

impl<'de> Visitor<'de> for ColorVisitor {
    type Value = Color;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("a linear RGB array of 3 floats or an RGBA array of 4 floats")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let r = required_channel(&mut seq, 0)?;
        let g = required_channel(&mut seq, 1)?;
        let b = required_channel(&mut seq, 2)?;
        let alpha = seq.next_element()?.unwrap_or(1.0);
        if seq.next_element::<f32>()?.is_some() {
            return Err(de::Error::invalid_length(5, &self));
        }
        Ok(Color::from_linear(r, g, b, alpha))
    }
}

fn required_channel<'de, A: SeqAccess<'de>>(seq: &mut A, read: usize) -> Result<f32, A::Error> {
    seq.next_element()?.ok_or_else(|| {
        de::Error::invalid_length(
            read,
            &"a linear RGB array of 3 floats or an RGBA array of 4",
        )
    })
}

fn srgb_channel_to_linear(channel: f32) -> f32 {
    if channel <= SRGB_LINEAR_THRESHOLD {
        channel / SRGB_LINEAR_SLOPE
    } else {
        ((channel + SRGB_POWER_OFFSET) / SRGB_POWER_SCALE).powf(SRGB_POWER_GAMMA)
    }
}

fn linear_channel_to_srgb(channel: f32) -> f32 {
    if channel <= SRGB_LINEAR_OUT_THRESHOLD {
        channel * SRGB_LINEAR_SLOPE
    } else {
        SRGB_POWER_SCALE * channel.powf(1.0 / SRGB_POWER_GAMMA) - SRGB_POWER_OFFSET
    }
}

const fn srgb_u8_channel(byte: u8) -> f32 {
    f32::from_bits(SRGB_U8_TO_LINEAR_BITS[byte as usize])
}

const fn hex_byte(hi: u8, lo: u8) -> Result<u8, HexColorError> {
    let hi = match hex_nibble(hi) {
        Ok(value) => value,
        Err(error) => return Err(error),
    };
    let lo = match hex_nibble(lo) {
        Ok(value) => value,
        Err(error) => return Err(error),
    };
    Ok((hi << 4) | lo)
}

const fn hex_nibble(byte: u8) -> Result<u8, HexColorError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(HexColorError::BadDigit),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AMBIENT_HEX: Color = match Color::hex("#1A1A26") {
        Ok(color) => color,
        Err(_) => Color::BLACK,
    };

    #[test]
    fn endpoints_and_constants() {
        assert_eq!(Color::srgb(0.0, 0.0, 0.0), Color::BLACK);
        assert_eq!(Color::srgb(1.0, 1.0, 1.0), Color::WHITE);
        assert_eq!(Color::srgb_u8(0, 0, 0), Color::BLACK);
        assert_eq!(Color::srgb_u8(255, 255, 255), Color::WHITE);
        assert_eq!(
            Color::linear_rgb(0.25, 0.5, 0.75)
                .to_linear_rgb()
                .as_array(),
            [0.25, 0.5, 0.75]
        );
        assert_eq!(Color::default(), Color::BLACK);
        assert_eq!(Color::BLACK.alpha(), 1.0);
        assert_eq!(Color::WHITE.with_alpha(0.25).alpha(), 0.25);
    }

    #[test]
    fn srgb_curve_matches_the_iec_transfer() {
        let mid = srgb_channel_to_linear(0.5);
        let expected = ((0.5 + 0.055) / 1.055_f32).powf(2.4);
        assert!((mid - expected).abs() < 1e-6);
        let low = srgb_channel_to_linear(0.02);
        assert!((low - 0.02 / 12.92).abs() < 1e-7);
        let encoded = Color::srgb(0.4, 0.5, 0.6).to_srgb();
        assert!((encoded[0] - 0.4).abs() < 1e-5);
        assert!((encoded[1] - 0.5).abs() < 1e-5);
        assert!((encoded[2] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn srgb_u8_table_matches_the_runtime_curve() {
        for byte in 0..=255 {
            let from_table = srgb_u8_channel(byte);
            let from_curve = srgb_channel_to_linear(f32::from(byte) / U8_SCALE);
            assert_eq!(from_table.to_bits(), from_curve.to_bits());
        }
    }

    #[test]
    fn hex_decodes_srgb_and_is_const() {
        assert_eq!(AMBIENT_HEX, Color::srgb_u8(0x1A, 0x1A, 0x26));
        assert_eq!(Color::hex("#1a1a26").unwrap(), AMBIENT_HEX);
        assert_eq!(Color::hex("#FFFFFF").unwrap(), Color::WHITE);
        assert_eq!(Color::hex("#000000").unwrap(), Color::BLACK);
        let with_alpha = Color::hex("#1A1A2680").unwrap();
        assert_eq!(with_alpha.to_linear_rgb(), AMBIENT_HEX.to_linear_rgb());
        assert!((with_alpha.alpha() - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(Color::hex("#FFFFFFFF").unwrap(), Color::WHITE);
        assert_eq!(Color::hex("1A1A26"), Err(HexColorError::MissingHash));
        assert_eq!(Color::hex("#1A1A"), Err(HexColorError::BadLength));
        assert_eq!(Color::hex("#GGGGGG"), Err(HexColorError::BadDigit));
    }

    #[test]
    fn color_ron_matches_linear_rgb_arrays() {
        let linear = [0.1_f32, 0.1, 0.15];
        let color = Color::linear_rgb(linear[0], linear[1], linear[2]);
        let color_ron = ron::ser::to_string(&color).unwrap();
        let array_ron = ron::ser::to_string(&linear).unwrap();
        assert_eq!(color_ron, array_ron);
        let parsed: Color = ron::de::from_str(&color_ron).unwrap();
        assert_eq!(parsed, color);
        assert_eq!(parsed.alpha(), 1.0);

        let scene_ambient: Color = ron::de::from_str("(0.1, 0.1, 0.15)").unwrap();
        assert_eq!(scene_ambient.to_linear_rgb().as_array(), linear);

        let rgba: Color = ron::de::from_str("(0.2, 0.4, 0.6, 0.5)").unwrap();
        assert_eq!(rgba.to_linear_rgb().as_array(), [0.2, 0.4, 0.6]);
        assert_eq!(rgba.alpha(), 0.5);
        let rgba_ron = ron::ser::to_string(&rgba).unwrap();
        let back: Color = ron::de::from_str(&rgba_ron).unwrap();
        assert_eq!(back, rgba);
        assert_ne!(
            rgba_ron,
            ron::ser::to_string(&rgba.to_linear_rgb().as_array()).unwrap()
        );
    }

    #[test]
    fn color_json_accepts_three_and_four_channels() {
        let rgb: Color = serde_json::from_str("[0.1, 0.1, 0.15]").unwrap();
        assert_eq!(rgb.to_linear_rgb().as_array(), [0.1, 0.1, 0.15]);
        assert_eq!(rgb.alpha(), 1.0);
        let rgba: Color = serde_json::from_str("[0.2, 0.3, 0.4, 0.25]").unwrap();
        assert_eq!(rgba.alpha(), 0.25);
        assert!(serde_json::from_str::<Color>("[0.1, 0.2]").is_err());
        assert!(serde_json::from_str::<Color>("[0.1, 0.2, 0.3, 0.4, 0.5]").is_err());
    }

    #[test]
    fn scene_field_shape_round_trips() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct LightFields {
            color: Color,
            intensity: Lux,
        }

        let ron_text = "(color: (1.0, 1.0, 1.0), intensity: 0.6)";
        let parsed: LightFields = ron::de::from_str(ron_text).unwrap();
        assert_eq!(parsed.color, Color::WHITE);
        assert_eq!(parsed.intensity, Lux(0.6));
        let again: LightFields = ron::de::from_str(&ron::ser::to_string(&parsed).unwrap()).unwrap();
        assert_eq!(again, parsed);
    }

    #[test]
    fn lux_is_transparent_f32() {
        assert_eq!(Lux::default(), Lux::ZERO);
        assert_eq!(Lux::from(0.6).get(), 0.6);
        assert_eq!(f32::from(Lux(0.6)), 0.6);
        let text = ron::ser::to_string(&Lux(0.6)).unwrap();
        assert_eq!(text, ron::ser::to_string(&0.6_f32).unwrap());
        assert!(!text.contains("Lux"));
        let parsed: Lux = ron::de::from_str(&text).unwrap();
        assert_eq!(parsed, Lux(0.6));
        let negative: Lux = ron::de::from_str("-1.5").unwrap();
        assert_eq!(negative, Lux(-1.5));
    }
}
