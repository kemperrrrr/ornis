//! Property checks for arbitrary input and structured mutations of a valid KTX2.

use ornis_texture::{Ktx2Image, MAX_TEXTURE_EDGE};
use proptest::prelude::*;

const VALID: &[u8] = include_bytes!("data/uastc.ktx2");

proptest! {
    #[test]
    fn arbitrary_bytes_never_panic(data in prop::collection::vec(any::<u8>(), 0..4096)) {
        if let Ok(image) = Ktx2Image::from_bytes(data.as_slice()) {
            let (w, h) = image.dimensions();
            prop_assert!((1..=MAX_TEXTURE_EDGE).contains(&w));
            prop_assert!((1..=MAX_TEXTURE_EDGE).contains(&h));
            prop_assert_eq!(image.bytes(), data.as_slice());
        }
    }

    #[test]
    fn mutations_preserve_bounded_dimensions_or_return_errors(
        changes in prop::collection::vec((0..VALID.len(), any::<u8>()), 0..16),
    ) {
        let mut bytes = VALID.to_vec();
        for (offset, value) in changes { bytes[offset] = value; }
        if let Ok(image) = Ktx2Image::from_bytes(bytes.as_slice()) {
            let (w, h) = image.dimensions();
            prop_assert!((1..=MAX_TEXTURE_EDGE).contains(&w));
            prop_assert!((1..=MAX_TEXTURE_EDGE).contains(&h));
            prop_assert_eq!(image.bytes(), bytes.as_slice());
        }
    }
}
