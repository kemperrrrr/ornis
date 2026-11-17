//! Shader metadata the builder reads back out of the LLVM IR.
//!
//! `#[gpu_entry]` writes one ASCII record into a `#[used]` global named
//! `ornis_gpu_shader_meta`. The record carries the entry name, the workgroup
//! size, and the
//! field tags of every uniform struct the signature names, so
//! `xtask build-shaders` needs no CLI flags for any of them.
//!
//! This module owns the writer and the format; `ornis-llvm2gpu` owns the
//! reader, because only the translator has to agree with an IR artifact that
//! may predate the current attribute. The record is:
//!
//! ```text
//! entry=<name>;workgroup=<x>,<y>,<z>[;uniform=<id>:<tags>[,<id>:<tags>]…]
//! ```
//!
//! The tags come from the parameter types (`GpuStruct::FIELD_TAGS`), not from
//! the attribute, so the published table cannot disagree with the fields the
//! kernel reads.

/// Name of the global the entry attribute emits.
///
/// The attribute appends the entry name (`ornis_gpu_shader_meta_vs_main`), so a
/// crate with a vertex and a fragment entry publishes two records and the
/// translator can pick either. A record that predates the suffix reads as
/// [`GLOBAL`] itself.
pub const GLOBAL: &str = "ornis_gpu_shader_meta";

/// Key of the workgroup-size part.
pub const KEY_WORKGROUP: &str = "workgroup";

/// Key of the uniform-struct part.
pub const KEY_UNIFORM: &str = "uniform";

/// Bytes the first struct adds before its tags: `;`, the key, `=`, the id and
/// the `:`.
const UNIFORM_OVERHEAD: usize = 1 + KEY_UNIFORM.len() + 1 + 1 + 1;

/// Bytes each later struct adds before its tags: `,`, the id and the `:`.
const UNIFORM_SEPARATOR: usize = 3;

/// Offset of the id inside `,<id>:`.
const ID_AFTER_SEPARATOR: usize = 1;

/// Offset of the `:` after the id inside `,<id>:`.
const COLON_AFTER_ID: usize = 2;

/// ASCII zero, the base of every decimal digit the record carries.
const DIGIT_ZERO: u8 = b'0';

/// Decimal base of the one-digit ids and tags.
const DIGIT_BASE: u32 = 10;

/// Bytes an entry at position `index` adds before its tags.
const fn overhead(index: usize) -> usize {
    if index == 0 {
        UNIFORM_OVERHEAD
    } else {
        UNIFORM_SEPARATOR
    }
}

/// Byte length of a record with `prefix` and the given uniform tag lists.
///
/// A record starts with `prefix` (`entry=<name>;workgroup=<x>,<y>,<z>`), then
/// one `,<id>:<tags>` per struct after the first, which contributes
/// `;uniform=<id>:<tags>` instead. `nested` holds the structs each uniform
/// reaches, and those entries follow their own uniform's tags. Ids and tags are
/// one decimal digit each.
#[must_use]
pub const fn len(prefix: &[u8], uniform_tags: &[&[u8]], nested: &[&[(u32, &[u8])]]) -> usize {
    let mut total = prefix.len();
    let mut index = 0;
    while index < uniform_tags.len() {
        total += overhead(index) + uniform_tags[index].len();
        let mut inner = 0;
        while inner < nested[index].len() {
            total += UNIFORM_SEPARATOR + nested[index][inner].1.len();
            inner += 1;
        }
        index += 1;
    }
    total
}

/// Byte offset where uniform struct `index` starts contributing.
///
/// `tags` carries one tag list per struct, in signature order, and `nested`
/// the structs each of them reached, whose entries sit between this struct and
/// the next.
#[must_use]
pub const fn cursor(
    prefix_len: usize,
    tags: &[&[u8]],
    nested: &[&[(u32, &[u8])]],
    index: usize,
) -> usize {
    let mut offset = prefix_len;
    let mut step = 0;
    while step < index {
        offset += overhead(step) + tags[step].len();
        let mut inner = 0;
        while inner < nested[step].len() {
            offset += UNIFORM_SEPARATOR + nested[step][inner].1.len();
            inner += 1;
        }
        step += 1;
    }
    offset
}

/// Copy `prefix` into a zeroed record of its own length.
#[must_use]
pub const fn entry<const N: usize>(prefix: &[u8]) -> [u8; N] {
    let mut out = [0u8; N];
    let mut index = 0;
    while index < prefix.len() && index < N {
        out[index] = prefix[index];
        index += 1;
    }
    out
}

/// Append `,<id>:<tags>` — or `;uniform=<id>:<tags>` when `first` — at `at`.
///
/// `id` and every tag are one decimal digit, which is what the
/// `ornis_gpu.types` table allows. A longer id or tag is truncated rather than
/// wrapped, so the translator rejects the record instead of reading a
/// different struct.
// qual:allow(srp, max_parameters=6) reason: "the record, where to write, the entry separators, the struct id, its tags and the structs it reaches"
#[must_use]
pub const fn with_uniforms<const N: usize>(
    base: [u8; N],
    at: usize,
    first: bool,
    id: u32,
    tags: &[u8],
    nested: &[(u32, &[u8])],
) -> [u8; N] {
    let mut out = base;
    let mut cursor = at;
    out[cursor] = if first { b';' } else { b',' };
    cursor += 1;
    if first {
        let key = KEY_UNIFORM.as_bytes();
        let mut index = 0;
        while index < key.len() {
            out[cursor] = key[index];
            cursor += 1;
            index += 1;
        }
        out[cursor] = b'=';
        cursor += 1;
    }
    out[cursor] = DIGIT_ZERO + (id % DIGIT_BASE) as u8;
    out[cursor + 1] = b':';
    cursor += 2;
    let mut index = 0;
    while index < tags.len() {
        out[cursor] = DIGIT_ZERO + (tags[index] % DIGIT_BASE as u8);
        cursor += 1;
        index += 1;
    }
    write_nested(&mut out, cursor, nested);
    out
}

/// Append `,<id>:<tags>` for every struct the uniform reaches.
const fn write_nested<const N: usize>(
    out: &mut [u8; N],
    mut cursor: usize,
    nested: &[(u32, &[u8])],
) {
    let mut index = 0;
    while index < nested.len() {
        out[cursor] = b',';
        out[cursor + ID_AFTER_SEPARATOR] = DIGIT_ZERO + (nested[index].0 % DIGIT_BASE) as u8;
        out[cursor + COLON_AFTER_ID] = b':';
        cursor += UNIFORM_SEPARATOR;
        let tags = nested[index].1;
        let mut inner = 0;
        while inner < tags.len() {
            out[cursor] = DIGIT_ZERO + (tags[inner] % DIGIT_BASE as u8);
            cursor += 1;
            inner += 1;
        }
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREFIX: &[u8] = b"entry=cs_main;workgroup=64,1,1";

    #[test]
    fn record_without_uniforms_is_the_prefix() {
        const LEN: usize = len(PREFIX, &[], &[]);
        assert_eq!(LEN, PREFIX.len());
        let record = entry::<LEN>(PREFIX);
        assert_eq!(&record[..], PREFIX);
    }

    #[test]
    fn record_lists_uniform_structs_in_order() {
        const TAGS_A: [u8; 2] = [6, 2];
        const TAGS_B: [u8; 4] = [1, 1, 1, 1];
        const TAGS: [&[u8]; 2] = [&TAGS_A, &TAGS_B];
        const NESTED: [&[(u32, &[u8])]; 2] = [&[], &[]];
        const LEN: usize = len(PREFIX, &TAGS, &NESTED);
        let first_at = cursor(PREFIX.len(), &TAGS, &NESTED, 0);
        let second_at = cursor(PREFIX.len(), &TAGS, &NESTED, 1);
        let first = with_uniforms(entry::<LEN>(PREFIX), first_at, true, 0, &TAGS_A, &[]);
        let record = with_uniforms(first, second_at, false, 1, &TAGS_B, &[]);
        assert_eq!(
            &record[..],
            b"entry=cs_main;workgroup=64,1,1;uniform=0:62,1:1111"
        );
    }

    #[test]
    fn cursor_lands_where_the_previous_struct_ended() {
        const TAGS: [&[u8]; 2] = [&[1, 1, 1, 1, 1, 1, 1], &[6]];
        const NESTED: [&[(u32, &[u8])]; 2] = [&[], &[]];
        assert_eq!(cursor(10, &TAGS, &NESTED, 0), 10);
        assert_eq!(cursor(10, &TAGS, &NESTED, 1), 10 + UNIFORM_OVERHEAD + 7);
    }

    /// A nested struct's own fields travel in the record: the translator needs
    /// them to lay the enclosing block out, and a nested field only names its
    /// struct by id.
    #[test]
    fn a_nested_struct_carries_its_own_tags() {
        const TAGS: [&[u8]; 1] = [&[6, 7, 3]];
        const INNER: [u8; 4] = [6, 6, 6, 1];
        const NESTED: [&[(u32, &[u8])]; 1] = [&[(3, &INNER)]];
        const LEN: usize = len(PREFIX, &TAGS, &NESTED);
        let at = cursor(PREFIX.len(), &TAGS, &NESTED, 0);
        let record = with_uniforms(entry::<LEN>(PREFIX), at, true, 0, TAGS[0], NESTED[0]);
        assert_eq!(
            &record[..],
            b"entry=cs_main;workgroup=64,1,1;uniform=0:673,3:6661"
        );
    }
}
