//! Reader for the `ornis_gpu_shader_meta*` records.
//!
//! `#[gpu_entry]` (the `ornis-gpu-shader-macros` attribute) writes one ASCII
//! record into a `#[used]` global, and `xtask build-shaders` translates the IR
//! without any CLI flags for the entry, the workgroup size, or the uniform
//! field tags. The global's name carries the entry
//! (`ornis_gpu_shader_meta_vs_main`), so a crate holds a vertex and a fragment
//! entry side by side and [`extract`] returns both records; the record body
//! repeats the entry name, which is what [`extract_entry`] matches on.
//!
//! The format is defined by `ornis_gpu_shader::meta`; this module
//! only reads it, so an artifact produced by an older attribute still
//! translates as long as it carries the parts it knew about:
//!
//! ```text
//! entry=cs_main;workgroup=64,1,1[;types=6,6,2][;uniform=0:662,1:11]
//! ```
//!
//! `types=` is the pre-typed form (one flat tag list for struct 0);
//! `uniform=` is the current one, one `<id>:<tags>` entry per uniform binding.
//! A nested struct field is the digit `7` followed by that struct's id digit
//! (`671` is a `vec4` and then struct 1); the parsed tag is
//! [`ornis_gpu_abi::TYPE_TAG_NESTED`] plus the id.

use std::collections::BTreeMap;

use crate::TranslateError;

/// Hex radix of the `\XX` escapes LLVM uses inside a string constant.
const HEX_RADIX: u32 = 16;

/// Bytes one `\XX` escape occupies: the backslash and two hex digits.
const ESCAPE_LEN: usize = 3;

/// Axes of a `@workgroup_size`.
const AXES: usize = 3;

/// Metadata one shader crate publishes about its entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderMeta {
    /// Entry function name (`cs_main` today).
    pub entry: String,
    /// `@workgroup_size(x, y, z)`.
    pub workgroup: [u32; 3],
    /// `ornis_gpu.types` field tags per struct id.
    pub structs: BTreeMap<u32, Vec<u8>>,
}

impl ShaderMeta {
    /// Tag list of struct `id`, in declaration order.
    #[must_use]
    pub fn tags(&self, id: u32) -> Option<&[u8]> {
        self.structs.get(&id).map(Vec::as_slice)
    }

    /// Struct table indexed by id, with an empty list for ids the record omits.
    ///
    /// The lowerer indexes structs by id, so a gap has to exist rather than
    /// shift later ids down.
    #[must_use]
    pub fn table(&self) -> Vec<Vec<u8>> {
        let mut table: Vec<Vec<u8>> = Vec::new();
        for (id, tags) in &self.structs {
            let index = *id as usize;
            if table.len() <= index {
                table.resize(index + 1, Vec::new());
            }
            table[index] = tags.clone();
        }
        table
    }
}

/// Read every metadata record out of textual LLVM IR.
///
/// An empty vector means the shader has no record at all (a kernel written
/// against the raw `ornis_gpu` imports); a record that does not parse is an
/// error, because translating it with guessed metadata would silently produce a
/// different shader.
pub fn extract(llvm_ir: &str) -> Result<Vec<ShaderMeta>, TranslateError> {
    let mut records = Vec::new();
    for text in record_texts(llvm_ir)? {
        records.push(parse(&text)?);
    }
    Ok(records)
}

/// The record of entry `entry`, when the module carries one.
///
/// # Errors
///
/// Same as [`extract`]: an unparsable record is never skipped.
pub fn extract_entry(llvm_ir: &str, entry: &str) -> Result<Option<ShaderMeta>, TranslateError> {
    Ok(extract(llvm_ir)?
        .into_iter()
        .find(|meta| meta.entry == entry))
}

/// The unescaped contents of every metadata global, in IR order.
fn record_texts(llvm_ir: &str) -> Result<Vec<String>, TranslateError> {
    let mut texts = Vec::new();
    for initializer in global_initializers(llvm_ir) {
        let text = unescape(initializer).ok_or_else(|| {
            TranslateError::UnsupportedType("shader metadata is not a byte string".to_string())
        })?;
        texts.push(text);
    }
    Ok(texts)
}

/// The `c"…"` literals of the metadata globals, quotes included.
fn global_initializers(llvm_ir: &str) -> Vec<&str> {
    llvm_ir
        .lines()
        .filter(|line| line.contains(ornis_gpu_shader::meta::GLOBAL))
        .filter_map(|line| {
            let start = line.find("c\"")?;
            Some(&line[start..])
        })
        .collect()
}

/// Decode the `\\XX` escapes of one LLVM string constant.
fn unescape(literal: &str) -> Option<String> {
    let rest = literal.strip_prefix("c\"")?;
    let mut end = None;
    let mut index = 0;
    while index < rest.len() {
        match rest.as_bytes()[index] {
            b'\\' => index += ESCAPE_LEN,
            b'"' => {
                end = Some(index);
                break;
            }
            _ => index += 1,
        }
    }
    let mut text = String::new();
    let mut chars = rest[..end?].chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            let high = chars.next()?;
            let low = chars.next()?;
            let byte = u8::from_str_radix(&format!("{high}{low}"), HEX_RADIX).ok()?;
            text.push(char::from(byte));
        } else {
            text.push(ch);
        }
    }
    Some(text)
}

/// Parse a record body.
///
/// # Errors
///
/// When a part has no `=`, the entry or workgroup part is missing, or a tag
/// list is malformed. Unknown keys are ignored so a newer attribute can add
/// parts without breaking this reader.
pub fn parse(text: &str) -> Result<ShaderMeta, TranslateError> {
    let mut entry = None;
    let mut workgroup = None;
    let mut structs = BTreeMap::new();
    for part in text.split(';') {
        if part.is_empty() {
            continue;
        }
        let (key, value) = part.split_once('=').ok_or_else(|| {
            TranslateError::UnsupportedType(format!("shader metadata part `{part}` has no `=`"))
        })?;
        match key {
            "entry" => entry = Some(value.to_string()),
            key if key == ornis_gpu_shader::meta::KEY_WORKGROUP => {
                workgroup = Some(parse_workgroup(value)?);
            }
            "types" => {
                structs.insert(0, parse_tags(value)?);
            }
            key if key == ornis_gpu_shader::meta::KEY_UNIFORM => {
                parse_uniforms(value, &mut structs)?;
            }
            _ => {}
        }
    }
    Ok(ShaderMeta {
        entry: entry.ok_or_else(|| {
            TranslateError::UnsupportedType("shader metadata has no entry".to_string())
        })?,
        workgroup: workgroup.ok_or_else(|| {
            TranslateError::UnsupportedType("shader metadata has no workgroup".to_string())
        })?,
        structs,
    })
}

fn parse_workgroup(value: &str) -> Result<[u32; 3], TranslateError> {
    let mut sizes = value.split(',');
    let mut axes = [0u32; AXES];
    for axis in &mut axes {
        let text = sizes.next().ok_or_else(|| {
            TranslateError::UnsupportedType(format!(
                "workgroup size needs {AXES} comma-separated axes"
            ))
        })?;
        *axis = text.parse().map_err(|_| {
            TranslateError::UnsupportedType(format!("workgroup size `{text}` is not a number"))
        })?;
    }
    if sizes.next().is_some() {
        return Err(TranslateError::UnsupportedType(format!(
            "workgroup size needs {AXES} comma-separated axes"
        )));
    }
    if axes.contains(&0) {
        return Err(TranslateError::UnsupportedType(
            "workgroup size has a zero component".to_string(),
        ));
    }
    Ok(axes)
}

fn parse_tags(value: &str) -> Result<Vec<u8>, TranslateError> {
    if value.is_empty() {
        return Err(TranslateError::UnsupportedType(
            "uniform struct has no fields".to_string(),
        ));
    }
    value
        .split(',')
        .map(|tag| {
            tag.parse::<u8>().map_err(|_| {
                TranslateError::UnsupportedType(format!("field tag `{tag}` is not a number"))
            })
        })
        .collect()
}

fn parse_uniforms(value: &str, structs: &mut BTreeMap<u32, Vec<u8>>) -> Result<(), TranslateError> {
    for part in value.split(',') {
        if part.is_empty() {
            continue;
        }
        let (id, digits) = part.split_once(':').ok_or_else(|| {
            TranslateError::UnsupportedType(format!("uniform part `{part}` has no `:`"))
        })?;
        let id = id.parse::<u32>().map_err(|_| {
            TranslateError::UnsupportedType(format!("uniform struct id `{id}` is not a number"))
        })?;
        // One decimal digit by construction: the writer refuses anything wider
        // (the record is one byte per tag, so `10` would read as struct `1`
        // followed by a stray tag). A record that carries a wider id did not
        // come from the current SDK.
        if id > ornis_gpu_shader::meta::MAX_DIGIT {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform struct id {id} does not fit the one-digit metadata encoding"
            )));
        }
        // A `7` consumes the next digit as the nested struct id.
        let bytes = digits.as_bytes();
        let mut tags = Vec::new();
        let mut index = 0;
        while index < bytes.len() {
            let Some(tag) = digit(bytes[index]) else {
                return Err(TranslateError::UnsupportedType(format!(
                    "uniform field tag `{}` is not a digit",
                    char::from(bytes[index])
                )));
            };
            index += 1;
            if tag == ornis_gpu_abi::TYPE_TAG_STRUCT {
                tags.push(nested_tag(bytes, &mut index)?);
            } else if tag == ornis_gpu_abi::TYPE_TAG_ARRAY {
                // `8`, the element's tag (itself `7` plus an id for an array of
                // structs) and the length. The parsed form keeps the three
                // bytes, so every later walk over the list needs a cursor.
                let element = match bytes.get(index).copied().and_then(digit) {
                    Some(ornis_gpu_abi::TYPE_TAG_STRUCT) => {
                        index += 1;
                        nested_tag(bytes, &mut index)?
                    }
                    Some(tag) if tag < ornis_gpu_abi::TYPE_TAG_ARRAY => {
                        index += 1;
                        tag
                    }
                    _ => {
                        return Err(TranslateError::UnsupportedType(
                            "array field needs an element tag".to_string(),
                        ));
                    }
                };
                let Some(length) = bytes.get(index).copied().and_then(digit) else {
                    return Err(TranslateError::UnsupportedType(
                        "array field needs a length".to_string(),
                    ));
                };
                index += 1;
                if length == 0 {
                    return Err(TranslateError::UnsupportedType(
                        "uniform array has no elements".to_string(),
                    ));
                }
                tags.extend_from_slice(&[ornis_gpu_abi::TYPE_TAG_ARRAY, element, length]);
            } else {
                tags.push(tag);
            }
        }
        if tags.is_empty() {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform struct {id} has no fields"
            )));
        }
        structs.insert(id, tags);
    }
    Ok(())
}

/// Value of one base-10 digit byte, or `None` for anything else.
///
/// The wire format is one byte per tag, so every field of the encoding is a
/// digit. Subtracting `b'0'` without checking underflowed on a stray byte —
/// panicking in a debug build and wrapping in a release one.
fn digit(byte: u8) -> Option<u8> {
    byte.is_ascii_digit().then(|| byte - b'0')
}

/// Parse the struct id that follows a nested-struct tag at `index`.
fn nested_tag(bytes: &[u8], index: &mut usize) -> Result<u8, TranslateError> {
    let Some(id_byte) = bytes.get(*index).copied() else {
        return Err(TranslateError::UnsupportedType(
            "nested struct tag needs an id".to_string(),
        ));
    };
    *index += 1;
    let id = digit(id_byte).ok_or_else(|| {
        TranslateError::UnsupportedType(format!(
            "uniform field tag `{}` is not a digit",
            char::from(id_byte)
        ))
    })?;
    Ok(ornis_gpu_abi::TYPE_TAG_NESTED + id)
}

#[cfg(test)]
mod tests {
    use super::parse_uniforms;
    use std::collections::BTreeMap;

    /// A byte that is not a digit is a malformed record, not a wrap-around.
    ///
    /// The parser subtracted `b'0'` from every tag byte, so `'a'` underflowed —
    /// a panic in a debug build and a nonsense tag in a release one.
    #[test]
    fn a_non_digit_tag_is_an_error() {
        let mut structs = BTreeMap::new();
        let error = parse_uniforms("0:1a", &mut structs).expect_err("not a digit");
        assert!(
            error.to_string().contains("not a digit"),
            "the rejection must name the byte: {error}"
        );
    }

    /// A struct id wider than one digit did not come from the current SDK's
    /// writer, and reading it would be reading a different record: the id part
    /// is one byte, so `uniform=12:` means struct `1` and a stray `2:`.
    #[test]
    fn a_two_digit_struct_id_is_rejected() {
        let mut structs = BTreeMap::new();
        let error = parse_uniforms("12:62", &mut structs).expect_err("two-digit id");
        assert!(
            error.to_string().contains("one-digit"),
            "the rejection must name the encoding: {error}"
        );
    }

    /// A `vec4` followed by a `u32` is two scalar tags, in order.
    #[test]
    fn scalar_tags_keep_their_order() {
        let mut structs = BTreeMap::new();
        parse_uniforms("0:62", &mut structs).expect("two scalar tags");
        assert_eq!(structs.get(&0), Some(&vec![6, 2]));
    }
}
