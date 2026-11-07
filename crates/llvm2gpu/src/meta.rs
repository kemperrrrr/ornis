//! Reader for the `ornis_gpu_shader_meta` record.
//!
//! `#[gpu_entry]` (the `ornis-gpu-shader-macros` attribute) writes one ASCII
//! record into a `#[used]` global, and `xtask build-shaders` translates the IR
//! without any CLI flags for the entry, the workgroup size, or the uniform
//! field tags. The format is defined by `ornis_gpu_shader::meta`; this module
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

/// Read the metadata global out of textual LLVM IR.
///
/// Returns `None` when the shader has no record at all (a kernel written
/// against the raw `ornis_gpu` imports); returns an error when a record exists
/// but does not parse, because translating it with guessed metadata would
/// silently produce a different shader.
pub fn extract(llvm_ir: &str) -> Result<Option<ShaderMeta>, TranslateError> {
    let Some(text) = record_text(llvm_ir) else {
        return Ok(None);
    };
    parse(&text).map(Some)
}

/// The record's contents, unescaped, from the global's initializer.
fn record_text(llvm_ir: &str) -> Option<String> {
    let initializer = global_initializer(llvm_ir)?;
    unescape(initializer)
}

/// The `c"…"` literal of the metadata global, quotes included.
fn global_initializer(llvm_ir: &str) -> Option<&str> {
    let line = llvm_ir
        .lines()
        .find(|line| line.contains(ornis_gpu_shader::meta::GLOBAL))?;
    let start = line.find("c\"")?;
    Some(&line[start..])
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
        // A `7` consumes the next digit as the nested struct id.
        let bytes = digits.as_bytes();
        let mut tags = Vec::new();
        let mut index = 0;
        while index < bytes.len() {
            let tag = bytes[index];
            index += 1;
            if !tag.is_ascii_digit() {
                return Err(TranslateError::UnsupportedType(format!(
                    "uniform field tag `{}` is not a digit",
                    char::from(tag)
                )));
            }
            let tag = tag - b'0';
            if tag == ornis_gpu_abi::TYPE_TAG_STRUCT {
                let Some(id_byte) = bytes.get(index).copied() else {
                    return Err(TranslateError::UnsupportedType(
                        "nested struct tag needs an id".to_string(),
                    ));
                };
                index += 1;
                if !id_byte.is_ascii_digit() {
                    return Err(TranslateError::UnsupportedType(format!(
                        "uniform field tag `{}` is not a digit",
                        char::from(id_byte)
                    )));
                }
                tags.push(ornis_gpu_abi::TYPE_TAG_NESTED + (id_byte - b'0'));
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
