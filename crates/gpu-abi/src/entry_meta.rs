//! Entry-point metadata emitted by the `#[shader::*]` attribute macros.
//!
//! A wasm guest declares its stage and workgroup size on the function. The
//! macro records the same values in the [`crate::symbols::ENTRY_SECTION`]
//! custom section so the offline translator can read them from the module
//! instead of repeating them as literals. One line per entry:
//!
//! ```text
//! compute,cull,4,1,1
//! ```
//!
//! Fields are stage, export name, and the three workgroup dimensions. The
//! format is plain ASCII: Rust identifiers cannot contain a comma or a
//! newline, so no escaping is needed.

use std::fmt;

/// Number of dimensions in a workgroup size.
pub const WORKGROUP_AXES: usize = 3;

/// Shader stage an entry point belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Stage {
    /// Vertex stage.
    Vertex = 0,
    /// Fragment stage.
    Fragment = 1,
    /// Compute stage.
    Compute = 2,
}

const STAGE_TAGS: [&str; Stage::Compute as usize + 1] = ["vertex", "fragment", "compute"];

impl Stage {
    /// Tag written to the metadata section and accepted by [`Stage::parse`].
    pub const fn as_str(self) -> &'static str {
        STAGE_TAGS[self as usize]
    }

    /// Parse the tag written by [`Stage::as_str`].
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "vertex" => Some(Self::Vertex),
            "fragment" => Some(Self::Fragment),
            "compute" => Some(Self::Compute),
            _ => None,
        }
    }
}

/// One entry point's stage and dispatch shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryPoint {
    /// Shader stage the export runs in.
    pub stage: Stage,
    /// Export name the translator looks up.
    pub export: String,
    /// Workgroup dimensions. Vertex and fragment entries use `[1, 1, 1]`.
    pub workgroup: [u32; WORKGROUP_AXES],
}

impl EntryPoint {
    /// One metadata line, newline-terminated.
    pub fn encode(&self) -> String {
        format!(
            "{},{},{},{},{}\n",
            self.stage.as_str(),
            self.export,
            self.workgroup[0],
            self.workgroup[1],
            self.workgroup[2],
        )
    }
}

/// A metadata section that does not follow the record format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetadataError {
    /// A line has the wrong field count or a field that cannot be parsed.
    Malformed {
        /// 1-based line number in the section.
        line: usize,
        /// What was wrong with the line.
        reason: &'static str,
    },
    /// The stage tag is not one of `vertex`, `fragment`, or `compute`.
    UnknownStage {
        /// 1-based line number in the section.
        line: usize,
        /// The unrecognised tag.
        stage: String,
    },
    /// A workgroup dimension is zero.
    ZeroWorkgroup {
        /// 1-based line number in the section.
        line: usize,
    },
}

impl fmt::Display for MetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed { line, reason } => {
                write!(f, "entry metadata line {line}: {reason}")
            }
            Self::UnknownStage { line, stage } => {
                write!(f, "entry metadata line {line}: unknown stage {stage:?}")
            }
            Self::ZeroWorkgroup { line } => {
                write!(f, "entry metadata line {line}: zero workgroup dimension")
            }
        }
    }
}

impl std::error::Error for MetadataError {}

/// Parse every non-empty line of an entry metadata section.
pub fn decode(data: &str) -> Result<Vec<EntryPoint>, MetadataError> {
    let mut entries = Vec::new();
    for (index, line) in data.lines().enumerate() {
        let line_number = index + 1;
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split(',');
        let stage = fields.next().unwrap_or_default();
        let export = fields.next().ok_or(MetadataError::Malformed {
            line: line_number,
            reason: "missing export name",
        })?;
        let stage = Stage::parse(stage).ok_or_else(|| MetadataError::UnknownStage {
            line: line_number,
            stage: stage.to_string(),
        })?;
        let mut workgroup = [0u32; WORKGROUP_AXES];
        for slot in &mut workgroup {
            let field = fields.next().ok_or(MetadataError::Malformed {
                line: line_number,
                reason: "missing workgroup dimension",
            })?;
            *slot = field.parse().map_err(|_| MetadataError::Malformed {
                line: line_number,
                reason: "workgroup dimension is not a u32",
            })?;
        }
        if fields.next().is_some() {
            return Err(MetadataError::Malformed {
                line: line_number,
                reason: "too many fields",
            });
        }
        if export.is_empty() {
            return Err(MetadataError::Malformed {
                line: line_number,
                reason: "empty export name",
            });
        }
        if workgroup.contains(&0) {
            return Err(MetadataError::ZeroWorkgroup { line: line_number });
        }
        entries.push(EntryPoint {
            stage,
            export: export.to_string(),
            workgroup,
        });
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::{EntryPoint, MetadataError, Stage, decode};

    #[test]
    fn round_trips_a_compute_record() {
        let entry = EntryPoint {
            stage: Stage::Compute,
            export: "cull".to_string(),
            workgroup: [4, 1, 1],
        };
        assert_eq!(decode(&entry.encode()), Ok(vec![entry]));
    }

    #[test]
    fn parses_one_record_per_line() {
        let parsed = decode("compute,cull,4,1,1\nfragment,shade,1,1,1\n").expect("decode");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].stage, Stage::Compute);
        assert_eq!(parsed[0].export, "cull");
        assert_eq!(parsed[1].stage, Stage::Fragment);
        assert_eq!(parsed[1].workgroup, [1, 1, 1]);
    }

    #[test]
    fn rejects_an_unknown_stage() {
        assert_eq!(
            decode("raygen,main,1,1,1\n"),
            Err(MetadataError::UnknownStage {
                line: 1,
                stage: "raygen".to_string(),
            })
        );
    }

    #[test]
    fn rejects_a_zero_workgroup_dimension() {
        assert_eq!(
            decode("compute,cull,0,1,1\n"),
            Err(MetadataError::ZeroWorkgroup { line: 1 })
        );
    }

    #[test]
    fn rejects_extra_fields() {
        assert!(matches!(
            decode("compute,cull,4,1,1,x\n"),
            Err(MetadataError::Malformed {
                line: 1,
                reason: "too many fields"
            })
        ));
    }
}
