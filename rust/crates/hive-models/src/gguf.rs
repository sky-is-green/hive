//! Minimal GGUF metadata reader (versions 2 and 3).

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde_json::{Map, Number, Value};

const MAX_STRING_LEN: u64 = 1 << 20;
const MAX_ARRAY_ELEMS: u64 = 1 << 24;
const MAX_LAYER_TYPES: usize = 4096;

const FILE_TYPE_NAMES: &[(u64, &str)] = &[
    (0, "F32"),
    (1, "F16"),
    (2, "Q4_0"),
    (3, "Q4_1"),
    (4, "Q4_1_SOME_F16"),
    (5, "Q4_2"),
    (6, "Q4_3"),
    (7, "Q8_0"),
    (8, "Q5_0"),
    (9, "Q5_1"),
    (10, "Q2_K"),
    (11, "Q3_K_S"),
    (12, "Q3_K_M"),
    (13, "Q3_K_L"),
    (14, "Q4_K_S"),
    (15, "Q4_K_M"),
    (16, "Q5_K_S"),
    (17, "Q5_K_M"),
    (18, "Q6_K"),
    (19, "IQ2_XXS"),
    (20, "IQ2_XS"),
    (21, "Q2_K_S"),
    (22, "IQ3_XXS"),
    (23, "IQ1_S"),
    (24, "IQ4_NL"),
    (25, "IQ3_S"),
    (26, "IQ2_S"),
    (27, "IQ4_XS"),
    (28, "IQ1_M"),
    (29, "BF16"),
    (30, "Q4_0_4_4"),
    (31, "Q4_0_4_8"),
    (32, "Q4_0_8_8"),
    (33, "TQ1_0"),
    (34, "TQ2_0"),
    (141, "PQ2_0"),
];

/// Read one file's metadata. Missing, truncated or non-GGUF files yield `{}`.
pub fn read_metadata(path: &Path) -> Map<String, Value> {
    match parse(path) {
        Ok(meta) => meta,
        Err(_) => Map::new(),
    }
}

fn parse(path: &Path) -> Result<Map<String, Value>, ()> {
    let file = File::open(path).map_err(|_| ())?;
    // A zero-length file cannot be mapped.
    if file.metadata().map_err(|_| ())?.len() == 0 {
        return Err(());
    }
    let map = unsafe { Mmap::map(&file) }.map_err(|_| ())?;
    let mut cursor = Cursor::new(&map);
    if cursor.take(4)? != b"GGUF" {
        return Err(());
    }
    let version = cursor.u32()?;
    if version != 2 && version != 3 {
        return Err(());
    }
    let position = cursor.position();
    let (tensor_count, metadata_count) = match (cursor.u64(), cursor.u64()) {
        (Ok(tensor), Ok(meta)) if meta <= 10_000 => (tensor, meta),
        _ => {
            cursor.seek(position);
            (cursor.u32()? as u64, cursor.u32()? as u64)
        }
    };
    let _ = tensor_count;
    if metadata_count > 10_000 {
        return Err(());
    }

    let mut out: Map<String, Value> = Map::new();
    let mut arch: Option<String> = None;
    let mut file_type: Option<u64> = None;
    let mut contexts: Vec<(String, i64)> = Vec::new();

    for _ in 0..metadata_count {
        let Ok(key_len) = (if version >= 3 {
            cursor.u64()
        } else {
            cursor.u32().map(u64::from)
        }) else {
            break;
        };
        if key_len > 500 {
            break;
        }
        let Ok(key_bytes) = cursor.take(key_len as usize) else {
            break;
        };
        let key = String::from_utf8_lossy(key_bytes).into_owned();
        let Ok(kind) = cursor.u32() else {
            break;
        };
        let Ok(value) = read_value(&mut cursor, version, kind, &key) else {
            break;
        };

        match (&key[..], &value) {
            ("general.architecture", Value::String(text)) => {
                arch = Some(text.clone());
                out.insert(key.clone(), value.clone());
            }
            ("general.file_type", Value::Number(number)) => {
                if let Some(number) = number.as_u64() {
                    file_type = Some(number);
                    out.insert(key.clone(), Value::Number(Number::from(number)));
                }
            }
            (_, Value::Number(number)) if key.ends_with(".context_length") => {
                if let Some(ctx) = number.as_i64() {
                    let prefix = key.trim_end_matches(".context_length").to_owned();
                    contexts.push((prefix, ctx));
                    out.insert(key.clone(), Value::Number(Number::from(ctx)));
                }
            }
            ("general.name" | "general.parameter_count", _) => {
                out.insert(key.clone(), value.clone());
            }
            (_, Value::Number(number)) if is_fit_key(&key) => {
                if let Some(value) = number.as_i64() {
                    out.insert(key.clone(), Value::Number(Number::from(value)));
                } else {
                    out.insert(key.clone(), value.clone());
                }
            }
            (_, Value::Array(_)) if key.ends_with(".attention.layer_types") => {
                out.insert(key.clone(), value.clone());
            }
            _ => {}
        }

        if let Some(arch_name) = &arch
            && file_type.is_some()
            && has_fit(&out)
            && (has_attention(&out) || key.starts_with("tokenizer."))
        {
            let _ = arch_name;
            break;
        }
    }

    if let Some(arch_name) = &arch {
        out.insert("architecture".into(), Value::String(arch_name.clone()));
    }
    if let Some(file_type) = file_type {
        let name = FILE_TYPE_NAMES
            .iter()
            .find(|(id, _)| *id == file_type)
            .map(|(_, name)| (*name).to_owned())
            .unwrap_or_else(|| format!("ftype-{file_type}"));
        out.insert("quantization".into(), Value::String(name));
        out.insert("file_type".into(), Value::Number(Number::from(file_type)));
    }
    if let Some(arch_name) = &arch {
        if let Some((_, ctx)) = contexts.iter().find(|(prefix, _)| prefix == arch_name) {
            out.insert("context_length".into(), Value::Number(Number::from(*ctx)));
        } else if let Some((_, ctx)) = contexts.first() {
            out.insert("context_length".into(), Value::Number(Number::from(*ctx)));
        }
    }
    Ok(out)
}

fn is_fit_key(key: &str) -> bool {
    const SUFFIXES: &[&str] = &[
        ".block_count",
        ".embedding_length",
        ".attention.head_count",
        ".attention.head_count_kv",
        ".attention.key_length",
        ".attention.value_length",
        ".attention.sliding_window",
        ".attention.sliding_window_pattern",
    ];
    SUFFIXES.iter().any(|suffix| key.ends_with(suffix))
}

fn has_fit(out: &Map<String, Value>) -> bool {
    out.keys().any(|key| key.ends_with(".block_count"))
        && out
            .keys()
            .any(|key| key.ends_with(".attention.head_count_kv"))
}

fn has_attention(out: &Map<String, Value>) -> bool {
    out.keys()
        .any(|key| key.ends_with(".attention.layer_types"))
        || (out
            .keys()
            .any(|key| key.ends_with(".attention.sliding_window"))
            && out
                .keys()
                .any(|key| key.ends_with(".attention.sliding_window_pattern")))
}

fn read_value(cursor: &mut Cursor<'_>, version: u32, kind: u32, key: &str) -> Result<Value, ()> {
    Ok(match kind {
        0 => Value::from(cursor.u8()?),
        1 => Value::from(cursor.u8()? as i8),
        2 => Value::from(cursor.u16()?),
        3 => Value::from(cursor.u16()? as i16),
        4 => Value::from(cursor.u32()?),
        5 => Value::from(cursor.u32()? as i32),
        6 => number(cursor.u32()? as f32 as f64),
        7 => Value::Bool(cursor.u8()? != 0),
        8 => {
            let len = if version >= 3 {
                cursor.u64()?
            } else {
                u64::from(cursor.u32()?)
            };
            if len > MAX_STRING_LEN {
                return Err(());
            }
            Value::String(String::from_utf8_lossy(cursor.take(len as usize)?).into_owned())
        }
        9 => {
            let element_kind = cursor.u32()?;
            let len = if version >= 3 {
                cursor.u64()?
            } else {
                u64::from(cursor.u32()?)
            };
            if len > MAX_ARRAY_ELEMS {
                return Err(());
            }
            if element_kind == 8 {
                let keep = key.ends_with(".attention.layer_types");
                let mut strings: Vec<Value> = Vec::new();
                for _ in 0..len {
                    let slen = if version >= 3 {
                        cursor.u64()?
                    } else {
                        u64::from(cursor.u32()?)
                    };
                    if slen > MAX_STRING_LEN {
                        return Err(());
                    }
                    let raw = cursor.take(slen as usize)?;
                    if keep && strings.len() < MAX_LAYER_TYPES {
                        strings.push(Value::String(String::from_utf8_lossy(raw).into_owned()));
                    }
                }
                if keep {
                    Value::Array(strings)
                } else {
                    Value::String(format!("<array:{len}>"))
                }
            } else {
                let size = match element_kind {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4..=6 => 4,
                    10..=12 => 8,
                    _ => 1,
                };
                cursor.skip(len.saturating_mul(size))?;
                Value::String(format!("<array:{len}>"))
            }
        }
        10 => Value::from(cursor.u64()?),
        11 => Value::from(cursor.u64()? as i64),
        12 => number(cursor.u64()? as f64),
        _ => return Err(()),
    })
}

fn number(value: f64) -> Value {
    Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn seek(&mut self, position: usize) {
        self.pos = position;
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], ()> {
        let end = self.pos.checked_add(count).ok_or(())?;
        if end > self.data.len() {
            return Err(());
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn skip(&mut self, count: u64) -> Result<(), ()> {
        let count = usize::try_from(count).map_err(|_| ())?;
        self.take(count).map(|_| ())
    }

    fn u8(&mut self) -> Result<u8, ()> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ()> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, ()> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, ()> {
        let bytes = self.take(8)?;
        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(array))
    }
}

/// Type ids from the spec, for tests and future fields.
pub fn type_name(kind: u32) -> &'static str {
    match kind {
        0 => "uint8",
        1 => "int8",
        2 => "uint16",
        3 => "int16",
        4 => "uint32",
        5 => "int32",
        6 => "float32",
        7 => "bool",
        8 => "string",
        9 => "array",
        10 => "uint64",
        11 => "int64",
        12 => "float64",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_non_gguf_file_yields_no_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not.gguf");
        std::fs::write(&path, b"not a gguf").unwrap();
        assert!(read_metadata(&path).is_empty());
    }
}
