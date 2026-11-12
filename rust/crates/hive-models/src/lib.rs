//! The local GGUF library: a header parser and the `models list` rows.
//!
//! The parser is best-effort by design. It reads the metadata KV section,
//! keeps only the keys the library hover card, fit gate and KV estimator need,
//! and gives up on a truncated or hostile header instead of risking a large
//! allocation. It mirrors `harness.models._read_gguf_metadata`.

pub mod gguf;
pub mod library;
pub mod ops;

pub use gguf::read_metadata;
pub use library::Library;
pub use ops::resolve_model;
