//! Safe Rust SDK for the Keryx inference engine.
//!
//! The engine is single-card and zero-dup only: `Engine::load_bound` reads weights from the
//! canonical walk blob through `keryx_infer_load_bound`. There is no layer-split load path.
//!
//! The protocol types are re-exported so a miner can build the prompt, arm the trace, replay an
//! opening and validate a custody record without depending on the native header directly.

pub mod engine;
pub mod error;
pub mod trace;

mod native;

pub use engine::{Engine, Tensor};
pub use error::Error;
pub use keryx_infer_protocol as protocol;
pub use keryx_infer_protocol::{
    b3_root, backend_patch_set, backend_patch_set_for, expected_ubatch, manifest, manifest_bytes, manifest_patch_set,
    tier_spec, tokens_root, trace_root, Backend, CardClass, CompatClass, Hash, Record, RecordBody, RecordHeader,
    RecordTrace, SeqTrace, TierSpec, UnitKind, Vendor, MAGIC, MAX_RECORD_BYTES, PROTOCOL, STAGE_CHAIN, TIERS, UBATCH,
};
pub use trace::{ReplayWants, Replayed, TraceDims, TraceSink, TracedRun};

/// Engine arithmetic ABI this SDK speaks. A library with another value is refused.
pub const ENGINE_ABI: u32 = keryx_infer_sys::ENGINE_ABI;
/// C ABI this SDK speaks.
pub const C_ABI: u32 = keryx_infer_sys::C_ABI;
