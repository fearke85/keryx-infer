//! Protocol types for the Keryx inference compatibility class and custody record.
//!
//! This crate carries no inference engine and no CUDA code. It defines what two engines must
//! agree on before one may replay the other's traced answer:
//!
//! - the [`CardClass`] of a backend and architecture (not a card model);
//! - the [`CompatClass`] identifier over the engine ABI, semantic arithmetic manifest and cards;
//! - the custody record that carries the class and the trace geometry;
//! - the tree helpers the record validation needs.
//!
//! The normative description is `SPEC.md` at the repository root.

pub mod class;
pub mod manifest;
pub mod record;
pub mod trace;

pub use class::{Backend, CardClass, CompatClass, Hash, UnitKind, Vendor, COMPAT_DOMAIN, PROTOCOL};
pub use manifest::{
    backend_patch_set, backend_patch_set_for, manifest, manifest_bytes, manifest_patch_set, BackendManifest, Manifest,
    PlanRow, BACKEND_DOMAIN, MANIFEST_DOMAIN,
};
pub use record::{Record, RecordBody, RecordHeader, RecordTrace, SeqTrace, MAGIC, MAX_RECORD_BYTES};
pub use trace::{b3_root, expected_ubatch, tier_spec, tokens_root, trace_root, TierSpec, STAGE_CHAIN, TIERS, UBATCH};
