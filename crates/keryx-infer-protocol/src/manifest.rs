//! The canonical arithmetic manifest.
//!
//! The manifest is the semantic identity of an engine build. There are two hashes:
//!
//! - `manifest_patch_set()` covers the whole file and is useful as a build artifact id.
//! - `backend_patch_set(backend)` covers only the global protocol fields and one backend's
//!   arithmetic. A class uses the backend hash, so adding or validating another backend, or adding
//!   an architecture row, does not change existing classes.
//!
//! A plan row's architecture, units, vendor and unit kind are carried by the card itself, so
//! changing a row changes only that row's classes. How the engine turns a row into a work split is
//! the `plan` arithmetic key: changing that policy without changing any number MUST bump it.
//!
//! Every change that can move a trace root — llama tag, unit plan policy, MMQ mode, GEMM path,
//! softplus, recurrent rounding, attention/RoPE/LSE capture, trace layout, prompt template —
//! belongs here and changes the affected classes' ids.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::Deserialize;

use crate::class::{Hash, UnitKind, Vendor};

/// Domain separator of the whole-manifest hash. The bytes include the NUL.
pub const MANIFEST_DOMAIN: &[u8] = b"keryx-infer/manifest/v2\0";
/// Domain separator of one backend's class hash. The bytes include the NUL.
pub const BACKEND_DOMAIN: &[u8] = b"keryx-infer/backend/v2\0";

/// Exact bytes that are hashed by `manifest_patch_set`. Formatting is part of the identity.
pub const MANIFEST_JSON: &str = include_str!("../data/manifest-v2.json");

#[derive(Clone, Debug, Deserialize)]
pub struct Manifest {
    pub protocol: u16,
    pub engine_abi: u32,
    pub trace_layout: String,
    pub prompt_template: String,
    pub tier_leaf_counts: String,
    pub backends: BTreeMap<String, BackendManifest>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BackendManifest {
    pub status: String,
    /// Default vendor of this backend's plan rows. A row may override it.
    #[serde(default)]
    pub vendor: Option<String>,
    /// Default unit kind of this backend's plan rows. A row may override it.
    #[serde(default)]
    pub unit_kind: Option<String>,
    #[serde(default)]
    pub plan: BTreeMap<String, PlanRow>,
    #[serde(default)]
    pub arithmetic: BTreeMap<String, String>,
}

/// One plan row: the planned units, plus the vendor and unit kind when they differ from the
/// backend's defaults (for example an NVIDIA card counted in SMs on a Vulkan backend).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum PlanRow {
    Units(u16),
    Card { units: u16, vendor: String, unit_kind: String },
}

impl PlanRow {
    pub fn units(&self) -> u16 {
        match self {
            Self::Units(units) | Self::Card { units, .. } => *units,
        }
    }
}

impl Manifest {
    pub fn backend(&self, name: &str) -> Option<&BackendManifest> {
        self.backends.get(name)
    }

    /// Planned units of `arch` on `backend`, regardless of the backend's administrative status.
    /// Whether a class may be advertised is decided by `CompatClass::single`.
    pub fn plan_units(&self, backend: &str, arch: &str) -> Option<u16> {
        Some(self.backend(backend)?.plan.get(arch)?.units()).filter(|n| *n > 0)
    }

    /// The vendor and unit kind a card of `backend`/`arch` must carry: the row's own, or the
    /// backend's defaults.
    pub fn card_kind(&self, backend: &str, arch: &str) -> Option<(Vendor, UnitKind)> {
        let section = self.backend(backend)?;
        let (vendor, unit_kind) = match section.plan.get(arch)? {
            PlanRow::Card { vendor, unit_kind, .. } => (vendor.as_str(), unit_kind.as_str()),
            PlanRow::Units(_) => (section.vendor.as_deref()?, section.unit_kind.as_deref()?),
        };
        Some((Vendor::parse(vendor)?, UnitKind::parse(unit_kind)?))
    }

    /// Whether the manifest lets a miner advertise classes for this backend.
    pub fn backend_ready(&self, backend: &str) -> bool {
        self.backend(backend).is_some_and(|b| b.status == "validated")
    }

    /// The class hash for one backend. It excludes the administrative `status` and the plan table,
    /// so validating a backend, adding another backend or adding an architecture row does not
    /// change this value.
    pub fn backend_patch_set(&self, backend: &str) -> Option<Hash> {
        backend_patch_set_for(self, backend)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.protocol != crate::class::PROTOCOL {
            return Err(format!(
                "manifest protocol {} is not supported (expected {})",
                self.protocol,
                crate::class::PROTOCOL
            ));
        }
        if self.engine_abi == 0 {
            return Err("manifest has no engine ABI".into());
        }
        for (name, value) in [
            ("trace_layout", &self.trace_layout),
            ("prompt_template", &self.prompt_template),
            ("tier_leaf_counts", &self.tier_leaf_counts),
        ] {
            if value.is_empty() {
                return Err(format!("manifest has an empty {name}"));
            }
        }
        if self.backends.is_empty() {
            return Err("manifest has no backends".into());
        }
        for (name, backend) in &self.backends {
            if !matches!(backend.status.as_str(), "validated" | "pending" | "planned" | "unsupported") {
                return Err(format!("backend {name} has unknown status {}", backend.status));
            }
            if backend.status == "validated" && backend.plan.is_empty() {
                return Err(format!("validated backend {name} has no plan table"));
            }
            for (arch, row) in &backend.plan {
                if arch.is_empty() || !arch.bytes().all(|b| b.is_ascii_graphic()) {
                    return Err(format!("backend {name} has an invalid architecture id"));
                }
                if row.units() == 0 {
                    return Err(format!("backend {name}/{arch} has zero planned units"));
                }
                if self.card_kind(name, arch).is_none() {
                    return Err(format!("backend {name}/{arch} has no valid vendor and unit kind"));
                }
            }
        }
        Ok(())
    }
}

fn push_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u32).to_le_bytes());
    hasher.update(value.as_bytes());
}

/// Backend class hash over a parsed manifest. Public so a test can prove that adding another
/// backend or architecture does not move an existing backend's class.
pub fn backend_patch_set_for(manifest: &Manifest, backend: &str) -> Option<Hash> {
    let section = manifest.backend(backend)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(BACKEND_DOMAIN);
    hasher.update(&manifest.protocol.to_le_bytes());
    hasher.update(&manifest.engine_abi.to_le_bytes());
    push_str(&mut hasher, &manifest.trace_layout);
    push_str(&mut hasher, &manifest.prompt_template);
    push_str(&mut hasher, &manifest.tier_leaf_counts);
    push_str(&mut hasher, backend);
    hasher.update(&(section.arithmetic.len() as u32).to_le_bytes());
    for (key, value) in &section.arithmetic {
        push_str(&mut hasher, key);
        push_str(&mut hasher, value);
    }
    Some(*hasher.finalize().as_bytes())
}

/// Parsed manifest, cached for the process.
pub fn manifest() -> &'static Manifest {
    static MANIFEST: OnceLock<Manifest> = OnceLock::new();
    MANIFEST.get_or_init(|| {
        let manifest: Manifest = serde_json::from_str(MANIFEST_JSON).expect("embedded manifest-v2.json must parse");
        manifest.validate().expect("embedded manifest-v2.json must validate");
        manifest
    })
}

/// Exact bytes of the canonical manifest.
pub fn manifest_bytes() -> &'static [u8] {
    MANIFEST_JSON.as_bytes()
}

/// BLAKE3 of the whole canonical manifest. This is a build artifact id, not a class id.
pub fn manifest_patch_set() -> Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(MANIFEST_DOMAIN);
    hasher.update(manifest_bytes());
    *hasher.finalize().as_bytes()
}

/// BLAKE3 of one backend's class section.
pub fn backend_patch_set(backend: &str) -> Option<Hash> {
    manifest().backend_patch_set(backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class::PROTOCOL;

    #[test]
    fn the_manifest_parses_and_validates() {
        let m = manifest();
        assert_eq!(m.protocol, PROTOCOL);
        assert_eq!(m.engine_abi, 9);
        assert!(m.backend_ready("cuda"));
        assert_eq!(m.plan_units("cuda", "cc8.6"), Some(84));
        assert_eq!(m.plan_units("cuda", "cc12.0"), Some(170));
        assert!(!m.backend_ready("vulkan"));
        assert_eq!(m.plan_units("vulkan", "gfx1100"), Some(96));
    }

    #[test]
    fn the_manifest_hash_is_stable_and_not_empty() {
        let first = manifest_patch_set();
        let second = manifest_patch_set();
        assert_eq!(first, second);
        assert_ne!(first, [0u8; 32]);
    }

    #[test]
    fn a_backend_hash_ignores_other_backends() {
        let mut m = manifest().clone();
        let before = backend_patch_set_for(&m, "cuda").unwrap();
        m.backends.insert(
            "intel".into(),
            BackendManifest {
                status: "validated".into(),
                vendor: Some("intel".into()),
                unit_kind: Some("xe_core".into()),
                plan: BTreeMap::from([("xe-hpg".to_string(), PlanRow::Units(32))]),
                arithmetic: BTreeMap::from([("dispatch".to_string(), "fixed".to_string())]),
            },
        );
        let after = backend_patch_set_for(&m, "cuda").unwrap();
        assert_eq!(before, after, "adding a backend must not move CUDA's class hash");
    }

    #[test]
    fn a_backend_hash_ignores_plan_rows() {
        let mut m = manifest().clone();
        let before = backend_patch_set_for(&m, "cuda").unwrap();
        m.backends.get_mut("cuda").unwrap().plan.insert("cc9.0".into(), PlanRow::Units(132));
        let after = backend_patch_set_for(&m, "cuda").unwrap();
        assert_eq!(before, after, "adding an architecture must not move CUDA's class hash");
    }

    #[test]
    fn a_backend_hash_changes_with_its_own_arithmetic() {
        let mut m = manifest().clone();
        let before = backend_patch_set_for(&m, "cuda").unwrap();
        m.backends.get_mut("cuda").unwrap().arithmetic.insert("mmq".into(), "split_tile".into());
        let after = backend_patch_set_for(&m, "cuda").unwrap();
        assert_ne!(before, after);
    }

    #[test]
    fn a_planned_backend_needs_a_card_kind() {
        let mut m = manifest().clone();
        m.backends.get_mut("vulkan").unwrap().unit_kind = None;
        assert!(m.validate().is_err());
    }

    #[test]
    fn a_row_may_override_the_card_kind() {
        let json = r#"{ "units": 46, "vendor": "nvidia", "unit_kind": "sm" }"#;
        let row: PlanRow = serde_json::from_str(json).unwrap();
        let mut m = manifest().clone();
        m.backends.get_mut("vulkan").unwrap().plan.insert("nv-cc8.6".into(), row);
        m.validate().unwrap();
        assert_eq!(m.card_kind("vulkan", "nv-cc8.6"), Some((Vendor::Nvidia, UnitKind::Sm)));
        assert_eq!(m.card_kind("vulkan", "gfx1100"), Some((Vendor::Amd, UnitKind::Cu)));
        assert_eq!(m.plan_units("vulkan", "nv-cc8.6"), Some(46));
    }
}
