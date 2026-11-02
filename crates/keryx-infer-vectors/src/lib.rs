//! Published compatibility-class vectors.
//!
//! A vector names one card class, the backend patch set and the `compat_id` the protocol must
//! derive from them. A backend's patch set must not change when another backend is added.

use keryx_infer_protocol::{manifest, Backend, CardClass, CompatClass, Hash, UnitKind, Vendor, PROTOCOL};
use serde::Deserialize;

pub const COMPAT_IDS_JSON: &str = include_str!("../../../vectors/compat-ids-v2.json");

#[derive(Clone, Debug, Deserialize)]
pub struct VectorSet {
    pub protocol: u16,
    pub engine_abi: u32,
    pub vectors: Vec<Vector>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Vector {
    pub name: String,
    pub patch_set: String,
    pub cards: Vec<CardSpec>,
    pub compat_id: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CardSpec {
    pub backend: String,
    pub vendor: String,
    pub arch: String,
    pub planned_units: u16,
    pub unit_kind: String,
}

impl VectorSet {
    pub fn parse() -> Result<Self, String> {
        let set: Self = serde_json::from_str(COMPAT_IDS_JSON).map_err(|e| format!("vectors: {e}"))?;
        if set.protocol != PROTOCOL {
            return Err(format!("vectors protocol {} does not match {PROTOCOL}", set.protocol));
        }
        Ok(set)
    }

    pub fn check(&self) -> Result<(), String> {
        let manifest = manifest();
        if self.engine_abi != manifest.engine_abi {
            return Err(format!(
                "vectors engine ABI {} does not match the manifest's {}",
                self.engine_abi, manifest.engine_abi
            ));
        }
        for vector in &self.vectors {
            let cards: Vec<CardClass> = vector.cards.iter().map(CardSpec::to_card).collect::<Result<_, _>>()?;
            let [card] = cards.as_slice() else {
                return Err(format!(
                    "{}: protocol 2 vectors carry exactly one card, the vector has {}",
                    vector.name,
                    cards.len()
                ));
            };
            let class = CompatClass::from_card(self.engine_abi, card.clone(), false)
                .map_err(|e| format!("{}: {e}", vector.name))?;
            let want_patch: Hash = hex::decode(&vector.patch_set)
                .map_err(|e| format!("{}: patch_set: {e}", vector.name))?
                .try_into()
                .map_err(|_| format!("{}: patch_set is not 32 bytes", vector.name))?;
            if want_patch != class.patch_set {
                return Err(format!(
                    "{}: patch_set {} does not match the backend manifest {}",
                    vector.name,
                    vector.patch_set,
                    hex::encode(class.patch_set)
                ));
            }
            let want: Hash = hex::decode(&vector.compat_id)
                .map_err(|e| format!("{}: {e}", vector.name))?
                .try_into()
                .map_err(|_| format!("{}: compat_id is not 32 bytes", vector.name))?;
            if class.id() != want {
                return Err(format!(
                    "{}: class id {} does not match {}",
                    vector.name,
                    hex::encode(class.id()),
                    vector.compat_id
                ));
            }
        }
        Ok(())
    }
}

impl CardSpec {
    pub fn to_card(&self) -> Result<CardClass, String> {
        let backend = Backend::parse(&self.backend).ok_or_else(|| format!("unknown backend {}", self.backend))?;
        let vendor = Vendor::parse(&self.vendor).ok_or_else(|| format!("unknown vendor {}", self.vendor))?;
        let unit_kind =
            UnitKind::parse(&self.unit_kind).ok_or_else(|| format!("unknown unit kind {}", self.unit_kind))?;
        Ok(CardClass::new(backend, vendor, self.arch.clone(), self.planned_units, unit_kind))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_vectors_match_the_protocol() {
        let set = VectorSet::parse().expect("vectors parse");
        set.check().expect("vectors match the class id algorithm");
    }
}
