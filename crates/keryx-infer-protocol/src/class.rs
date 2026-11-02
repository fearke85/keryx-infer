//! Compatibility classes for the H15 inference protocol.
//!
//! A class is the tuple `(protocol, engine_abi, patch_set, cards)`. The cards are
//! [`CardClass`] entries: backend, vendor, canonical architecture, planned execution units and
//! the kind of those units. The card's real unit count is deliberately absent.
//!
//! Protocol 2 is single-card. The vector stays in the type so the wire format can grow without
//! changing the record layout; [`CompatClass::validate`] refuses anything but one card today.

use borsh::{BorshDeserialize, BorshSerialize};

use crate::manifest::{backend_patch_set, manifest};

/// A 32-byte hash.
pub type Hash = [u8; 32];

/// Wire/semantic version of the compatibility protocol.
pub const PROTOCOL: u16 = 2;

/// Domain separator of the class id. The bytes include the NUL.
pub const COMPAT_DOMAIN: &[u8] = b"keryx-interop/compat/v2\0";

/// Execution backend. The numeric values are part of the class id; never renumber them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Backend {
    Cuda = 1,
    Vulkan = 2,
    Sycl = 3,
    Metal = 4,
    Cpu = 5,
}

impl Backend {
    pub const fn code(self) -> u16 {
        self as u16
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Sycl => "sycl",
            Self::Metal => "metal",
            Self::Cpu => "cpu",
        }
    }

    pub fn from_code(code: u16) -> Option<Self> {
        Some(match code {
            1 => Self::Cuda,
            2 => Self::Vulkan,
            3 => Self::Sycl,
            4 => Self::Metal,
            5 => Self::Cpu,
            _ => return None,
        })
    }

    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "cuda" => Self::Cuda,
            "vulkan" => Self::Vulkan,
            "sycl" => Self::Sycl,
            "metal" => Self::Metal,
            "cpu" => Self::Cpu,
            _ => return None,
        })
    }
}

/// Vendor of the card. The backend, not the vendor, decides the arithmetic path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Vendor {
    Nvidia = 1,
    Amd = 2,
    Intel = 3,
    Apple = 4,
    Other = 255,
}

impl Vendor {
    pub const fn code(self) -> u16 {
        self as u16
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Nvidia => "nvidia",
            Self::Amd => "amd",
            Self::Intel => "intel",
            Self::Apple => "apple",
            Self::Other => "other",
        }
    }

    pub fn from_code(code: u16) -> Option<Self> {
        Some(match code {
            1 => Self::Nvidia,
            2 => Self::Amd,
            3 => Self::Intel,
            4 => Self::Apple,
            255 => Self::Other,
            _ => return None,
        })
    }

    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "nvidia" => Self::Nvidia,
            "amd" => Self::Amd,
            "intel" => Self::Intel,
            "apple" => Self::Apple,
            "other" => Self::Other,
            _ => return None,
        })
    }
}

/// The kind of execution unit the planned count refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum UnitKind {
    Sm = 1,
    Cu = 2,
    XeCore = 3,
    GpuCore = 4,
    Thread = 5,
}

impl UnitKind {
    pub const fn code(self) -> u8 {
        self as u8
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Sm => "sm",
            Self::Cu => "cu",
            Self::XeCore => "xe_core",
            Self::GpuCore => "gpu_core",
            Self::Thread => "thread",
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            1 => Self::Sm,
            2 => Self::Cu,
            3 => Self::XeCore,
            4 => Self::GpuCore,
            5 => Self::Thread,
            _ => return None,
        })
    }

    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "sm" => Self::Sm,
            "cu" => Self::Cu,
            "xe_core" => Self::XeCore,
            "gpu_core" => Self::GpuCore,
            "thread" => Self::Thread,
            _ => return None,
        })
    }
}

/// One card in a trace class: a backend and architecture, not a card model.
///
/// `planned_units` is the count the engine splits work for on this architecture, not the card's
/// real count. Two cards of the same architecture with different real counts share this value.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, BorshSerialize, BorshDeserialize)]
pub struct CardClass {
    /// [`Backend::code`].
    pub backend: u16,
    /// [`Vendor::code`].
    pub vendor: u16,
    /// Canonical architecture id: `cc8.6`, `gfx1100`, `xe-hpg`, `apple-g13`, `x86_64-avx2`.
    pub arch: String,
    /// Planned execution units for the architecture.
    pub planned_units: u16,
    /// [`UnitKind::code`].
    pub unit_kind: u8,
}

impl CardClass {
    pub fn new(
        backend: Backend,
        vendor: Vendor,
        arch: impl Into<String>,
        planned_units: u16,
        unit_kind: UnitKind,
    ) -> Self {
        Self {
            backend: backend.code(),
            vendor: vendor.code(),
            arch: arch.into(),
            planned_units,
            unit_kind: unit_kind.code(),
        }
    }

    pub fn backend(&self) -> Option<Backend> {
        Backend::from_code(self.backend)
    }

    pub fn vendor(&self) -> Option<Vendor> {
        Vendor::from_code(self.vendor)
    }

    pub fn unit_kind(&self) -> Option<UnitKind> {
        UnitKind::from_code(self.unit_kind)
    }

    /// A human-readable label, for logs and hello display. Never used for routing.
    pub fn label(&self) -> String {
        format!(
            "{}/{}/{}",
            self.backend().map_or("unknown", Backend::name),
            self.arch,
            self.vendor().map_or("unknown", Vendor::name)
        )
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.backend().is_none() {
            return Err(format!("unknown backend code {}", self.backend));
        }
        if self.vendor().is_none() {
            return Err(format!("unknown vendor code {}", self.vendor));
        }
        if self.unit_kind().is_none() {
            return Err(format!("unknown unit kind code {}", self.unit_kind));
        }
        if self.arch.is_empty() {
            return Err("card class has no architecture id".into());
        }
        if self.arch.len() > u16::MAX as usize {
            return Err("architecture id is too long".into());
        }
        if !self.arch.bytes().all(|b| b.is_ascii_graphic()) {
            return Err("architecture id must be printable ASCII without spaces".into());
        }
        if self.planned_units == 0 {
            return Err("card class has no planned unit count".into());
        }
        Ok(())
    }
}

/// Everything two engines must share for one to replay the other's trace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompatClass {
    pub protocol: u16,
    pub engine_abi: u32,
    pub patch_set: Hash,
    pub cards: Vec<CardClass>,
}

impl CompatClass {
    /// Builds an advertiseable class for one card. The backend must be `validated` in the
    /// manifest, and the card's vendor, unit kind and planned units must match the backend's
    /// section and plan row.
    pub fn single(engine_abi: u32, card: CardClass) -> Result<Self, String> {
        Self::from_card(engine_abi, card, true)
    }

    /// Builds a class for a published vector. `require_ready` is false for a backend whose plan
    /// and arithmetic are published but whose vectors have not been validated yet.
    pub fn from_card(engine_abi: u32, card: CardClass, require_ready: bool) -> Result<Self, String> {
        card.validate()?;
        let backend = card.backend().ok_or("card has no known backend")?;
        let name = backend.name();
        let manifest = manifest();
        if engine_abi != manifest.engine_abi {
            return Err(format!("class engine ABI {engine_abi} does not match the manifest's {}", manifest.engine_abi));
        }
        if require_ready && !manifest.backend_ready(name) {
            return Err(format!("backend {name} is not validated in the manifest"));
        }
        let planned = manifest
            .plan_units(name, &card.arch)
            .ok_or_else(|| format!("backend {name}/{} has no published plan", card.arch))?;
        if planned != card.planned_units {
            return Err(format!(
                "backend {name}/{} plans {planned} units, the class says {}",
                card.arch, card.planned_units
            ));
        }
        let (vendor, unit_kind) = manifest
            .card_kind(name, &card.arch)
            .ok_or_else(|| format!("backend {name}/{} has no published card kind", card.arch))?;
        if card.vendor != vendor.code() || card.unit_kind != unit_kind.code() {
            return Err(format!(
                "backend {name}/{} cards are {}/{}, the class says {}/{}",
                card.arch,
                vendor.name(),
                unit_kind.name(),
                card.vendor().map_or("unknown", Vendor::name),
                card.unit_kind().map_or("unknown", UnitKind::name)
            ));
        }
        let patch_set = backend_patch_set(name).ok_or_else(|| format!("backend {name} has no class hash"))?;
        let class = Self { protocol: PROTOCOL, engine_abi, patch_set, cards: vec![card] };
        class.validate()?;
        Ok(class)
    }

    /// Structural validation for a record or a foreign class. It does not require the class to
    /// match this build's manifest; the replayer does that comparison separately.
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol != PROTOCOL {
            return Err(format!("protocol {} is not supported (expected {PROTOCOL})", self.protocol));
        }
        if self.engine_abi == 0 {
            return Err("class has no engine ABI".into());
        }
        if self.patch_set == [0u8; 32] {
            return Err("class has no patch set".into());
        }
        if self.cards.len() != 1 {
            return Err(format!("protocol 2 is single-card; the class has {} cards", self.cards.len()));
        }
        self.cards[0].validate()
    }

    /// Validation against this build's manifest: the class must be exactly what
    /// [`CompatClass::single`] builds for its card. This is what a miner calls before advertising
    /// a class.
    pub fn validate_local(&self) -> Result<(), String> {
        let card = self.cards.first().ok_or("class has no card")?;
        let expected = Self::single(self.engine_abi, card.clone())?;
        if *self != expected {
            return Err("class does not match what this build's manifest derives for its card".into());
        }
        Ok(())
    }

    /// The 32-byte class id. Cards are sorted by their canonical fields before hashing, so the
    /// order a miner enumerated its devices does not change the id.
    pub fn id(&self) -> Hash {
        let mut cards = self.cards.clone();
        cards.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(COMPAT_DOMAIN);
        hasher.update(&self.protocol.to_le_bytes());
        hasher.update(&self.engine_abi.to_le_bytes());
        hasher.update(&self.patch_set);
        hasher.update(&[cards.len() as u8]);
        for card in cards {
            hasher.update(&card.backend.to_le_bytes());
            hasher.update(&card.vendor.to_le_bytes());
            let arch = card.arch.as_bytes();
            hasher.update(&(arch.len() as u16).to_le_bytes());
            hasher.update(arch);
            hasher.update(&card.planned_units.to_le_bytes());
            hasher.update(&[card.unit_kind]);
        }
        *hasher.finalize().as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nvidia(arch: &str, units: u16) -> CardClass {
        CardClass::new(Backend::Cuda, Vendor::Nvidia, arch, units, UnitKind::Sm)
    }

    #[test]
    fn a_class_has_a_stable_id() {
        let class = CompatClass::single(9, nvidia("cc8.6", 84)).unwrap();
        let first = class.id();
        let second = class.id();
        assert_eq!(first, second);
        assert_eq!(first.len(), 32);
    }

    #[test]
    fn different_backends_are_different_classes() {
        let cuda = CompatClass::single(9, nvidia("cc8.6", 84)).unwrap();
        let vulkan =
            CompatClass::from_card(9, CardClass::new(Backend::Vulkan, Vendor::Amd, "gfx1100", 96, UnitKind::Cu), false)
                .unwrap();
        assert_ne!(cuda.id(), vulkan.id());
    }

    #[test]
    fn real_sm_count_is_not_part_of_the_class() {
        let a = CompatClass::single(9, nvidia("cc8.6", 84)).unwrap();
        let b = CompatClass::single(9, nvidia("cc8.6", 84)).unwrap();
        assert_eq!(a.id(), b.id());
        // The class only carries the planned count; a real 3060/3070/3080 is not representable.
        assert_eq!(a.cards[0].planned_units, 84);
    }

    #[test]
    fn a_class_must_match_the_manifest_plan() {
        assert!(CompatClass::single(9, nvidia("cc8.6", 1)).is_err());
        assert!(CompatClass::single(9, nvidia("cc8.6", 84)).is_ok());
        assert!(CompatClass::single(9, nvidia("cc8.6", 84)).unwrap().validate_local().is_ok());
    }

    #[test]
    fn a_pending_backend_is_not_advertiseable() {
        let vulkan = CardClass::new(Backend::Vulkan, Vendor::Amd, "gfx1100", 96, UnitKind::Cu);
        assert!(CompatClass::single(9, vulkan.clone()).is_err());
        let vector = CompatClass::from_card(9, vulkan, false).unwrap();
        assert_eq!(vector.cards[0].planned_units, 96);
    }

    #[test]
    fn a_card_must_match_its_backend_kind() {
        let amd = CardClass::new(Backend::Cuda, Vendor::Amd, "cc8.6", 84, UnitKind::Sm);
        assert!(CompatClass::single(9, amd).is_err());
        let cu = CardClass::new(Backend::Cuda, Vendor::Nvidia, "cc8.6", 84, UnitKind::Cu);
        assert!(CompatClass::single(9, cu).is_err());
        let mut class = CompatClass::single(9, nvidia("cc8.6", 84)).unwrap();
        class.cards[0].vendor = Vendor::Amd.code();
        assert!(class.validate_local().is_err());
        let mut extra = CompatClass::single(9, nvidia("cc8.6", 84)).unwrap();
        extra.cards.push(nvidia("cc8.6", 84));
        assert!(extra.validate_local().is_err());
    }

    #[test]
    fn protocol_2_is_single_card() {
        let mut class = CompatClass::single(9, nvidia("cc8.6", 84)).unwrap();
        class.cards.push(nvidia("cc12.0", 170));
        assert!(class.validate().is_err());
    }

    #[test]
    fn unvalidated_cards_are_refused() {
        let mut card = nvidia("cc8.6", 84);
        card.planned_units = 0;
        assert!(card.validate().is_err());
        card.planned_units = 84;
        card.arch = "cc 8.6".into();
        assert!(card.validate().is_err());
    }
}
