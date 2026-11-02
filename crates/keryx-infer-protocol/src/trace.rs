//! Trace layout constants and the BLAKE3 tree helpers the record validation needs.
//!
//! These are public consensus constants. The engine remains the authority for which model and
//! leaves it produces; this module only says how a record is checked.

use blake3::hazmat::{merge_subtrees_non_root, merge_subtrees_root, Mode};

use crate::class::Hash;

/// Commitment stage of a chain trace.
pub const STAGE_CHAIN: u8 = 2;

/// Prompt positions per batch, as the consensus trace layout fixes them.
pub const UBATCH: u32 = 512;

/// Per-tier trace layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierSpec {
    pub pos_leaves: u64,
    pub seq_leaves: u64,
}

/// Tiers 0..=4, in the order the protocol numbers them.
pub const TIERS: [TierSpec; 5] = [
    TierSpec { pos_leaves: 563, seq_leaves: 2304 },
    TierSpec { pos_leaves: 563, seq_leaves: 240 },
    TierSpec { pos_leaves: 755, seq_leaves: 1152 },
    TierSpec { pos_leaves: 1123, seq_leaves: 6912 },
    TierSpec { pos_leaves: 832, seq_leaves: 1952 },
];

pub fn tier_spec(tier: u8) -> Option<TierSpec> {
    TIERS.get(tier as usize).copied()
}

/// Batch size of position `t` in a prompt of `n_prompt` positions: 512-token prompt batches,
/// then one token per generated position.
pub fn expected_ubatch(t: u64, n_prompt: u32) -> u32 {
    if t >= n_prompt as u64 {
        return 1;
    }
    let start = t / UBATCH as u64 * UBATCH as u64;
    UBATCH.min(n_prompt - start as u32)
}

/// BLAKE3 of the generated token ids, each as four little-endian bytes.
pub fn tokens_root(ids: &[u32]) -> Hash {
    let mut hasher = blake3::Hasher::new();
    for id in ids {
        hasher.update(&id.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn merge(left: &Hash, right: &Hash, root: bool) -> Hash {
    if root {
        *merge_subtrees_root(left, right, Mode::Hash).as_bytes()
    } else {
        merge_subtrees_non_root(left, right, Mode::Hash)
    }
}

fn b3_level(level: &[Hash]) -> Vec<Hash> {
    let top = level.len() == 2;
    let mut next = Vec::with_capacity(level.len().div_ceil(2));
    let (pairs, rest) = level.as_chunks::<2>();
    for pair in pairs {
        next.push(merge(&pair[0], &pair[1], top));
    }
    if let Some(&last) = rest.first() {
        next.push(last);
    }
    next
}

/// Root of a BLAKE3 tree over `leaves`; `None` when there is no leaf.
pub fn b3_root(leaves: &[Hash]) -> Option<Hash> {
    if leaves.is_empty() {
        return None;
    }
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        level = b3_level(&level);
    }
    Some(level[0])
}

/// The commitment root of a trace: the position tree and the sequence tree as two leaves.
pub fn trace_root(pos_tree_root: &Hash, seq_root: &Hash) -> Hash {
    b3_root(&[*pos_tree_root, *seq_root]).expect("two leaves")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_leaf_is_its_own_root() {
        let leaf = [7u8; 32];
        assert_eq!(b3_root(&[leaf]), Some(leaf));
    }

    #[test]
    fn ubatch_follows_the_consensus_layout() {
        assert_eq!(expected_ubatch(0, 600), 512);
        assert_eq!(expected_ubatch(511, 600), 512);
        assert_eq!(expected_ubatch(512, 600), 88);
        assert_eq!(expected_ubatch(599, 600), 88);
        assert_eq!(expected_ubatch(600, 600), 1);
        assert_eq!(expected_ubatch(601, 600), 1);
    }

    #[test]
    fn tokens_root_is_ordered_little_endian() {
        let a = tokens_root(&[1, 2]);
        let b = tokens_root(&[2, 1]);
        assert_ne!(a, b);
    }
}
