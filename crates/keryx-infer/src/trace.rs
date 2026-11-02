//! Trace types shared by the safe SDK and a miner's scheduler.

use keryx_infer_protocol::Hash;

/// The capture dimensions of the tier currently armed on an engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceDims {
    /// Leaves per position.
    pub leaves: u32,
    /// Sequence leaves in layout order.
    pub seq_leaves: u32,
}

/// Where a traced generation's tokens and leaf hashes go while it runs.
pub trait TraceSink {
    /// A sampled token and its text. Returning false stops generation.
    fn token(&mut self, token: u32, piece: &[u8]) -> bool;
    /// Leaf hashes of rows `t0 ..` of one batch (`n_rows * leaves * 32` bytes).
    fn rows(&mut self, t0: u32, n_ubatch: u32, hashes: &[u8]) -> bool;
}

/// What a finished traced generation hands back beside what its sink took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TracedRun {
    /// Generated tokens fed: positions after the prompt.
    pub generated: u32,
    /// BLAKE3 of every sequence leaf, in layout order.
    pub seq_hashes: Vec<Hash>,
}

/// What an opening wants copied out of a replay.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplayWants {
    /// `(position, leaf, offset)` triples.
    pub pos: Vec<(u64, u64, u64)>,
    pub pos_bytes: usize,
    /// `(leaf, offset)` pairs for whole sequence leaves.
    pub seq: Vec<(u64, u64)>,
    pub seq_bytes: usize,
}

/// The bytes a replay copied out and the hash of every sequence leaf it streamed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Replayed {
    pub pos: Vec<u8>,
    pub seq: Vec<u8>,
    pub seq_hashes: Vec<Hash>,
}
