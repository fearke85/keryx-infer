//! Custody record.
//!
//! The record is a compact copy of a traced answer that any verifier (a pool, another miner or a
//! solo node) can keep and replay: the class that produced it, the replay window, the cards, and
//! the trace geometry a replayer needs. It carries no model code
//! and no engine code; decoding it only says whether the record is structurally trustworthy.
//!
//! Layout:
//!
//! ```text
//! magic[8] = "KXIREC01"
//! u32_le header_len
//! header   borsh(RecordHeader)
//! body     borsh(RecordBody)
//! digest[32] BLAKE3(header || body)
//! ```

use borsh::{BorshDeserialize, BorshSerialize};

use crate::class::{CardClass, CompatClass, Hash};
use crate::trace::{b3_root, expected_ubatch, tier_spec, tokens_root, STAGE_CHAIN};

pub const MAGIC: &[u8; 8] = b"KXIREC01";
pub const MAX_RECORD_BYTES: usize = 512 << 20;

#[derive(Clone, Debug, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct RecordHeader {
    pub protocol: u16,
    pub engine_abi: u32,
    pub patch_set: Hash,
    pub compat_id: Hash,
    pub model_id: Hash,
    pub tier: u8,
    pub request_hash: Hash,
    pub max_tokens: u32,
    /// Context window the producer used. A replayer must load the same window or prove that
    /// another window yields the same roots.
    pub window: u32,
    /// Protocol 2 is single-card. The vector stays for wire compatibility.
    pub cards: Vec<CardClass>,
    /// Informational only; never part of the class id.
    pub driver_version: String,
}

#[derive(Clone, Debug, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct RecordTrace {
    pub stage: u8,
    pub n_prompt: u32,
    pub n_pos: u32,
    pub ubatch: Vec<u32>,
    pub n_leaves: u64,
    pub leaves: Vec<Hash>,
    pub pos_roots: Vec<Hash>,
    /// Root of the position tree, before the sequence tree is merged in.
    pub trace_root: Hash,
    pub tokens_root: Option<Hash>,
}

#[derive(Clone, Debug, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct SeqTrace {
    pub leaves: Vec<Hash>,
    pub root: Hash,
}

#[derive(Clone, Debug, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct RecordBody {
    pub record: RecordTrace,
    /// Prompt then generated token ids, as the engine consumed them.
    pub tokens: Vec<i32>,
    pub seq: Option<SeqTrace>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub header: RecordHeader,
    pub body: RecordBody,
}

impl Record {
    pub fn class(&self) -> CompatClass {
        CompatClass {
            protocol: self.header.protocol,
            engine_abi: self.header.engine_abi,
            patch_set: self.header.patch_set,
            cards: self.header.cards.clone(),
        }
    }

    /// Serializes the record: magic, header length, header, body, digest.
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let header = borsh::to_vec(&self.header).map_err(|e| format!("header: {e}"))?;
        let body = borsh::to_vec(&self.body).map_err(|e| format!("body: {e}"))?;
        if header.len() > u32::MAX as usize {
            return Err("header is too large".into());
        }
        let mut digest = blake3::Hasher::new();
        digest.update(&header);
        digest.update(&body);
        let mut out = Vec::with_capacity(12 + header.len() + body.len() + 32);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(header.len() as u32).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&body);
        out.extend_from_slice(digest.finalize().as_bytes());
        if out.len() > MAX_RECORD_BYTES {
            return Err("record exceeds the size cap".into());
        }
        Ok(out)
    }

    /// Parses and validates a record. The digest is checked before the body is trusted.
    pub fn decode(blob: &[u8]) -> Result<Self, String> {
        if blob.len() > MAX_RECORD_BYTES {
            return Err("record exceeds the size cap".into());
        }
        if blob.len() < 8 + 4 + 32 {
            return Err("record is too short".into());
        }
        if &blob[..8] != MAGIC {
            return Err("not a custody record".into());
        }
        let header_len =
            u32::from_le_bytes(blob[8..12].try_into().map_err(|_| "a cut record header".to_string())?) as usize;
        let header_end = 12usize.checked_add(header_len).ok_or("a cut record header")?;
        let body_end = blob.len().checked_sub(32).ok_or("a cut record digest")?;
        if header_end > body_end {
            return Err("a cut record header".into());
        }
        let header_bytes = &blob[12..header_end];
        let body_bytes = &blob[header_end..body_end];
        let digest = &blob[body_end..];
        let mut hasher = blake3::Hasher::new();
        hasher.update(header_bytes);
        hasher.update(body_bytes);
        if hasher.finalize().as_bytes() != digest {
            return Err("the record digest does not match".into());
        }
        let header = RecordHeader::try_from_slice(header_bytes).map_err(|e| format!("bad header: {e}"))?;
        let body = RecordBody::try_from_slice(body_bytes).map_err(|e| format!("bad body: {e}"))?;
        let record = Self { header, body };
        record.validate()?;
        Ok(record)
    }

    pub fn generated_ids(&self) -> Result<Vec<u32>, String> {
        let n_prompt = self.body.record.n_prompt as usize;
        if n_prompt > self.body.tokens.len() {
            return Err("the prompt is longer than the token list".into());
        }
        self.body.tokens[n_prompt..]
            .iter()
            .map(|&t| u32::try_from(t).map_err(|_| "a token id is negative".to_string()))
            .collect()
    }

    pub fn validate(&self) -> Result<(), String> {
        let header = &self.header;
        let class = self.class();
        class.validate()?;
        if header.cards.len() != 1 {
            return Err("protocol 2 records carry exactly one card".into());
        }
        if class.id() != header.compat_id {
            return Err("record compat id does not match its class fields".into());
        }
        if header.request_hash.iter().all(|&b| b == 0) {
            return Err("record has no request hash".into());
        }
        if header.max_tokens == 0 {
            return Err("record has no token budget".into());
        }
        if header.window == 0 {
            return Err("record has no replay window".into());
        }
        let tier = tier_spec(header.tier).ok_or("record tier has no trace layout")?;
        let record = &self.body.record;
        let n_pos = record.pos_roots.len();
        let n_prompt = record.n_prompt as usize;
        if record.stage != STAGE_CHAIN {
            return Err("record is not a chain trace".into());
        }
        if n_pos == 0
            || n_prompt == 0
            || n_prompt > n_pos
            || record.n_pos as usize != n_pos
            || self.body.tokens.len() != n_pos
            || record.ubatch.len() != n_pos
        {
            return Err("record positions, tokens and batches disagree".into());
        }
        if n_pos - n_prompt > header.max_tokens as usize {
            return Err("record generated more positions than its budget".into());
        }
        if record.n_leaves != tier.pos_leaves || record.leaves.len() != n_pos.saturating_mul(tier.pos_leaves as usize) {
            return Err("record does not carry every position leaf".into());
        }
        for (t, &n_ubatch) in record.ubatch.iter().enumerate() {
            if n_ubatch != expected_ubatch(t as u64, record.n_prompt) {
                return Err("record batches do not match their positions".into());
            }
        }
        for (t, root) in record.pos_roots.iter().enumerate() {
            let at = t * tier.pos_leaves as usize;
            let want = b3_root(&record.leaves[at..at + tier.pos_leaves as usize]).ok_or("empty position leaf set")?;
            if want != *root {
                return Err("record position root does not match its leaves".into());
            }
        }
        let seq = self.body.seq.as_ref().ok_or("record has no sequence leaves")?;
        if seq.leaves.len() as u64 != tier.seq_leaves
            || b3_root(&seq.leaves).ok_or("empty sequence leaf set")? != seq.root
        {
            return Err("record sequence leaves disagree".into());
        }
        let pos_tree_root = b3_root(&record.pos_roots).ok_or("record has no positions")?;
        if record.trace_root != pos_tree_root {
            return Err("record position root disagreement".into());
        }
        let token_root = record.tokens_root.ok_or("record has no generated-token root")?;
        if tokens_root(&self.generated_ids()?) != token_root {
            return Err("record token root disagreement".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class::{Backend, CardClass, UnitKind, Vendor, PROTOCOL};

    fn card() -> CardClass {
        CardClass::new(Backend::Cuda, Vendor::Nvidia, "cc12.0", 170, UnitKind::Sm)
    }

    fn record() -> Record {
        let tier = 2u8;
        let spec = tier_spec(tier).unwrap();
        let n_prompt = 2u32;
        let n_pos = 3usize;
        let tokens = vec![10, 11, 12];
        let leaves: Vec<Hash> = (0..n_pos * spec.pos_leaves as usize)
            .map(|i| *blake3::hash(&(i as u64).to_le_bytes()).as_bytes())
            .collect();
        let pos_roots: Vec<Hash> = (0..n_pos)
            .map(|t| b3_root(&leaves[t * spec.pos_leaves as usize..(t + 1) * spec.pos_leaves as usize]).unwrap())
            .collect();
        let seq: Vec<Hash> = vec![[4u8; 32]; spec.seq_leaves as usize];
        let class = CompatClass::single(9, card()).unwrap();
        let header = RecordHeader {
            protocol: PROTOCOL,
            engine_abi: 9,
            patch_set: class.patch_set,
            compat_id: class.id(),
            model_id: [2u8; 32],
            tier,
            request_hash: [1u8; 32],
            max_tokens: 8,
            window: 32_768,
            cards: vec![card()],
            driver_version: "nctx=32768".into(),
        };
        let body = RecordBody {
            record: RecordTrace {
                stage: STAGE_CHAIN,
                n_prompt,
                n_pos: n_pos as u32,
                ubatch: vec![2, 2, 1],
                n_leaves: spec.pos_leaves,
                pos_roots: pos_roots.clone(),
                trace_root: b3_root(&pos_roots).unwrap(),
                leaves,
                tokens_root: Some(tokens_root(&[12])),
            },
            tokens,
            seq: Some(SeqTrace { root: b3_root(&seq).unwrap(), leaves: seq }),
        };
        Record { header, body }
    }

    #[test]
    fn a_record_round_trips() {
        let record = record();
        let blob = record.encode().unwrap();
        assert_eq!(&blob[..8], MAGIC);
        let decoded = Record::decode(&blob).unwrap();
        assert_eq!(decoded.header.compat_id, record.header.compat_id);
        assert_eq!(decoded.header.window, 32_768);
        assert_eq!(decoded.generated_ids().unwrap(), vec![12]);
    }

    #[test]
    fn a_flipped_byte_is_refused() {
        let blob = record().encode().unwrap();
        let mut bad = blob.clone();
        bad[20] ^= 1;
        assert!(Record::decode(&bad).is_err());
    }

    #[test]
    fn a_multi_card_record_is_refused() {
        let mut record = record();
        record.header.cards.push(card());
        assert!(record.encode().is_err());
    }
}
