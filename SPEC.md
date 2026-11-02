# keryx-infer — specification

**Status:** draft 0.1
**Protocol version:** 2
**Record magic:** `KXIREC01`
**Engine ABI:** 9
**Normative language:** MUST, SHOULD, MAY as in RFC 2119.

## 1. Scope

This document defines the compatibility class, the canonical arithmetic manifest, the custody
record and the engine ABI surface used for Keryx H15 verifiable inference.

The engine is shared by every pool, every miner implementation and solo mining. It does not
define how a model is mined, how weights are distributed, or how a pool schedules work. It does
not require two different backend/architecture classes to produce identical bits. It defines how
a miner proves which class it can produce and replay, and how a verifier (for example a pool,
§8) routes an opening without guessing.

## 2. Terms

- **Producer** — the rig that generated and submitted a traced answer.
- **Replayer** — a rig that opens a custody record on a card of the record's class and window.
- **Compatibility class** — a backend, an architecture, a planned unit count and a canonical
  arithmetic manifest. Two engines of the same class MUST produce the same roots for the same
  request, model, tier and window.
- **`compat_id`** — the 32-byte identifier of a class.
- **Record** — the custody blob (§5).
- **Opening** — the signed evidence the node checks for an audited response.

## 3. Compatibility class

### 3.1 Fields

```
protocol   : u16      // 2
engine_abi : u32      // 9
patch_set  : [u8;32]  // BLAKE3 of the backend's canonical arithmetic section
cards      : [CardClass; 1]   // protocol 2 is single-card
```

```
CardClass {
  backend       : u16   // cuda=1, vulkan=2, sycl=3, metal=4, cpu=5
  vendor        : u16   // nvidia=1, amd=2, intel=3, apple=4, other=255
  arch          : string // "cc8.6", "gfx1100", "xe-hpg", "apple-g13", "x86_64-avx2"
  planned_units : u16   // the count the engine plans for the architecture
  unit_kind     : u8    // sm=1, cu=2, xe_core=3, gpu_core=4, thread=5
}
```

`planned_units` is not the card's real SM/CU/EU/core count. It is the count the engine uses to
split work for the architecture. Two cards of the same class with different real counts MUST
produce the same trace.

The card model, the driver version and the real unit count are not part of the class.

### 3.2 Class id

```
domain = "keryx-interop/compat/v2\0"   // ASCII, including the NUL

bytes =
    domain
 || u16_le(protocol)
 || u32_le(engine_abi)
 || patch_set[0..32]
 || u8(card_count)
 || for each card (sorted by backend, vendor, arch, planned_units, unit_kind):
      u16_le(backend)
   || u16_le(vendor)
   || u16_le(arch_len)
   || arch bytes
   || u16_le(planned_units)
   || u8(unit_kind)

compat_id = BLAKE3(bytes)
```

Implementations MUST reject a class with zero cards, more than one card in protocol 2, an unknown
backend/vendor/unit kind, an empty or non-printable architecture id, or zero planned units. A
class built against the manifest MUST also carry the vendor and unit kind of its backend's
section and the planned units of its plan row.

### 3.3 Planned unit table

The canonical table lives in the manifest, not in a miner. Protocol 2 starts with:

| Backend | Architecture | Unit | Planned | Group |
|---|---|---|---:|---|
| cuda | cc7.5 | SM | 72 | Turing |
| cuda | cc8.0 | SM | 108 | GA100 |
| cuda | cc8.6 | SM | 84 | Ampere consumer |
| cuda | cc8.9 | SM | 128 | Ada |
| cuda | cc12.0 | SM | 170 | Blackwell |
| vulkan | gfx906 | CU | 60 | Vega 20 |
| vulkan | gfx1030 | CU | 80 | RDNA2 Navi 21 |
| vulkan | gfx1031 | CU | 40 | RDNA2 Navi 22 |
| vulkan | gfx1032 | CU | 32 | RDNA2 Navi 23 |
| vulkan | gfx1100 | CU | 96 | RDNA3 Navi 31 |
| vulkan | gfx1101 | CU | 60 | RDNA3 Navi 32 |
| vulkan | gfx1102 | CU | 32 | RDNA3 Navi 33 |
| vulkan | gfx1200 | CU | 64 | RDNA4 Navi 48 |
| vulkan | gfx1201 | CU | 32 | RDNA4 Navi 44 |

Vulkan rows remain `pending` until their conformance vectors pass. SYCL, Metal and CPU rows are
`planned` and have no classes until their tables and vectors are published.

An engine that implements a row MUST split work by the planned count, not by the card's real
count. Changing a row changes the arithmetic and MUST publish a new engine ABI or a new manifest.

## 4. Canonical arithmetic manifest

The manifest is `data/manifest-v2.json` in `keryx-infer-protocol`. It carries:

- protocol and engine ABI;
- trace layout, prompt template and tier leaf-count versions;
- per backend: status (`validated`, `pending`, `planned`, `unsupported`), the default vendor and
  unit kind of its cards, planned-unit table and arithmetic fields (llama tag, unit plan policy,
  MMQ mode, GEMM path, vendor-product policy, softplus, recurrent rounding, attention/RoPE/LSE
  capture).

A plan row is either a unit count (`"cc8.6": 84`) or an object that also overrides the backend's
default vendor and unit kind (`"nv-cc8.6": { "units": 46, "vendor": "nvidia", "unit_kind": "sm" }`).
Every row MUST resolve to a known vendor and unit kind.

There are two hashes. `manifest_patch_set` covers the whole file and is a build artifact id. A
class uses the **backend** hash:

```
backend_patch_set(backend) = BLAKE3(
    "keryx-infer/backend/v2\0"
 || u16_le(protocol)
 || u32_le(engine_abi)
 || string(trace_layout)
 || string(prompt_template)
 || string(tier_leaf_counts)
 || string(backend)
 || u32_le(arithmetic_count)
 || for each arithmetic row sorted by key: string(key) || string(value)
)
```

`string` is `u32_le(byte_len) || bytes`. The administrative `status`, the vendor/unit kind and the
plan table are deliberately not hashed: the card already carries its vendor, unit kind,
architecture and planned units, so they enter `compat_id` directly. Adding a backend, adding an
architecture row, or changing `pending` to `validated` does not change any existing class.
Changing an arithmetic row changes that backend's `patch_set` and every class of the backend;
changing a plan row changes only the classes of that architecture. The arithmetic key `plan` names
the policy that turns a row into a work split; a change to that policy that keeps every number
MUST bump it, so it moves the `patch_set`.

Two miners with different source trees can share a backend `patch_set` only when the canonical
bytes are identical. The published conformance vectors prove that the claim is true.

## 5. Custody record

### 5.1 Layout

```
magic[8]   = "KXIREC01"
u32_le       header_len
header       borsh(RecordHeader)
body         borsh(RecordBody)
digest[32]   BLAKE3(header || body)
```

A record MUST be at most 512 MiB.

### 5.2 Header

Borsh field order:

```
protocol       : u16
engine_abi     : u32
patch_set      : [u8;32]
compat_id      : [u8;32]
model_id       : [u8;32]
tier           : u8
request_hash   : [u8;32]
max_tokens     : u32
window         : u32
cards          : Vec<CardClass>   // exactly one in protocol 2
driver_version : String           // informational only
```

`compat_id` MUST equal the id computed from `protocol`, `engine_abi`, `patch_set` and `cards`.
`window` is a replay hint, not part of the class id. `driver_version` is descriptive and MUST NOT
be used to decide compatibility.

### 5.3 Body

```
RecordTrace {
  stage       : u8
  n_prompt    : u32
  n_pos       : u32
  ubatch      : Vec<u32>
  n_leaves    : u64
  leaves      : Vec<[u8;32]>
  pos_roots   : Vec<[u8;32]>
  trace_root  : [u8;32]      // position tree only
  tokens_root : Option<[u8;32]>
}

RecordBody {
  record : RecordTrace
  tokens : Vec<i32>          // prompt then generated ids
  seq    : Option<SeqTrace>
}

SeqTrace {
  leaves : Vec<[u8;32]>
  root   : [u8;32]
}
```

`trace_root` is the position tree root before the sequence tree is merged. The commitment root is
`B3(trace_root, seq.root)`.

### 5.4 Validation

A decoder MUST reject a record unless:

- magic, header length and digest are valid;
- protocol is supported;
- `compat_id` equals the computed class id;
- `cards` has exactly one entry and validates;
- `patch_set` is the class's backend patch set (checked by the replayer against its manifest);
- `request_hash` is not all zero;
- `max_tokens > 0` and `window > 0`;
- `tier` has a known trace layout;
- `stage == 2` (chain trace);
- `n_prompt > 0`, `n_prompt <= n_pos`, `n_pos == pos_roots.len()`;
- `tokens.len() == n_pos` and `ubatch.len() == n_pos`;
- `n_pos - n_prompt <= max_tokens`;
- `n_leaves` equals the tier's position-leaf count and `leaves.len() == n_pos * n_leaves`;
- for every position `t`, `ubatch[t]` equals `expected_ubatch(t, n_prompt)`;
- for every position `t`, `B3(leaves[t*n_leaves .. (t+1)*n_leaves]) == pos_roots[t]`;
- `seq` is present and `B3(seq.leaves) == seq.root`, with the tier's sequence-leaf count;
- `B3(pos_roots) == trace_root`;
- `tokens_root == BLAKE3(concat(u32_le(id) for id in tokens[n_prompt..]))`;
- all token ids are non-negative.

A decoder MUST NOT attempt a replay when validation fails. A replayer MUST additionally check
that its replay reproduces `pos_roots`, `seq.root` and `tokens_root` before signing anything.

## 6. Engine ABI

The C ABI is `native/include/keryx_infer.h`, C ABI version 2, engine ABI 9.

- **Load:** `keryx_infer_load_bound` only. It reads the model's weights from the canonical walk
  blob and copies nothing. There is no `load_pair` and no layer split.
- **Context:** `keryx_infer_context_acquire` / `context_release` recreate only the KV cache and
  compute buffers; the weights stay in the blob. A miner may shrink the context for the mining
  pad and restore it for inference without reloading the model.
- **Trace:** `keryx_infer_trace_init`, `trace_leaves`, `trace_seq_leaves`,
  `generate_traced`, `replay`.
- **Identity:** `plan_units`, `card_for_device`, `manifest_patch_set(backend)`, `compat_id`.
- **Zero-dup walk:** `tensor_count` / `tensor_info` expose the canonical table for a miner's
  walk. This is optional for inference.
- **Fixed-order policy:** `vendor_products` MUST be 0 for ABI 9. Any product that leaves the
  fixed-order path invalidates the class.

## 7. Conformance

Each `(backend, architecture, tier)` class MUST publish vectors:

```
tier, model_id, prompt, max_tokens, window,
expected trace_root, seq_root, tokens_root, pos_roots, ubatch
```

A rig MUST run the vectors on a representative card of the class and MUST advertise only classes
that pass. The minimum matrix is:

- one card per backend/architecture;
- a second SKU or a test-only planned-unit override to prove the real count does not change bits;
- one cross-driver run per backend, or a `runtime_family` in the class if the backend is sensitive;
- node-oracle openings (256 seeds) per tier/class.

The vectors are public and versioned. A rig that cannot reproduce them MUST NOT advertise the
class.

## 8. Pool handshake and routing

`mining.hello` gains an `interop` member:

```json
{
  "protocol": 2,
  "replay": [
    {
      "backend": "cuda",
      "compat_id": "<64 hex>",
      "tiers": [0, 1, 2],
      "max_window": 32768,
      "engine_build": "<semver+hash>",
      "vectors_hash": "<64 hex>"
    }
  ],
  "portable": false
}
```

- The routing key is `(compat_id, tier)`.
- The pool MUST NOT route by GPU name or by `engine_abi` alone.
- `mining.ai_request.trace` MAY carry `want: <compat_id>`.
- `mining.ai_opening` gains an optional seventh params entry `{ "compat_id": "<hex>" }`.
- `mining.ai_opening_result` carries the class that produced the opening.

A replayer MUST refuse, before loading the model, when:

- the record's `compat_id` is not one it advertised;
- it does not serve the record's tier;
- it cannot load the record's `window`;
- its vectors for the class did not pass.

Error taxonomy:

- `no_replayer` — no live rig advertises the class;
- `class_mismatch` — a rig was asked for a class it does not implement;
- `window_too_small` — the rig cannot load the record's window;
- `replay_failed` — the replay ran but the roots did not match;
- `record_invalid` — the record failed section 5.4.

`replay_failed` and `record_invalid` MUST NOT be retried on the same class without a new record.
A refusal before replay MUST NOT be treated as a strike.

## 9. Versioning and migration

- A change to the class fields, the record layout or the validation rules requires a new protocol
  version or a new record magic.
- A change to arithmetic, the capture or the prompt/tokenizer template changes the affected
  backend's `patch_set` and therefore the `compat_id` of that backend's classes. A change to a plan
  row changes the `compat_id` of that architecture's classes only. These changes MUST NOT move
  the classes of other backends.
- Adding an architecture or validating a pending backend is a new row/section; existing backends
  keep their `patch_set` and classes.
- A miner SHOULD keep old classes replayable for as long as a verifier keeps their records.

## 10. Non-goals

- Cross-backend or cross-architecture bit-identical replay.
- Layer-split (`load_pair`) inference.
- PoM, walk kernels or context-pad generation.
- Model distribution, fee policy, scheduling or miner UX.
- Node-side tolerance verification.
