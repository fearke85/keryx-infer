# Miner integration

A miner links the safe SDK or loads the native library. It does not own tokenizer templates,
trace layout or the class table; those come from the protocol crate and the engine.

## 1. Open the library and the model

```rust
let engine = unsafe {
    Engine::load_bound(
        "/path/libkeryx_infer.so",
        &gguf,
        gpu,
        blob_ptr,
        held,
        cut,
        layout_len,
        n_ctx,
        card,
    )?
};
```

`blob_ptr` is the miner's canonical walk blob. The engine copies nothing. There is no `load_pair`;
a model that does not fit one card is not served by this library.

`card` comes from the manifest plus the device's architecture. The planned units come from the
manifest, never from the real SM/CU count. `load_bound` asks the library for the device's own class
(`card_for_device`) and refuses to load when the device cannot report it or reports a different
card, so a miner cannot advertise a class its hardware does not match.

## 2. Mining-pad VRAM dance

```rust
engine.context_acquire(64)?;   // release KV/compute for the pad, weights stay
// ... allocate the PoM v5 pad and mine ...
engine.context_release();
engine.context_acquire(n_ctx)?; // restore before inference
```

`context_acquire` only frees and recreates the `llama_context`. It does not reload weights and
does not change the trace.

## 3. Arm the trace and generate

The protocol crate builds the token stream (request prefix + template + prompt). The engine
tokenizes and generates:

```rust
let dims = engine.trace_init(
    tier_manifest,
    n_layers,
    n_used,
    n_attn,
    n_recur,
    seq_heads,
)?;
let run = engine.generate_traced(&prompt_tokens, max_tokens, &mut sink)?;
```

The sink receives every sampled token and every batch's leaf table. The miner assembles the
`RecordTrace` with the protocol crate and checks that the roots are consistent.

## 4. Replay an opening

```rust
let replayed = engine.replay(&tokens, n_prompt, &wants, &mut rows)?;
```

`rows` receives every batch's leaf table. The caller MUST compare the replayed `pos_roots`,
`seq.root` and `tokens_root` with the kept ones before signing. A mismatch means `replay_failed`
and nothing is signed.

## 5. Conformance

Before advertising a class/tier/window:

```bash
keryx-infer-conformance manifest
keryx-infer-conformance class-id --backend cuda --arch cc8.6 --vendor nvidia --units 84
keryx-infer-conformance vectors
```

The native runner (`run`) is added in the engine phase. It runs the published trace vectors on a
representative card and emits the `compat_id`, roots and `vendor_products` count. A miner stores
the verdict per `(class, tier, window)` and advertises only the passed entries.

## 6. What the miner must not do

- compute its own plan table;
- hash source files to decide compatibility;
- route by GPU name;
- replay a record of another backend/class;
- attempt an opening it has not proved;
- sign an opening whose roots do not match.
