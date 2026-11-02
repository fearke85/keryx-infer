# Native engine source

This directory is the ABI-9 inference engine source:

- `keryx_llama.cpp` — the C ABI over llama.cpp;
- `llama_patches.rs` — the textual llama.cpp patch list;
- `trace/` — the trace graph, ops, BLAKE3 and the SM plan.

It contains **no PoM code**: no walk, pad, nonce or mining kernel.

## Status

There is no `keryx_llama_load_pair`, copy loader or multi-card `keryx_llama_load_bound` path;
`load_bound` rejects anything but one card.
The engine phase will:

1. add `keryx_llama_context_acquire/release` to the shared source, preserving the weights in the
   blob for the mining-pad VRAM dance;
2. rename/export the C ABI symbols to `keryx_infer_*` and wire them to
   `native/include/keryx_infer.h`;
3. apply `llama_patches.rs` through the build glue in `native/build/`;
4. run the conformance vectors.

Until then, this tree is a reference and is not linked by the Rust crates.
