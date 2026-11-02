# Native build glue

The build phase produces `libkeryx_infer` from `native/engine/` and the pinned llama.cpp tree
(`b10015`).

Planned inputs:

- `KERYX_LLAMA_SRC` — clean `b10015` checkout;
- `KERYX_CUDA_ROOT` — toolkit root;
- `KERYX_ARCHS` — CUDA architecture list (for example `75;80;86;89;120`).

Planned outputs per platform line:

- `libkeryx_infer.so` / `libkeryx_infer.dylib` / `keryx_infer.dll`;
- an optional no-AVX CPU-backend variant;
- a manifest sidecar with `patch_set`, engine build and `vendor_products`.

The build will apply `native/engine/llama_patches.rs` through the same textual anchors the
reference uses, then compile `keryx_llama.cpp` and `trace/*` into the shared library. No PoM file
is part of the build.

There is no `load_pair` target. The only model load is the single-card zero-dup `load_bound`.
