# keryx-infer

Canonical inference engine and compatibility class for Keryx H15 verifiable inference.

The engine is for any miner and any verifier: several pools, several miner implementations and
solo mining use the same library. Every card runs **one arithmetic** and belongs to **one class
per architecture**, not one class per card model, so a rig with several cards of the same
architecture produces one trace. A verifier — a pool, another miner or a solo node — sends a
custody record only to a rig that can reproduce its trace; a rig refuses a class it has not
proved.

## Status

Draft 0.1 — protocol and SDK skeleton.

- The canonical arithmetic manifest, `compat_id` algorithm and custody record codec are
  implemented and tested.
- The ABI-9 native engine source is under `native/engine/`. It has no `load_pair`, copy loader
  or multi-card `load_bound` path; only the single-card zero-dup loader exists.
  It is not wired to a build yet; the next engine step adds the context-acquire/release path to the shared source.
- The safe SDK (`keryx-infer`) already loads a native library dynamically and exposes the
  single-card zero-dup API, trace generation and replay. It compiles without a built library.

## Why a class, not a card

A trace is a sequence of floating-point sums. The sums depend on the work split and on the
arithmetic path. Two RTX 3080s with different real SM counts must produce the same bits when the
engine plans the same count for their architecture. A class is:

```
backend + architecture + planned units + canonical arithmetic manifest
```

The card's real SM/CU count is deliberately absent. A smaller card runs the same blocks in more
waves. The `patch_set` is per backend and excludes the plan table: adding an architecture, or adding
or validating another backend, does not move existing classes.

The initial class table is in [`docs/class-table.md`](docs/class-table.md):

- CUDA: 7.5→72, 8.0→108, 8.6→84, 8.9→128, 12.0→170 SM.
- Vulkan/AMD: planned CU per `gfx` target.
- SYCL/Intel, Metal/Apple and CPU are reserved; they need their own plan and vectors.

## Scope

**In scope**

- llama.cpp pinned to `b10015` and the ABI-9 trace/arithmetic patches.
- Single-card, zero-dup load from the miner's canonical walk blob (`keryx_infer_load_bound`).
- Context acquire/release for the mining-pad VRAM dance, without reloading weights.
- Trace generation, replay, class identity, conformance vectors and the custody record.
- Multi-backend class model (CUDA, Vulkan, SYCL, Metal, CPU).

**Out of scope**

- `load_pair` / layer split. This project is single-card.
- PoM, walk kernels, context-pad generation or any mining code.
- Model distribution, fee logic, scheduling, TUI.
- Cross-backend bit-identical traces. Each backend/architecture is its own class.

## Layout

```
crates/
  keryx-infer/              safe Rust SDK (dynamic native library)
  keryx-infer-sys/          raw C ABI
  keryx-infer-protocol/     class, manifest, custody record, tree helpers
  keryx-infer-conformance/  CLI: manifest, class-id, inspect, vectors
  keryx-infer-vectors/      published compat-id vectors
native/
  include/keryx_infer.h     stable C ABI
  engine/                   ABI-9 llama.cpp/trace source
  build/                    build glue (next phase)
vectors/                    published vectors
docs/                       class table, integration, pool routing, release, adding hardware
```

## Quick start

```bash
cargo test --workspace

cargo run -p keryx-infer-conformance -- manifest
cargo run -p keryx-infer-conformance -- class-id \
  --backend cuda --arch cc8.6 --vendor nvidia --units 84
cargo run -p keryx-infer-conformance -- vectors
```

The `class-id` command prints the same `compat_id` for every card of compute capability 8.6,
regardless of the real SM count.

## Adding hardware

A new SKU of a known architecture needs no change. A new architecture, vendor or backend follows
[`docs/adding-hardware.md`](docs/adding-hardware.md) and is submitted with the hardware PR
template (`gh pr create --template hardware.md`).

## Replay safety

A rig advertises only classes it has proved with the published vectors. On an opening:

1. check `compat_id`, tier and window before loading the model;
2. replay on the same class and window;
3. verify every kept root and the token root before signing;
4. otherwise answer `class_mismatch`, `window_too_small`, `vector_failed` or `replay_failed` and
   sign nothing.

A verifier (a pool, another miner or a solo node) routes by `(compat_id, tier)` and never guesses
from a GPU name or `engine_abi`. The pool handshake is in [`docs/pool-routing.md`](docs/pool-routing.md).

## License

Apache-2.0 OR MIT. The engine keeps the upstream llama.cpp and Keryx notices in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
