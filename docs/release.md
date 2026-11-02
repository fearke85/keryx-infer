# Release policy

## Versions

- **C ABI** (`keryx_infer_c_abi`): changes only when the C function table or struct layout breaks.
- **Engine ABI** (`keryx_infer_engine_abi`): the arithmetic/semantic version. ABI 9 is the first
  version of this project.
- **Protocol** (`PROTOCOL`): the class/record wire version. Protocol 2 is the first multi-backend,
  single-card version.
- **Manifest**: the arithmetic identity. A change to a kernel path, capture, prompt template or
  trace layout updates `data/manifest-v2.json` and the affected backend's `patch_set` and
  `compat_id`s. A change to a plan row changes only that architecture's `compat_id`. Other
  backends keep theirs.

## What a release ships

- `libkeryx_infer` for each supported backend/platform line.
- The Rust crates (`keryx-infer`, `keryx-infer-sys`, `keryx-infer-protocol`).
- The conformance CLI and the published vectors.
- `SPEC.md`, `docs/`, `THIRD_PARTY_NOTICES.md`.
- Checksums, SBOM and the exact manifest bytes.

## Class freeze

- A published plan row is frozen for the engine ABI.
- Adding an architecture is a new row and new vectors; it does not change existing classes.
- Changing an existing row requires a new engine ABI or manifest version and invalidates the
  classes of that architecture, not the classes of other architectures or backends.
- Old classes remain readable for the pool's audit window.

## Validation gate

Before publishing a class:

1. run the vectors on a representative card of the backend/architecture;
2. run them under a second SKU or a test-only planned-unit override;
3. run one cross-driver check;
4. run the 256-seed node oracle for each tier;
5. require `vendor_products == 0`;
6. record the `compat_id`, `vectors_hash` and engine build in the release notes.

## Migration

- Protocol 2 miners advertise only protocol 2 classes.
- A miner that cannot pass a class's vectors does not advertise it; the pool returns
  `no_replayer` instead of routing a doomed opening.
