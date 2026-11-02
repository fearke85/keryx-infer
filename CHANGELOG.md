# Changelog

## 0.1.0 — project skeleton

- Add the `keryx-infer` workspace: protocol, safe SDK, raw ABI, conformance CLI and vectors.
- Define protocol 2: single-card, multi-backend compatibility classes, `compat_id` v2.
- Add the canonical arithmetic manifest and `patch_set`.
- Add the custody record codec and validation.
- Add the ABI-9 inference engine source under `native/engine/`, with no PoM code.
- Add the stable C header (C ABI 2) with the single-card `load_bound` API, context
  acquire/release, trace generation and replay. No `load_pair`.
- Use a per-backend `patch_set`, so adding or validating a backend does not move existing
  classes.
- Validate a class's planned units against the manifest and cross-check the native class at
  engine load.
- Leave the plan table out of the backend `patch_set`, so adding an architecture keeps existing
  classes; the card carries its own planned units.
- Publish each backend's default vendor and unit kind in the manifest, let a plan row override
  them, and refuse classes that differ.
- Add the `plan` arithmetic key, so a change of unit-plan policy moves the backend `patch_set`.
- Refuse to load when the device cannot report its class or reports a different card.
- Cache the native library per path and content hash; refuse a library whose bytes changed after
  it was loaded.
- `keryx-infer-conformance class-id` rejects unknown, repeated or value-less flags.
- Document how and when to add a card, architecture, vendor or backend, with a hardware PR
  template.
- Document the class table, miner integration, pool routing and release policy.
