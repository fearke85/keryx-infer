# Published vectors

`compat-ids-v2.json` is the class-id conformance set. Each entry names one card class, that
backend's `patch_set`, and the `compat_id` the protocol must derive from them.

Each entry's `patch_set` must equal `backend_patch_set(backend)` of the embedded manifest. A test
in `keryx-infer-vectors` recomputes every id and fails when the file drifts. Adding or validating
another backend must not change an existing entry.

Trace vectors (expected roots for a fixed model/tier/prompt/window) are added with the native
engine runner. They are published per `(backend, architecture, tier, window)` and are the proof a
miner runs before advertising a class.
