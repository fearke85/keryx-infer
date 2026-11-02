# Pool routing and strike safety

The pool treats a compatibility class as an opaque 32-byte id. It does not need to understand
CUDA, Vulkan or Intel; it only needs to route a record to a rig that advertised the same class.

## Hello

```json
{
  "interop": {
    "protocol": 2,
    "replay": [
      {
        "backend": "cuda",
        "compat_id": "<64 hex>",
        "tiers": [0, 1, 2],
        "max_window": 32768,
        "engine_build": "keryx-infer 0.1.0+<hash>",
        "vectors_hash": "<64 hex>"
      }
    ],
    "portable": false
  }
}
```

- `replay` is per class, not per card. A rig with two classes advertises two entries.
- `tiers` lists only tiers whose vectors passed on this rig.
- `max_window` is the largest window the rig can load for those tiers.
- `engine_build` and `vectors_hash` are diagnostic; routing uses `compat_id`.

## Record

The record carries `compat_id`, `tier`, `window` and one `CardClass`. The pool validates the
record against the protocol before asking any rig.

## Opening

`mining.ai_opening` gains an optional seventh params entry:

```json
{ "compat_id": "<64 hex>" }
```

The miner performs the preflight before loading the model:

1. `compat_id` equals a class it advertised;
2. it serves the record's tier;
3. it can load `window`;
4. its vectors for the class passed;
5. the record decodes and validates.

If any check fails, it returns `class_mismatch`, `window_too_small`, `vector_failed` or
`record_invalid` and signs nothing. This is not a strike.

If the checks pass, the miner replays on the same class and window and compares all roots. A
mismatch returns `replay_failed` and signs nothing.

## Pool error policy

| Error | Meaning | Strike? |
|---|---|---|
| `no_replayer` | no live rig advertises the class | no |
| `class_mismatch` | rig asked for a class it does not implement | no |
| `window_too_small` | rig cannot load the record's window | no |
| `vector_failed` | rig's vectors do not pass | no |
| `replay_failed` | replay ran but roots differ | investigate |
| `record_invalid` | record failed structural validation | investigate producer |

The pool MUST NOT silently fall back to a different class. If no rig of the class is live, it
returns `no_replayer` with the class, tier and deadline.

## Why this is safe

- The class is proved by vectors before it is advertised.
- The replayer refuses before spending the audit window on a class it cannot reproduce.
- The replayer verifies the replay before signing.
- The pool routes by class, not by card name or `engine_abi` alone.
