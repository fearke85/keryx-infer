<!--
Hardware PR template: new architecture row, vendor/unit-kind code, or backend.
Guide: docs/adding-hardware.md. Read it before filling this in.
Title: `hw(<backend>): <change>`, e.g. `hw(cuda): add cc9.0 (Hopper, 132 SM)`.
Delete the sections that do not apply; do not leave unchecked boxes unexplained.
-->

## Change kind

<!-- Exactly one. Section numbers refer to docs/adding-hardware.md. -->
- [ ] New architecture row on an existing backend (§3)
- [ ] New vendor / unit kind code (§4)
- [ ] Activate a declared backend: `planned` → `pending` (§5)
- [ ] Promote a backend: `pending` → `validated` (§5, needs the validation gate)
- [ ] New backend code (§6)

## Rows

<!-- One line per added row. Planned units = largest shipping SKU of the architecture (§3.1). -->

| Backend | Vendor | Arch | Unit kind | Planned | Reference SKU | compat_id |
|---|---|---|---|---:|---|---|
| | | | | | | |

`patch_set` of the backend: `<64 hex>`

## Files

- [ ] `crates/keryx-infer-protocol/data/manifest-v2.json`
- [ ] Native plan table (CUDA: `native/engine/trace/keryx_plan.cu`) with the **same** value, or
      "native engine not in tree yet" stated below
- [ ] `vectors/compat-ids-v2.json`: one vector per row, values from
      `keryx-infer-conformance class-id`
- [ ] `docs/class-table.md` and `SPEC.md` §3.3
- [ ] Enum codes in `class.rs`, `native/include/keryx_infer.h` and `keryx-infer-sys` (§4/§6 only)
- [ ] `CHANGELOG.md`

## Class safety

<!-- Every box must be checked. If one cannot be, the PR hits a stop condition (§7): say why. -->
- [ ] No existing plan row, `arithmetic` value, global manifest field or enum code changed
- [ ] `git diff vectors/compat-ids-v2.json` only **adds** entries; no existing `patch_set` or
      `compat_id` moved
- [ ] No real unit count, card name or vendor-library path reaches the class or the arithmetic
- [ ] `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets` and
      `cargo test --workspace` pass
- [ ] `keryx-infer-conformance vectors` passes

## Hardware validation

<!--
Required to set or keep `status: validated` (docs/release.md, "Validation gate").
If nothing was run on hardware, write "not run" here; the backend stays `pending`
or the row is not advertised. Never claim a run that did not happen.
-->

| Check | Card / driver | Result |
|---|---|---|
| Trace vectors on a representative card | | |
| Second SKU or planned-unit override | | |
| Cross-driver run | | |
| 256-seed node oracle per tier | | |
| `vendor_products == 0` | | |

## Notes

<!-- Anything a reviewer must know: missing native support, open questions, follow-ups. -->
