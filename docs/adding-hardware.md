# Adding hardware: card, architecture, vendor, backend

This guide is written for an automated agent (or a human) asked to "add support for card X". Read
it fully before editing anything. It tells you **whether** a change is needed, **which files**
change, **what must never change**, and **how to prove** the change is correct.

## 0. The one rule

The engine exists so that every card of one class produces **bit-identical traces**. A class is:

```
backend + vendor + architecture + planned units + unit kind + backend patch_set
```

The card model, its marketing name and its **real** SM/CU count are never part of a class and must
never reach the arithmetic. The engine splits work by the architecture's *planned* units; a card
with fewer real units runs the same blocks in more waves and gets the same bits.

If a change you are about to make lets the real unit count, a driver, or a vendor library decide a
sum, stop: it breaks the project's purpose.

## 1. Decide what kind of change it is

Answer in order. Stop at the first match.

| Question | If yes |
|---|---|
| Is the card a new SKU of an architecture already in the manifest plan (e.g. an RTX 3050 is `cc8.6`, an RX 7600 is `gfx1102`)? | **Do nothing.** It is already the same class. See §2. |
| Is it a new architecture on a backend that already has a plan table (`cuda`, `vulkan`)? | **New architecture row.** See §3. |
| Is it a backend already declared in the manifest as `planned` (`sycl`, `metal`, `cpu`) that now gets an engine? | **Activate a declared backend.** See §5. |
| Is it a vendor or unit kind that has no code in `Vendor` / `UnitKind`? | **New vendor / unit kind code.** See §4. |
| Is it a second vendor on an existing backend (e.g. NVIDIA or Intel cards over Vulkan)? | **New architecture row with a card-kind override.** See §3.3. |
| Is it a backend with no code in `Backend` at all (e.g. ROCm/HIP)? | **New backend code**, then §5. See §6. |
| Does it require changing an existing plan row, an arithmetic key, or any hash/encoding? | **Stop.** That is a class-breaking change. See §7. |

How to find a card's architecture id:

| Backend | Architecture id format | Source of truth |
|---|---|---|
| cuda | `cc<major>.<minor>` (e.g. `cc8.9`, `cc12.0`) | CUDA compute capability (`cudaDeviceProp.major/minor`) |
| vulkan (AMD) | `gfx<id>` (e.g. `gfx1100`) | the LLVM AMDGPU target id of the GPU |
| sycl (Intel) | `xe-hpg`, `xe2`, ... | the Xe generation |
| metal | `apple-g13`, ... | the Apple GPU family |
| cpu | `x86_64-avx2`, ... | ISA + vector extension |

Architecture ids must be printable ASCII with no spaces and at most 63 bytes (the C struct holds
64 bytes including the NUL; see `KERYX_INFER_ARCH_MAX`). `CardClass::validate` and
`Manifest::validate` enforce this.

## 2. New SKU of a known architecture: nothing to do

Examples already true today: RTX 3060 (28 SMs), 3070 (46), 3080 (68) are all `cuda/cc8.6/84`. A
GPU split by MPS or a cut-down die is still the same class.

Do **not**:

- add a row per SKU;
- add the real unit count anywhere in the manifest, class or record;
- add the card name to the class (a name may appear only in comments and docs).

If the user insists the SKU behaves differently, the correct response is to run the trace vectors
on it (§8). A mismatch there is an engine bug, not a reason for a new class.

## 3. New architecture on an existing backend

Example: adding Hopper `cc9.0` to CUDA, or `gfx1150` to Vulkan.

This change is **additive**: it must not change any existing `patch_set` or `compat_id`. The plan
table is deliberately not part of `backend_patch_set`; the card carries its own planned units.

### 3.1 Choose the planned unit count

Use the unit count of the **largest shipping SKU** of that architecture, and name it in a comment.
Every existing row follows this convention:

| Row | Planned | Reference SKU |
|---|---:|---|
| cuda cc7.5 | 72 | TITAN RTX |
| cuda cc8.0 | 108 | A100 |
| cuda cc8.6 | 84 | RTX 3090 Ti |
| cuda cc8.9 | 128 | RTX 4090 |
| cuda cc12.0 | 170 | RTX 5090 |
| vulkan gfx1100 | 96 | RX 7900 XTX |

The value must be non-zero and fit `u16`. Once published it is frozen for the engine ABI (§7).

### 3.2 Files to change

1. `crates/keryx-infer-protocol/data/manifest-v2.json` — add `"<arch>": <units>` to the backend's
   `plan` object. Keep the existing key order style (the file is parsed into a `BTreeMap`, so
   order does not affect hashing, but keep it readable). Do not touch `arithmetic`, the backend's
   default `vendor`/`unit_kind` or `status`.
2. Native plan table — the engine must plan the same number:
   - CUDA: `native/engine/trace/keryx_plan.cu`, function `keryx_plan_sms`. Add
     `case <major*10+minor>: return <units>;   // <reference SKU>`. Example: `cc9.0` →
     `case 90: return 132;`.
   - Vulkan: the Vulkan engine is not in this tree yet. Record the row in the manifest only and
     leave the backend `pending`.
   The manifest and the native table must hold identical values. Nothing checks this
   automatically until the native ABI is wired (`native/engine/README.md`); check it by eye.
3. `vectors/compat-ids-v2.json` — add one vector for the new row. Name it
   `<backend>-<group>-single`. Fill `patch_set` and `compat_id` from the CLI (§8), never by hand.
4. Docs that mirror the table (keep all three in sync with the manifest):
   - `docs/class-table.md` — add the row to the backend's table.
   - `SPEC.md` §3.3 — add the row to the planned unit table.
   - `CHANGELOG.md` — one line under the current version.

### 3.3 A row of another vendor on the same backend

A backend section has a default `vendor` and `unit_kind`. A row of another vendor uses the object
form and overrides both:

```json
"plan": {
  "gfx1100": 96,
  "nv-cc8.6": { "units": 84, "vendor": "nvidia", "unit_kind": "sm" }
}
```

Give the row an architecture id that cannot collide with the backend's other rows (prefix it with
the vendor). Everything else in §3 applies. Cross-vendor rows on one backend are separate classes;
never assume they share bits with the default vendor's rows.

### 3.4 What must not change

Run `git diff vectors/compat-ids-v2.json`. The only change must be the **added** entry. If any
existing `patch_set` or `compat_id` moved, you changed something hashed: revert and find it.

## 4. New vendor or unit kind code

Only needed when the enum has no suitable value. `Other = 255` exists but should not be used for a
vendor that will have its own rows.

Codes are part of `compat_id`. **Never renumber or reuse a code.** Append the next free value.

Change all of these together:

| What | Rust (`keryx-infer-protocol`) | C header | Raw Rust ABI |
|---|---|---|---|
| Vendor | `class.rs`: `Vendor` variant, `name`, `from_code`, `parse` | `native/include/keryx_infer.h`: `enum keryx_infer_vendor` | `crates/keryx-infer-sys/src/lib.rs`: `VENDOR_*` |
| Unit kind | `class.rs`: `UnitKind` variant, `name`, `from_code`, `parse` | `enum keryx_infer_unit_kind` | `UNIT_*` |

Then update `SPEC.md` §3.1 (the comment listing the codes). The string names used in the manifest
and vectors (`"nvidia"`, `"xe_core"`, ...) are what `parse` accepts; keep them lowercase
snake_case.

Adding a code alone does not change any existing class. It does nothing useful until a backend
section uses it (§5).

## 5. Activating a declared backend (`planned` → `pending` → `validated`)

`sycl`, `metal` and `cpu` already have backend codes and empty manifest sections. To give one a
plan:

1. In its manifest section, set `vendor`, `unit_kind`, the `plan` rows (§3.1) and the
   `arithmetic` keys that describe its fixed-order path. `Manifest::validate` refuses a row that
   resolves to no valid vendor and unit kind.
2. Set `status` to `pending`. A pending backend can compute class ids for vectors
   (`CompatClass::from_card(.., false)`) but cannot be advertised (`CompatClass::single` refuses
   it).
3. Implement the native side: the engine for that backend, its unit plan (the equivalent of
   `keryx_plan_sms`, which must never fall back to the real count), and
   `keryx_infer_card_for_device` for it. `Engine::load_bound` refuses to load on a device that
   cannot report its class.
4. Add vectors and docs as in §3.2.
5. Only after the validation gate (§8) passes, change `status` to `validated`.

`status` is not hashed, so promoting `pending` → `validated` does not move any `patch_set` or
`compat_id`. Filling `arithmetic` for the first time does define the backend's `patch_set`; that is
expected because the backend had no classes before.

## 6. A backend with no code yet

Append a variant to `Backend` in `class.rs` (next free code, never reuse one), the matching
`KERYX_INFER_BACKEND_*` in the C header and `BACKEND_*` in `keryx-infer-sys`, then add a manifest
section with `status: "planned"` and follow §5. Update `SPEC.md` §3.1.

## 7. Stop conditions: escalate to a human

Do not perform these on your own. Explain the situation and ask.

- **Changing an existing plan row** (e.g. "cc8.6 should plan 82"). It changes the `compat_id` of
  every card of that architecture and makes published records unreplayable. It requires a new
  engine ABI or manifest version (`docs/release.md`, "Class freeze").
- **Changing an `arithmetic` value, `trace_layout`, `prompt_template`, `tier_leaf_counts`,
  `protocol` or `engine_abi`.** These are hashed into `patch_set` and move every class of the
  backend (or of every backend, for the global fields).
- **Changing any hashing or encoding**: `backend_patch_set_for`, `CompatClass::id`, the domain
  separators, or the enum codes.
- **Changing how the engine maps a row to a work split without changing a number.** That needs
  a new value of the `plan` arithmetic key (it moves the backend's `patch_set`).
- **Changing a backend's default `vendor`/`unit_kind`.** It changes every row that relies on the
  default.
- **A request to plan by the real unit count**, or to let a vendor library (cuBLAS, rocBLAS,
  oneMKL, MPS) decide a sum. `vendor_products` must stay 0.

## 8. Verify

Run all of these. Every one must pass.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test --workspace

# Recompute the class of the new row; paste patch_set and compat_id into the vector file.
cargo run -q -p keryx-infer-conformance -- class-id \
  --backend cuda --arch cc9.0 --vendor nvidia --units 132

# Recheck every published vector against the manifest.
cargo run -q -p keryx-infer-conformance -- vectors
```

The unit tests already prove that adding a backend or a plan row does not move an existing
backend's `patch_set` (`a_backend_hash_ignores_other_backends`, `a_backend_hash_ignores_plan_rows`
in `manifest.rs`).

Class-id vectors only prove the id algorithm. Before a row may be advertised (`validated`
backend), a human must run the **validation gate** from `docs/release.md` on real hardware:

1. trace vectors on a representative card of the architecture;
2. the same vectors on a second SKU, or with a test-only planned-unit override, to prove the real
   count is ignored;
3. one cross-driver run;
4. the 256-seed node oracle per tier;
5. `vendor_products == 0`.

An agent cannot run these without the hardware. Say so in the change description instead of
claiming the row is validated.

## 9. Pull request

Every hardware change goes in its own PR using the template in
`.github/PULL_REQUEST_TEMPLATE/hardware.md`:

```bash
gh pr create --template hardware.md --title "hw(cuda): add cc9.0 (Hopper, 132 SM)"
```

On the GitHub web form, append `?template=hardware.md` to the compare URL.

Rules:

- **Title:** `hw(<backend>): <change>`. Examples: `hw(cuda): add cc9.0 (Hopper, 132 SM)`,
  `hw(vulkan): add gfx1150`, `hw(sycl): activate backend as pending`,
  `hw(vulkan): promote to validated`, `hw(protocol): add vendor code qualcomm`.
- **One concern per PR.** A new row, a new enum code, activating a backend and promoting it to
  `validated` are separate PRs. Promotion is always its own PR, because it is the only one that
  needs hardware evidence.
- **Several rows of one backend** may share a PR when they are all additive.
- **Fill every section that applies.** The class-safety boxes must all be checked; if one cannot
  be, the PR hits a stop condition (§7) and must say so instead of being merged.
- **Hardware results are evidence, not claims.** Write the card, driver and outcome of each run,
  or `not run`. An agent without the hardware writes `not run` and leaves the backend `pending`.
- **Commits:** one commit per logical step is fine (`manifest`, `native plan`, `vectors`,
  `docs`), but the PR must pass the checks of §8 as a whole.

## 10. File map

| File | Role |
|---|---|
| `crates/keryx-infer-protocol/data/manifest-v2.json` | Authoritative plan table, card kind and arithmetic per backend |
| `crates/keryx-infer-protocol/src/manifest.rs` | Manifest parsing, validation, `backend_patch_set` |
| `crates/keryx-infer-protocol/src/class.rs` | `Backend`/`Vendor`/`UnitKind` codes, `CardClass`, `CompatClass`, `compat_id` |
| `crates/keryx-infer-sys/src/lib.rs` | Raw C ABI constants mirroring the header |
| `native/include/keryx_infer.h` | Stable C ABI, enum codes, `card_for_device` |
| `native/engine/trace/keryx_plan.cu` | CUDA planned-SM table used by the engine |
| `native/engine/llama_patches.rs` | Patch that routes llama.cpp's SM count through `keryx_plan_sms` |
| `crates/keryx-infer/src/engine.rs` | Load-time checks: class, patch set, device-reported card |
| `vectors/compat-ids-v2.json` | Published class-id vectors, one per row |
| `docs/class-table.md`, `SPEC.md` §3.3 | Human-readable mirrors of the plan table |
| `docs/release.md` | Class freeze rules and validation gate |
| `.github/PULL_REQUEST_TEMPLATE/hardware.md` | PR template for every hardware change |

## 11. Known gap

`keryx_plan_sms` in `native/engine/trace/keryx_plan.cu` ends with `default: return sms;`: an
architecture missing from the table plans by its **real** SM count, so two cards of that
architecture with different SM counts can diverge. The Rust SDK refuses such a card (no plan row →
no class, and `load_bound` requires a device-reported class), but a caller of the native engine
that bypasses the SDK is not protected. When you add a CUDA row, add it to both the manifest and
this switch. Do not rely on the fallback.
