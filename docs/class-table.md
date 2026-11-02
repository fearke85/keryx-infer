# Compatibility class table

A class is `backend + architecture + planned units + manifest`. The card model and the real unit
count are not part of it.

The authoritative table is `crates/keryx-infer-protocol/data/manifest-v2.json`. This page mirrors
it for readers. A class uses its backend's `patch_set`, which covers the arithmetic but not the
plan table; changing a plan row changes the classes of that architecture only. Adding an
architecture, or adding or validating another backend, does not move existing classes.

To add a card, architecture, vendor or backend, follow [`adding-hardware.md`](adding-hardware.md).

## CUDA — validated

| Architecture | Planned SM | Group |
|---|---:|---|
| cc7.5 | 72 | Turing |
| cc8.0 | 108 | GA100 |
| cc8.6 | 84 | Ampere consumer |
| cc8.9 | 128 | Ada |
| cc12.0 | 170 | Blackwell |

Examples:

- RTX 3060 (28 SMs), 3070 (46), 3080 (68): the same class.
- RTX 4060 Ti (34), 4070 (46), 4070 SUPER (56): the same class.
- RTX 4090 (cc8.9) and 5090 (cc12.0): different classes.
- A 5070 Ti under MPS with half the SMs: the same class; the engine ignores the real count.

## Vulkan / AMD — pending vectors

| Architecture | Planned CU | Group |
|---|---:|---|
| gfx906 | 60 | Vega 20 |
| gfx1030 | 80 | RDNA2 Navi 21 |
| gfx1031 | 40 | RDNA2 Navi 22 |
| gfx1032 | 32 | RDNA2 Navi 23 |
| gfx1100 | 96 | RDNA3 Navi 31 |
| gfx1101 | 60 | RDNA3 Navi 32 |
| gfx1102 | 32 | RDNA3 Navi 33 |
| gfx1200 | 64 | RDNA4 Navi 48 |
| gfx1201 | 32 | RDNA4 Navi 44 |

These rows are `pending` until the Vulkan engine publishes trace vectors and passes them on a
representative card. A row is not advertised before that.

## SYCL / Intel — planned

No class until the Xe architecture table and vectors exist. The expected shape is one row per Xe
generation (`xe-hpg`, `xe2`, ...) with planned Xe cores or vector engines.

## Metal / Apple — planned

No class until the GPU-family table and vectors exist. Metal may need a `runtime_family` field if
OS/driver updates change bits.

## CPU — planned

Not intended for pool inference. If used, the class must include the ISA and the fixed thread
count.

## Validation matrix

| Check | What it proves |
|---|---|
| one card per architecture | the class exists and reproduces |
| second SKU or test-only unit override | the plan ignores the real count |
| one cross-driver run | the backend is not driver-sensitive |
| 256-seed node oracle per tier | openings are accepted |
| `vendor_products == 0` | no vendor library decided a sum |

There is no per-SKU validation row. A new SKU of a known architecture is the same class.
