# Third-party notices

## llama.cpp

The native engine builds against llama.cpp, pinned to tag `b10015`.

- Project: https://github.com/ggml-org/llama.cpp
- License: MIT

The engine source under `native/engine/` is derived from the Keryx reference inference engine and
keeps the upstream copyright and license notices. See the headers of the imported files.

## BLAKE3

The Rust crates use the `blake3` crate for the commitment tree and the class id.

- Project: https://github.com/BLAKE3-team/BLAKE3
- License: Apache-2.0 OR MIT (double licensed)

## borsh

The custody record uses `borsh` for the header and body encoding.

- Project: https://github.com/near/borsh-rs
- License: Apache-2.0

## Project license

`keryx-infer` is Apache-2.0 OR MIT. See `LICENSE-APACHE` and `LICENSE-MIT`.
