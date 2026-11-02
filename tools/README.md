# Tools

Planned tools:

- `import-engine.sh` — refresh `native/engine/` from a reference checkout, refusing to copy any
  PoM file.
- `gen-vectors` — run the engine on a representative card and emit published vectors.
- `package.sh` — produce the per-backend library artifacts, checksums and SBOM.

The first tool is a shell script; vectors and packaging stay Rust/CI once the engine build exists.
