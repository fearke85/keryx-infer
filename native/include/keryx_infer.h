// Keryx inference engine C ABI.
//
// One card, one model, zero-dup only: the engine reads weights from the canonical walk blob
// through keryx_infer_load_bound.
//
// The context functions release only the KV cache and compute buffers; the weights stay in the
// blob. This lets a miner take the context down to a small mining context and restore it for
// inference without reloading the model.
#ifndef KERYX_INFER_H
#define KERYX_INFER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KERYX_INFER_C_ABI 2
#define KERYX_INFER_ENGINE_ABI 9
#define KERYX_INFER_ARCH_MAX 64

typedef struct keryx_infer_model keryx_infer_model;

enum keryx_infer_backend {
    KERYX_INFER_BACKEND_CUDA = 1,
    KERYX_INFER_BACKEND_VULKAN = 2,
    KERYX_INFER_BACKEND_SYCL = 3,
    KERYX_INFER_BACKEND_METAL = 4,
    KERYX_INFER_BACKEND_CPU = 5,
};

enum keryx_infer_vendor {
    KERYX_INFER_VENDOR_NVIDIA = 1,
    KERYX_INFER_VENDOR_AMD = 2,
    KERYX_INFER_VENDOR_INTEL = 3,
    KERYX_INFER_VENDOR_APPLE = 4,
    KERYX_INFER_VENDOR_OTHER = 255,
};

enum keryx_infer_unit_kind {
    KERYX_INFER_UNIT_SM = 1,
    KERYX_INFER_UNIT_CU = 2,
    KERYX_INFER_UNIT_XE_CORE = 3,
    KERYX_INFER_UNIT_GPU_CORE = 4,
    KERYX_INFER_UNIT_THREAD = 5,
};

// One card of a compatibility class: backend + architecture + planned units, never a model name.
typedef struct keryx_infer_card_class {
    uint16_t backend;
    uint16_t vendor;
    char arch[KERYX_INFER_ARCH_MAX];
    uint16_t planned_units;
    uint8_t unit_kind;
} keryx_infer_card_class;

typedef int (*keryx_infer_on_token)(void* user, int32_t token);
typedef int (*keryx_infer_on_rows)(void* user, int32_t t0, int32_t n_rows, int32_t n_ubatch,
                                   const uint8_t* hashes);

// C ABI and engine ABI. A miner must refuse a library whose values differ.
int keryx_infer_c_abi(void);
int keryx_infer_engine_abi(void);
const char* keryx_infer_version(void);

// Last failed call on this thread; NUL-terminated and valid until the next call.
const char* keryx_infer_last_error(void);

// Planned execution units for a backend/architecture. The card's real count is ignored.
int keryx_infer_plan_units(uint16_t backend, const char* arch, int real_units);

// Class identity of the device, when the backend can report it. 0 on success. The SDK refuses to
// load a model on a device that cannot report its class.
int keryx_infer_card_for_device(uint16_t backend, int device, keryx_infer_card_class* out);

// BLAKE3 of one backend's canonical arithmetic section (SPEC §4; the plan table is not hashed).
// Returns 0, or -1 for an unknown backend.
int keryx_infer_manifest_patch_set(uint16_t backend, uint8_t out[32]);

// Protocol-2 compat id over one card. Returns 0 on success.
int keryx_infer_compat_id(const keryx_infer_card_class* cards, size_t n_cards, uint8_t out[32]);

// Loads the model on one GPU, reading its weights from the canonical walk blob. Copies nothing.
// `held` is how many bytes of the blob the card holds; `cut` is the prefix every tensor but the
// output projection is read from; `layout_len` is the canonical blob length. 0 on failure.
keryx_infer_model* keryx_infer_load_bound(const char* gguf_path, int gpu, void* blob,
                                          size_t held, size_t cut, size_t layout_len, int n_ctx);

// Re-creates the llama context at `n_ctx` tokens. Weights stay in the blob. Returns the context
// size, or -1. Used for the mining-pad dance.
int keryx_infer_context_acquire(keryx_infer_model* model, int n_ctx);
void keryx_infer_context_release(keryx_infer_model* model);

// Arms the trace for a tier. `manifest` is the canonical tier manifest; the counts are the
// tier context the protocol crate supplies. Returns 0, or -1 with last_error.
int keryx_infer_trace_init(keryx_infer_model* model, const uint8_t* manifest, size_t len,
                           uint32_t n_layers, uint32_t n_used, uint32_t n_attn,
                           uint32_t n_recur, uint32_t seq_heads);
uint32_t keryx_infer_trace_leaves(keryx_infer_model* model);
uint32_t keryx_infer_trace_seq_leaves(keryx_infer_model* model);

// Traced generation. `prompt` is the full token stream (prompt + request prefix), already built
// by the protocol crate. The engine feeds 512-token prompt batches and one token per generated
// position. `seq_out` receives every sequence-leaf hash (seq_leaves * 32 bytes).
int keryx_infer_generate_traced(keryx_infer_model* model, const int32_t* prompt, int n_prompt,
                                int max_tokens, keryx_infer_on_token on_token,
                                keryx_infer_on_rows on_rows, void* user, uint8_t* seq_out,
                                size_t seq_cap);

// Replays a kept run exactly as it was generated. `pos_wants` holds triples
// (position, leaf, offset); `seq_wants` holds pairs (leaf, offset). `on_rows` receives every
// batch's leaf table so the caller checks the kept roots before signing.
int keryx_infer_replay(keryx_infer_model* model, const int32_t* tokens, int n_pos, int n_prompt,
                       const int64_t* pos_wants, int n_pos_wants, uint8_t* pos_out,
                       size_t pos_bytes, const int64_t* seq_wants, int n_seq_wants,
                       uint8_t* seq_out, size_t seq_bytes, keryx_infer_on_rows on_rows,
                       void* user, uint8_t* seq_hashes, size_t seq_hashes_cap);

// Tokenizer. `add_special` adds BOS; `parse_special` parses control tokens. Returns the number of
// ids written, or a negative required capacity.
int keryx_infer_tokenize(keryx_infer_model* model, const char* text, int len, int add_special,
                         int parse_special, int32_t* out, int cap);
int keryx_infer_token_piece(keryx_infer_model* model, int32_t token, char* out, int cap);
int keryx_infer_is_eog(keryx_infer_model* model, int32_t token);

// Canonical tensor table for the PoM walk over the engine's weights. Not used by load_bound
// inference itself; it lets a miner keep the walk on the same bytes.
size_t keryx_infer_tensor_count(keryx_infer_model* model);
int keryx_infer_tensor_info(keryx_infer_model* model, size_t index, const char** name,
                            void** data, size_t* nbytes, int* is_device);

// Products that left the fixed-order path for a vendor library. ABI 9 requires 0.
uint64_t keryx_infer_vendor_products(void);

void keryx_infer_free(keryx_infer_model* model);

#ifdef __cplusplus
}
#endif

#endif
