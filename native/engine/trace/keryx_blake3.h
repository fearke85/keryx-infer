// BLAKE3 on the GPU for the H15 trace (ADR 0023): leaf hashes, streamed sequence leaves and B3
// roots, all byte-exact with blake3::hash and the node's `b3_root`.
//
// Every launcher only enqueues on `stream`: no host sync, no allocation, so a CUDA graph can
// capture it. Errors are argument checks and launch errors; device faults surface at the
// caller's next sync. Hashes and outputs are 32 bytes, 4-byte aligned.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace kb3 {

// Levels of the chaining-value stack: enough for 2^54 chunks, as in BLAKE3.
constexpr int STACK_LEVELS = 54;

// One streamed leaf, resident in device memory. Zeroed = empty (kb3::stream_reset).
// `stack[k]` is the subtree of 2^k chunks that ends at chunk `chunks_done`'s bit k; it is valid
// only while that bit is set. The last chunk stays in `pending` until finalize, even when full:
// BLAKE3 must not hash it before it knows whether it is the root.
struct alignas(16) stream_state {
    uint32_t stack[STACK_LEVELS][8];
    uint8_t  pending[1024];
    uint64_t total_len;  // bytes appended so far
    uint64_t reserved;
};

// Rows appended to one state: row j = `n_rows` rows of `row_len` bytes at base + j * stride.
// Each state appears at most once per call. Per call n_rows * row_len must stay below 4 GiB.
struct append_desc {
    const void* base;
    int64_t     stride;
    uint32_t    n_rows;
    uint32_t    row_len;
    uint32_t    state;  // index into the states array
    uint32_t    reserved;
};

// Scratch bytes hash_strided needs for messages of `len` bytes (0 when no scratch is needed).
size_t hash_strided_scratch(uint32_t count, uint64_t len);

// out + i * out_stride = blake3::hash(segment0(i) || segment1(i)) for i < count, with
// segment0(i) = base0 + i * stride0 (len0 bytes) and segment1(i) = base1 + i * stride1 (len1
// bytes, none when len1 == 0). `scratch` holds hash_strided_scratch(count, len0 + len1) bytes.
cudaError_t hash_strided(const void* base0, int64_t stride0, uint64_t len0,
                         const void* base1, int64_t stride1, uint64_t len1,
                         uint32_t count, void* out, int64_t out_stride,
                         void* scratch, size_t scratch_bytes, cudaStream_t stream);

// Empties n states (an async memset).
cudaError_t stream_reset(stream_state* states, uint32_t n, cudaStream_t stream);

// Appends rows to states: descs[d] (device memory, n_descs entries) names its state.
cudaError_t stream_append(stream_state* states, const append_desc* descs, uint32_t n_descs,
                          cudaStream_t stream);

// out + i * out_stride = blake3::hash of every byte appended to states[i], for i < n. The
// states are left as they were: more rows can follow and be finalized again.
cudaError_t stream_finalize(const stream_state* states, uint32_t n, void* out, int64_t out_stride,
                            cudaStream_t stream);

// out + g * out_stride = b3_root of the n hashes at base + g * stride (contiguous), g < groups:
// pairs merged, the odd node carried up, ROOT only on the last merge, one hash = itself.
cudaError_t b3_root(const void* base, int64_t stride, uint32_t n, uint32_t groups,
                    void* out, int64_t out_stride, cudaStream_t stream);

}  // namespace kb3
