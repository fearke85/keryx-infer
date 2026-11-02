// The trace's capture nodes, run by ggml-cuda (ADR 0023). build.rs copies this file into
// ggml/src/ggml-cuda/ and patches ggml-cuda.cu to send GGML_OP_CUSTOM here.
//
// Every op only enqueues on the backend's stream and takes its scratch from the backend's
// pool, as llama's own ops do, so CUDA graphs capture it like any other node.
#include "common.cuh"
#include "quantize.cuh"

#include "keryx_blake3.cuh"
#include "keryx_blake3.h"
#include "keryx_trace_abi.h"
#include "keryx_trace_ops.cuh"

#include <algorithm>
#include <cstring>
#include <vector>

extern "C" void keryx_trace_cpu_fun(ggml_tensor *, int, int, void *) {
    GGML_ABORT("keryx trace: a capture node ran on the CPU");
}

static const keryx_trace_op * keryx_trace_op_of(const ggml_tensor * t) {
    if (t->op != GGML_OP_CUSTOM) {
        return nullptr;
    }
    ggml_custom_op_params p;
    memcpy(&p, t->op_params, sizeof(p));
    if (p.fun != keryx_trace_cpu_fun || !p.userdata) {
        return nullptr;
    }
    const auto * op = static_cast<const keryx_trace_op *>(p.userdata);
    return op->magic == KERYX_TRACE_MAGIC ? op : nullptr;
}

// Rows of `t` as one run of equal strides: n rows of ne0 elements, `stride` bytes apart.
static bool keryx_flat_rows(const ggml_tensor * t, int64_t & n, int64_t & stride) {
    if (t->nb[0] != ggml_type_size(t->type)) {
        return false;
    }
    n = t->ne[1] * t->ne[2] * t->ne[3];
    stride = t->nb[1];
    return (t->ne[2] == 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
           (t->ne[3] == 1 || t->nb[3] == t->nb[2] * t->ne[2]);
}

bool keryx_trace_supports(const ggml_tensor * t) {
    const keryx_trace_op * op = keryx_trace_op_of(t);
    if (!op || !t->src[0]) {
        return false;
    }
    switch (op->kind) {
        case KERYX_TRACE_F32:
            return t->src[0]->type == GGML_TYPE_F32 && (!t->src[1] || t->src[1]->type == GGML_TYPE_I32);
        case KERYX_TRACE_X:
            return t->src[0]->type == GGML_TYPE_F32;
        case KERYX_TRACE_KV:
        case KERYX_TRACE_LSE:
        case KERYX_TRACE_RECUR:
        case KERYX_TRACE_FLUSH:
            return true;  // shapes and types are checked when it runs
        default:
            return false;
    }
}

static uint8_t * keryx_table_of(const keryx_trace_op * op, int device) {
    for (const keryx_trace_table & tb : op->state->tables) {
        if (tb.device == device) {
            return tb.hashes;
        }
    }
    return nullptr;
}

// Row-major copy of an MMQ buffer: llama keeps block b of row r at (b * n_rows + r). One thread
// per 16-byte word of a block.
static constexpr int KERYX_MMQ_WORDS = sizeof(block_q8_1_mmq) / sizeof(int4);
static_assert(sizeof(block_q8_1_mmq) % sizeof(int4) == 0, "an MMQ block is whole 16-byte words");

static __global__ void keryx_mmq_rows(const int4 * src, int4 * dst, int64_t n_rows, int64_t n_blocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows * n_blocks * KERYX_MMQ_WORDS) {
        return;
    }
    const int64_t w = i % KERYX_MMQ_WORDS;
    const int64_t rb = i / KERYX_MMQ_WORDS;
    const int64_t r = rb / n_blocks;
    const int64_t b = rb % n_blocks;
    dst[(r * n_blocks + b) * KERYX_MMQ_WORDS + w] = src[(b * n_rows + r) * KERYX_MMQ_WORDS + w];
}

static keryx_trace_seq * keryx_seq_of(const keryx_trace_op * op, int device);

// Replay: message j goes to arena + off[j * off_stride] when that offset is set.
static __global__ void keryx_dump_messages(const char * b0, int64_t s0, uint64_t l0, const char * b1, int64_t s1,
                                           uint64_t l1, const int64_t * off, int64_t off_stride, uint8_t * const * arena) {
    const int64_t o = off[(int64_t) blockIdx.x * off_stride];
    if (o < 0) {
        return;
    }
    uint8_t * dst = *arena + o;
    const char * a = b0 + (int64_t) blockIdx.x * s0;
    for (uint64_t i = threadIdx.x; i < l0; i += blockDim.x) dst[i] = a[i];
    const char * b = b1 + (int64_t) blockIdx.x * s1;
    for (uint64_t i = threadIdx.x; i < l1; i += blockDim.x) dst[l0 + i] = b[i];
}

// n bytes from a to d by the blocks of gridDim.x, in the widest word both ends and n allow.
static __device__ void keryx_copy_span(uint8_t * d, const char * a, uint64_t n) {
    const uint64_t t = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x, step = (uint64_t) gridDim.x * blockDim.x;
    const uint64_t mis = (uint64_t) (uintptr_t) d | (uint64_t) (uintptr_t) a | n;
    if ((mis & 15) == 0) {
        for (uint64_t i = t; i < n / 16; i += step) ((uint4 *) d)[i] = ((const uint4 *) a)[i];
    } else if ((mis & 3) == 0) {
        for (uint64_t i = t; i < n / 4; i += step) ((uint32_t *) d)[i] = ((const uint32_t *) a)[i];
    } else {
        for (uint64_t i = t; i < n; i += step) d[i] = a[i];
    }
}

// Decode staging: message blockIdx.y (both segments) to dst + j * (l0 + l1).
static __global__ void keryx_stage_messages(const char * b0, int64_t s0, uint64_t l0, const char * b1, int64_t s1,
                                            uint64_t l1, uint8_t * dst) {
    const int64_t j = blockIdx.y;
    uint8_t * d = dst + j * (int64_t) (l0 + l1);
    keryx_copy_span(d, b0 + j * s0, l0);
    if (l1) {
        keryx_copy_span(d + l0, b1 + j * s1, l1);
    }
}

static bool keryx_hash_messages(ggml_backend_cuda_context & ctx, const keryx_trace_op * op,
                                const void * base0, int64_t stride0, uint64_t len0,
                                const void * base1, int64_t stride1, uint64_t len1, int64_t count) {
    uint8_t * table = keryx_table_of(op, ctx.device);
    if (!table || op->row0 + count > op->state->row_cap) {
        return false;
    }
    if (count == 0) {
        return true;
    }
    if (op->defer_off >= 0) {
        // Decode: stage message j at defer_off + j * len; the run's flush hashes it.
        keryx_trace_seq * sq = keryx_seq_of(op, ctx.device);
        if (!sq || !sq->defer || count > 65535) {
            return false;
        }
        if (base0 == sq->defer + op->defer_off && stride0 == (int64_t) len0 && len1 == 0) {
            return true;  // the capture wrote its messages in place
        }
        const unsigned blocks = (unsigned) std::min<uint64_t>((len0 + len1 + 4095) / 4096, 128);
        keryx_stage_messages<<<dim3(blocks, (unsigned) count), 256, 0, ctx.stream()>>>(
            (const char *) base0, stride0, len0, (const char *) base1, stride1, len1, sq->defer + op->defer_off);
        return true;
    }
    const int64_t row_bytes = (int64_t) op->state->n_leaves * KERYX_TRACE_HASH;
    uint8_t * out = table + op->row0 * row_bytes + (int64_t) op->leaf * KERYX_TRACE_HASH;
    if (keryx_trace_seq * sq = keryx_seq_of(op, ctx.device); sq && sq->pos_off) {
        keryx_dump_messages<<<(unsigned) count, 256, 0, ctx.stream()>>>(
            (const char *) base0, stride0, len0, (const char *) base1, stride1, len1,
            sq->pos_off + (int64_t) op->row0 * op->state->n_leaves + op->leaf, op->state->n_leaves, sq->arena);
    }
    const size_t need = kb3::hash_strided_scratch((uint32_t) count, len0 + len1);
    ggml_cuda_pool_alloc<char> scratch(ctx.pool(), need > 0 ? need : 1);
    return kb3::hash_strided(base0, stride0, len0, base1, stride1, len1, (uint32_t) count, out, row_bytes,
                             scratch.get(), need, ctx.stream()) == cudaSuccess;
}

// f32 copy of a [k, rows, heads] source with every head of a row together: out[r][h][i].
static __global__ void keryx_heads_rows(const char * src, int64_t nb0, int64_t nb1, int64_t nb2, int64_t k,
                                        int64_t rows, int64_t heads, float * out) {
    const int64_t n = k * rows * heads;
    for (int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; j < n; j += (int64_t) gridDim.x * blockDim.x) {
        const int64_t i = j % k, h = (j / k) % heads, r = j / (k * heads);
        out[j] = *(const float *) (src + r * nb1 + h * nb2 + i * nb0);
    }
}

static float * keryx_gather_heads(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_cuda_pool_alloc<float> & buf) {
    const int64_t n = src->ne[0] * src->ne[1] * src->ne[2];
    buf.alloc(n);
    keryx_heads_rows<<<(unsigned) std::min<int64_t>((n + 255) / 256, 4096), 256, 0, ctx.stream()>>>(
        (const char *) src->data, src->nb[0], src->nb[1], src->nb[2], src->ne[0], src->ne[1], src->ne[2], buf.get());
    return buf.get();
}

static bool keryx_trace_f32(ggml_backend_cuda_context & ctx, const keryx_trace_op * op, const ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * ids = dst->src[1];
    if (op->heads_dim == 2) {
        if (ids || src->ne[3] != 1) {
            return false;
        }
        ggml_cuda_pool_alloc<float> buf(ctx.pool());
        const float * rows = keryx_gather_heads(ctx, src, buf);
        const int64_t len = src->ne[0] * src->ne[2] * 4;
        return keryx_hash_messages(ctx, op, rows, len, (uint64_t) len, nullptr, 0, 0, src->ne[1]);
    }
    if (!ids) {
        int64_t n, stride;
        if (!keryx_flat_rows(src, n, stride)) {
            return false;
        }
        return keryx_hash_messages(ctx, op, src->data, stride, (uint64_t) src->ne[0] * 4, nullptr, 0, 0, n);
    }
    // Experts: per token the n_used slot rows ([m, n_used, n_tok], slots contiguous), then its ids.
    if (src->nb[0] != 4 || src->nb[1] != src->ne[0] * 4 || src->ne[3] != 1 || ids->nb[0] != 4 ||
        ids->ne[0] != src->ne[1] || ids->ne[1] != src->ne[2]) {
        return false;
    }
    return keryx_hash_messages(ctx, op, src->data, src->nb[2], (uint64_t) src->ne[0] * src->ne[1] * 4,
                               ids->data, ids->nb[1], (uint64_t) ids->ne[0] * 4, src->ne[2]);
}

// Decode: the Q8_1 rows X captures staged during the running traced ubatch, by the f32 tensor they
// quantise. A product reading that tensor uses them instead of quantising it again: the same
// quantiser over the same rows writes the same bytes, so what it reads is what the trace commits.
// Host side, on the thread computing the graph; forgotten by each flush, on a failed capture and
// by the engine after every traced ubatch.
struct keryx_q8_1_rows {
    int device;
    const ggml_tensor * src;
    const void * data;
    int64_t k, rows;
    const void * q;
};
static thread_local std::vector<keryx_q8_1_rows> keryx_q8_1_lent;

static void keryx_forget_q8_1_on(int device) {
    auto & v = keryx_q8_1_lent;
    v.erase(std::remove_if(v.begin(), v.end(), [&](const keryx_q8_1_rows & r) { return r.device == device; }), v.end());
}

extern "C" void keryx_trace_forget_q8_1(void) {
    keryx_q8_1_lent.clear();
}

const void * keryx_trace_q8_1(int device, const ggml_tensor * src1, int64_t k, int64_t rows) {
    for (const keryx_q8_1_rows & r : keryx_q8_1_lent) {
        if (r.device == device && r.src == src1 && r.data == src1->data && r.k == k && r.rows == rows) {
            return r.q;
        }
    }
    return nullptr;
}

// Decode staging: where the op's messages go in the defer arena, for a capture that can write them
// there itself; null outside staging.
static char * keryx_staged_at(ggml_backend_cuda_context & ctx, const keryx_trace_op * op) {
    if (op->defer_off < 0) {
        return nullptr;
    }
    keryx_trace_seq * sq = keryx_seq_of(op, ctx.device);
    return sq && sq->defer ? (char *) sq->defer + op->defer_off : nullptr;
}

static bool keryx_trace_x(ggml_backend_cuda_context & ctx, const keryx_trace_op * op, const ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    int64_t n_rows, stride;
    // A per-head source is copied to rows in leaf order first: each row quantises on its own, so
    // its bytes are those llama's strided quantiser writes.
    ggml_cuda_pool_alloc<float> heads(ctx.pool());
    const float * data = (const float *) src->data;
    if (op->heads_dim == 2) {
        if (src->ne[3] != 1 || (int64_t) op->per_leaf != src->ne[2]) {
            return false;
        }
        data = keryx_gather_heads(ctx, src, heads);
        n_rows = src->ne[1] * src->ne[2];
        stride = src->ne[0] * 4;
    } else if (!keryx_flat_rows(src, n_rows, stride)) {
        return false;
    }
    if (op->per_leaf == 0 || n_rows % op->per_leaf != 0) {
        return false;
    }
    if (n_rows == 0) {
        return true;
    }
    if (op->alias) {
        return op->defer_off >= 0 && op->layout == KERYX_TRACE_MMVQ;  // staged by the first capture of src
    }
    const ggml_type wtype = (ggml_type) op->wtype;
    const int64_t k = src->ne[0];
    const int64_t kp = GGML_PAD(k, MATRIX_ROW_PADDING);
    const int64_t s01 = stride / 4;
    cudaStream_t stream = ctx.stream();
    if (op->layout == KERYX_TRACE_MMVQ) {
        // llama's MMVQ input: block_q8_1 rows one after the other.
        const int64_t row = kp / QK8_1 * (int64_t) sizeof(block_q8_1);
        ggml_cuda_pool_alloc<char> q(ctx.pool());
        char * out = keryx_staged_at(ctx, op);
        if (!out) {
            out = q.alloc(n_rows * row);
        }
        quantize_row_q8_1_cuda(data, nullptr, out, wtype, k, s01, s01 * n_rows,
                               s01 * n_rows, kp, n_rows, 1, 1, stream);
        if (op->defer_off >= 0 && op->heads_dim != 2) {
            keryx_q8_1_lent.push_back({ctx.device, src, src->data, k, n_rows, out});
        }
        const int64_t len = row * op->per_leaf;
        return keryx_hash_messages(ctx, op, out, len, len, nullptr, 0, 0, n_rows / op->per_leaf);
    }
    // llama's MMQ input, in the layout its quantiser picks for this weight type; that must be the
    // one the node expects for it.
    const bool ds4 = mmq_get_q8_1_ds_layout(wtype) == MMQ_Q8_1_DS_LAYOUT_DS4;
    const bool d4 = mmq_get_q8_1_ds_layout(wtype) == MMQ_Q8_1_DS_LAYOUT_D4;
    if (!(op->layout == KERYX_TRACE_MMQ_DS4 ? ds4 : op->layout == KERYX_TRACE_MMQ_D4 && d4)) {
        return false;
    }
    const int64_t n_blocks = kp / (4 * QK8_1);
    const int64_t bytes = n_rows * n_blocks * (int64_t) sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> q(ctx.pool(), bytes);
    ggml_cuda_pool_alloc<char> rows(ctx.pool());
    char * out = keryx_staged_at(ctx, op);
    if (!out) {
        out = rows.alloc(bytes);
    }
    quantize_mmq_q8_1_cuda(data, nullptr, q.get(), wtype, k, s01, s01 * n_rows, s01 * n_rows,
                           kp, n_rows, 1, 1, stream);
    const int64_t n = n_rows * n_blocks * KERYX_MMQ_WORDS;
    keryx_mmq_rows<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>((const int4 *) q.get(), (int4 *) out,
                                                                     n_rows, n_blocks);
    const int64_t len = n_blocks * (int64_t) sizeof(block_q8_1_mmq) * op->per_leaf;
    return keryx_hash_messages(ctx, op, out, len, len, nullptr, 0, 0, n_rows / op->per_leaf);
}

static keryx_trace_seq * keryx_seq_of(const keryx_trace_op * op, int device) {
    for (keryx_trace_seq & sq : op->state->seq) {
        if (sq.device == device && sq.streams) {
            return &sq;
        }
    }
    return nullptr;
}

// Replay: the rows a descriptor appends go to seq_arena at the leaf's offset plus what the leaf
// already holds. Runs before the append moves total_len.
static __global__ void keryx_dump_append(const kb3::append_desc * descs, const kb3::stream_state * states,
                                         const int64_t * seq_off, uint8_t * const * arena) {
    const kb3::append_desc d = descs[blockIdx.x];
    const int64_t o = seq_off[d.state];
    if (o < 0) {
        return;
    }
    uint8_t * dst = *arena + o + states[d.state].total_len;
    const uint64_t n = (uint64_t) d.n_rows * d.row_len;
    for (uint64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const uint64_t r = i / d.row_len, c = i % d.row_len;
        dst[i] = ((const uint8_t *) d.base)[r * d.stride + c];
    }
}

static bool keryx_append(ggml_backend_cuda_context & ctx, keryx_trace_seq * sq, const kb3::append_desc * descs,
                         uint32_t n_descs) {
    if (sq->seq_off) {
        keryx_dump_append<<<n_descs, 256, 0, ctx.stream()>>>(descs, (const kb3::stream_state *) sq->streams,
                                                             sq->seq_off, sq->seq_arena);
    }
    return kb3::stream_append((kb3::stream_state *) sq->streams, descs, n_descs, ctx.stream()) == cudaSuccess;
}

// One append per head: n_rows rows of row_len bytes at base + h * head_off, stride apart, into
// stream leaf + h * leaf_step. Built on the device so a captured graph keeps no host pointer.
static __global__ void keryx_heads_descs(kb3::append_desc * out, const char * base, int64_t head_off, int64_t stride,
                                         uint32_t n_rows, uint32_t row_len, uint32_t leaf, uint32_t leaf_step) {
    const uint32_t h = threadIdx.x;
    out[h] = {base + h * head_off, stride, n_rows, row_len, leaf + h * leaf_step, 0};
}

static bool keryx_append_heads(ggml_backend_cuda_context & ctx, keryx_trace_seq * sq, const keryx_trace_op * op,
                               const void * base, int64_t head_off, int64_t stride, uint32_t n_rows,
                               uint32_t row_len, uint32_t leaf) {
    if (n_rows == 0 || op->n_heads == 0) {
        return true;
    }
    if (op->defer_off >= 0) {
        // Decode: the descriptors wait in the device's table for the layer's flush.
        keryx_heads_descs<<<1, op->n_heads, 0, ctx.stream()>>>((kb3::append_desc *) sq->descs + op->desc,
            (const char *) base, head_off, stride, n_rows, row_len, leaf, op->leaf_step);
        return true;
    }
    ggml_cuda_pool_alloc<kb3::append_desc> descs(ctx.pool(), op->n_heads);
    keryx_heads_descs<<<1, op->n_heads, 0, ctx.stream()>>>(descs.get(), (const char *) base, head_off, stride, n_rows,
                                                            row_len, leaf, op->leaf_step);
    return keryx_append(ctx, sq, descs.get(), op->n_heads);
}

// Q8_0 cache rows of the ubatch's cells, one head after the other: out[h][j] = head h of cell idx[j].
static __global__ void keryx_kv_gather(const char * cache, int64_t nb1, const int64_t * idx, uint16_t * out,
                                       int64_t n_rows, int64_t head_bytes) {
    const int64_t h = blockIdx.y;
    const int64_t words = head_bytes / 2;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n_rows * words; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t j = i / words, w = i % words;
        const uint16_t * row = (const uint16_t *) (cache + idx[j] * nb1 + h * head_bytes);
        out[(h * n_rows + j) * words + w] = row[w];
    }
}

static bool keryx_trace_kv(ggml_backend_cuda_context & ctx, const keryx_trace_op * op, const ggml_tensor * dst) {
    const ggml_tensor * cache = dst->src[0];
    const ggml_tensor * idx = dst->src[1];
    keryx_trace_seq * sq = keryx_seq_of(op, ctx.device);
    const int64_t head_bytes = (int64_t) op->head_dim / QK8_0 * (int64_t) sizeof(block_q8_0);
    if (!sq || cache->type != GGML_TYPE_Q8_0 || idx->type != GGML_TYPE_I64 || op->head_dim % QK8_0 != 0 ||
        (int64_t) op->n_heads * op->head_dim > cache->ne[0]) {
        return false;
    }
    const int64_t n_rows = idx->ne[0];
    if (n_rows == 0) {
        return true;
    }
    ggml_cuda_pool_alloc<char> pool(ctx.pool());
    char * rows = op->defer_off >= 0 ? (char *) sq->defer + op->defer_off : pool.alloc(n_rows * op->n_heads * head_bytes);
    const dim3 grid((unsigned) std::min<int64_t>((n_rows * head_bytes / 2 + 255) / 256, 1024), op->n_heads);
    keryx_kv_gather<<<grid, 256, 0, ctx.stream()>>>((const char *) cache->data, cache->nb[1], (const int64_t *) idx->data,
                                                    (uint16_t *) rows, n_rows, head_bytes);
    return keryx_append_heads(ctx, sq, op, rows, n_rows * head_bytes, head_bytes, (uint32_t) n_rows,
                              (uint32_t) head_bytes, op->leaf);
}

// lse[t][qh] = m + ln(l) of scale * q·k over the cells the mask lets row t see, with k the
// dequantised Q8_0 cache row: what FLASH_ATTN_EXT attended, in f32. A block takes one kv head,
// a tile of NP (row, query head) pairs and the cells [z * chunk, (z + 1) * chunk); each warp takes
// one cell at a time, its lanes one value of every Q8_0 block, so a row loads coalesced. The NP
// dot products are reduced by transposition: after it, lane l holds the sum of pair l >> (5 - LOG)
// and runs that pair's online softmax. With more than one chunk, each writes its (m, l) to part[z]
// and keryx_lse_combine folds them (a decode row has few pairs, so the cells carry the parallelism).
#define KERYX_LSE_THREADS 128
#define KERYX_LSE_WARPS (KERYX_LSE_THREADS / 32)

// The query and key rounding of the FLASH_ATTN_EXT kernel ggml picks (fattn.cu,
// keryx_fattn_kernel): the vector kernel quantises scale·q to Q8_1 per 32 values against the
// Q8_0 keys; the MMA kernel takes half(scale)·half(q) against keys converted to f16. The node
// checks the committed mass and the attention output under one rounding, so the mass must be
// computed the way the output was.
enum { KERYX_LSE_Q8_1 = 1, KERYX_LSE_F16 = 2 };
enum { KERYX_FATTN_VEC = 100, KERYX_FATTN_MMA_F16 = 400 };  // ggml's best_fattn_kernel values
int keryx_fattn_kernel(int device, const ggml_tensor * dst);

template <int LOG>
static __global__ void keryx_lse(const char * q, int64_t q1, int64_t q2, const char * k, int64_t k1, int64_t k2,
                                 const half * mask, int64_t m1, int n_kv, int n_rows, int hd, int g, float scale,
                                 float * lse, int n_head, int chunk, float2 * part, int arith) {
    constexpr int NP = 1 << LOG;
    extern __shared__ float sh[];
    const int kvh = blockIdx.x;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int p0 = blockIdx.y * NP;
    const int np = min(NP, n_rows * g - p0);
    float * qs = sh;  // [NP][hd], zero past np
    for (int i = threadIdx.x; i < NP * hd; i += blockDim.x) {
        const int p = p0 + i / hd, d = i % hd;
        const float x = i / hd < np ? *(const float *) (q + (p / g) * q1 + (kvh * g + p % g) * q2 + d * 4) : 0.0f;
        qs[i] = arith == KERYX_LSE_F16 ? __half2float(__hmul(__float2half(scale), __float2half(x))) : scale * x;
    }
    __syncthreads();
    if (arith == KERYX_LSE_Q8_1) {
        for (int b = threadIdx.x; b < NP * hd / QK8_1; b += blockDim.x) {
            float * v = qs + b * QK8_1;
            float amax = 0.0f;
            for (int j = 0; j < QK8_1; j++) amax = fmaxf(amax, fabsf(v[j]));
            const float dq = amax / 127;
            for (int j = 0; j < QK8_1; j++) v[j] = dq != 0.0f ? dq * roundf(v[j] / dq) : 0.0f;
        }
        __syncthreads();
    }
    const int mine = lane >> (5 - LOG);  // the pair this lane ends up holding
    const bool live = mine < np;
    const half * mrow = mask + (int64_t) ((p0 + (live ? mine : 0)) / g) * m1;
    float m = -INFINITY, l = 0.0f;
    const int nb = hd / QK8_0;
    const int c1 = min(n_kv, (int) (blockIdx.z + 1) * chunk);
    for (int c = blockIdx.z * chunk + warp; c < c1; c += KERYX_LSE_WARPS) {
        const float mk = live ? __half2float(mrow[c]) : -INFINITY;
        if (__all_sync(0xffffffffu, mk == -INFINITY)) {
            continue;
        }
        const block_q8_0 * row = (const block_q8_0 *) (k + (int64_t) c * k1 + kvh * k2);
        float s[NP];
#pragma unroll
        for (int i = 0; i < NP; i++) s[i] = 0.0f;
        for (int b = 0; b < nb; b++) {
            float kv = __half2float(row[b].d) * row[b].qs[lane];
            if (arith == KERYX_LSE_F16) {
                kv = __half2float(__float2half(kv));
            }
            const float * qb = qs + b * QK8_0 + lane;
#pragma unroll
            for (int i = 0; i < NP; i++) s[i] += qb[i * hd] * kv;
        }
        // Transposed reduction: each step halves the values a lane holds and the lanes they span.
#pragma unroll
        for (int step = 0; step < LOG; step++) {
            const int bit = 16 >> step, half_n = NP >> (step + 1);
            const bool up = lane & bit;
#pragma unroll
            for (int i = 0; i < half_n; i++) {
                const float send = up ? s[i] : s[i + half_n];
                const float keep = up ? s[i + half_n] : s[i];
                s[i] = keep + __shfl_xor_sync(0xffffffffu, send, bit);
            }
        }
#pragma unroll
        for (int bit = 16 >> LOG; bit > 0; bit >>= 1) s[0] += __shfl_xor_sync(0xffffffffu, s[0], bit);
        if (mk != -INFINITY) {
            const float x = s[0] + mk;
            if (x > m) {
                l = l * expf(m - x) + 1.0f;
                m = x;
            } else {
                l += expf(x - m);
            }
        }
    }
    // Combine the warps' (m, l) per pair: the first lane holding a pair speaks for the warp.
    float * rm = sh + NP * hd;
    float * rl = rm + KERYX_LSE_WARPS * NP;
    if ((lane & ((32 >> LOG) - 1)) == 0) {
        rm[warp * NP + mine] = m;
        rl[warp * NP + mine] = l;
    }
    __syncthreads();
    if (threadIdx.x < np) {
        const int i = threadIdx.x;
        float mm = -INFINITY;
        for (int w = 0; w < KERYX_LSE_WARPS; w++) mm = fmaxf(mm, rm[w * NP + i]);
        float ll = 0.0f;
        for (int w = 0; w < KERYX_LSE_WARPS; w++) {
            const float mw = rm[w * NP + i];
            if (mw != -INFINITY) ll += rl[w * NP + i] * expf(mw - mm);
        }
        const int p = p0 + i;
        const int o = (p / g) * n_head + kvh * g + p % g;
        if (part) {
            part[(int64_t) blockIdx.z * n_rows * n_head + o] = make_float2(mm, ll);
        } else {
            lse[o] = mm + logf(ll);
        }
    }
}

static __global__ void keryx_lse_combine(const float2 * part, int n_chunks, int n, float * lse) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) {
        return;
    }
    float mm = -INFINITY;
    for (int z = 0; z < n_chunks; z++) mm = fmaxf(mm, part[(int64_t) z * n + o].x);
    float ll = 0.0f;
    for (int z = 0; z < n_chunks; z++) {
        const float2 c = part[(int64_t) z * n + o];
        if (c.x != -INFINITY) ll += c.y * expf(c.x - mm);
    }
    lse[o] = mm + logf(ll);
}

static bool keryx_trace_lse(ggml_backend_cuda_context & ctx, const keryx_trace_op * op, const ggml_tensor * dst) {
    const ggml_tensor * fa = dst->src[0];
    const ggml_tensor * q = fa->src[0];
    const ggml_tensor * k = fa->src[1];
    const ggml_tensor * mask = fa->src[3];
    keryx_trace_seq * sq = keryx_seq_of(op, ctx.device);
    float scale, max_bias, softcap;
    memcpy(&scale, (const float *) fa->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) fa->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *) fa->op_params + 2, sizeof(float));
    const int hd = (int) q->ne[0], n_rows = (int) q->ne[1], n_head = (int) q->ne[2], n_kv = (int) k->ne[1];
    const int g = (int) op->group;
    if (!sq || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_Q8_0 || !mask || mask->type != GGML_TYPE_F16 ||
        max_bias != 0.0f || softcap != 0.0f || q->nb[0] != 4 || k->ne[0] != hd || hd % QK8_0 != 0 ||
        n_head != (int) op->n_heads * g || k->ne[2] != (int64_t) op->n_heads || q->ne[3] != 1 || mask->ne[0] < n_kv) {
        return false;
    }
    // As many (row, head) pairs per block as 32 KiB of queries hold, a power of two: decode tiles
    // just its query heads.
    int log = 0;
    while (log < 4 && (1 << log) < n_rows * g && (2 << log) * hd * 4 <= 32 * 1024) log++;
    const int tile = 1 << log;
    const size_t smem = (size_t) tile * hd * 4 + 2 * KERYX_LSE_WARPS * tile * 4;
    if (tile * hd * 4 > 32 * 1024) {
        return false;
    }
    const int fattn = keryx_fattn_kernel(ctx.device, fa);
    const int arith = fattn == KERYX_FATTN_VEC ? KERYX_LSE_Q8_1 : fattn == KERYX_FATTN_MMA_F16 ? KERYX_LSE_F16 : 0;
    if (!arith) {
        return false;  // a kernel whose rounding the trace does not know (not on Turing or newer)
    }
    ggml_cuda_pool_alloc<float> pool(ctx.pool());
    float * lse = op->defer_off >= 0 ? (float *) (sq->defer + op->defer_off) : pool.alloc((size_t) n_rows * n_head);
    // Split the cells until the grid fills the card twice, a warp per cell at least.
    const int tiles = (n_rows * g + tile - 1) / tile;
    const int sms = ggml_cuda_info().devices[ctx.device].nsm;
    const int max_chunks = std::max(1, (n_kv + KERYX_LSE_WARPS - 1) / KERYX_LSE_WARPS / 8);
    const int n_chunks = std::min(max_chunks, std::max(1, 2 * sms / (int) (op->n_heads * tiles)));
    const int chunk = (n_kv + n_chunks - 1) / n_chunks;
    ggml_cuda_pool_alloc<float2> part(ctx.pool());
    if (n_chunks > 1) {
        part.alloc((size_t) n_chunks * n_rows * n_head);
    }
    const dim3 grid(op->n_heads, tiles, n_chunks);
    auto kernel = log == 0 ? keryx_lse<0> : log == 1 ? keryx_lse<1> : log == 2 ? keryx_lse<2> : log == 3 ? keryx_lse<3>
                                                                                                          : keryx_lse<4>;
    kernel<<<grid, KERYX_LSE_THREADS, smem, ctx.stream()>>>((const char *) q->data, q->nb[1], q->nb[2],
        (const char *) k->data, k->nb[1], k->nb[2], (const half *) mask->data, mask->nb[1] / 2, n_kv, n_rows, hd, g,
        scale, lse, n_head, chunk, n_chunks > 1 ? part.get() : nullptr, arith);
    if (n_chunks > 1) {
        const int n = n_rows * n_head;
        keryx_lse_combine<<<(n + 127) / 128, 128, 0, ctx.stream()>>>(part.get(), n_chunks, n, lse);
    }
    return keryx_append_heads(ctx, sq, op, lse, (int64_t) g * 4, (int64_t) n_head * 4, (uint32_t) n_rows,
                              (uint32_t) g * 4, op->leaf);
}

// Record of value head h at row t (node seq.rs `Record::encode`): q, k, v, o as f16 (round to
// nearest even), beta f32, then g (GDA one f32, KDA hd f16). q and k come from key head h % n_k.
static __global__ void keryx_records(const float * q, int64_t q1, int64_t q2, const float * k, const float * v,
                                     int64_t v1, int64_t v2, const float * gg, const float * beta, int64_t b1,
                                     int64_t b2, const float * o, int n_k, int n_v, int n_rows, int hd, int kda,
                                     char * out, int64_t rl) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int hk = h % n_k;
    char * rec = out + ((int64_t) h * n_rows + t) * rl;
    half * f = (half *) rec;
    f[i] = __float2half_rn(q[t * q2 + hk * q1 + i]);
    f[hd + i] = __float2half_rn(k[t * q2 + hk * q1 + i]);
    f[2 * hd + i] = __float2half_rn(v[t * v2 + h * v1 + i]);
    f[3 * hd + i] = __float2half_rn(o[((int64_t) t * n_v + h) * hd + i]);
    const int64_t gb = t * b2 + h * b1;
    float * tail = (float *) (rec + 8 * hd);
    if (i == 0) tail[0] = beta[gb];
    if (kda) {
        ((half *) (tail + 1))[i] = __float2half_rn(gg[gb * hd + i]);
    } else if (i == 0) {
        tail[1] = gg[gb];
    }
}

// Checkpoints the ubatch produced, per head (see keryx_trace_gdn): built from pos0 on the device.
static __global__ void keryx_ckpt_descs(kb3::append_desc * out, const float * ckpt, const int32_t * pos0, int n_rows,
                                        int interval, int n_v, int hd, uint32_t leaf, uint32_t leaf_step) {
    const int h = threadIdx.x;
    const int first = (*pos0 + interval) / interval, last = (*pos0 + n_rows) / interval;
    const uint32_t n = last >= first ? (uint32_t) (last - first + 1) : 0u;
    const int64_t state = (int64_t) hd * hd * 4;
    out[h] = {(const char *) ckpt + h * state, state * n_v, n, (uint32_t) state, leaf + 1 + h * leaf_step, 0};
}

static bool keryx_trace_recur(ggml_backend_cuda_context & ctx, const keryx_trace_op * op, const ggml_tensor * dst) {
    const ggml_tensor * gdn = dst->src[0];
    const ggml_tensor * q = gdn->src[0];
    const ggml_tensor * k = gdn->src[1];
    const ggml_tensor * v = gdn->src[2];
    const ggml_tensor * g = gdn->src[3];
    const ggml_tensor * beta = gdn->src[4];
    keryx_trace_seq * sq = keryx_seq_of(op, ctx.device);
    const int hd = (int) v->ne[0], n_v = (int) v->ne[1], n_rows = (int) v->ne[2], n_k = (int) q->ne[1];
    const int kda = g->ne[0] == hd ? 1 : 0;
    if (!sq || hd != (int) op->head_dim || n_v != (int) op->n_heads || kda != (int) op->recur_kind || v->ne[3] != 1 ||
        q->nb[0] != 4 || k->nb[1] != q->nb[1] || k->nb[2] != q->nb[2] || v->nb[0] != 4 || hd > 1024 ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta)) {
        return false;
    }
    if (n_rows == 0) {
        return true;
    }
    const int64_t rl = 8 * (int64_t) hd + 4 + (kda ? 2 * (int64_t) hd : 4);
    ggml_cuda_pool_alloc<char> pool(ctx.pool());
    char * recs = op->defer_off >= 0 ? (char *) sq->defer + op->defer_off : pool.alloc((size_t) n_v * n_rows * rl);
    keryx_records<<<dim3(n_v, n_rows), hd, 0, ctx.stream()>>>((const float *) q->data, q->nb[1] / 4, q->nb[2] / 4,
        (const float *) k->data, (const float *) v->data, v->nb[1] / 4, v->nb[2] / 4, (const float *) g->data,
        (const float *) beta->data, beta->nb[1] / 4, beta->nb[2] / 4, (const float *) gdn->data, n_k, n_v, n_rows, hd,
        kda, recs, rl);
    if (!keryx_append_heads(ctx, sq, op, recs, n_rows * rl, rl, (uint32_t) n_rows, (uint32_t) rl, op->leaf)) {
        return false;
    }
    // The checkpoints append now, even in decode: the next recurrent layer's GDN kernel writes the
    // same staging.
    ggml_cuda_pool_alloc<kb3::append_desc> descs(ctx.pool(), op->n_heads);
    keryx_ckpt_descs<<<1, op->n_heads, 0, ctx.stream()>>>(descs.get(), sq->ckpt, sq->pos0, n_rows, (int) op->interval,
                                                           n_v, hd, op->leaf, op->leaf_step);
    return keryx_append(ctx, sq, descs.get(), op->n_heads);
}

// Decode flush: one block per staged message. Each thread takes chunks; their chaining values
// meet in shared memory and fold up to the root (pairs merged, the odd node carried, ROOT on the
// last merge), which is BLAKE3 of the message. A message of one chunk is its root chunk.
#define KERYX_FLUSH_THREADS 128
#define KERYX_FLUSH_CHUNKS (KERYX_DEFER_MAX_MSG / 1024)

static __global__ void keryx_flush_hash(const keryx_trace_msg * msgs, const uint8_t * arena, uint8_t * table,
                                        const int64_t * pos_off, uint8_t * const * dump) {
    extern __shared__ uint32_t flush_sh[];
    uint32_t (*cvs)[8] = (uint32_t (*)[8]) flush_sh;            // [KERYX_FLUSH_CHUNKS][8]
    uint32_t * stage = flush_sh + KERYX_FLUSH_CHUNKS * 8 + threadIdx.x * 16;
    const keryx_trace_msg m = msgs[blockIdx.x];
    const uint8_t * p = arena + m.off;
    const uint32_t n = m.len == 0 ? 1 : (m.len + 1023) / 1024;
    uint32_t * out = (uint32_t *) (table + (int64_t) m.table * 32);
    if (n == 1) {
        if (threadIdx.x == 0) {
            kb3::cursor c{p, m.len, nullptr, 0, 0, 0};
            uint32_t cv[8];
            kb3::chunk(c, m.len, 0, kb3::ROOT, stage, cv);
            for (int i = 0; i < 8; i++) out[i] = cv[i];
        }
    } else {
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t len = min(1024u, m.len - i * 1024);
            kb3::cursor c{p + (int64_t) i * 1024, len, nullptr, 0, 0, 0};
            kb3::chunk(c, len, i, 0, stage, cvs[i]);
        }
        __syncthreads();
        uint32_t level = n;
        while (level > 1) {
            const uint32_t pairs = level / 2;
            const bool top = level == 2;
            uint32_t merged[(KERYX_FLUSH_CHUNKS + 2 * KERYX_FLUSH_THREADS - 1) / (2 * KERYX_FLUSH_THREADS)][8];
            uint32_t k = 0;
            for (uint32_t i = threadIdx.x; i < pairs; i += blockDim.x, k++) kb3::merge(cvs[2 * i], cvs[2 * i + 1], top, merged[k]);
            __syncthreads();
            k = 0;
            for (uint32_t i = threadIdx.x; i < pairs; i += blockDim.x, k++) {
                for (int w = 0; w < 8; w++) cvs[i][w] = merged[k][w];
            }
            if (level % 2 == 1 && threadIdx.x == 0) {
                for (int w = 0; w < 8; w++) cvs[pairs][w] = cvs[level - 1][w];
            }
            __syncthreads();
            level = pairs + level % 2;
        }
        if (threadIdx.x == 0) {
            for (int i = 0; i < 8; i++) out[i] = cvs[0][i];
        }
    }
    const int64_t o = pos_off[m.table];
    if (o >= 0) {
        uint8_t * d = *dump + o;
        for (uint32_t i = threadIdx.x; i < m.len; i += blockDim.x) d[i] = p[i];
    }
}

static bool keryx_trace_flush(ggml_backend_cuda_context & ctx, const keryx_trace_op * op) {
    keryx_trace_seq * sq = keryx_seq_of(op, ctx.device);
    uint8_t * table = keryx_table_of(op, ctx.device);
    if (!sq || !table || !sq->defer || !sq->msgs || !sq->pos_off) {
        return false;
    }
    keryx_forget_q8_1_on(ctx.device);
    // The run's bytes are in this card's arena only if every capture of it ran here.
    for (uint32_t i = 0; i < op->n_members; i++) {
        if (op->members[i]->ran_on != ctx.device) {
            GGML_LOG_ERROR("keryx trace: a decode capture of leaf %u ran on another card\n", op->members[i]->leaf);
            return false;
        }
    }
    if (op->count > 0) {
        const size_t smem = (size_t) KERYX_FLUSH_CHUNKS * 32 + (size_t) KERYX_FLUSH_THREADS * 64;
        keryx_flush_hash<<<op->count, KERYX_FLUSH_THREADS, smem, ctx.stream()>>>(sq->msgs + op->first, sq->defer, table,
                                                                                sq->pos_off, sq->arena);
    }
    return op->n_descs == 0 || keryx_append(ctx, sq, (const kb3::append_desc *) sq->descs + op->desc, op->n_descs);
}

bool keryx_trace_compute(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    keryx_trace_op * op = const_cast<keryx_trace_op *>(keryx_trace_op_of(dst));
    bool ok = false;
    if (op) {
        op->ran_on = ctx.device;
    }
    if (op && op->kind == KERYX_TRACE_F32) {
        ok = keryx_trace_f32(ctx, op, dst);
    } else if (op && op->kind == KERYX_TRACE_X) {
        ok = keryx_trace_x(ctx, op, dst);
    } else if (op && op->kind == KERYX_TRACE_KV) {
        ok = keryx_trace_kv(ctx, op, dst);
    } else if (op && op->kind == KERYX_TRACE_LSE) {
        ok = keryx_trace_lse(ctx, op, dst);
    } else if (op && op->kind == KERYX_TRACE_RECUR) {
        ok = keryx_trace_recur(ctx, op, dst);
    } else if (op && op->kind == KERYX_TRACE_FLUSH) {
        ok = keryx_trace_flush(ctx, op);
    }
    if (!ok) {
        keryx_trace_forget_q8_1();
        GGML_LOG_ERROR("keryx trace: capture node %s cannot run (leaf %u)\n", dst->name, op ? op->leaf : 0u);
    }
    return ok;
}
