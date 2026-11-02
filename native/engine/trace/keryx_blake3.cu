// BLAKE3 kernels and launchers for the H15 trace (keryx_blake3.h). One thread hashes one 1 KiB
// chunk; the tree above the chunks is merged level by level in shared memory.
#include "keryx_blake3.cuh"

#include <algorithm>

namespace kb3 {

namespace {

constexpr uint32_t STAGE_WORDS  = 17;   // 16 words per thread, one pad word: no bank conflicts
constexpr uint32_t GROUP_CHUNKS = 128;  // chunks one block reduces for a long message
constexpr uint32_t ROOT_RUN     = 4;     // leaves a thread merges in registers first
constexpr uint32_t ROOT_WINDOW  = 1024;  // runs one in-place pass merges (32 KiB of shared)
constexpr uint32_t ROOT_THREADS = 128;  // 512 groups of 1123: 60 us, 72 us at 256
constexpr uint32_t FINAL_THREADS = 128;
constexpr uint32_t APPEND_THREADS = 32;  // a warp per state (append_kernel)

// Bytes kept pending after `total` bytes: the last chunk, never empty unless nothing came in.
__device__ __forceinline__ uint32_t pending_len(uint64_t total) {
    return total == 0 ? 0u : (uint32_t) ((total - 1) % CHUNK_LEN) + 1;
}

// Cursor at byte `dd` of a descriptor's rows. Rows back to back (stride == row_len) are one
// piece, so a block never splits at their seams.
__device__ __forceinline__ cursor rows_at(const append_desc& d, uint32_t dd) {
    const uint8_t* base = (const uint8_t*) d.base;
    cursor c;
    c.stride = d.stride;
    if (d.stride == (int64_t) d.row_len) {
        const uint32_t total = d.n_rows * d.row_len;
        c.p = base + dd; c.rem = total - dd; c.next_row = nullptr; c.row_len = 0; c.rows_left = 0;
    } else {
        const uint32_t row = dd / d.row_len, col = dd % d.row_len;
        c.p         = base + (int64_t) row * d.stride + col;
        c.rem       = d.row_len - col;
        c.next_row  = base + (int64_t) (row + 1) * d.stride;
        c.row_len   = d.row_len;
        c.rows_left = d.n_rows - row - 1;
    }
    return c;
}

__device__ __forceinline__ void put_hash(uint8_t* out, const uint32_t cv[8]) {
    uint32_t* o = (uint32_t*) out;
#pragma unroll
    for (int i = 0; i < 8; i++) o[i] = cv[i];
}

// hash_strided. Short messages (<= GROUP_CHUNKS chunks): `per_block` whole messages per block,
// their tree merged here and the hash written out. Long ones: one block per GROUP_CHUNKS-chunk
// group, the group's subtree written to scratch for b3_root_kernel. An aligned group of the
// left-complete tree reduces to the same node whether the tree is merged by levels or by
// groups, so the two stages agree with b3_root over all chunk CVs.
__global__ void strided_kernel(const uint8_t* base0, int64_t stride0, uint64_t len0,
                               const uint8_t* base1, int64_t stride1, uint64_t len1,
                               uint32_t count, uint64_t n_chunks, uint32_t per_block,
                               uint32_t groups_per_msg, uint8_t* out, int64_t out_stride,
                               uint8_t* scratch) {
    extern __shared__ uint32_t smem[];
    uint32_t* stage = smem + threadIdx.x * STAGE_WORDS;
    uint32_t* nodes = smem + blockDim.x * STAGE_WORDS;
    const uint32_t cap = blockDim.x;

    uint64_t msg, c, n_group;
    uint32_t l;  // index in the group
    if (groups_per_msg == 0) {
        const uint32_t m = threadIdx.x / (uint32_t) n_chunks;
        l       = threadIdx.x % (uint32_t) n_chunks;
        msg     = (uint64_t) blockIdx.x * per_block + m;
        c       = l;
        n_group = n_chunks;
        if (m >= per_block) msg = count;  // the block's tail threads
    } else {
        const uint64_t grp = blockIdx.x % groups_per_msg;
        msg     = blockIdx.x / groups_per_msg;
        l       = threadIdx.x;
        c       = grp * GROUP_CHUNKS + l;
        n_group = min((uint64_t) GROUP_CHUNKS, n_chunks - grp * GROUP_CHUNKS);
    }
    const bool active = msg < count && l < n_group;
    const bool whole  = groups_per_msg == 0;

    uint32_t cv[8];
    if (active) {
        const uint64_t len = len0 + len1;
        const uint64_t o   = c * CHUNK_LEN;
        cursor cur;
        if (o < len0) {
            cur.p         = base0 + (int64_t) msg * stride0 + o;
            cur.rem       = len0 - o;
            cur.next_row  = base1 + (int64_t) msg * stride1;
            cur.rows_left = len1 != 0 ? 1 : 0;
        } else {
            cur.p         = base1 + (int64_t) msg * stride1 + (o - len0);
            cur.rem       = len1 - (o - len0);
            cur.next_row  = nullptr;
            cur.rows_left = 0;
        }
        cur.stride  = 0;
        cur.row_len = (uint32_t) len1;  // a single row: only len1 < 4 GiB matters here
        const uint32_t clen = (uint32_t) min((uint64_t) CHUNK_LEN, len - o);
        chunk(cur, clen, c, n_chunks == 1 ? ROOT : 0u, stage, cv);
        node_put(nodes, cap, threadIdx.x, cv);
    }
    for (uint32_t s = 1; s < GROUP_CHUNKS && s < n_group; s <<= 1) {
        __syncthreads();
        if (active && (l & (2 * s - 1)) == 0 && l + s < n_group) {
            uint32_t r[8];
            node_get(nodes, cap, threadIdx.x, cv);
            node_get(nodes, cap, threadIdx.x + s, r);
            merge(cv, r, whole && l == 0 && 2 * s >= n_group, cv);
            node_put(nodes, cap, threadIdx.x, cv);
        }
    }
    if (active && l == 0) {
        if (whole) {
            put_hash(out + (int64_t) msg * out_stride, cv);
        } else {
            put_hash(scratch + (blockIdx.x * (uint64_t) 32), cv);
        }
    }
}

// Subtree of an aligned run of `cnt` (1..4) leaves, merged in registers; `root` = the run is the
// whole tree.
__device__ __forceinline__ void run4(const uint32_t* leaf, uint32_t cnt, bool root, uint32_t out[8]) {
    uint32_t a[8], b[8], c[8], d[8];
#pragma unroll
    for (int w = 0; w < 8; w++) {
        a[w] = leaf[w];
        b[w] = cnt > 1 ? leaf[8 + w] : 0u;
        c[w] = cnt > 2 ? leaf[16 + w] : 0u;
        d[w] = cnt > 3 ? leaf[24 + w] : 0u;
    }
    if (cnt >= 2) merge(a, b, root && cnt == 2, a);
    if (cnt == 4) merge(c, d, false, c);
    if (cnt >= 3) merge(a, c, root, a);
#pragma unroll
    for (int w = 0; w < 8; w++) out[w] = a[w];
}

// b3_root of `n` hashes per group. Each thread merges an aligned run of ROOT_RUN leaves, then a
// window of ROOT_WINDOW runs is merged in place, node l taking l + s while l is a multiple of
// 2s: about log2(n) serial merges (a push per window level through the stack cost 40 for
// n = 1123, the in-place pass 19, with runs 11). Runs and window roots are aligned subtrees,
// so they reduce to the nodes the level-by-level tree has; window roots go through the stack
// like chunk CVs.
__global__ void b3_root_kernel(const uint8_t* base, int64_t stride, uint32_t n, uint8_t* out,
                               int64_t out_stride) {
    extern __shared__ uint32_t nodes[];  // [8][cap]
    __shared__ uint32_t stk[STACK_LEVELS][8];
    const uint32_t* leaves = (const uint32_t*) (base + (int64_t) blockIdx.x * stride);
    const uint32_t  runs   = (n + ROOT_RUN - 1) / ROOT_RUN;
    const uint32_t  cap    = min(runs, ROOT_WINDOW);
    const uint32_t  n_win  = (runs + ROOT_WINDOW - 1) / ROOT_WINDOW;

    for (uint32_t w = 0; w < n_win; w++) {
        const uint32_t first = w * ROOT_WINDOW, cnt = min(ROOT_WINDOW, runs - first);
        const bool     whole = n_win == 1;
        for (uint32_t r = threadIdx.x; r < cnt; r += blockDim.x) {
            const uint64_t leaf0 = (uint64_t) (first + r) * ROOT_RUN;
            uint32_t cv[8];
            run4(leaves + leaf0 * 8, (uint32_t) min((uint64_t) ROOT_RUN, n - leaf0), whole && cnt == 1, cv);
            node_put(nodes, cap, r, cv);
        }
        __syncthreads();
        for (uint32_t s = 1; s < cnt; s <<= 1) {
            for (uint32_t l = threadIdx.x * 2 * s; l + s < cnt; l += blockDim.x * 2 * s) {
                uint32_t a[8], b[8];
                node_get(nodes, cap, l, a);
                node_get(nodes, cap, l + s, b);
                merge(a, b, whole && l == 0 && 2 * s >= cnt, a);
                node_put(nodes, cap, l, a);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            uint32_t cv[8];
            node_get(nodes, cap, 0, cv);
            if (w + 1 < n_win) {
                stack_push(stk, w, cv);
            } else {
                fold_stack(stk, w, cv);
                put_hash(out + (int64_t) blockIdx.x * out_stride, cv);
            }
        }
        __syncthreads();
    }
}

// One block per descriptor: the chunks the new rows complete are hashed in windows of 2 per
// thread and pushed into the state's stack; the last chunk stays pending. A warp per state
// keeps 1-row appends (every generated token) to one wave: 128 threads and 22 KiB of shared
// per block took three.
template <uint32_t THREADS>
__global__ void __launch_bounds__(THREADS) append_kernel(stream_state* states, const append_desc* descs) {
    constexpr uint32_t W = 2 * THREADS;
    __shared__ uint32_t stk[STACK_LEVELS][8];
    __shared__ uint32_t cur[8 * W];
    // The stage is used while chunks are hashed, nxt while they merge: one area for both.
    __shared__ uint32_t area[THREADS * STAGE_WORDS > 8 * (W / 2 + 1) ? THREADS * STAGE_WORDS : 8 * (W / 2 + 1)];
    uint32_t* stage = area + threadIdx.x * STAGE_WORDS;

    const append_desc d  = descs[blockIdx.x];
    stream_state*     st = states + d.state;
    const uint64_t    t0 = st->total_len;
    const uint64_t    t1 = t0 + (uint64_t) d.n_rows * d.row_len;
    if (t1 == t0) return;
    const uint32_t p0 = pending_len(t0), p1 = pending_len(t1);
    const uint64_t c0 = (t0 - p0) / CHUNK_LEN, c1 = (t1 - p1) / CHUNK_LEN;
    const uint8_t* base = (const uint8_t*) d.base;

    if (c1 > c0) {
        // Only levels below the highest bit where c0 and c1 differ are read or written.
        const uint32_t levels = min((uint32_t) STACK_LEVELS, 64u - (uint32_t) __clzll((long long) (c0 ^ c1)));
        for (uint32_t i = threadIdx.x; i < levels * 8; i += THREADS) stk[i >> 3][i & 7] = st->stack[i >> 3][i & 7];
        for (uint64_t wc = c0; wc < c1; wc += W) {
            const uint32_t cnt = (uint32_t) min((uint64_t) W, c1 - wc);
            for (uint32_t j = threadIdx.x; j < cnt; j += THREADS) {
                const uint64_t c   = wc + j;
                const uint32_t rel = (uint32_t) ((c - c0) * CHUNK_LEN);  // from the old pending start
                cursor cur_;
                if (rel < p0) {
                    // The old pending bytes, then the rows (back to back ones as one).
                    const bool packed = d.stride == (int64_t) d.row_len;
                    cur_.p         = st->pending + rel;
                    cur_.rem       = p0 - rel;
                    cur_.next_row  = base;
                    cur_.stride    = d.stride;
                    cur_.row_len   = packed ? d.n_rows * d.row_len : d.row_len;
                    cur_.rows_left = packed ? 1 : d.n_rows;
                } else {
                    cur_ = rows_at(d, rel - p0);
                }
                uint32_t cv[8];
                chunk(cur_, CHUNK_LEN, c, 0, stage, cv);
                node_put(cur, W, j, cv);
            }
            __syncthreads();
            absorb(stk, wc, cnt, cur, area, W);
        }
        for (uint32_t i = threadIdx.x; i < levels * 8; i += THREADS) st->stack[i >> 3][i & 7] = stk[i >> 3][i & 7];
    }
    // absorb ended on a barrier, so chunk c0 has read the old pending bytes. The new pending
    // bytes [lo, hi) of the last chunk go in by 64-byte block, one thread each.
    const uint32_t lo = (uint32_t) (max(t0, c1 * CHUNK_LEN) - c1 * CHUNK_LEN), hi = p1;
    for (uint32_t j = threadIdx.x; j < CHUNK_LEN / 64; j += THREADS) {
        const uint32_t blo = max(lo, 64 * j), bhi = min(hi, 64 * j + 64);
        if (blo >= bhi) continue;
        cursor src = rows_at(d, (uint32_t) (c1 * CHUNK_LEN + blo - t0));
        uint32_t m[16];
        load_block(src, bhi - blo, m, stage);
        if (bhi - blo == 64) {
            uint4* o = (uint4*) (st->pending + blo);
#pragma unroll
            for (int i = 0; i < 4; i++) o[i] = make_uint4(m[4 * i], m[4 * i + 1], m[4 * i + 2], m[4 * i + 3]);
        } else {
            const uint8_t* sb = (const uint8_t*) stage;
            for (uint32_t b = blo; b < bhi; b++) st->pending[b] = sb[b - blo];
        }
    }
    if (threadIdx.x == 0) st->total_len = t1;
}

__global__ void finalize_kernel(const stream_state* states, uint32_t n, uint8_t* out, int64_t out_stride) {
    __shared__ uint32_t stage_all[FINAL_THREADS * STAGE_WORDS];
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const stream_state* st = states + i;
    const uint64_t t  = st->total_len;
    const uint32_t p  = pending_len(t);
    const uint64_t cd = (t - p) / CHUNK_LEN;
    cursor c;
    c.p = st->pending; c.rem = p; c.next_row = nullptr; c.stride = 0; c.row_len = 0; c.rows_left = 0;
    uint32_t cv[8];
    chunk(c, p, cd, cd == 0 ? ROOT : 0u, stage_all + threadIdx.x * STAGE_WORDS, cv);
    fold_stack(st->stack, cd, cv);
    put_hash(out + (int64_t) i * out_stride, cv);
}

bool aligned4(const void* p, int64_t stride) {
    return ((uintptr_t) p & 3) == 0 && (stride & 3) == 0;
}

uint64_t chunks_of(uint64_t len) {
    return len == 0 ? 1 : (len + CHUNK_LEN - 1) / CHUNK_LEN;
}

}  // namespace

size_t hash_strided_scratch(uint32_t count, uint64_t len) {
    const uint64_t c = chunks_of(len);
    return c <= GROUP_CHUNKS ? 0 : (size_t) count * ((c + GROUP_CHUNKS - 1) / GROUP_CHUNKS) * 32;
}

cudaError_t hash_strided(const void* base0, int64_t stride0, uint64_t len0,
                         const void* base1, int64_t stride1, uint64_t len1,
                         uint32_t count, void* out, int64_t out_stride,
                         void* scratch, size_t scratch_bytes, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (!aligned4(out, out_stride) || len1 >= (1ull << 32)) return cudaErrorInvalidValue;
    const uint64_t n_chunks = chunks_of(len0 + len1);
    if (n_chunks <= GROUP_CHUNKS) {
        // Whole messages per block: the largest block that still leaves ~256 blocks, then the
        // fewest idle lanes. A small call on a few big blocks crowds a few SMs: 512 x 4.6 KB
        // took 23 us on 11 blocks of 256 threads, 13 us on 86 blocks of 32.
        const uint32_t c = (uint32_t) n_chunks;
        const uint64_t want = ((uint64_t) count * c + 255) / 256;
        uint32_t max_threads = 32;
        while (max_threads < 256 && max_threads < want) max_threads *= 2;
        uint32_t best_k = 1, best_threads = (c + 31) / 32 * 32;
        for (uint32_t k = 2; k <= count && (k * c + 31) / 32 * 32 <= max_threads; k++) {
            const uint32_t th = (k * c + 31) / 32 * 32;
            if ((uint64_t) k * best_threads > (uint64_t) best_k * th) {
                best_k = k;
                best_threads = th;
            }
        }
        const uint32_t blocks = (count + best_k - 1) / best_k;
        const size_t   shmem  = (size_t) best_threads * (STAGE_WORDS + 8) * 4;
        strided_kernel<<<blocks, best_threads, shmem, stream>>>(
            (const uint8_t*) base0, stride0, len0, (const uint8_t*) base1, stride1, len1, count,
            n_chunks, best_k, 0, (uint8_t*) out, out_stride, nullptr);
        return cudaGetLastError();
    }
    const uint64_t groups = (n_chunks + GROUP_CHUNKS - 1) / GROUP_CHUNKS;
    if (scratch == nullptr || scratch_bytes < hash_strided_scratch(count, len0 + len1) ||
        ((uintptr_t) scratch & 3) != 0 || groups * count >= (1ull << 31)) {
        return cudaErrorInvalidValue;
    }
    const size_t shmem = (size_t) GROUP_CHUNKS * (STAGE_WORDS + 8) * 4;
    strided_kernel<<<(uint32_t) (groups * count), GROUP_CHUNKS, shmem, stream>>>(
        (const uint8_t*) base0, stride0, len0, (const uint8_t*) base1, stride1, len1, count,
        n_chunks, 0, (uint32_t) groups, nullptr, 0, (uint8_t*) scratch);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    return b3_root(scratch, (int64_t) groups * 32, (uint32_t) groups, count, out, out_stride, stream);
}

cudaError_t stream_reset(stream_state* states, uint32_t n, cudaStream_t stream) {
    return cudaMemsetAsync(states, 0, (size_t) n * sizeof(stream_state), stream);
}

cudaError_t stream_append(stream_state* states, const append_desc* descs, uint32_t n_descs,
                          cudaStream_t stream) {
    if (n_descs == 0) return cudaSuccess;
    append_kernel<APPEND_THREADS><<<n_descs, APPEND_THREADS, 0, stream>>>(states, descs);
    return cudaGetLastError();
}

cudaError_t stream_finalize(const stream_state* states, uint32_t n, void* out, int64_t out_stride,
                            cudaStream_t stream) {
    if (n == 0) return cudaSuccess;
    if (!aligned4(out, out_stride)) return cudaErrorInvalidValue;
    finalize_kernel<<<(n + FINAL_THREADS - 1) / FINAL_THREADS, FINAL_THREADS, 0, stream>>>(
        states, n, (uint8_t*) out, out_stride);
    return cudaGetLastError();
}

cudaError_t b3_root(const void* base, int64_t stride, uint32_t n, uint32_t groups, void* out,
                    int64_t out_stride, cudaStream_t stream) {
    if (groups == 0) return cudaSuccess;
    if (n == 0 || !aligned4(base, stride) || !aligned4(out, out_stride)) return cudaErrorInvalidValue;
    const size_t shmem = (size_t) std::min((n + ROOT_RUN - 1) / ROOT_RUN, ROOT_WINDOW) * 32;
    b3_root_kernel<<<groups, ROOT_THREADS, shmem, stream>>>((const uint8_t*) base, stride, n,
                                                           (uint8_t*) out, out_stride);
    return cudaGetLastError();
}

}  // namespace kb3
