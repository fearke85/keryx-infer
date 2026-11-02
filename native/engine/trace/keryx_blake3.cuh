// BLAKE3 device functions for the H15 trace (hash mode only, spec h15-spec-commitment.md 1.1).
// Plain CUDA (sm_75 and up), no ggml headers: ggml-cuda compiles this file as it is.
#pragma once

#include <cstdint>

#include "keryx_blake3.h"

namespace kb3 {

constexpr uint32_t CHUNK_START = 1;
constexpr uint32_t CHUNK_END   = 2;
constexpr uint32_t PARENT      = 4;
constexpr uint32_t ROOT        = 8;
constexpr uint32_t CHUNK_LEN   = 1024;

__device__ __forceinline__ uint32_t iv(int i) {
    // A switch on a constant folds away; a table would need a __constant__ symbol.
    switch (i) {
        case 0: return 0x6A09E667u;
        case 1: return 0xBB67AE85u;
        case 2: return 0x3C6EF372u;
        case 3: return 0xA54FF53Au;
        case 4: return 0x510E527Fu;
        case 5: return 0x9B05688Cu;
        case 6: return 0x1F83D9ABu;
        default: return 0x5BE0CD19u;
    }
}

__device__ __forceinline__ void g(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
                                  uint32_t mx, uint32_t my) {
    a = a + b + mx;
    d = __byte_perm(d ^ a, 0, 0x1032);  // rotr 16
    c = c + d;
    b = __funnelshift_r(b ^ c, b ^ c, 12);
    a = a + b + my;
    d = __byte_perm(d ^ a, 0, 0x0321);  // rotr 8
    c = c + d;
    b = __funnelshift_r(b ^ c, b ^ c, 7);
}

__device__ __forceinline__ void round_fn(uint32_t v[16], const uint32_t m[16]) {
    g(v[0], v[4], v[8],  v[12], m[0],  m[1]);
    g(v[1], v[5], v[9],  v[13], m[2],  m[3]);
    g(v[2], v[6], v[10], v[14], m[4],  m[5]);
    g(v[3], v[7], v[11], v[15], m[6],  m[7]);
    g(v[0], v[5], v[10], v[15], m[8],  m[9]);
    g(v[1], v[6], v[11], v[12], m[10], m[11]);
    g(v[2], v[7], v[8],  v[13], m[12], m[13]);
    g(v[3], v[4], v[9],  v[14], m[14], m[15]);
}

// BLAKE3's message permutation; unrolled, it is only register renaming.
__device__ __forceinline__ void permute(uint32_t m[16]) {
    const uint32_t t0 = m[2], t1 = m[6], t2 = m[3], t3 = m[10], t4 = m[7], t5 = m[0],
                   t6 = m[4], t7 = m[13], t8 = m[1], t9 = m[11], t10 = m[12], t11 = m[5],
                   t12 = m[9], t13 = m[14], t14 = m[15], t15 = m[8];
    m[0] = t0;   m[1] = t1;   m[2] = t2;   m[3] = t3;
    m[4] = t4;   m[5] = t5;   m[6] = t6;   m[7] = t7;
    m[8] = t8;   m[9] = t9;   m[10] = t10; m[11] = t11;
    m[12] = t12; m[13] = t13; m[14] = t14; m[15] = t15;
}

// cv = compress(cv, m, counter, block_len, flags), first 8 output words (all a hash needs).
// `m` is clobbered.
__device__ __forceinline__ void compress(uint32_t cv[8], uint32_t m[16], uint64_t counter,
                                         uint32_t block_len, uint32_t flags) {
    uint32_t v[16];
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = cv[i];
#pragma unroll
    for (int i = 0; i < 4; i++) v[8 + i] = iv(i);
    v[12] = (uint32_t) counter;
    v[13] = (uint32_t) (counter >> 32);
    v[14] = block_len;
    v[15] = flags;
#pragma unroll
    for (int r = 0; r < 7; r++) {
        round_fn(v, m);
        if (r < 6) permute(m);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = v[i] ^ v[i + 8];
}

// Parent node of two chaining values (`merge_subtrees_non_root` / `_root`).
__device__ __forceinline__ void merge(const uint32_t l[8], const uint32_t r[8], bool root,
                                      uint32_t out[8]) {
    uint32_t m[16], cv[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        m[i]     = l[i];
        m[8 + i] = r[i];
        cv[i]    = iv(i);
    }
    compress(cv, m, 0, 64, PARENT | (root ? ROOT : 0u));
#pragma unroll
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}

// Reads a byte string made of pieces: the current piece (p, rem), then `rows_left` rows of
// `row_len` bytes, the next one at `next_row`, each `stride` after the previous. A strided
// message is two pieces, a streamed append the pending bytes and its rows.
struct cursor {
    const uint8_t* p;
    uint64_t       rem;
    const uint8_t* next_row;
    int64_t        stride;
    uint32_t       row_len;
    uint32_t       rows_left;

    __device__ __forceinline__ void skip_empty() {
        while (rem == 0 && rows_left != 0) {
            p   = next_row;
            rem = row_len;
            next_row += stride;
            rows_left--;
        }
    }
};

// Read-only loads that ask L2 for the whole 128-byte line. One thread walks one 1 KiB chunk, so
// a warp's lanes sit 1 KiB apart and plain loads fetch each line sector by sector: 341 GB/s
// against 837 GB/s with the hint on a 5070 Ti (memory-only probe, 512 MB).
__device__ __forceinline__ uint4 ld_v4(const void* p) {
    uint4 v;
    asm("ld.global.nc.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint2 ld_v2(const void* p) {
    uint2 v;
    asm("ld.global.nc.L2::128B.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t ld_u32(const void* p) {
    uint32_t v;
    asm("ld.global.nc.L2::128B.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

// Loads 64 contiguous bytes at p by its alignment; 16-byte loads in the common case.
__device__ __forceinline__ void load64(const uint8_t* p, uint32_t m[16]) {
    const uintptr_t a = (uintptr_t) p;
    if ((a & 15) == 0) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint4 q = ld_v4(p + 16 * i);
            m[4 * i] = q.x; m[4 * i + 1] = q.y; m[4 * i + 2] = q.z; m[4 * i + 3] = q.w;
        }
    } else if ((a & 7) == 0) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint2 q = ld_v2(p + 8 * i);
            m[2 * i] = q.x; m[2 * i + 1] = q.y;
        }
    } else if ((a & 3) == 0) {
#pragma unroll
        for (int i = 0; i < 16; i++) m[i] = ld_u32(p + 4 * i);
    } else if ((a & 1) == 0) {
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint32_t lo = __ldg((const unsigned short*) p + 2 * i);
            const uint32_t hi = __ldg((const unsigned short*) p + 2 * i + 1);
            m[i] = lo | (hi << 16);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint32_t b0 = __ldg(p + 4 * i), b1 = __ldg(p + 4 * i + 1);
            const uint32_t b2 = __ldg(p + 4 * i + 2), b3 = __ldg(p + 4 * i + 3);
            m[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
    }
}

// Stage bytes [k, k + n) = bytes [p, p + n) (k + n <= 64). Every aligned source word the range
// touches is loaded at once, so a piece costs one memory round trip whatever its alignment; a
// byte loop waits on each load in turn. Such a word may hold bytes outside the range but never
// leaves the 4-byte granule of a byte inside it.
__device__ __forceinline__ void gather(uint32_t* stage, uint32_t k, const uint8_t* p, uint32_t n) {
    const uint32_t  off = (uint32_t) ((uintptr_t) p & 3);
    const uint32_t* a   = (const uint32_t*) (p - off);
    const int       e   = (int) off - (int) k;  // dest byte 4 * dw lies at a + e + 4 * dw
    const int       ew  = e >> 2;               // floor: e may be negative
    const uint32_t  sh  = (uint32_t) (e & 3) * 8;
    const int       jmax = (int) ((off + n - 1) >> 2);
    uint32_t x[17];
#pragma unroll
    for (int i = 0; i < 17; i++) {
        const int j = ew + i;
        x[i] = j >= 0 && j <= jmax ? __ldg(a + j) : 0u;
    }
#pragma unroll
    for (int dw = 0; dw < 16; dw++) {
        const int lo = max((int) k, 4 * dw) - 4 * dw, hi = min((int) (k + n), 4 * dw + 4) - 4 * dw;
        if (lo < hi) {
            const uint32_t v    = __funnelshift_r(x[dw], x[dw + 1], sh);
            const uint32_t mask = (uint32_t) (((1ull << (8 * hi)) - 1) & ~((1ull << (8 * lo)) - 1));
            stage[dw] = (stage[dw] & ~mask) | (v & mask);
        }
    }
}

// Next block of `len` bytes (1..64, zero-padded). A block that crosses pieces or ends short is
// assembled in this thread's 16 words of shared `stage`, where a dynamic index costs no local
// memory; its bytes are left there too.
__device__ __forceinline__ void load_block(cursor& c, uint32_t len, uint32_t m[16],
                                           uint32_t* stage) {
    c.skip_empty();
    if (len == 64 && c.rem >= 64) {
        load64(c.p, m);
        c.p += 64;
        c.rem -= 64;
        return;
    }
#pragma unroll
    for (int i = 0; i < 16; i++) stage[i] = 0;
    uint32_t k = 0;
    while (k < len) {
        c.skip_empty();
        if (c.rem == 0) break;  // the caller asked past the end: the rest stays zero
        const uint32_t n = (uint32_t) min((uint64_t) (len - k), c.rem);
        gather(stage, k, c.p, n);
        k += n;
        c.p += n;
        c.rem -= n;
    }
#pragma unroll
    for (int i = 0; i < 16; i++) m[i] = stage[i];
}

// Chaining value of one chunk of `len` bytes (0..1024) read from `c`: counter = chunk index.
// `last_flags` = ROOT makes it the hash of a message of at most one chunk.
__device__ __forceinline__ void chunk(cursor& c, uint32_t len, uint64_t counter, uint32_t last_flags,
                                      uint32_t* stage, uint32_t cv[8]) {
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = iv(i);
    const uint32_t nblocks = len == 0 ? 1 : (len + 63) / 64;
    uint32_t b = 0;
    c.skip_empty();
    if (len >= 64 && c.rem >= len) {
        // Contiguous: the next block's load is in flight while this one compresses, so a thread
        // waits on memory once per chunk instead of once per block.
        const uint32_t nfull = len / 64;
        uint32_t m[16];
        load64(c.p, m);
        for (; b < nfull; b++) {
            uint32_t nx[16];
            if (b + 1 < nfull) load64(c.p + 64 * (b + 1), nx);
            const uint32_t flags = (b == 0 ? CHUNK_START : 0u) | (b + 1 == nblocks ? CHUNK_END | last_flags : 0u);
            compress(cv, m, counter, 64, flags);
#pragma unroll
            for (int i = 0; i < 16; i++) m[i] = nx[i];
        }
        c.p += 64 * nfull;
        c.rem -= 64 * nfull;
    }
    for (; b < nblocks; b++) {
        const uint32_t blen = min(64u, len - 64 * b);
        uint32_t m[16];
        if (blen == 0) {
#pragma unroll
            for (int i = 0; i < 16; i++) m[i] = 0;
        } else {
            load_block(c, blen, m, stage);
        }
        const uint32_t flags = (b == 0 ? CHUNK_START : 0u) | (b + 1 == nblocks ? CHUNK_END | last_flags : 0u);
        compress(cv, m, counter, blen, flags);
    }
}

// Node arrays in shared memory are word-major ([8][cap]) so a warp's lanes hit 32 banks.
__device__ __forceinline__ void node_get(const uint32_t* a, uint32_t cap, uint32_t i, uint32_t v[8]) {
#pragma unroll
    for (int w = 0; w < 8; w++) v[w] = a[w * cap + i];
}

__device__ __forceinline__ void node_put(uint32_t* a, uint32_t cap, uint32_t i, const uint32_t v[8]) {
#pragma unroll
    for (int w = 0; w < 8; w++) a[w * cap + i] = v[w];
}

// Pushes level-0 nodes [c0, c0 + cnt) (in `cur`, capacity `cap`) into the level-indexed stack
// `stk` (shared, [STACK_LEVELS][8]) by merging whole levels in parallel. Node i of level k+1 is
// merge(node 2i, node 2i+1); a left child older than c0 is the stack entry of level k. Every
// merge is non-root: the caller keeps the last node out until it folds the stack. `nxt` holds
// cap / 2 + 1 nodes. Every thread of the block calls it; `cur` is clobbered.
__device__ __forceinline__ void absorb(uint32_t (*stk)[8], uint64_t c0, uint32_t cnt,
                                       uint32_t* cur, uint32_t* nxt, uint32_t cap) {
    const uint64_t c1 = c0 + cnt;
    uint32_t cur_cap = cap, nxt_cap = cap / 2 + 1;
    for (int k = 0; k < STACK_LEVELS; k++) {
        const uint64_t ak = c0 >> k, bk = c1 >> k;
        if (ak == bk) break;
        const uint64_t an = ak >> 1, bn = bk >> 1;
        for (uint64_t i = an + threadIdx.x; i < bn; i += blockDim.x) {
            uint32_t l[8], r[8], o[8];
            if (2 * i >= ak) {
                node_get(cur, cur_cap, (uint32_t) (2 * i - ak), l);
            } else {
#pragma unroll
                for (int w = 0; w < 8; w++) l[w] = stk[k][w];
            }
            node_get(cur, cur_cap, (uint32_t) (2 * i + 1 - ak), r);
            merge(l, r, false, o);
            node_put(nxt, nxt_cap, (uint32_t) (i - an), o);
        }
        __syncthreads();
        // Bit k of c1 set: its newest level-k node becomes the stack entry.
        if ((bk & 1) && threadIdx.x < 8) stk[k][threadIdx.x] = cur[threadIdx.x * cur_cap + (uint32_t) (bk - 1 - ak)];
        __syncthreads();
        uint32_t* t = cur; cur = nxt; nxt = t;
        const uint32_t tc = cur_cap; cur_cap = nxt_cap; nxt_cap = tc;
    }
}

// Pushes node number `index` (nodes 0..index-1 already pushed) into the level-indexed stack,
// merging while the count carries. One thread; all merges non-root.
__device__ __forceinline__ void stack_push(uint32_t (*stk)[8], uint64_t index, uint32_t cv[8]) {
    for (int k = 0; k < STACK_LEVELS; k++) {
        if (((index >> k) & 1) == 0) {
#pragma unroll
            for (int w = 0; w < 8; w++) stk[k][w] = cv[w];
            return;
        }
        merge(stk[k], cv, false, cv);
    }
}

// Root over the stack of `n` pushed nodes and the last node `cv`: smallest subtree first, ROOT
// on the final merge. n == 0 leaves `cv` as it is.
template <typename Stack>
__device__ __forceinline__ void fold_stack(const Stack& stk, uint64_t n, uint32_t cv[8]) {
    for (int k = 0; k < STACK_LEVELS && (n >> k) != 0; k++) {
        if ((n >> k) & 1) {
            uint32_t l[8];
#pragma unroll
            for (int w = 0; w < 8; w++) l[w] = stk[k][w];
            merge(l, cv, (n >> (k + 1)) == 0, cv);
        }
    }
}

}  // namespace kb3
