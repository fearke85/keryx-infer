// Host side of the trace (ADR 0023): which tensor of a ubatch graph feeds which leaf of the
// position tree, and the capture nodes that hash it. Included by keryx_llama.cpp, which owns
// the context; the nodes run in ggml-cuda (keryx_trace_ops.cu).
//
// The leaf numbering mirrors the node's `chain.rs` (X of rank r at 2r, Y at 2r+1, the two
// residuals of layer l at 2np+2l and 2np+2l+1, router scores at 2np+2nl+l); the Rust side
// derives the same map from the same manifest and refuses a ubatch whose hashes disagree with it.
#pragma once

#include "keryx_blake3.h"
#include "keryx_trace_abi.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "llama-arch.h"
#include "llama-model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

extern "C" void keryx_trace_cpu_fun(ggml_tensor *, int, int, void *);

namespace keryx_trace {

constexpr uint32_t NONE = 0xFFFFFFFFu;
constexpr uint8_t ROLE_PRODUCT = 0, ROLE_OUTPUT = 2, ROLE_HEADS = 7;
constexpr uint8_t FORM_NONE = 0, FORM_SPLIT = 1, FORM_PAIR_GATE = 2, FORM_PAIR_UP = 3;
constexpr uint32_t UBATCH = 512, MMVQ_MAX_BATCH = 8;

// The attention and recurrent fields of a layer entry the sequence leaves need (manifest.rs
// `AttnLinks`, `RecurLinks`).
struct Attn {
    uint32_t n_head = 0, n_head_kv = 0, head_dim = 0, slot = 0, mla_rank = 0, v = NONE;
};
struct Recur {
    uint32_t kind = 0, slot = 0, n_v_heads = 0, n_k_heads = 0, head_dim = 0, interval = 0;
};

// One manifest entry, as much of it as capture needs (node `manifest.rs`, KXMANIF2).
struct Entry {
    std::string name;
    uint32_t wtype = 0;
    uint64_t k = 0, m = 0;
    uint32_t n_exp = 0;
    uint8_t role = 0, glu_op = 0, glu_form = 0;
    uint32_t partner = NONE, bias = NONE;
    uint8_t tag = 0;
    uint32_t layer = NONE;  // layer entries: the layer they describe
    bool has_attn = false, has_recur = false;
    Attn attn;
    Recur recur;
};

struct Manifest {
    std::vector<Entry> entries;
    uint32_t n_products = 0;
};

// Parses KXMANIF2 bytes; false on any malformed or truncated entry.
inline bool parse_manifest(const uint8_t * p, size_t len, Manifest & out) {
    size_t o = 0;
    auto take = [&](void * dst, size_t n) {
        if (o + n > len) return false;
        memcpy(dst, p + o, n);
        o += n;
        return true;
    };
    char magic[8];
    uint32_t n_entries = 0;
    if (!take(magic, 8) || memcmp(magic, "KXMANIF2", 8) != 0 || !take(&n_entries, 4) || !take(&out.n_products, 4)) {
        return false;
    }
    // Bytes of each chain extension after its tag (manifest.rs `TAG_*`).
    static const size_t ext_len[8] = {0, 13, 24, 48, 24 + 84, 48 + 84, 24 + 76, 48 + 76};
    for (uint32_t i = 0; i < n_entries; i++) {
        Entry e;
        uint16_t name_len = 0;
        if (!take(&name_len, 2) || o + name_len > len) return false;
        e.name.assign(reinterpret_cast<const char *>(p + o), name_len);
        o += name_len;
        uint8_t base[55];
        if (!take(base, sizeof(base))) return false;
        memcpy(&e.wtype, base + 0, 4);
        memcpy(&e.k, base + 4, 8);
        memcpy(&e.m, base + 12, 8);
        memcpy(&e.n_exp, base + 20, 4);
        e.role = base[44];
        e.glu_op = base[45];
        e.glu_form = base[46];
        memcpy(&e.partner, base + 47, 4);
        memcpy(&e.bias, base + 51, 4);
        uint8_t tag = 0;
        if (!take(&tag, 1) || tag > 7 || o + ext_len[tag] > len) return false;
        const uint8_t * ext = p + o;
        o += ext_len[tag];
        e.tag = tag;
        if (tag == 1) {
            memcpy(&e.layer, ext, 4);  // ProductLink.layer: the layer the product runs in (n_layers = the head)
        }
        if (tag >= 2) {
            auto u32 = [&](size_t word) {
                uint32_t v;
                memcpy(&v, ext + 4 * word, 4);
                return v;
            };
            e.layer = u32(0);
            const size_t at = (tag == 3 || tag == 5 || tag == 7) ? 12 : 6;  // past the MoE links
            if (tag == 4 || tag == 5) {
                e.has_attn = true;
                e.attn = {u32(at + 5), u32(at + 6), u32(at + 7), u32(at + 17), u32(at + 18), u32(at + 2)};
            } else if (tag == 6 || tag == 7) {
                e.has_recur = true;
                e.recur = {u32(at + 0), u32(at + 1), u32(at + 2), u32(at + 3), u32(at + 4), u32(at + 17)};
            }
        }
        out.entries.push_back(std::move(e));
    }
    return o == len && out.n_products <= out.entries.size();
}

// The residual and router tensors of one llama architecture, by the names its graph builder
// gives them (suffixed "-<layer>").
struct ArchNames {
    const char * resid_attn;
    const char * resid_out;
    const char * router;  // nullptr: no MoE layers
};

inline bool arch_names(llm_arch arch, ArchNames & out) {
    switch (arch) {
        case LLM_ARCH_QWEN35:
            out = {"attn_residual", "l_out", nullptr};
            return true;
        case LLM_ARCH_GLM4:
            out = {"ffn_inp", "l_out", nullptr};
            return true;
        case LLM_ARCH_GEMMA4:
            out = {"attn_out", "l_out", nullptr};
            return true;
        case LLM_ARCH_KIMI_LINEAR:
            out = {"ffn_inp", "l_out", "ffn_moe_logits"};
            return true;
        default:
            return false;
    }
}

// Per-tier facts the manifest does not carry (node params.rs, TRACE_CHAINS_H15).
struct Tier {
    uint32_t n_layers = 0;
    uint32_t n_used = 0;  // experts per token, 0 = dense
    uint32_t n_attn = 0, n_recur = 0, seq_heads = 0;  // TRACE_ATTN_H_NEXT
};

class Plan {
public:
    Manifest manifest;
    Tier tier;
    ArchNames names{};
    std::map<std::string, uint32_t> rank_of;  // product weight name -> rank
    std::vector<std::pair<uint32_t, Attn>> attn_layers;    // by layer, in layer order
    std::vector<std::pair<uint32_t, Recur>> recur_layers;  // by layer, in layer order
    keryx_trace_state state{};
    std::string error;  // why the last graph could not be traced
    // The device of each layer, then of the output (llama's layer placement); empty = unknown,
    // and decode then flushes every layer on its own.
    std::vector<const void *> layer_dev;

    uint32_t n_leaves() const {
        const uint32_t np = manifest.n_products, nl = tier.n_layers;
        return 2 * np + 2 * nl + (tier.n_used > 0 ? nl : 0) + 1;
    }
    uint32_t resid_leaf(uint32_t layer, bool out) const { return 2 * manifest.n_products + 2 * layer + (out ? 1 : 0); }
    uint32_t router_leaf(uint32_t layer) const { return 2 * manifest.n_products + 2 * tier.n_layers + layer; }
    // Sequence tree (node seq.rs): kcache(s,h) = 2(s·H + h), vcache = +1, record(s,h) = 2H·n_attn +
    // 2(s·H + h), checkpoint = +1, lse(s,h) = 2H(n_attn + n_recur) + s·H + h.
    uint32_t kcache_leaf(uint32_t slot) const { return 2 * slot * tier.seq_heads; }
    uint32_t record_leaf(uint32_t slot) const { return 2 * tier.seq_heads * tier.n_attn + 2 * slot * tier.seq_heads; }
    uint32_t lse_leaf(uint32_t slot) const {
        return 2 * tier.seq_heads * (tier.n_attn + tier.n_recur) + slot * tier.seq_heads;
    }

    bool init(const uint8_t * bytes, size_t len, const Tier & t, llm_arch arch) {
        if (!parse_manifest(bytes, len, manifest)) {
            error = "malformed manifest";
            return false;
        }
        if (!arch_names(arch, names)) {
            error = std::string("no trace names for architecture ") + llm_arch_name(arch);
            return false;
        }
        tier = t;
        for (uint32_t r = 0; r < manifest.n_products; r++) {
            rank_of[manifest.entries[r].name] = r;
        }
        for (const Entry & e : manifest.entries) {
            if (e.has_attn) attn_layers.push_back({e.layer, e.attn});
            if (e.has_recur) recur_layers.push_back({e.layer, e.recur});
        }
        std::sort(attn_layers.begin(), attn_layers.end(), [](auto & a, auto & b) { return a.first < b.first; });
        std::sort(recur_layers.begin(), recur_layers.end(), [](auto & a, auto & b) { return a.first < b.first; });
        for (size_t i = 0; i < attn_layers.size(); i++) {
            const Attn & a = attn_layers[i].second;
            if (a.slot != i || a.n_head_kv == 0 || a.n_head_kv > tier.seq_heads || a.n_head % a.n_head_kv != 0 ||
                a.head_dim % 32 != 0) {
                error = "attention layer " + std::to_string(attn_layers[i].first) + " does not fit the sequence tree";
                return false;
            }
        }
        for (size_t i = 0; i < recur_layers.size(); i++) {
            const Recur & r = recur_layers[i].second;
            if (r.slot != i || r.n_v_heads == 0 || r.n_v_heads > tier.seq_heads || r.interval == 0 || r.kind > 1) {
                error = "recurrent layer " + std::to_string(recur_layers[i].first) + " does not fit the sequence tree";
                return false;
            }
        }
        if (attn_layers.size() != tier.n_attn || recur_layers.size() != tier.n_recur) {
            error = "the manifest's attention and recurrent layers are not the tier's";
            return false;
        }
        state.n_leaves = n_leaves();
        state.row_cap = UBATCH;
        state.n_seq_leaves = 2 * tier.seq_heads * (tier.n_attn + tier.n_recur) + tier.seq_heads * tier.n_attn;
        for (auto & tb : state.tables) {
            tb.hashes = nullptr;
            tb.device = -1;
        }
        for (auto & sq : state.seq) {
            sq = {};
            sq.device = -1;
        }
        return true;
    }

    // Bytes the GDN kernel copies out per ubatch at most: every checkpoint a ubatch can end.
    uint64_t ckpt_bytes() const {
        uint64_t most = 0;
        for (const auto & [layer, r] : recur_layers) {
            const uint64_t n = UBATCH / r.interval + 1;
            most = std::max<uint64_t>(most, n * r.n_v_heads * (uint64_t) r.head_dim * r.head_dim * 4);
        }
        return most;
    }

    // One table per card the context runs on.
    bool alloc_tables(const std::vector<int> & devices) {
        const size_t bytes = (size_t) state.row_cap * state.n_leaves * KERYX_TRACE_HASH;
        for (size_t i = 0; i < devices.size() && i < KERYX_TRACE_MAX_DEVICES; i++) {
            void * p = nullptr;
            if (cudaSetDevice(devices[i]) != cudaSuccess || cudaMalloc(&p, bytes) != cudaSuccess) {
                error = "cannot allocate the leaf table on gpu " + std::to_string(devices[i]);
                return false;
            }
            state.tables[i].hashes = static_cast<uint8_t *>(p);
            state.tables[i].device = devices[i];
            keryx_trace_seq & sq = state.seq[i];
            sq.device = devices[i];
            sq.ckpt_bytes = ckpt_bytes();
            void * streams = nullptr;
            void * pos0 = nullptr;
            void * ckpt = nullptr;
            if (cudaMalloc(&streams, std::max<size_t>(1, state.n_seq_leaves) * sizeof(kb3::stream_state)) != cudaSuccess ||
                cudaMalloc(&pos0, sizeof(int32_t)) != cudaSuccess ||
                (sq.ckpt_bytes && cudaMalloc(&ckpt, sq.ckpt_bytes) != cudaSuccess)) {
                error = "cannot allocate the sequence leaves on gpu " + std::to_string(devices[i]);
                return false;
            }
            sq.streams = streams;
            sq.pos0 = static_cast<int32_t *>(pos0);
            sq.ckpt = static_cast<float *>(ckpt);
            void * pos_off = nullptr;
            void * seq_off = nullptr;
            void * slots = nullptr;
            if (cudaMalloc(&pos_off, pos_off_bytes()) != cudaSuccess ||
                cudaMalloc(&seq_off, std::max<size_t>(1, state.n_seq_leaves) * sizeof(int64_t)) != cudaSuccess ||
                cudaMalloc(&slots, 2 * sizeof(uint8_t *)) != cudaSuccess) {
                error = "cannot allocate the replay offsets on gpu " + std::to_string(devices[i]);
                return false;
            }
            sq.pos_off = static_cast<int64_t *>(pos_off);
            sq.seq_off = static_cast<int64_t *>(seq_off);
            sq.arena = static_cast<uint8_t **>(slots);
            sq.seq_arena = sq.arena + 1;
            if (!reset_dumps(sq)) {
                error = "cannot clear the replay offsets on gpu " + std::to_string(devices[i]);
                return false;
            }
        }
        return true;
    }

    size_t pos_off_bytes() const { return (size_t) state.row_cap * state.n_leaves * sizeof(int64_t); }

    // No leaf wanted, no arena: every dump copy is a no-op.
    bool reset_dumps(keryx_trace_seq & sq) {
        const uint8_t * none[2] = {nullptr, nullptr};
        return cudaSetDevice(sq.device) == cudaSuccess && cudaMemset(sq.pos_off, 0xFF, pos_off_bytes()) == cudaSuccess &&
               cudaMemset(sq.seq_off, 0xFF, std::max<size_t>(1, state.n_seq_leaves) * sizeof(int64_t)) == cudaSuccess &&
               cudaMemcpy(sq.arena, none, sizeof(none), cudaMemcpyHostToDevice) == cudaSuccess &&
               cudaDeviceSynchronize() == cudaSuccess;
    }

    // Replay: the arenas live in pinned host memory every card writes through (UVA), so a card
    // that holds a whole tier needs no VRAM for them and the cards' copies land in one place.
    uint8_t * arena_host = nullptr;
    uint8_t * seq_arena_host = nullptr;

    bool replay_begin(uint64_t pos_bytes, uint64_t seq_bytes, const std::vector<std::pair<uint32_t, int64_t>> & seq_wants) {
        std::vector<int64_t> offs(std::max<size_t>(1, state.n_seq_leaves), -1);
        for (const auto & [leaf, off] : seq_wants) {
            if (leaf >= state.n_seq_leaves) return false;
            offs[leaf] = off;
        }
        void * a = nullptr;
        void * b = nullptr;
        const unsigned flags = cudaHostAllocMapped | cudaHostAllocPortable;
        if (cudaHostAlloc(&a, std::max<uint64_t>(1, pos_bytes), flags) != cudaSuccess ||
            cudaHostAlloc(&b, std::max<uint64_t>(1, seq_bytes), flags) != cudaSuccess) {
            if (a) cudaFreeHost(a);
            return false;
        }
        arena_host = static_cast<uint8_t *>(a);
        seq_arena_host = static_cast<uint8_t *>(b);
        memset(a, 0, std::max<uint64_t>(1, pos_bytes));
        memset(b, 0, std::max<uint64_t>(1, seq_bytes));
        for (auto & sq : state.seq) {
            if (!sq.streams) continue;
            uint8_t * slots[2] = {arena_host, seq_arena_host};
            if (cudaSetDevice(sq.device) != cudaSuccess ||
                cudaMemcpy(sq.arena, slots, sizeof(slots), cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(sq.seq_off, offs.data(), offs.size() * sizeof(int64_t), cudaMemcpyHostToDevice) != cudaSuccess) {
                return false;
            }
        }
        return true;
    }

    // The leaves wanted in the next ubatch: offs = [rows][n_leaves], -1 = not wanted.
    bool replay_rows(const std::vector<int64_t> & offs) {
        for (auto & sq : state.seq) {
            if (!sq.streams) continue;
            if (cudaSetDevice(sq.device) != cudaSuccess ||
                cudaMemcpy(sq.pos_off, offs.data(), offs.size() * sizeof(int64_t), cudaMemcpyHostToDevice) != cudaSuccess) {
                return false;
            }
        }
        return true;
    }

    // Copies the arenas out, frees them and turns the copies off again.
    bool replay_end(uint8_t * pos_out, uint64_t pos_bytes, uint8_t * seq_out, uint64_t seq_bytes) {
        bool ok = true;
        for (auto & sq : state.seq) {
            if (!sq.streams) continue;
            ok = cudaSetDevice(sq.device) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess && reset_dumps(sq) && ok;
        }
        if (pos_out && arena_host) memcpy(pos_out, arena_host, pos_bytes);
        if (seq_out && seq_arena_host) memcpy(seq_out, seq_arena_host, seq_bytes);
        if (arena_host) cudaFreeHost(arena_host);
        if (seq_arena_host) cudaFreeHost(seq_arena_host);
        arena_host = seq_arena_host = nullptr;
        return ok;
    }

    // Empties every sequence leaf: a new generation starts at position 0.
    bool begin() {
        for (auto & sq : state.seq) {
            if (!sq.streams) continue;
            if (cudaSetDevice(sq.device) != cudaSuccess ||
                kb3::stream_reset((kb3::stream_state *) sq.streams, state.n_seq_leaves, nullptr) != cudaSuccess ||
                cudaDeviceSynchronize() != cudaSuccess) {
                return false;
            }
        }
        return true;
    }

    // The position of the next ubatch's first row, which the recurrent checkpoints count from.
    bool set_pos0(int32_t pos) {
        for (auto & sq : state.seq) {
            if (!sq.pos0) continue;
            if (cudaSetDevice(sq.device) != cudaSuccess ||
                cudaMemcpy(sq.pos0, &pos, sizeof(pos), cudaMemcpyHostToDevice) != cudaSuccess) {
                return false;
            }
        }
        return true;
    }

    // BLAKE3 of every sequence leaf, after the last ubatch. A leaf streams on the card that ran
    // its layer; a leaf no card wrote (a head the layer does not have) is H("").
    bool finish(std::vector<uint8_t> & out) {
        const uint32_t n = state.n_seq_leaves;
        out.assign((size_t) n * KERYX_TRACE_HASH, 0);
        std::vector<uint8_t> hashes((size_t) n * KERYX_TRACE_HASH);
        std::vector<uint64_t> lens(n);
        std::vector<int> owner(n, -1);
        std::vector<bool> filled(n, false);
        for (int d = 0; d < KERYX_TRACE_MAX_DEVICES; d++) {
            keryx_trace_seq & sq = state.seq[d];
            if (!sq.streams) continue;
            void * dev = nullptr;
            const bool ok = cudaSetDevice(sq.device) == cudaSuccess && cudaMalloc(&dev, hashes.size()) == cudaSuccess &&
                kb3::stream_finalize((const kb3::stream_state *) sq.streams, n, dev, KERYX_TRACE_HASH, nullptr) == cudaSuccess &&
                cudaMemcpy(hashes.data(), dev, hashes.size(), cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy2D(lens.data(), sizeof(uint64_t), (const char *) sq.streams + offsetof(kb3::stream_state, total_len),
                             sizeof(kb3::stream_state), sizeof(uint64_t), n, cudaMemcpyDeviceToHost) == cudaSuccess;
            if (dev) cudaFree(dev);
            if (!ok) return false;
            for (uint32_t i = 0; i < n; i++) {
                if (lens[i] > 0) {
                    if (owner[i] >= 0) return false;  // two cards streamed the same leaf
                    owner[i] = d;
                } else if (owner[i] >= 0 || filled[i]) {
                    continue;
                }
                memcpy(&out[(size_t) i * KERYX_TRACE_HASH], &hashes[(size_t) i * KERYX_TRACE_HASH], KERYX_TRACE_HASH);
                filled[i] = true;
            }
        }
        return true;
    }

    void free_tables() {
        for (auto & tb : state.tables) {
            if (tb.hashes) {
                cudaSetDevice(tb.device);
                cudaFree(tb.hashes);
            }
            tb.hashes = nullptr;
            tb.device = -1;
        }
        for (auto & sq : state.seq) {
            if (sq.device >= 0) {
                cudaSetDevice(sq.device);
                for (void * p : {sq.streams, (void *) sq.pos0, (void *) sq.ckpt, (void *) sq.pos_off, (void *) sq.seq_off,
                                 (void *) sq.arena}) {
                    if (p) cudaFree(p);
                }
            }
            sq = {};
            sq.device = -1;
        }
    }

    // Zeroes every table: a leaf still zero after the ubatch was not captured.
    bool clear_tables() {
        const size_t bytes = (size_t) state.row_cap * state.n_leaves * KERYX_TRACE_HASH;
        for (auto & tb : state.tables) {
            if (!tb.hashes) continue;
            if (cudaSetDevice(tb.device) != cudaSuccess || cudaMemset(tb.hashes, 0, bytes) != cudaSuccess ||
                cudaDeviceSynchronize() != cudaSuccess) {
                return false;
            }
        }
        return true;
    }

    // The rows of the ubatch just computed, every table merged: a leaf comes from one card.
    // False when two cards wrote the same leaf.
    bool read_tables(uint32_t rows, std::vector<uint8_t> & out) {
        const size_t bytes = (size_t) rows * state.n_leaves * KERYX_TRACE_HASH;
        out.assign(bytes, 0);
        std::vector<uint8_t> part(bytes);
        bool first = true;
        for (auto & tb : state.tables) {
            if (!tb.hashes) continue;
            if (cudaSetDevice(tb.device) != cudaSuccess ||
                cudaMemcpy(first ? out.data() : part.data(), tb.hashes, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
                return false;
            }
            if (!first) {
                for (size_t h = 0; h < bytes; h += KERYX_TRACE_HASH) {
                    if (!written(part.data() + h)) continue;
                    if (written(out.data() + h)) return false;
                    memcpy(out.data() + h, part.data() + h, KERYX_TRACE_HASH);
                }
            }
            first = false;
        }
        return true;
    }

    static bool written(const uint8_t * h) {
        for (int i = 0; i < KERYX_TRACE_HASH; i++) {
            if (h[i]) return true;
        }
        return false;
    }

    // Adds the capture nodes to a ubatch graph of `n_tokens` rows. False (with `error`) when a
    // product or residual the manifest names is not where the plan looks for it.
    bool extend(ggml_context * ctx, ggml_cgraph * gf, uint32_t n_tokens, ggml_backend_sched_t sched = nullptr) {
        error.clear();
        sched_ = sched;
        // A generated token runs alone: its captures stage their bytes and each layer hashes
        // them in one launch (keryx_trace_ops.cu), instead of one launch per leaf.
        defer_ = n_tokens == 1;
        const int n = ggml_graph_n_nodes(gf);
        std::vector<ggml_tensor *> nodes(n);
        std::map<const ggml_tensor *, int> index;
        for (int i = 0; i < n; i++) {
            nodes[i] = ggml_graph_node(gf, i);
            index[nodes[i]] = i;
        }
        // consumers[t] = nodes reading t or a view of it, in graph order.
        std::map<const ggml_tensor *, std::vector<int>> consumers;
        for (int i = 0; i < n; i++) {
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                const ggml_tensor * src = nodes[i]->src[s];
                if (!src) continue;
                consumers[src].push_back(i);
                if (src->view_src) consumers[src->view_src].push_back(i);
            }
        }
        std::vector<Capture> caps;
        std::vector<bool> layer_seen(2 * tier.n_layers, false);
        size_t n_fa = 0, n_gdn = 0, n_kv = 0;

        for (int i = 0; i < n && error.empty(); i++) {
            ggml_tensor * node = nodes[i];
            uint32_t layer = 0;
            bool out = false;
            // The router scores are the output of the router's own product (not a trace product).
            if (names.router && tier.n_used > 0 && layer_name(node->name, names.router, layer) && !is_view(node)) {
                caps.push_back({f32_op(router_leaf(layer), rows_of(node), n_tokens), {node}});
                continue;
            }
            if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
                const auto r = rank_of.find(node->src[0]->name);
                if (r != rank_of.end()) product(r->second, node, nodes, consumers, n_tokens, caps);
                continue;
            }
            // Attention layers run FLASH_ATTN_EXT once each, in layer order; recurrent layers
            // GATED_DELTA_NET.
            if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                if (n_fa >= attn_layers.size()) {
                    error = "more attention ops than attention layers";
                    break;
                }
                const Attn & a = attn_layers[n_fa++].second;
                keryx_trace_op * op = seq_op(KERYX_TRACE_LSE, lse_leaf(a.slot), 1, a.n_head_kv, a.head_dim);
                op->group = a.n_head / a.n_head_kv;
                caps.push_back({op, {node, node->src[0], node->src[1], node->src[3]}});
                continue;
            }
            if (node->op == GGML_OP_GATED_DELTA_NET) {
                if (n_gdn >= recur_layers.size()) {
                    error = "more recurrent ops than recurrent layers";
                    break;
                }
                const Recur & r = recur_layers[n_gdn++].second;
                keryx_trace_op * op = seq_op(KERYX_TRACE_RECUR, record_leaf(r.slot), 2, r.n_v_heads, r.head_dim);
                op->interval = r.interval;
                op->recur_kind = r.kind;
                keryx_trace_gdn * kx = gdn_for(r.interval);
                memcpy((char *) node->op_params + KERYX_TRACE_GDN_PARAM, &kx, sizeof(kx));
                caps.push_back({op, {node, node->src[0], node->src[1], node->src[2], node->src[3], node->src[4]}});
                continue;
            }
            unsigned kv_layer = 0;
            char kv = 0;
            if (node->op == GGML_OP_SET_ROWS && sscanf(node->name, "cache_%c_l%u", &kv, &kv_layer) == 2 &&
                (kv == 'k' || kv == 'v')) {
                const Attn * a = attn_of(kv_layer);
                if (!a) {
                    error = std::string("cache write of a layer without attention: ") + node->name;
                    break;
                }
                if (kv == 'v' && a->mla_rank != 0) continue;  // a latent layer keeps no value leaf
                n_kv += kv == 'k';
                keryx_trace_op * op = seq_op(KERYX_TRACE_KV, kcache_leaf(a->slot) + (kv == 'v'), 2, a->n_head_kv, a->head_dim);
                caps.push_back({op, {node, node->src[1]}});
                continue;
            }
            if (resid_name(node->name, layer, out) && !is_view(node)) {
                const uint32_t slot = 2 * layer + (out ? 1 : 0);
                if (layer_seen[slot]) {
                    error = std::string("two tensors named ") + node->name;
                    break;
                }
                layer_seen[slot] = true;
                caps.push_back({f32_op(resid_leaf(layer, out), rows_of(node), n_tokens), {node}});
            }
        }
        for (uint32_t s = 0; s < layer_seen.size() && error.empty(); s++) {
            if (!layer_seen[s]) error = "no residual tensor for layer " + std::to_string(s / 2);
        }
        if (error.empty() && (n_fa != attn_layers.size() || n_gdn != recur_layers.size() || n_kv != attn_layers.size())) {
            error = "the graph's attention and recurrent ops are not the manifest's layers";
        }
        if (!error.empty()) return false;
        if (defer_ && !defer_plan(caps)) return false;
        return insert(ctx, gf, nodes, index, consumers, caps);
    }

private:
    struct Capture {
        keryx_trace_op * op;
        std::vector<ggml_tensor *> src;
        // A FLUSH node: the layers [flush_from, flush_layer] whose captures it hashes.
        int flush_layer = -1;
        int flush_from = -1;
    };

    bool defer_ = false;
    ggml_backend_sched_t sched_ = nullptr;

    // Pins a capture node to the card of its layer (layer_dev), as the scheduler would otherwise
    // place a node between two layers on either card: a decode capture stages its bytes on the card
    // it runs on, and its run's flush reads them there. A no-op when the placement is unknown.
    bool place(ggml_tensor * t, uint32_t layer) {
        if (!sched_ || layer_dev.size() != (size_t) tier.n_layers + 1) return true;
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched_); i++) {
            ggml_backend_t b = ggml_backend_sched_get_backend(sched_, i);
            if ((const void *) ggml_backend_get_device(b) == layer_dev[layer]) {
                ggml_backend_sched_set_tensor_backend(sched_, t, b);
                return true;
            }
        }
        error = "no backend runs the card of layer " + std::to_string(layer);
        return false;
    }

    // The layer a capture belongs to (n_layers = after the last layer: the LM head).
    uint32_t layer_of(const keryx_trace_op * op) const {
        const uint32_t np = manifest.n_products, nl = tier.n_layers, H = tier.seq_heads;
        switch (op->kind) {
            case KERYX_TRACE_X:
            case KERYX_TRACE_F32:
                if (op->leaf < 2 * np) {
                    const uint32_t l = manifest.entries[op->leaf / 2].layer;
                    return l < nl ? l : nl;
                }
                if (op->leaf < 2 * np + 2 * nl) return (op->leaf - 2 * np) / 2;
                return op->leaf - 2 * np - 2 * nl;
            case KERYX_TRACE_KV:
                return attn_layers[op->leaf / (2 * H)].first;
            case KERYX_TRACE_LSE:
                return attn_layers[(op->leaf - 2 * H * (tier.n_attn + tier.n_recur)) / H].first;
            case KERYX_TRACE_RECUR:
                return recur_layers[(op->leaf - 2 * H * tier.n_attn) / (2 * H)].first;
            default:
                return nl;
        }
    }

    // Decode staging: every capture's bytes get a place in the defer arena, every message a row
    // of the table the flush hashes, every sequence append a descriptor slot; then one FLUSH
    // capture per run of layers on one card (per layer when the placement is unknown). The arena
    // and tables are sized on the first decode graph and never move: a captured CUDA graph keeps
    // their addresses. A message longer than KERYX_DEFER_MAX_MSG is not staged.
    bool defer_plan(std::vector<Capture> & caps) {
        const uint32_t nl = tier.n_layers;
        std::vector<std::vector<size_t>> by_layer(nl + 1);
        for (size_t i = 0; i < caps.size(); i++) by_layer[layer_of(caps[i].op)].push_back(i);
        std::vector<keryx_trace_msg> msgs;
        uint64_t off = 0;
        uint32_t descs = 0;
        auto align = [](uint64_t v) { return (v + 255) & ~uint64_t(255); };
        const bool runs = layer_dev.size() == (size_t) nl + 1;
        std::vector<Capture> flushes;
        uint32_t first = 0, desc0 = 0;
        int run_from = -1;
        std::vector<keryx_trace_op *> run_ops;
        // MMVQ inputs by source and rows per leaf: X captures of one source (q, k and v; gate and
        // up) stage the same bytes once.
        std::map<std::pair<const ggml_tensor *, uint32_t>, uint64_t> x_staged;
        for (uint32_t l = 0; l <= nl; l++) {
            if (!by_layer[l].empty() && run_from < 0) run_from = (int) l;
            for (size_t i : by_layer[l]) {
                keryx_trace_op * op = caps[i].op;
                const ggml_tensor * src = caps[i].src[0];
                op->defer_off = (int64_t) off;
                op->desc = descs;
                op->alias = 0;
                uint64_t count = 0, len = 0, bytes = 0;
                if (op->kind == KERYX_TRACE_F32) {
                    if (op->heads_dim == 2) {
                        count = src->ne[1];
                        len = (uint64_t) src->ne[0] * src->ne[2] * 4;
                    } else if (caps[i].src.size() > 1) {
                        count = src->ne[2];
                        len = (uint64_t) src->ne[0] * src->ne[1] * 4 + (uint64_t) caps[i].src[1]->ne[0] * 4;
                    } else {
                        count = (uint64_t) rows_of(src);
                        len = (uint64_t) src->ne[0] * 4;
                    }
                } else if (op->kind == KERYX_TRACE_X) {
                    const uint64_t kp = (src->ne[0] + 511) / 512 * 512;
                    const uint64_t row = op->layout == KERYX_TRACE_MMVQ ? kp / 32 * 36 : kp / 128 * 144;
                    count = (uint64_t) rows_of(src) / op->per_leaf;
                    len = row * op->per_leaf;
                    if (op->layout == KERYX_TRACE_MMVQ && op->heads_dim != 2 && len <= (uint64_t) KERYX_DEFER_MAX_MSG) {
                        const auto key = std::make_pair(src, op->per_leaf);
                        const auto at = x_staged.find(key);
                        if (at != x_staged.end()) {
                            op->defer_off = (int64_t) at->second;
                            op->alias = 1;
                            for (uint64_t j = 0; j < count; j++) {
                                msgs.push_back({at->second + j * len, (uint32_t) len,
                                                (op->row0 + (uint32_t) j) * state.n_leaves + op->leaf});
                            }
                            run_ops.push_back(op);
                            continue;
                        }
                        x_staged[key] = off;
                    }
                } else if (op->kind == KERYX_TRACE_KV) {
                    bytes = (uint64_t) op->n_heads * caps[i].src[1]->ne[0] * (op->head_dim / 32 * 34);
                    descs += op->n_heads;
                } else if (op->kind == KERYX_TRACE_LSE) {
                    bytes = (uint64_t) caps[i].src[1]->ne[1] * op->n_heads * op->group * 4;
                    descs += op->n_heads;
                } else if (op->kind == KERYX_TRACE_RECUR) {
                    const uint64_t rl = 8ull * op->head_dim + 4 + (op->recur_kind ? 2ull * op->head_dim : 4);
                    bytes = (uint64_t) op->n_heads * caps[i].src[3]->ne[2] * rl;
                    descs += op->n_heads;  // the checkpoints append where they are captured
                }
                if (len > (uint64_t) KERYX_DEFER_MAX_MSG) {
                    op->defer_off = -1;
                    continue;
                }
                for (uint64_t j = 0; j < count; j++) {
                    msgs.push_back({off + j * len, (uint32_t) len, (op->row0 + (uint32_t) j) * state.n_leaves + op->leaf});
                }
                run_ops.push_back(op);
                off = align(off + count * len + bytes);
            }
            const bool run_ends = !runs || l == nl || layer_dev[l + 1] != layer_dev[l];
            if (run_from < 0 || !run_ends) continue;
            keryx_trace_op * fl = op_for(KERYX_TRACE_FLUSH, l, 0, 0, 1, 0);
            fl->first = first;
            fl->count = (uint32_t) msgs.size() - first;
            fl->desc = desc0;
            fl->n_descs = descs - desc0;
            auto & members = flush_members[fl];
            members = run_ops;
            fl->members = members.data();
            fl->n_members = (uint32_t) members.size();
            Capture f{fl, {}, (int) l, run_from};
            flushes.push_back(f);
            first = (uint32_t) msgs.size();
            desc0 = descs;
            run_from = -1;
            run_ops.clear();
            x_staged.clear();
        }
        caps.insert(caps.end(), flushes.begin(), flushes.end());
        for (auto & sq : state.seq) {
            if (!sq.streams) continue;
            if (cudaSetDevice(sq.device) != cudaSuccess) return false;
            if (!sq.defer) {
                // First decode graph: size everything once, with no room to grow.
                void * a = nullptr;
                void * m = nullptr;
                void * d = nullptr;
                if (cudaMalloc(&a, std::max<uint64_t>(off, 256)) != cudaSuccess ||
                    cudaMalloc(&m, std::max<size_t>(msgs.size(), 1) * sizeof(keryx_trace_msg)) != cudaSuccess ||
                    cudaMalloc(&d, std::max<uint32_t>(descs, 1) * sizeof(kb3::append_desc)) != cudaSuccess) {
                    error = "cannot allocate the decode staging on gpu " + std::to_string(sq.device);
                    return false;
                }
                sq.defer = static_cast<uint8_t *>(a);
                sq.defer_cap = std::max<uint64_t>(off, 256);
                sq.msgs = static_cast<keryx_trace_msg *>(m);
                sq.msgs_cap = (uint32_t) std::max<size_t>(msgs.size(), 1);
                sq.descs = d;
                sq.descs_cap = std::max<uint32_t>(descs, 1);
            }
            if (off > sq.defer_cap || msgs.size() > sq.msgs_cap || descs > sq.descs_cap) {
                error = "the decode graph outgrew its staging";
                return false;
            }
            if (!msgs.empty() &&
                cudaMemcpy(sq.msgs, msgs.data(), msgs.size() * sizeof(keryx_trace_msg), cudaMemcpyHostToDevice) != cudaSuccess) {
                return false;
            }
        }
        return true;
    }

    std::map<uint32_t, std::unique_ptr<keryx_trace_gdn>> gdns;
    std::map<const keryx_trace_op *, std::vector<keryx_trace_op *>> flush_members;

    keryx_trace_gdn * gdn_for(uint32_t interval) {
        auto & slot = gdns[interval];
        if (!slot) {
            slot = std::make_unique<keryx_trace_gdn>();
            *slot = {KERYX_TRACE_MAGIC, interval, &state};
        }
        return slot.get();
    }

    const Attn * attn_of(uint32_t layer) const {
        for (const auto & [l, a] : attn_layers) {
            if (l == layer) return &a;
        }
        return nullptr;
    }

    keryx_trace_op * seq_op(uint32_t kind, uint32_t leaf, uint32_t step, uint32_t heads, uint32_t hd) {
        keryx_trace_op * op = op_for(kind, leaf, 0, 0, 1, 0);
        op->leaf_step = step;
        op->n_heads = heads;
        op->head_dim = hd;
        return op;
    }

    // Ops live as long as the plan, one per distinct capture, so a graph rebuilt with the same
    // shape names the same pointers and a captured CUDA graph stays valid.
    // A decode op stages (defer_), the same op in a prompt graph hashes: two keys.
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool>, std::unique_ptr<keryx_trace_op>> ops;

    keryx_trace_op * op_for(uint32_t kind, uint32_t leaf, uint32_t layout, int32_t wtype, uint32_t per_leaf, uint32_t row0) {
        auto & slot = ops[{kind, leaf, layout, per_leaf, row0, defer_}];
        if (!slot) {
            slot = std::make_unique<keryx_trace_op>();
            *slot = {};
            slot->magic = KERYX_TRACE_MAGIC;
            slot->kind = kind;
            slot->leaf = leaf;
            slot->layout = layout;
            slot->wtype = wtype;
            slot->per_leaf = per_leaf;
            slot->row0 = row0;
            slot->defer_off = -1;
            slot->state = &state;
            slot->ran_on = -1;
        }
        return slot.get();
    }

    keryx_trace_op * f32_op(uint32_t leaf, int64_t rows, uint32_t n_tokens) {
        return op_for(KERYX_TRACE_F32, leaf, 0, 0, 1, (uint32_t) (n_tokens - rows));
    }

    static bool is_view(const ggml_tensor * t) {
        return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
    }

    static int64_t rows_of(const ggml_tensor * t) { return t->ne[1] * t->ne[2] * t->ne[3]; }

    // "<prefix>-<layer>" with a layer of this tier.
    bool layer_name(const char * name, const char * prefix, uint32_t & layer) const {
        const size_t len = strlen(prefix);
        if (strncmp(name, prefix, len) != 0 || name[len] != '-') return false;
        char * end = nullptr;
        const unsigned long l = strtoul(name + len + 1, &end, 10);
        if (end == name + len + 1 || *end != '\0' || l >= tier.n_layers) return false;
        layer = (uint32_t) l;
        return true;
    }

    bool resid_name(const char * name, uint32_t & layer, bool & out) const {
        for (int k = 0; k < 2; k++) {
            const char * prefix = k == 0 ? names.resid_attn : names.resid_out;
            const size_t len = strlen(prefix);
            if (strncmp(name, prefix, len) != 0 || name[len] != '-') continue;
            char * end = nullptr;
            const unsigned long l = strtoul(name + len + 1, &end, 10);
            if (end == name + len + 1 || *end != '\0' || l >= tier.n_layers) return false;
            layer = (uint32_t) l;
            out = k == 1;
            return true;
        }
        return false;
    }

    // The node consuming `t` (directly or through a view) with op `op`, or nullptr.
    static ggml_tensor * consumer(const ggml_tensor * t, ggml_op op, const std::vector<ggml_tensor *> & nodes,
                                  std::map<const ggml_tensor *, std::vector<int>> & consumers) {
        for (int i : consumers[t]) {
            if (nodes[i]->op == op) return nodes[i];
            if (is_view(nodes[i])) {
                if (ggml_tensor * c = consumer(nodes[i], op, nodes, consumers)) return c;
            }
        }
        return nullptr;
    }

    // Layout of a product's X for a ubatch of n rows (node verify.rs `input_layout`).
    static uint32_t layout_of(const Entry & e, uint32_t n_ubatch) {
        if (e.role == ROLE_OUTPUT || n_ubatch <= MMVQ_MAX_BATCH) return KERYX_TRACE_MMVQ;
        switch (e.wtype) {
            case 2: case 3: case 7: case 12: case 13:
                return KERYX_TRACE_MMQ_DS4;
            default:
                return KERYX_TRACE_MMQ_D4;
        }
    }

    void product(uint32_t rank, ggml_tensor * mm, const std::vector<ggml_tensor *> & nodes,
                 std::map<const ggml_tensor *, std::vector<int>> & consumers, uint32_t n_tokens, std::vector<Capture> & caps) {
        const Entry & e = manifest.entries[rank];
        if (e.role == ROLE_HEADS) {
            heads_product(rank, mm, n_tokens, caps);
            return;
        }
        const bool experts = mm->op == GGML_OP_MUL_MAT_ID;
        ggml_tensor * x = mm->src[1];
        ggml_tensor * ids = experts ? mm->src[2] : nullptr;
        if (x->type != GGML_TYPE_F32 || (uint64_t) x->ne[0] != e.k || mm->src[0]->type != (ggml_type) e.wtype ||
            experts != (e.n_exp > 1)) {
            error = "product " + e.name + " does not have the manifest's shape";
            return;
        }
        // X: one row per token, or the n_used inputs of a down-experts product per token.
        const uint32_t per_leaf = experts && x->ne[1] > 1 ? (uint32_t) x->ne[1] : 1;
        const int64_t leaves = rows_of(x) / per_leaf;
        caps.push_back({op_for(KERYX_TRACE_X, 2 * rank, layout_of(e, n_tokens), (int32_t) e.wtype, per_leaf,
                               (uint32_t) (n_tokens - leaves)),
                        {x}});
        // Y: what the product hands on (node verify.rs: Y includes the bias; a GLU pair commits
        // act(gate)*up on the gate's rank and nothing on the up's).
        ggml_tensor * y = mm;
        switch (e.glu_form) {
            case FORM_PAIR_UP:
                return;
            case FORM_PAIR_GATE:
            case FORM_SPLIT:
                y = consumer(mm, GGML_OP_GLU, nodes, consumers);
                break;
            case FORM_NONE:
                if (e.bias != NONE) {
                    y = nullptr;
                    const std::string & bias = manifest.entries[e.bias].name;
                    for (int i : consumers[mm]) {
                        ggml_tensor * c = nodes[i];
                        if (c->op == GGML_OP_ADD && c->src[1] && bias == c->src[1]->name) y = c;
                    }
                }
                break;
            default:
                y = nullptr;
        }
        // A split product's rows are gate then up; its Y is the m/2 values of act(gate)*up.
        const uint64_t width = e.glu_form == FORM_SPLIT ? e.m / 2 : e.m;
        if (!y || (uint64_t) y->ne[0] != width) {
            error = "no output tensor for product " + e.name;
            return;
        }
        const int64_t y_leaves = experts ? y->ne[2] * y->ne[3] : rows_of(y);
        caps.push_back({f32_op(2 * rank + 1, y_leaves, n_tokens), ids ? std::vector<ggml_tensor *>{y, ids} : std::vector<ggml_tensor *>{y}});
    }

    // A per-head product (ROLE_HEADS): weight [k, m, n_head], input [k, rows, n_head], output
    // [m, rows, n_head]. X holds the n_head input rows of a position, Y the n_head outputs
    // (node chainkit: head h at h·m).
    void heads_product(uint32_t rank, ggml_tensor * mm, uint32_t n_tokens, std::vector<Capture> & caps) {
        const Entry & e = manifest.entries[rank];
        ggml_tensor * x = mm->src[1];
        if (mm->op != GGML_OP_MUL_MAT || x->type != GGML_TYPE_F32 || (uint64_t) x->ne[0] != e.k ||
            (uint32_t) x->ne[2] != e.n_exp || (uint64_t) mm->ne[0] != e.m || mm->ne[2] != x->ne[2] ||
            mm->src[0]->type != (ggml_type) e.wtype) {
            error = "per-head product " + e.name + " does not have the manifest's shape";
            return;
        }
        keryx_trace_op * xo = op_for(KERYX_TRACE_X, 2 * rank, layout_of(e, n_tokens), (int32_t) e.wtype, e.n_exp,
                                     (uint32_t) (n_tokens - x->ne[1]));
        xo->heads_dim = 2;
        caps.push_back({xo, {x}});
        keryx_trace_op * yo = op_for(KERYX_TRACE_F32, 2 * rank + 1, 0, 0, e.n_exp, (uint32_t) (n_tokens - mm->ne[1]));
        yo->heads_dim = 2;
        caps.push_back({yo, {mm}});
    }

    // Rebuilds the node order with each capture just before the first later reader of what it
    // hashes: that is where a fused chain starts, never inside one, and no source lives longer.
    bool insert(ggml_context * ctx, ggml_cgraph * gf, const std::vector<ggml_tensor *> & nodes,
                std::map<const ggml_tensor *, int> & index, std::map<const ggml_tensor *, std::vector<int>> & consumers,
                std::vector<Capture> & caps) {
        const int n = (int) nodes.size();
        std::vector<std::vector<ggml_tensor *>> before(n + 1);
        // A flush goes after the last capture of its layers, on that capture's card.
        std::map<uint32_t, std::pair<int, ggml_tensor *>> layer_end;
        for (Capture & c : caps) {
            if (c.flush_layer >= 0) {
                std::pair<int, ggml_tensor *> end{-1, nullptr};
                for (int l = c.flush_from; l <= c.flush_layer; l++) {
                    const auto e = layer_end.find((uint32_t) l);
                    if (e != layer_end.end() && e->second.first >= end.first) end = e->second;
                }
                if (!end.second) {
                    error = "a flush has no capture";
                    return false;
                }
                ggml_tensor * src = end.second;
                ggml_tensor * t = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, &src, 1, keryx_trace_cpu_fun, 1, c.op);
                ggml_format_name(t, "kxtrace-flush-%d", c.flush_layer);
                if (!place(t, (uint32_t) c.flush_layer)) return false;
                before[end.first].push_back(t);
                continue;
            }
            int last = -1;
            for (ggml_tensor * s : c.src) {
                if (!s) continue;
                const auto at = index.find(s);
                if (at != index.end()) last = std::max(last, at->second);
            }
            int at = n;
            for (ggml_tensor * s : c.src) {
                if (!s) continue;
                for (int i : consumers[s]) {
                    if (i > last && i < at) at = i;
                }
            }
            if (last < 0) {
                error = "a capture has no source in the graph";
                return false;
            }
            ggml_tensor * t = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, c.src.data(), (int) c.src.size(),
                                             keryx_trace_cpu_fun, 1, c.op);
            ggml_format_name(t, "kxtrace-%u", c.op->leaf);
            if (!place(t, layer_of(c.op))) return false;
            before[at].push_back(t);
            if (defer_) {
                auto & e = layer_end[layer_of(c.op)];
                if (!e.second || at >= e.first) e = {at, c.src[0]};
            }
        }
        if (n + (int) caps.size() > ggml_graph_size(gf)) {
            error = "the graph has no room for the capture nodes";
            return false;
        }
        ggml_graph_clear(gf);
        for (int i = 0; i <= n; i++) {
            for (ggml_tensor * t : before[i]) ggml_build_forward_expand(gf, t);
            if (i < n) ggml_build_forward_expand(gf, nodes[i]);
        }
        return ggml_graph_n_nodes(gf) == n + (int) caps.size();
    }
};

}  // namespace keryx_trace
