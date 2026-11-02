// libkeryx-llama.{so,dylib} — the miner's in-process llama.cpp engine (Phase 2
// on CUDA, Phase 3b on Apple Silicon Metal).
//
// One llama.cpp instance per loaded model, for OPoI text generation. This tree exposes the
// single-card zero-dup path: `keryx_llama_load_bound` copies nothing, and the weights point into
// the miner's walk blob in the canonical layout (ADR 0012). Layer-split (`load_pair`) is not part
// of this project.
//
// The miner dlopens this next to its own binary; absent = inference is unavailable.
// Built by hiveos/build-keryx-llama.sh (CUDA) or hiveos/build-keryx-llama-macos.sh (Metal).
#include <cstdio>
#include <cstring>

#include "llama.h"
#include "llama-model.h"
#include "common.h"
#include "speculative.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include "llama-mmap.h"
#ifndef __APPLE__
#include "ggml-backend-impl.h"
#include "trace/keryx_trace_graph.hpp"
#endif
#ifdef __APPLE__
// Metal: llama.cpp's ggml-metal backend stores quantized tensors in unified-memory MTLBuffers.
// `t->data` is a CPU-readable pointer into that unified memory (also GPU-visible on Apple Silicon
// via the shared address space), so we don't need cudaPointerGetAttributes — `is_device` is
// always 1 for tensors llama.cpp reports.
#else
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <thread>
#include <mutex>
#include <string>
#include <vector>

// Windows DLLs export nothing by default — mark the ABI surface explicitly so the miner's
// GetProcAddress finds it. No-op on ELF/Mach-O (default visibility already exports).
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define KERYX_EXPORT __declspec(dllexport)
#define keryx_fseek _fseeki64
#else
#define KERYX_EXPORT
#define keryx_fseek fseeko
#endif

// Load failures are reported by a null return, which tells the miner nothing. Keep the reason
// (and the VRAM figures behind an OOM) for `keryx_llama_last_error`.
static thread_local std::string keryx_last_error;

static std::string keryx_cuda_diagnostics() {
#ifdef __APPLE__
    return std::string();
#else
    std::string out;
    const cudaError_t pending = cudaPeekAtLastError();
    if (pending != cudaSuccess) {
        out += std::string(" [cuda: ") + cudaGetErrorString(pending) + "]";
    }
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        out += " [vram: " + std::to_string(free_bytes / (1024 * 1024)) + " MiB free / "
             + std::to_string(total_bytes / (1024 * 1024)) + " MiB total]";
    }
    return out;
#endif
}

static void keryx_set_error(const char* stage, const std::string& detail) {
    keryx_last_error = std::string(stage) + ": " + detail + keryx_cuda_diagnostics();
}

struct KeryxLlama {
    llama_model*   model = nullptr;
    llama_context* ctx   = nullptr;
    llama_sampler* smpl  = nullptr;
    // A model with a multi-token-prediction layer (Qwen3.8) drafts tokens with it, and the
    // model checks them in one batch: `mtp` runs that layer, `spec` drives the drafts.
    llama_context*      mtp  = nullptr;
    common_speculative* spec = nullptr;
    std::vector<std::string> names; // canonical (byte-lexicographic) order — matches pom.rs
    std::mutex gen_lock;
    // The last generation, read under gen_lock by keryx_llama_last_stats.
    int    last_prompt_tokens = 0;
    int    last_tokens = 0;
    double last_prefill_ms = 0;
    double last_decode_ms = 0;
    int    last_drafted = 0;
    int    last_accepted = 0;
    // Zero-dup only: the buffers the weights were placed in, freed after the model, and the
    // mapping host tensors and copied tensors were read from.
    std::vector<ggml_backend_buffer_t> buffers;
    std::unique_ptr<llama_file> file;
    std::unique_ptr<llama_mmap> map;
    // The cards the model runs on, and whether some weights live in host memory.
    std::vector<int> gpus;
    bool host_weights = false;
    bool kv_q8 = false;  // the K cache is Q8_0, as the trace commits it
#ifndef __APPLE__
    // H15 trace (ADR 0023): set by keryx_llama_trace_init.
    std::unique_ptr<keryx_trace::Plan> trace;
#endif
};

// llama.cpp/ggml emit a large INFO-level dump on every model load (full tensor list, per-layer
// device assignment, kv-cache map, sched-reserve…). The miner's own Rust logging already covers
// what matters, so by default only WARN/ERROR are kept. KERYX_LLAMA_VERBOSE=1 also keeps INFO
// and the rest. The text stays in memory: fputs(stderr) paints the TUI, which redraws only the
// cells it believes changed. The miner drains the buffer with keryx_llama_drain_log.
static std::mutex keryx_log_mu;
static std::string keryx_log_buf;
static bool keryx_log_verbose = false;

static bool keryx_log_keep(enum ggml_log_level level) {
    if (keryx_log_verbose) return level != GGML_LOG_LEVEL_NONE;
    return level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR;
}

static void keryx_llama_log_cb(enum ggml_log_level level, const char* text, void* /*ud*/) {
    if (!text || !text[0] || !keryx_log_keep(level)) return;
    std::lock_guard<std::mutex> lock(keryx_log_mu);
    keryx_log_buf.append(text);
}

static void keryx_install_log_filter() {
    static bool done = false;
    if (done) return;
    done = true;
    keryx_log_verbose = getenv("KERYX_LLAMA_VERBOSE") != nullptr;
    llama_log_set(keryx_llama_log_cb, nullptr);
    ggml_log_set(keryx_llama_log_cb, nullptr);
}

extern "C" {

// ABI version — the miner refuses to use a mismatched .so.
KERYX_EXPORT int keryx_llama_abi() { return 9; }

// Reason for the last failed call on this thread; empty when none. Valid until the next call.
KERYX_EXPORT const char* keryx_llama_last_error() { return keryx_last_error.c_str(); }

// Copies pending log text into `out` (NUL-terminated) and drops what was copied.
// Returns the number of bytes copied, excluding the NUL. Text that does not fit stays
// for the next call. Returns 0 when there is nothing to copy.
KERYX_EXPORT int keryx_llama_drain_log(char* out, int cap) {
    if (!out || cap <= 1) return 0;
    std::lock_guard<std::mutex> lock(keryx_log_mu);
    const size_t room = static_cast<size_t>(cap - 1);
    const size_t n = std::min(keryx_log_buf.size(), room);
    if (n > 0) {
        memcpy(out, keryx_log_buf.data(), n);
        keryx_log_buf.erase(0, n);
    }
    out[n] = '\0';
    return static_cast<int>(n);
}

// Context, sampler and the canonical name list. The model is already on the GPUs.
// Cores, one thread each: the ops on weights a card reads from host memory run there, bound by
// memory bandwidth, and a second thread per core only fights the first (POLAR's 5700X decoded
// Qwen3.8 at 14.9 tokens a second on 8 threads, 13.0 on 16).
static int keryx_physical_cores() {
    const int logical = (int)std::max(1u, std::thread::hardware_concurrency());
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len);
    if (len == 0 || !GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) {
        return std::max(1, logical / 2);
    }
    int cores = 0;
    for (DWORD off = 0; off < len;) {
        const auto* p = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
        if (p->Relationship == RelationProcessorCore) cores++;
        off += p->Size;
    }
    return cores > 0 ? cores : std::max(1, logical / 2);
#else
    std::set<std::string> siblings;
    for (int cpu = 0; cpu < logical; cpu++) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings");
        std::string line;
        if (f && std::getline(f, line)) siblings.insert(line);
    }
    return siblings.empty() ? std::max(1, logical / 2) : (int)siblings.size();
#endif
}

// Tokens the MTP layer drafts ahead. A draft is kept while it matches what the model samples,
// and the model checks them all in one batch. The recurrent layers keep one state per drafted
// token, so a rejected draft is undone in place. Qwen3.8 alone on a 5070 Ti (2026-10-18), 4k:
// 2 drafted gave 38 tokens a second on nighthawk (61% kept) against 31 for 3 (41%) and 29
// without; on POLAR's 8 cores 14.6 against 13.1 and 14.9: the third draft was rarely kept and
// its state and check cost more than it saved.
static const int keryx_mtp_draft = 2;

#ifndef __APPLE__
// Defined by the build's patches of llama-context.cpp and llama-kv-cache.cpp (llama_patches.rs).
extern "C" {
extern bool (*keryx_llama_graph_hook)(const llama_context*, ggml_context*, ggml_cgraph*, uint32_t, ggml_backend_sched_t);
extern uint32_t keryx_llama_graph_margin;
extern bool keryx_llama_attn_rot;
}

// Contexts that trace, and their plans. The hook runs for every graph llama builds, the MTP
// layer's and a reserve included; only a registered context gets capture nodes.
static std::mutex keryx_trace_mu;
static std::map<const llama_context*, keryx_trace::Plan*> keryx_traces;

static bool keryx_trace_hook(const llama_context* c, ggml_context* ctx, ggml_cgraph* gf, uint32_t n_tokens,
                             ggml_backend_sched_t sched) {
    keryx_trace::Plan* plan = nullptr;
    {
        std::lock_guard<std::mutex> lock(keryx_trace_mu);
        const auto at = keryx_traces.find(c);
        if (at != keryx_traces.end()) plan = at->second;
    }
    if (!plan) return true;
    if (!plan->extend(ctx, gf, n_tokens, sched)) {
        keryx_llama_log_cb(GGML_LOG_LEVEL_ERROR, ("keryx: trace: " + plan->error + "\n").c_str(), nullptr);
        return false;
    }
    return true;
}

// Room for the capture nodes of the largest tier (about 1200) on top of llama's own graph.
static const uint32_t keryx_trace_margin = 4096;
#endif

// `host_weights`: some weights live in host memory and their ops run on the CPU.
static KeryxLlama* keryx_finish_load(llama_model* model, int n_ctx, bool host_weights = false) {
    llama_context_params cp = llama_context_default_params();
    if (host_weights) cp.n_threads = cp.n_threads_batch = keryx_physical_cores();
#ifndef __APPLE__
    // Every context can trace: the hook and its room are in place before llama sizes the
    // graph, and the KV cache keeps keys and values unrotated, as the trace commits them.
    keryx_llama_graph_hook = keryx_trace_hook;
    keryx_llama_graph_margin = keryx_trace_margin;
    keryx_llama_attn_rot = false;
    // No MTP drafts: a traced answer runs every generated token alone (node verify.rs
    // `expected_ubatch`), and the recurrent snapshots the drafts need change the state kernel.
    const bool mtp = false;
#else
    const bool mtp = llama_model_n_layer_nextn(model) > 0;
#endif
    if (mtp) cp.n_rs_seq = keryx_mtp_draft;
    cp.n_ctx = n_ctx > 0 ? n_ctx : 32768;
    cp.n_batch = std::min(cp.n_batch, cp.n_ctx);
    // 512 stays. A micro-batch of 128 or 64 would shrink Qwen3.5's logit buffer (505 MB at
    // 512, from the 248k vocabulary), but that prefill was not measured, so the window on a
    // tight card stays 4k rather than guessing that 32k still answers in time.
    cp.n_ubatch = std::min(cp.n_ubatch, std::max(1u, cp.n_ctx / 4));
    // q8_0: the trace commits the cache's keys and values as Q8_0 rows (node seq.rs), and flash
    // attention is what a quantized V needs. A V head that does not divide into 32-value blocks
    // (Kimi-Linear's 72) keeps V in f16; its latent layers keep no V cache and commit no V leaf.
    // A card without flash attention for the model falls back to f16 with llama.cpp's own choice.
    // Sliding-window layers keep only their window: the library default sizes them to the
    // whole context, which put Gemma 4's 40 SWA layers at ~5 GB for 32k. Each generation clears
    // the memory and no prompt is reused, so the full-size cache would buy nothing.
    cp.swa_full = false;
    struct KvTry {
        ggml_type k;
        ggml_type v;
        llama_flash_attn_type fa;
        const char* name;
    };
    const KvTry tries[] = {
        { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, LLAMA_FLASH_ATTN_TYPE_ENABLED, "q8_0" },
        { GGML_TYPE_Q8_0, GGML_TYPE_F16, LLAMA_FLASH_ATTN_TYPE_ENABLED, "q8_0/f16" },
        { GGML_TYPE_F16, GGML_TYPE_F16, LLAMA_FLASH_ATTN_TYPE_AUTO, "f16" },
    };
    llama_context* ctx = nullptr;
    const char* kv = nullptr;
    for (const KvTry& t : tries) {
        cp.type_k = t.k;
        cp.type_v = t.v;
        cp.flash_attn_type = t.fa;
        ctx = llama_init_from_model(model, cp);
        if (ctx) {
            kv = t.name;
            break;
        }
    }
    if (ctx && std::strcmp(kv, "q8_0") != 0) {
        char line[96];
        snprintf(line, sizeof(line), "keryx: KV cache %s; q8_0 did not build\n", kv);
        keryx_llama_log_cb(GGML_LOG_LEVEL_WARN, line, nullptr);
    }
    if (!ctx) {
        keryx_set_error("context", "llama_init_from_model failed");
        llama_model_free(model);
        return nullptr;
    }

    // User-facing sampling (DRY -> repeat penalty -> temperature 0.7 / top_p 0.9) — the OPoI
    // text is not consensus-relevant, but keep the flavor consistent.
    //
    // The flat repeat penalty alone does not stop these models: it charges a token once for
    // having appeared, no matter how many times, so a repeated *sentence* pays almost nothing
    // per token and the loop survives (observed on-chain with GLM-4-9B at 1.10 / 256). DRY
    // penalises the continuation of an already-seen sequence, growing with its length, which is
    // the failure mode we actually have. Standard settings: multiplier 0.8, base 1.75, loops
    // allowed up to 2 tokens, scanning the whole context (-1). The break tokens keep normal
    // structure (newlines, list markers, quotes) from counting as repetition.
    //
    // Top-k 64 goes first. Top-p sorted the whole vocabulary on the CPU for every token: 2.4-2.6
    // ms a token on Qwen3.5 (248k tokens) and Gemma 4, 13-17% of the decode. A token outside
    // the 64 likeliest is almost never inside top-p 0.9 anyway (2026-10-17).
    static const char* dry_breakers[] = { "\n", ":", "\"", "*" };
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(64));
    llama_sampler_chain_add(smpl, llama_sampler_init_dry(
        llama_model_get_vocab(model), llama_model_n_ctx_train(model),
        0.8f, 1.75f, 2, -1, dry_breakers, sizeof(dry_breakers) / sizeof(dry_breakers[0])));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(256, 1.10f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    auto* h = new KeryxLlama();
    h->model = model; h->ctx = ctx; h->smpl = smpl;
    h->host_weights = host_weights;
    h->kv_q8 = std::strncmp(kv, "q8_0", 4) == 0;
    if (mtp) {
        // The MTP layer's own context over the same weights: its KV for that one layer.
        llama_context_params dp = cp;
        dp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        dp.n_rs_seq = 0;
        dp.ctx_other = ctx;
        // It runs one layer: small micro-batches cost a prefill nothing, and its compute
        // buffers are sized for the vocabulary's logits of a whole micro-batch (505 MB at 512
        // tokens, 76-142 at 64).
        dp.n_ubatch = std::min(cp.n_ubatch, 64u);
        h->mtp = llama_init_from_model(model, dp);
        if (h->mtp) {
            common_params_speculative sp;
            sp.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            sp.draft.n_max = keryx_mtp_draft;
            sp.draft.ctx_tgt = ctx;
            sp.draft.ctx_dft = h->mtp;
            h->spec = common_speculative_init(sp, 1);
        }
        if (!h->spec) {
            keryx_llama_log_cb(GGML_LOG_LEVEL_WARN, "keryx: no MTP context, one token a decode\n", nullptr);
            if (h->mtp) llama_free(h->mtp);
            h->mtp = nullptr;
        }
    }
    for (auto& p : model->tensors_by_name) h->names.push_back(p.first);
    std::sort(h->names.begin(), h->names.end());
    return h;
}

// Defined by the build's patch of llama-model.cpp. On, the loader populates the whole GGUF
// mapping before it copies the weights, which a host that cannot cache the model pays twice:
// a read to populate it, and after the eviction a second read for the copy. Off, each page
// is read once, in file order. The mapping stays either way: the token embeddings live on the
// host and are paged in only as tokens need them.
// C linkage on both sides (this whole block is extern "C"): MSVC decorates a C++ global's
// name, GCC does not, so a C++ definition links on Linux only.
extern bool keryx_llama_prefetch;

KERYX_EXPORT void keryx_llama_set_prefetch(int on) { keryx_llama_prefetch = on != 0; }

// The multiprocessor count the kernels split their work for on a card of compute capability
// major.minor with `sms` multiprocessors (keryx_plan.cu): cards that answer alike plan alike.
int keryx_plan_sms(int major, int minor, int sms);
KERYX_EXPORT int keryx_llama_plan_sms(int major, int minor, int sms) { return keryx_plan_sms(major, minor, sms); }

// Products that left the fixed-order path for cuBLAS in this process (tests: must stay 0).
uint64_t keryx_plan_vendor_products(void);
KERYX_EXPORT uint64_t keryx_llama_vendor_products() { return keryx_plan_vendor_products(); }

}  // extern "C": the helpers below return and take C++ types, which MSVC refuses there.

// Metadata a pinned GGUF lacks, supplied at load: the weights are pinned by consensus, so the
// file cannot be fixed. GLM-4-0414 rotates half of each attention head (partial_rotary_factor
// 0.5), but the tier-1 GGUF was converted without `glm4.rope.dimension_count`, so llama.cpp
// rotated all 128 dimensions: a one-line question still came out right, any prompt past a few
// hundred tokens degenerated into loops (2026-10-16, CPU and CUDA alike). Returns an array
// terminated by an empty key, or an empty vector when nothing is missing.
static std::vector<llama_model_kv_override> keryx_metadata_fixes(const char* gguf_path) {
    std::vector<llama_model_kv_override> fixes;
    gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context* g = gguf_init_from_file(gguf_path, gp);
    if (!g) return fixes;
    const int64_t arch = gguf_find_key(g, "general.architecture");
    const bool glm4 = arch >= 0 && gguf_get_kv_type(g, arch) == GGUF_TYPE_STRING
        && std::strcmp(gguf_get_val_str(g, arch), "glm4") == 0;
    if (glm4 && gguf_find_key(g, "glm4.rope.dimension_count") < 0) {
        const int64_t kl = gguf_find_key(g, "glm4.attention.key_length");
        const int64_t head = kl >= 0 && gguf_get_kv_type(g, kl) == GGUF_TYPE_UINT32
            ? gguf_get_val_u32(g, kl) : 128;
        llama_model_kv_override o{};
        o.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        std::snprintf(o.key, sizeof(o.key), "%s", "glm4.rope.dimension_count");
        o.val_i64 = head / 2;
        fixes.push_back(o);
    }
    gguf_free(g);
    if (!fixes.empty()) fixes.push_back(llama_model_kv_override{});
    return fixes;
}

#ifndef __APPLE__
// Where one tensor sits in the walk blob: names sorted bytewise, each tensor's whole 32-byte
// chunks back to back, tensors under 32 bytes left out (`WeightIndex::open_gguf_with`).
struct KeryxCanon {
    size_t off = 0;    // in the blob
    size_t full = 0;   // bytes the blob holds of it
    size_t nbytes = 0; // the tensor's own size
    size_t file = 0;   // in the GGUF
};

// A lent split asks for the tensor table and its bound layout on every request; both depend
// only on the file, the cards and how much of each blob they hold.
struct KeryxRange {
    int card;
    size_t off, len;
    int copied;
};
static std::mutex keryx_memo_lock;
static std::map<std::string, std::pair<std::map<std::string, KeryxCanon>, size_t>> keryx_canon_memo;
static std::map<std::string, std::vector<KeryxRange>> keryx_layout_memo;

static bool keryx_canonical_read(const char* path, std::map<std::string, KeryxCanon>& out, size_t& total) {
    gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context* g = gguf_init_from_file(path, gp);
    if (!g) return false;
    std::vector<std::string> names;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) names.push_back(gguf_get_tensor_name(g, i));
    std::sort(names.begin(), names.end());
    total = 0;
    for (const auto& name : names) {
        const int64_t i = gguf_find_tensor(g, name.c_str());
        KeryxCanon c;
        c.nbytes = gguf_get_tensor_size(g, i);
        c.full = c.nbytes / 32 * 32;
        c.off = total;
        c.file = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i);
        out[name] = c;
        total += c.full;
    }
    gguf_free(g);
    return true;
}

static bool keryx_canonical(const char* path, std::map<std::string, KeryxCanon>& out, size_t& total) {
    {
        std::lock_guard<std::mutex> lock(keryx_memo_lock);
        const auto at = keryx_canon_memo.find(path);
        if (at != keryx_canon_memo.end()) {
            out = at->second.first;
            total = at->second.second;
            return true;
        }
    }
    if (!keryx_canonical_read(path, out, total)) return false;
    std::lock_guard<std::mutex> lock(keryx_memo_lock);
    keryx_canon_memo[path] = {out, total};
    return true;
}

// A CUDA-typed buffer over memory the miner owns: the scheduler runs the ops on the card, while
// nothing is allocated or freed here. No init_tensor: llama's would zero the row padding past
// each tensor, which in the blob is the next tensor.
static void* keryx_bound_base(ggml_backend_buffer_t b) { return b->context; }
static void keryx_bound_set(ggml_backend_buffer_t, ggml_tensor* t, const void* d, size_t o, size_t n) {
    cudaMemcpy((char*)t->data + o, d, n, cudaMemcpyHostToDevice);
}
static void keryx_bound_get(ggml_backend_buffer_t, const ggml_tensor* t, void* d, size_t o, size_t n) {
    cudaMemcpy(d, (const char*)t->data + o, n, cudaMemcpyDeviceToHost);
}
static void keryx_bound_memset(ggml_backend_buffer_t, ggml_tensor* t, uint8_t v, size_t o, size_t n) {
    cudaMemset((char*)t->data + o, v, n);
}
static void keryx_bound_clear(ggml_backend_buffer_t, uint8_t) {}

static ggml_backend_buffer_t keryx_bound_buffer(int gpu, void* base, size_t size) {
    ggml_backend_buffer_i iface = {};
    iface.get_base = keryx_bound_base;
    iface.set_tensor = keryx_bound_set;
    iface.get_tensor = keryx_bound_get;
    iface.memset_tensor = keryx_bound_memset;
    iface.clear = keryx_bound_clear;
    return ggml_backend_buffer_init(ggml_backend_cuda_buffer_type(gpu), iface, base, size);
}
#endif

#ifndef __APPLE__
// The model llama would load on `gpus` (one card, or a contiguous layer split over two), its
// tensors placed but no weights read.
static llama_model* keryx_place(const char* gguf_path, int n, const int* gpus) {
    if (n < 1 || n > 2 || gpus[0] < 0 || gpus[0] >= 128 || (n == 2 && (gpus[1] < 0 || gpus[1] >= 128 || gpus[1] == gpus[0]))) {
        keryx_set_error("bound", "need one card or two distinct ones");
        return nullptr;
    }
    llama_backend_init();
    static thread_local float split[128];
    std::fill(split, split + 128, 0.f);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.main_gpu     = gpus[0];
    if (n == 1) {
        // Without a split llama shares the layers out by free VRAM, which is 0 on a tail card
        // whose walk fills it: 0/0, and the layer's device index lands past the list (MSVC
        // throws "invalid vector subscript"). The one device takes them all.
        split[0] = 1.f;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.tensor_split = split;
    } else {
        split[gpus[0]] = 1.f;
        split[gpus[1]] = 1.f;
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
        mp.tensor_split = split;
    }
    mp.use_mmap  = false;
    mp.use_mlock = false;
    mp.no_alloc  = true;  // tensors and their placement, no weights
    mp.use_extra_bufts = false;
    const auto fixes = keryx_metadata_fixes(gguf_path);
    if (!fixes.empty()) mp.kv_overrides = fixes.data();
    llama_model* model = llama_model_load_from_file(gguf_path, mp);
    if (!model) keryx_set_error("model", std::string("llama_model_load_from_file failed for ") + gguf_path);
    return model;
}

// Which of `gpus` a placed tensor went to, -1 for the host. -2 for a card not in the list.
static int keryx_card_of(const ggml_tensor* t, int n, const int* gpus) {
    ggml_backend_buffer_type_t buft = t->buffer ? ggml_backend_buffer_get_type(t->buffer) : nullptr;
    if (!buft || ggml_backend_buft_is_host(buft)) return -1;
    for (int i = 0; i < n; i++) {
        if (buft == ggml_backend_cuda_buffer_type(gpus[i])) return i;
    }
    return -2;
}

// The bytes MMQ reads past the last row of the tensor at `it`, `over` of them, all f16 fields
// finite. MMQ steps 256 values at a time, so a 32-value block type whose rows are not a multiple
// of 512 values reads that far into the padding llama allocates and zeroes; in the blob it is the
// next tensors' bytes. The padded part of every activation row is zero, so those blocks add
// nothing unless a scale (or a min) is Inf or NaN: Kimi-Linear's 128-wide Q5_0 rows came out NaN
// (2026-10-17), GLM's 13,696-wide Q8_0 ffn_down read finite scales. The blob is fixed by the model,
// so this is read once from the GGUF.
static bool keryx_overread_safe(const char* path, const std::map<std::string, KeryxCanon>& canon,
                                std::map<std::string, KeryxCanon>::const_iterator it, size_t over,
                                ggml_type type) {
    int fields;
    switch (type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q5_0: case GGML_TYPE_Q8_0: fields = 1; break;
        case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_1: fields = 2; break;
        default: return false;
    }
    const size_t block = ggml_type_size(type);
    if (over % block != 0) return false;
    std::vector<uint8_t> bytes(over);
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    size_t got = 0;
    for (auto next = std::next(it); next != canon.end() && got < over; ++next) {
        const size_t take = std::min(next->second.full, over - got);
        if (take == 0) continue;
        if (keryx_fseek(f, (int64_t)next->second.file, SEEK_SET) != 0 || fread(bytes.data() + got, 1, take, f) != take) {
            fclose(f);
            return false;
        }
        got += take;
    }
    fclose(f);
    if (got < over) return false;
    for (size_t b = 0; b < over; b += block) {
        for (int i = 0; i < fields; i++) {
            uint16_t h;
            memcpy(&h, &bytes[b + 2 * i], 2);
            if ((h & 0x7C00) == 0x7C00) return false;
        }
    }
    return true;
}

// A tensor the blob of its card cannot serve, held to `held` bytes: not whole 32-byte chunks
// (the blob drops the remainder); read past what the card holds (a tail card keeps only the head
// of the blob, and the last tensor's row padding ends past the blob); or read past its last row
// into bytes that are not safe to multiply by zero (`keryx_overread_safe`). `reach` is what llama
// allocates for it, row padding included; only a 32-value block type reads that padding.
static bool keryx_copied(const char* path, const std::map<std::string, KeryxCanon>& canon,
                         std::map<std::string, KeryxCanon>::const_iterator it, const ggml_tensor* t,
                         size_t reach, size_t held) {
    const KeryxCanon& c = it->second;
    const size_t nbytes = ggml_nbytes(t);
    const bool over_read = reach > nbytes && ggml_blck_size(t->type) < 256;
    if (c.full != c.nbytes || c.full == 0 || c.off + (over_read ? reach : nbytes) > held) return true;
    return over_read && !keryx_overread_safe(path, canon, it, reach - nbytes, t->type);
}
#endif

extern "C" {

// The bytes of card `card`'s blob llama may read `t` from: all it holds for the output projection
// (read for every token and by every speculative draft; the tied token embeddings on Gemma), the
// first cut[card] for the rest. A card serving alone unmaps what lies between.
static size_t keryx_reach_of(const llama_model* model, const ggml_tensor* t, const size_t* held,
                             const size_t* cut, int card) {
    return t == model->output ? held[card] : std::min(held[card], cut[card]);
}

// What llama would read from each card's walk blob if it were loaded on `gpus` with the blob of
// card i holding its first held[i] bytes, of which it reads every tensor but the output
// projection in the first cut[i]: one call of `range` per tensor placed on a card, with the
// card's index, the tensor's canonical range (row padding included) and whether it would be
// copied instead (bit 1), and bit 2 on the output projection, which is copied to VRAM even on a
// card that spills (see keryx_llama_load_bound). Returns 0, or -1 with keryx_llama_last_error.
KERYX_EXPORT int keryx_llama_bound_layout(const char* gguf_path, int n, const int* gpus, const size_t* held,
                                          const size_t* cut, void (*range)(void*, int, size_t, size_t, int),
                                          void* user) {
    keryx_last_error.clear();
    keryx_install_log_filter();
#ifdef __APPLE__
    (void)gguf_path; (void)n; (void)gpus; (void)held; (void)cut; (void)range; (void)user;
    keryx_set_error("bound", "zero-dup needs CUDA");
    return -1;
#else
    std::string key = gguf_path;
    for (int i = 0; i < n; i++) {
        key += "|" + std::to_string(gpus[i]) + ":" + std::to_string(held[i]) + ":" + std::to_string(cut[i]);
    }
    {
        std::lock_guard<std::mutex> lock(keryx_memo_lock);
        const auto at = keryx_layout_memo.find(key);
        if (at != keryx_layout_memo.end()) {
            for (const auto& r : at->second) range(user, r.card, r.off, r.len, r.copied);
            return 0;
        }
    }
    std::map<std::string, KeryxCanon> canon;
    size_t total = 0;
    if (!keryx_canonical(gguf_path, canon, total)) {
        keryx_set_error("bound", std::string("cannot read the tensor table of ") + gguf_path);
        return -1;
    }
    llama_model* model = keryx_place(gguf_path, n, gpus);
    if (!model) return -1;
    int rc = 0;
    std::vector<KeryxRange> ranges;
    for (auto& [name, t] : model->tensors_by_name) {
        const auto at = canon.find(name);
        const int card = keryx_card_of(t, n, gpus);
        if (card == -1) continue;
        if (at == canon.end() || card < 0) {
            keryx_set_error("bound", "tensor " + name + " is not in the GGUF table or off the cards");
            rc = -1;
            break;
        }
        const size_t reach = ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(t->buffer), t);
        const int copied = (keryx_copied(gguf_path, canon, at, t, reach, keryx_reach_of(model, t, held, cut, card)) ? 1 : 0)
                           | (t == model->output ? 2 : 0);
        ranges.push_back({card, at->second.off, std::max(reach, at->second.full), copied});
    }
    llama_model_free(model);
    if (rc != 0) return rc;
    for (const auto& r : ranges) range(user, r.card, r.off, r.len, r.copied);
    std::lock_guard<std::mutex> lock(keryx_memo_lock);
    keryx_layout_memo[key] = std::move(ranges);
    return 0;
#endif
}

// llama on `gpus` (one card, or a layer split over two), every weight read in place from the
// walk blob of the card it lands on: blobs[i] is card i's copy of the GGUF in the canonical
// layout (not permuted while llama runs), of which it holds the first held[i] bytes of
// `layout_len`, and serves the output projection there and every other tensor in the first
// cut[i] (a card serving alone unmaps what lies between for the context). Only the context is
// allocated, and the tensors the blobs cannot serve
// (`keryx_copied`) are copied to a small buffer per card in its VRAM. Where spill[i] is set (a
// tail card serving alone has no VRAM for them) they go to pinned host memory instead, and the
// CPU runs their ops for a decoded token while a prompt's batches take them to the card: three
// times the speed of the card reading them over PCIe (2026-10-18). The output projection still
// goes to VRAM there: it is read for every token, and by every draft of a speculative decode.
// Host tensors (the token embeddings) are read from a mapping of the GGUF.
KERYX_EXPORT KeryxLlama* keryx_llama_load_bound(const char* gguf_path, int n, const int* gpus, void* const* blobs,
                                                const size_t* held, const size_t* cut, size_t layout_len,
                                                int n_ctx, const int* spill) {
    keryx_last_error.clear();
    keryx_install_log_filter();
#ifdef __APPLE__
    (void)gguf_path; (void)n; (void)gpus; (void)blobs; (void)held; (void)cut; (void)layout_len; (void)n_ctx;
    (void)spill;
    keryx_set_error("bound", "zero-dup needs CUDA");
    return nullptr;
#else
    if (n != 1 || gpus == nullptr || blobs == nullptr || held == nullptr || cut == nullptr) {
        keryx_set_error("bound", "the canonical engine is single-card");
        return nullptr;
    }
    std::map<std::string, KeryxCanon> canon;
    size_t total = 0;
    if (!keryx_canonical(gguf_path, canon, total)) {
        keryx_set_error("bound", std::string("cannot read the tensor table of ") + gguf_path);
        return nullptr;
    }
    if (total != layout_len) {
        keryx_set_error("bound", "the GGUF lays out " + std::to_string(total) + " bytes, the walk "
                        + std::to_string(layout_len));
        return nullptr;
    }
    for (int i = 0; i < n; i++) {
        if (!blobs[i] || held[i] > layout_len) {
            keryx_set_error("bound", "no walk blob on gpu " + std::to_string(gpus[i]));
            return nullptr;
        }
    }
    llama_model* model = keryx_place(gguf_path, n, gpus);
    if (!model) return nullptr;

    // Nothing is read ahead: a host that cannot cache the GGUF pages in only what it touches.
    std::unique_ptr<llama_file> file;
    std::unique_ptr<llama_mmap> map;
    try {
        file = std::make_unique<llama_file>(gguf_path, "rb");
        map = std::make_unique<llama_mmap>(file.get(), /*prefetch =*/ 0);
    } catch (const std::exception& err) {
        keryx_set_error("bound", std::string("cannot map ") + gguf_path + ": " + err.what());
        llama_model_free(model);
        return nullptr;
    }
    char* const mapped = static_cast<char*>(map->addr());
    std::vector<ggml_backend_buffer_t> buffers;
    const auto fail = [&](const char* stage, const std::string& detail) {
        keryx_set_error(stage, detail);
        llama_model_free(model);
        for (auto b : buffers) ggml_backend_buffer_free(b);
        return nullptr;
    };
    ggml_backend_buffer_t host = ggml_backend_cpu_buffer_from_ptr(mapped, map->size());
    buffers.push_back(host);
    std::vector<ggml_backend_buffer_t> shared(n);
    // The buffer claims a little past what the card holds: ggml checks each tensor's allocated
    // size against it, and the last tensor's row padding ends past the blob. `keryx_copied`
    // keeps every read inside `held`; only a 256-value block type gets this far, and it never
    // reads its padding.
    const size_t row_pad = 64 * 1024;
    for (int i = 0; i < n; i++) {
        shared[i] = keryx_bound_buffer(gpus[i], blobs[i], std::max<size_t>(held[i], 1) + row_pad);
        buffers.push_back(shared[i]);
    }

    // Per card: copied to VRAM, and (spill) copied to host memory.
    std::vector<std::vector<std::pair<ggml_tensor*, const KeryxCanon*>>> copied(n), hosted(n);
    std::vector<size_t> copied_bytes(n, 0), hosted_bytes(n, 0);
    for (auto& [name, t] : model->tensors_by_name) {
        const auto at = canon.find(name);
        if (at == canon.end()) return fail("bound", "tensor " + name + " is not in the GGUF table");
        const KeryxCanon& c = at->second;
        const int card = keryx_card_of(t, n, gpus);
        if (card == -2) return fail("bound", "tensor " + name + " was placed off the cards");
        if (card == -1) {
            t->buffer = nullptr;
            t->data = nullptr;
            ggml_backend_tensor_alloc(host, t, mapped + c.file);
            continue;
        }
        ggml_backend_buffer_type_t cuda = ggml_backend_cuda_buffer_type(gpus[card]);
        const size_t reach = ggml_backend_buft_get_alloc_size(cuda, t);
        t->buffer = nullptr;
        t->data = nullptr;
        if (keryx_copied(gguf_path, canon, at, t, reach, keryx_reach_of(model, t, held, cut, card))) {
            const bool host = spill && spill[card] && t != model->output;
            (host ? hosted : copied)[card].push_back({t, &c});
            (host ? hosted_bytes : copied_bytes)[card] += GGML_PAD(reach, ggml_backend_buft_get_alignment(cuda));
        } else {
            ggml_backend_tensor_alloc(shared[card], t, static_cast<char*>(blobs[card]) + c.off);
        }
    }
    bool host_weights = false;
    for (int i = 0; i < n; i++) {
        ggml_backend_buffer_type_t cuda = ggml_backend_cuda_buffer_type(gpus[i]);
        ggml_backend_buffer_type_t pinned = ggml_backend_cuda_host_buffer_type();
        for (const bool host : {false, true}) {
            const auto& list = host ? hosted[i] : copied[i];
            const size_t bytes = host ? hosted_bytes[i] : copied_bytes[i];
            if (list.empty()) continue;
            ggml_backend_buffer_t own = ggml_backend_buft_alloc_buffer(host ? pinned : cuda, bytes);
            if (!own) {
                return fail("bound", "cannot allocate " + std::to_string(bytes) + " bytes "
                            + (host ? std::string("of pinned host memory") : "on gpu " + std::to_string(gpus[i]))
                            + " for the copied tensors");
            }
            buffers.push_back(own);
            host_weights = host_weights || host;
            char* p = static_cast<char*>(ggml_backend_buffer_get_base(own));
            for (auto& [t, c] : list) {
                ggml_backend_tensor_alloc(own, t, p);
                ggml_backend_tensor_set(t, mapped + c->file, 0, ggml_nbytes(t));
                p += GGML_PAD(ggml_backend_buft_get_alloc_size(cuda, t), ggml_backend_buft_get_alignment(cuda));
            }
        }
    }
    // As llama marks its own: the scheduler runs an op where its weight lives.
    for (auto b : buffers) ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    model->hparams.no_alloc = false;

    KeryxLlama* h = keryx_finish_load(model, n_ctx, host_weights);
    if (h) h->gpus.assign(gpus, gpus + n);
    if (!h) {
        for (auto b : buffers) ggml_backend_buffer_free(b);
        return nullptr;
    }
    h->buffers = std::move(buffers);
    h->file = std::move(file);
    h->map = std::move(map);
    return h;
#endif
}

KERYX_EXPORT void keryx_llama_free(KeryxLlama* h);
}  // extern "C"

// Prefills `prompt` and samples up to `max_tokens` after it, handing each token to `emit` until
// it returns false; fills the last_* figures. With an MTP layer each step checks the token and
// what the layer drafted after it in one batch, keeps the drafts the sampler agrees with, and
// drops the rest from both contexts. False when the prefill fails; a decode that fails later
// ends the answer there.
static bool keryx_run(KeryxLlama* h, const std::vector<llama_token>& prompt, int max_tokens,
                      const std::function<bool(llama_token)>& emit) {
    using clock = std::chrono::steady_clock;
    const auto ms_since = [](clock::time_point t) {
        return std::chrono::duration<double, std::milli>(clock::now() - t).count();
    };
    const auto started = clock::now();
    h->last_prompt_tokens = 0; h->last_tokens = 0;
    h->last_prefill_ms = 0; h->last_decode_ms = 0;
    h->last_drafted = 0; h->last_accepted = 0;
    const int n = (int)prompt.size();
    if (n == 0 || max_tokens <= 0) return n > 0;
    llama_memory_clear(llama_get_memory(h->ctx), true);
    if (h->mtp) llama_memory_clear(llama_get_memory(h->mtp), true);
    // One llama_decode cannot exceed n_batch (default 2048, even when n_ctx is 4096).
    // A prompt that fills the window is prefilled in batches; sampling starts after that.
    const int n_batch = std::max(1, (int)llama_n_batch(h->ctx));
    const int n_ctx = (int)llama_n_ctx(h->ctx);
    std::unique_ptr<llama_batch, void (*)(llama_batch*)> held(
        new llama_batch(llama_batch_init(std::max(n_batch, keryx_mtp_draft + 1), 0, 1)),
        [](llama_batch* b) { llama_batch_free(*b); delete b; });
    llama_batch& batch = *held;
    const auto decode = [&] {
        if (llama_decode(h->ctx, batch) != 0) return false;
        return !h->spec || common_speculative_process(h->spec, batch);
    };
    for (int at = 0; at < n; at += n_batch) {
        common_batch_clear(batch);
        for (int i = at; i < std::min(n, at + n_batch); i++) common_batch_add(batch, prompt[i], i, { 0 }, i == n - 1);
        if (!decode()) return false;
    }
    h->last_prompt_tokens = n;
    h->last_prefill_ms = ms_since(started);
    const auto decoding = clock::now();
    std::vector<llama_token> history = prompt; // what the context holds
    if (h->spec) common_speculative_begin(h->spec, 0, history);
    llama_token tok = llama_sampler_sample(h->smpl, h->ctx, -1);
    int sampled = 1;
    int n_past = n;
    std::vector<llama_token> draft;
    std::vector<llama_token> ids;
    for (;;) {
        h->last_tokens = sampled;
        h->last_decode_ms = ms_since(decoding);
        if (!emit(tok) || sampled >= max_tokens || n_past >= n_ctx) return true;
        draft.clear();
        const int n_draft = h->spec ? std::min({ keryx_mtp_draft, max_tokens - sampled - 1, n_ctx - n_past - 1 }) : 0;
        if (n_draft > 0) {
            common_speculative_get_draft_params(h->spec, 0) = { true, n_draft, n_past, tok, &history, &draft };
            common_speculative_draft(h->spec);
            // The draft left its tokens in the MTP layer's KV; the check below writes them again.
            llama_memory_seq_rm(llama_get_memory(h->mtp), 0, n_past, -1);
        }
        common_batch_clear(batch);
        common_batch_add(batch, tok, n_past, { 0 }, true);
        for (size_t i = 0; i < draft.size(); i++) common_batch_add(batch, draft[i], n_past + 1 + (int)i, { 0 }, true);
        if (!decode()) return true;
        history.push_back(tok);
        ids.clear();
        for (size_t i = 0;; i++) {
            ids.push_back(llama_sampler_sample(h->smpl, h->ctx, (int)i));
            if (i == draft.size() || ids.back() != draft[i]) break;
        }
        const int accepted = (int)ids.size() - 1;
        h->last_drafted += (int)draft.size();
        h->last_accepted += accepted;
        n_past += 1 + accepted;
        if (accepted < (int)draft.size()) {
            llama_memory_seq_rm(llama_get_memory(h->ctx), 0, n_past, -1);
            llama_memory_seq_rm(llama_get_memory(h->mtp), 0, n_past, -1);
        }
        if (h->spec) common_speculative_accept(h->spec, 0, (uint16_t)accepted);
        for (int k = 0; k < accepted; k++) {
            h->last_tokens = ++sampled;
            h->last_decode_ms = ms_since(decoding);
            if (!emit(ids[k])) return true;
            history.push_back(ids[k]);
        }
        tok = ids.back();
        sampled++;
    }
}

extern "C" {

// Runs `gguf_path` once on `gpu` with throwaway weights (every CUDA tensor over one zeroed
// buffer, every eighth one in pinned host memory as a card serving alone keeps them): a prefill
// of two micro-batches and one decoded token. The CUDA modules its kernels need load on first
// use and stay for the life of the process, 34 MiB on a 5070 Ti for Qwen3.8 (2026-10-18); run
// before the walk is sized, they are not taken from what a tail card lends.
// Returns 0, or -1 with keryx_llama_last_error.
KERYX_EXPORT int keryx_llama_warm(const char* gguf_path, int gpu, int n_ctx) {
    keryx_last_error.clear();
    keryx_install_log_filter();
#ifdef __APPLE__
    (void)gguf_path; (void)gpu; (void)n_ctx;
    keryx_set_error("warm", "needs CUDA");
    return -1;
#else
    llama_model* model = keryx_place(gguf_path, 1, &gpu);
    if (!model) return -1;
    std::unique_ptr<llama_file> file;
    std::unique_ptr<llama_mmap> map;
    try {
        file = std::make_unique<llama_file>(gguf_path, "rb");
        map = std::make_unique<llama_mmap>(file.get(), /*prefetch =*/ 0);
    } catch (const std::exception& err) {
        keryx_set_error("warm", std::string("cannot map ") + gguf_path + ": " + err.what());
        llama_model_free(model);
        return -1;
    }
    std::map<std::string, KeryxCanon> canon;
    size_t total = 0;
    if (!keryx_canonical(gguf_path, canon, total)) {
        keryx_set_error("warm", std::string("cannot read the tensor table of ") + gguf_path);
        llama_model_free(model);
        return -1;
    }
    ggml_backend_buffer_type_t cuda = ggml_backend_cuda_buffer_type(gpu);
    size_t most = 1;
    for (auto& [name, t] : model->tensors_by_name) {
        if (keryx_card_of(t, 1, &gpu) == 0) most = std::max(most, ggml_backend_buft_get_alloc_size(cuda, t));
    }
    std::vector<ggml_backend_buffer_t> buffers;
    ggml_backend_buffer_t scratch = ggml_backend_buft_alloc_buffer(cuda, most);
    if (!scratch) {
        keryx_set_error("warm", "cannot allocate " + std::to_string(most) + " bytes on gpu " + std::to_string(gpu));
        llama_model_free(model);
        return -1;
    }
    ggml_backend_buffer_clear(scratch, 0);
    buffers.push_back(scratch);
    ggml_backend_buffer_t pinned = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), most);
    if (!pinned) {
        keryx_set_error("warm", "cannot pin " + std::to_string(most) + " bytes of host memory");
        llama_model_free(model);
        for (auto b : buffers) ggml_backend_buffer_free(b);
        return -1;
    }
    ggml_backend_buffer_clear(pinned, 0);
    buffers.push_back(pinned);
    char* const mapped = static_cast<char*>(map->addr());
    ggml_backend_buffer_t host = ggml_backend_cpu_buffer_from_ptr(mapped, map->size());
    buffers.push_back(host);
    size_t on_card = 0;
    for (auto& [name, t] : model->tensors_by_name) {
        const int card = keryx_card_of(t, 1, &gpu);
        t->buffer = nullptr;
        t->data = nullptr;
        if (card == 0) {
            const bool spill = t != model->output && on_card++ % 8 == 7;
            ggml_backend_buffer_t into = spill ? pinned : scratch;
            ggml_backend_tensor_alloc(into, t, ggml_backend_buffer_get_base(into));
        } else {
            const auto at = canon.find(name);
            if (at == canon.end()) {
                keryx_set_error("warm", "tensor " + name + " is not in the GGUF table");
                llama_model_free(model);
                for (auto b : buffers) ggml_backend_buffer_free(b);
                return -1;
            }
            ggml_backend_tensor_alloc(host, t, mapped + at->second.file);
        }
    }
    for (auto b : buffers) ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    model->hparams.no_alloc = false;
    KeryxLlama* h = keryx_finish_load(model, n_ctx, true);
    if (!h) {
        for (auto b : buffers) ggml_backend_buffer_free(b);
        return -1;
    }
    h->buffers = std::move(buffers);
    h->file = std::move(file);
    h->map = std::move(map);
    // Two micro-batches of prefill and a few tokens, drafts and their check included.
    const std::vector<llama_token> toks(2 * llama_n_ubatch(h->ctx), 0);
    const bool ok = keryx_run(h, toks, keryx_mtp_draft + 2, [](llama_token) { return true; });
    if (!ok) keryx_set_error("warm", "llama_decode failed");
    keryx_llama_free(h);
    return ok ? 0 : -1;
#endif
}

KERYX_EXPORT size_t keryx_llama_tensor_count(KeryxLlama* h) { return h ? h->names.size() : 0; }

// Tensor i in CANONICAL order. *is_device = the data pointer is CUDA device memory (walkable
// in-place); 0 = host memory (the caller uploads its own device copy for the walk).
KERYX_EXPORT bool keryx_llama_tensor_info(KeryxLlama* h, size_t i, const char** name, void** data,
                                          size_t* nbytes, int* is_device) {
    if (!h || i >= h->names.size()) return false;
    const ggml_tensor* t = h->model->get_tensor(h->names[i].c_str());
    if (!t || !t->data) return false;
    *name = h->names[i].c_str();
    *data = t->data;
    *nbytes = ggml_nbytes(t);
#ifdef __APPLE__
    // Metal / Apple Silicon unified memory: tensor bytes are in an MTLBuffer that's both CPU- and
    // GPU-visible via the same address. The Metal PoM walk (Phase 3a) doesn't consume `data` for
    // its own gather (it pre-packs from GGUF), so the semantic here is "there's a live pointer
    // to the tensor bytes for anyone who wants to check byte-exactness against GGUF".
    *is_device = 1;
#else
    cudaPointerAttributes attr{};
    cudaPointerGetAttributes(&attr, t->data);
    *is_device = attr.type == cudaMemoryTypeDevice ? 1 : 0;
#endif
    return true;
}

// Generate up to max_tokens; writes UTF-8 into out (cap bytes, NUL-terminated). Returns written
// length, or -1 on error. Serialized — one generation at a time (OPoI challenges are rare).
// CUDA ordinal owning tensor i's bytes, or -1 (host memory, unified memory, unknown, or a
// context in error). The walk gathers over these pointers, so it must launch on this device.
KERYX_EXPORT int keryx_llama_tensor_device(KeryxLlama* h, size_t i) {
#ifdef __APPLE__
    (void)h; (void)i;
    return -1;
#else
    if (!h || i >= h->names.size()) return -1;
    const ggml_tensor* t = h->model->get_tensor(h->names[i].c_str());
    if (!t || !t->data) return -1;
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, t->data) != cudaSuccess) return -1;
    return attr.type == cudaMemoryTypeDevice ? attr.device : -1;
#endif
}

KERYX_EXPORT int keryx_llama_generate(KeryxLlama* h, const char* prompt, int max_tokens, char* out, int cap) {
    if (!h || !prompt || !out || cap < 2) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    const llama_vocab* vocab = llama_model_get_vocab(h->model);

    std::vector<llama_token> toks(strlen(prompt) + 16);
    int n = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt), toks.data(), (int32_t)toks.size(), true, true);
    if (n < 0) return -1;
    toks.resize(n);
    // A prompt longer than the window fails its prefill, and a sealed request left without an
    // answer is a strike. Keep the head (the system prompt) and the tail (the end of the user
    // turn and the assistant prefix) and drop the middle, leaving room for the answer.
    const int n_ctx = (int)llama_n_ctx(h->ctx);
    const int reserve = std::max(1, std::min(max_tokens, n_ctx / 4));
    const int keep = n_ctx - reserve;
    if (n > keep && keep > 2) {
        const int head = keep / 3;
        toks.erase(toks.begin() + head, toks.begin() + (n - (keep - head)));
        n = (int)toks.size();
    }

    int written = 0;
    const bool ok = keryx_run(h, toks, max_tokens, [&](llama_token tok) {
        if (llama_vocab_is_eog(vocab, tok)) return false;
        char piece[256];
        const int pn = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
        if (pn < 0 || written + pn >= cap - 1) return false;
        memcpy(out + written, piece, pn);
        written += pn;
        return true;
    });
    if (!ok) {
        out[0] = 0;
        return -1;
    }
    out[written] = 0;
    return written;
}

// H15 trace (ADR 0023). Tracing needs every layer on a Turing or newer card, the K cache in
// Q8_0, 512-position ubatches and the manifest of the model's tier; the plan finds every leaf
// in the graph the first time a ubatch runs. Returns 0, or -1 with keryx_llama_last_error.
KERYX_EXPORT int keryx_llama_trace_init(KeryxLlama* h, const uint8_t* manifest, size_t len, uint32_t n_layers,
                                        uint32_t n_used, uint32_t n_attn, uint32_t n_recur, uint32_t seq_heads) {
    keryx_last_error.clear();
#ifdef __APPLE__
    (void)h; (void)manifest; (void)len; (void)n_layers; (void)n_used; (void)n_attn; (void)n_recur; (void)seq_heads;
    keryx_set_error("trace", "tracing needs CUDA");
    return -1;
#else
    if (!h || !manifest) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    if (h->trace) return 0;
    if (h->gpus.empty() || h->host_weights) {
        keryx_set_error("trace", "every layer must run on the cards");
        return -1;
    }
    if (!h->kv_q8) {
        keryx_set_error("trace", "the K cache is not Q8_0");
        return -1;
    }
    if (llama_n_ubatch(h->ctx) != keryx_trace::UBATCH) {
        keryx_set_error("trace", "the micro-batch is not 512 positions");
        return -1;
    }
    for (int gpu : h->gpus) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, gpu) != cudaSuccess || prop.major * 10 + prop.minor < 75) {
            keryx_set_error("trace", "gpu " + std::to_string(gpu) + " is older than Turing");
            return -1;
        }
    }
    auto plan = std::make_unique<keryx_trace::Plan>();
    keryx_trace::Tier tier{ n_layers, n_used, n_attn, n_recur, seq_heads };
    if (!plan->init(manifest, len, tier, h->model->arch) || !plan->alloc_tables(h->gpus)) {
        keryx_set_error("trace", plan->error);
        plan->free_tables();
        return -1;
    }
    // Each capture runs on its layer's card; decode hashes once per run of layers on one card.
    if (n_layers <= (uint32_t) llama_model_n_layer(h->model)) {
        for (uint32_t il = 0; il < n_layers; il++) plan->layer_dev.push_back(h->model->dev_layer((int) il));
        plan->layer_dev.push_back(h->model->dev_output());
    }
    {
        std::lock_guard<std::mutex> lock(keryx_trace_mu);
        keryx_traces[h->ctx] = plan.get();
    }
    h->trace = std::move(plan);
    return 0;
#endif
}

KERYX_EXPORT uint32_t keryx_llama_trace_leaves(KeryxLlama* h) {
#ifdef __APPLE__
    (void)h;
    return 0;
#else
    return h && h->trace ? h->trace->state.n_leaves : 0;
#endif
}

KERYX_EXPORT uint32_t keryx_llama_trace_seq_leaves(KeryxLlama* h) {
#ifdef __APPLE__
    (void)h;
    return 0;
#else
    return h && h->trace ? h->trace->state.n_seq_leaves : 0;
#endif
}

// The token count, or -(needed) when `cap` is too small.
KERYX_EXPORT int keryx_llama_tokenize(KeryxLlama* h, const char* text, int len, int add_special, int parse_special,
                                      int32_t* out, int cap) {
    if (!h || !text || len < 0) return 0;
    return llama_tokenize(llama_model_get_vocab(h->model), text, len, out, cap, add_special != 0, parse_special != 0);
}

// Bytes of `tok`'s text written to `out` (special tokens included as text), -1 on error.
KERYX_EXPORT int keryx_llama_token_piece(KeryxLlama* h, int32_t tok, char* out, int cap) {
    if (!h || !out || cap <= 0) return -1;
    const int n = llama_token_to_piece(llama_model_get_vocab(h->model), tok, out, cap, 0, true);
    return n < 0 ? -1 : n;
}

KERYX_EXPORT int keryx_llama_is_eog(KeryxLlama* h, int32_t tok) {
    return h && llama_vocab_is_eog(llama_model_get_vocab(h->model), tok) ? 1 : 0;
}

#ifndef __APPLE__
// One traced ubatch: the positions [pos0, pos0+n) of `tokens`, logits on the last row when
// `logits_last` (the LM head is committed only at output positions). The leaf table is cleared
// before and read after, so `rows` holds exactly this ubatch.
static bool keryx_trace_ubatch(KeryxLlama* h, llama_batch& batch, const int32_t* tokens, int pos0, int n,
                               bool logits_last, std::vector<uint8_t>& rows,
                               int (*on_rows)(void*, int32_t, int32_t, int32_t, const uint8_t*), void* user) {
    keryx_trace::Plan* plan = h->trace.get();
    if (!plan->clear_tables() || !plan->set_pos0(pos0)) {
        keryx_set_error("trace", "cannot reset the leaf table");
        return false;
    }
    batch.n_tokens = n;
    for (int i = 0; i < n; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos0 + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = logits_last && i == n - 1;
    }
    const bool decoded = llama_decode(h->ctx, batch) == 0;
    llama_synchronize(h->ctx);
    // The rows decode captures lent to the products belong to this ubatch only.
    keryx_trace_forget_q8_1();
    if (!decoded) {
        keryx_set_error("trace", "decode failed at position " + std::to_string(pos0));
        return false;
    }
    if (!plan->read_tables((uint32_t)n, rows)) {
        keryx_set_error("trace", "cannot read the leaf table");
        return false;
    }
    if (!on_rows(user, pos0, n, n, rows.data())) {
        keryx_set_error("trace", "the leaf table was refused");
        return false;
    }
    return true;
}

// Starts a traced run: no cached positions, empty sequence leaves.
static bool keryx_trace_start(KeryxLlama* h, int n_prompt) {
    if (n_prompt >= (int)llama_n_ctx(h->ctx)) {
        keryx_set_error("trace", "the prompt does not fit the window");
        return false;
    }
    llama_memory_clear(llama_get_memory(h->ctx), true);
    if (!h->trace->begin()) {
        keryx_set_error("trace", "cannot reset the sequence leaves");
        return false;
    }
    return true;
}
#endif

// Traced generation (see keryx_llama_trace_init). The prompt runs in 512-position ubatches from
// position 0, then every generated token alone, its logits computed so the LM head is committed
// at every output position. on_token gets each sampled token before it is fed and returns 1 to
// feed it, 0 to stop; end-of-generation tokens are never offered. on_rows gets the leaf hashes of
// every ubatch (zero = not captured) and returns 0 to abort. Returns the number of fed generated
// tokens, or -1 with keryx_llama_last_error; seq_out then holds every sequence leaf's hash.
KERYX_EXPORT int keryx_llama_generate_traced(KeryxLlama* h, const int32_t* prompt, int n_prompt, int max_tokens,
                                             int (*on_token)(void*, int32_t),
                                             int (*on_rows)(void*, int32_t, int32_t, int32_t, const uint8_t*),
                                             void* user, uint8_t* seq_out, size_t seq_cap) {
    keryx_last_error.clear();
#ifdef __APPLE__
    (void)h; (void)prompt; (void)n_prompt; (void)max_tokens; (void)on_token; (void)on_rows; (void)user;
    (void)seq_out; (void)seq_cap;
    keryx_set_error("trace", "tracing needs CUDA");
    return -1;
#else
    if (!h || !prompt || n_prompt <= 0 || !on_token || !on_rows || !seq_out) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    keryx_trace::Plan* plan = h->trace.get();
    if (!plan) {
        keryx_set_error("trace", "tracing is not set up");
        return -1;
    }
    if (seq_cap < (size_t) plan->state.n_seq_leaves * KERYX_TRACE_HASH) {
        keryx_set_error("trace", "the sequence buffer is too small");
        return -1;
    }
    using clock = std::chrono::steady_clock;
    const auto started = clock::now();
    const auto ms_since = [](clock::time_point t) {
        return std::chrono::duration<double, std::milli>(clock::now() - t).count();
    };
    h->last_prompt_tokens = 0; h->last_tokens = 0;
    h->last_prefill_ms = 0; h->last_decode_ms = 0;
    h->last_drafted = 0; h->last_accepted = 0;
    if (!keryx_trace_start(h, n_prompt)) return -1;
    const int n_ctx = (int)llama_n_ctx(h->ctx);
    const llama_vocab* vocab = llama_model_get_vocab(h->model);
    llama_batch batch = llama_batch_init((int32_t)keryx_trace::UBATCH, 0, 1);
    std::vector<uint8_t> rows;
    int fed = -1;
    for (int at = 0; at < n_prompt; at += (int)keryx_trace::UBATCH) {
        const int n = std::min(n_prompt - at, (int)keryx_trace::UBATCH);
        if (!keryx_trace_ubatch(h, batch, prompt + at, at, n, at + n == n_prompt, rows, on_rows, user)) {
            llama_batch_free(batch);
            return -1;
        }
    }
    h->last_prompt_tokens = n_prompt;
    h->last_prefill_ms = ms_since(started);
    const auto decoding = clock::now();
    fed = 0;
    int n_past = n_prompt;
    while (fed < max_tokens && n_past < n_ctx) {
        const llama_token tok = llama_sampler_sample(h->smpl, h->ctx, -1);
        if (llama_vocab_is_eog(vocab, tok) || !on_token(user, tok)) break;
        if (!keryx_trace_ubatch(h, batch, &tok, n_past, 1, true, rows, on_rows, user)) {
            fed = -1;
            break;
        }
        n_past++;
        fed++;
    }
    llama_batch_free(batch);
    h->last_tokens = std::max(fed, 0);
    h->last_decode_ms = ms_since(decoding);
    if (fed < 0) return -1;
    std::vector<uint8_t> seq;
    if (!plan->finish(seq)) {
        keryx_set_error("trace", "cannot finish the sequence leaves");
        return -1;
    }
    memcpy(seq_out, seq.data(), seq.size());
    return fed;
#endif
}

// Replays a traced generation for an opening: `tokens` = the prompt then every fed answer token
// (n_pos in all), run exactly as keryx_llama_generate_traced ran them, so every leaf comes out
// byte for byte (the caller checks the roots it gets through on_rows against the ones it kept).
// pos_wants holds triples (position, leaf, offset): that position leaf is copied to
// pos_out + offset. seq_wants holds pairs (leaf, offset): that whole sequence leaf is copied to
// seq_bytes_out + offset. The caller sizes both from the leaves' lengths. Returns 0, or -1 with
// keryx_llama_last_error; seq_out then holds every sequence leaf's hash.
KERYX_EXPORT int keryx_llama_replay(KeryxLlama* h, const int32_t* tokens, int n_pos, int n_prompt,
                                    const int64_t* pos_wants, int n_pos_wants, uint8_t* pos_out, size_t pos_bytes,
                                    const int64_t* seq_wants, int n_seq_wants, uint8_t* seq_bytes_out, size_t seq_bytes,
                                    int (*on_rows)(void*, int32_t, int32_t, int32_t, const uint8_t*), void* user,
                                    uint8_t* seq_out, size_t seq_cap) {
    keryx_last_error.clear();
#ifdef __APPLE__
    (void)h; (void)tokens; (void)n_pos; (void)n_prompt; (void)pos_wants; (void)n_pos_wants; (void)pos_out;
    (void)pos_bytes; (void)seq_wants; (void)n_seq_wants; (void)seq_bytes_out; (void)seq_bytes; (void)on_rows;
    (void)user; (void)seq_out; (void)seq_cap;
    keryx_set_error("trace", "tracing needs CUDA");
    return -1;
#else
    if (!h || !tokens || n_prompt <= 0 || n_pos < n_prompt || !on_rows || !seq_out) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    keryx_trace::Plan* plan = h->trace.get();
    if (!plan) {
        keryx_set_error("trace", "tracing is not set up");
        return -1;
    }
    const uint32_t L = plan->state.n_leaves;
    if (seq_cap < (size_t)plan->state.n_seq_leaves * KERYX_TRACE_HASH || n_pos > (int)llama_n_ctx(h->ctx)) {
        keryx_set_error("trace", "the replay does not fit");
        return -1;
    }
    std::vector<std::pair<uint32_t, int64_t>> seq;
    for (int i = 0; i < n_seq_wants; i++) seq.push_back({(uint32_t)seq_wants[2 * i], seq_wants[2 * i + 1]});
    if (!keryx_trace_start(h, n_prompt) || !plan->replay_begin(pos_bytes, seq_bytes, seq)) {
        if (keryx_last_error.empty()) keryx_set_error("trace", "cannot set up the replay");
        plan->replay_end(nullptr, 0, nullptr, 0);
        return -1;
    }
    llama_batch batch = llama_batch_init((int32_t)keryx_trace::UBATCH, 0, 1);
    std::vector<uint8_t> rows;
    std::vector<int64_t> offs;
    bool ok = true;
    for (int at = 0; at < n_pos && ok;) {
        const int n = at < n_prompt ? std::min(n_prompt - at, (int)keryx_trace::UBATCH) : 1;
        offs.assign((size_t)n * L, -1);
        for (int i = 0; i < n_pos_wants; i++) {
            const int64_t pos = pos_wants[3 * i], leaf = pos_wants[3 * i + 1];
            if (pos >= at && pos < at + n && leaf >= 0 && leaf < L) offs[(pos - at) * L + leaf] = pos_wants[3 * i + 2];
        }
        ok = plan->replay_rows(offs) &&
             keryx_trace_ubatch(h, batch, tokens + at, at, n, at >= n_prompt || at + n == n_prompt, rows, on_rows, user);
        at += n;
    }
    llama_batch_free(batch);
    std::vector<uint8_t> hashes;
    ok = ok && plan->finish(hashes);
    ok = plan->replay_end(pos_out, pos_bytes, seq_bytes_out, seq_bytes) && ok;
    if (!ok) {
        if (keryx_last_error.empty()) keryx_set_error("trace", "the replay failed");
        return -1;
    }
    memcpy(seq_out, hashes.data(), hashes.size());
    return 0;
#endif
}

// What the last generate on `h` spent: prompt tokens and their prefill, sampled tokens and
// their decode, tokens the MTP layer drafted and how many of them were kept. Returns false for
// a null handle.
KERYX_EXPORT bool keryx_llama_last_stats(KeryxLlama* h, int* prompt_tokens, int* tokens,
                                         double* prefill_ms, double* decode_ms, int* drafted,
                                         int* accepted) {
    if (!h) return false;
    std::lock_guard<std::mutex> g(h->gen_lock);
    *prompt_tokens = h->last_prompt_tokens;
    *tokens = h->last_tokens;
    *prefill_ms = h->last_prefill_ms;
    *decode_ms = h->last_decode_ms;
    *drafted = h->last_drafted;
    *accepted = h->last_accepted;
    return true;
}

// Runs one tiny graph on `gpu` so a library without kernels for that device fails here rather
// than on the first request. ggml aborts the process on a missing kernel image: call from a
// child process. Returns 0 on success.
KERYX_EXPORT int keryx_llama_probe_device(int gpu) {
    keryx_last_error.clear();
    keryx_install_log_filter();
    if (gpu < 0 || gpu >= ggml_backend_cuda_get_device_count()) {
        keryx_set_error("probe", "no such CUDA device");
        return 1;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(gpu);
    if (!backend) {
        keryx_set_error("probe", "ggml_backend_cuda_init failed");
        return 2;
    }
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* c = ggml_add(ctx, a, b);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    int rc = 0;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        keryx_set_error("probe", "ggml_backend_alloc_ctx_tensors failed");
        rc = 3;
    } else {
        const float v[4] = { 1.f, 2.f, 3.f, 4.f };
        ggml_backend_tensor_set(a, v, 0, sizeof v);
        ggml_backend_tensor_set(b, v, 0, sizeof v);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            keryx_set_error("probe", "graph compute failed");
            rc = 4;
        } else {
            float out[4] = { 0.f, 0.f, 0.f, 0.f };
            ggml_backend_tensor_get(c, out, 0, sizeof out);
            if (out[0] != 2.f || out[3] != 8.f) {
                keryx_set_error("probe", "graph compute returned wrong values");
                rc = 5;
            }
        }
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    ggml_backend_free(backend);
    return rc;
}

KERYX_EXPORT void keryx_llama_free(KeryxLlama* h) {
    if (!h) return;
#ifndef __APPLE__
    if (h->trace) {
        {
            std::lock_guard<std::mutex> lock(keryx_trace_mu);
            keryx_traces.erase(h->ctx);
        }
        h->trace->free_tables();
    }
#endif
    if (h->spec) common_speculative_free(h->spec);
    if (h->mtp) llama_free(h->mtp);
    if (h->smpl) llama_sampler_free(h->smpl);
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    for (auto b : h->buffers) ggml_backend_buffer_free(b);
    delete h;
}

} // extern "C"
