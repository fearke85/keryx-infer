// Textual edits of llama.cpp for the H15 trace (ADR 0023), applied by build.rs in order.
// Each is (file, text to find, replacement, how many times the text must occur): a new llama
// tag that moves an anchor fails the build instead of building an engine without the trace.
// The trace's own CUDA files (tools/keryx-llama/trace) are copied into ggml-cuda beside them.
//
// What they do:
// - llama-context.cpp: hands every ubatch graph to the engine before it is allocated
//   (keryx_llama_graph_hook), with room for the capture nodes it adds.
// - llama-kv-cache.cpp: lets the engine turn off the Hadamard rotation of a quantised KV cache,
//   which would put rotated keys and values in the cache the trace commits.
// - ggml-cuda.cu: runs the capture nodes (GGML_OP_CUSTOM) on the card. Every card of one
//   architecture splits its work for the same multiprocessor count, and the products llama
//   would hand to cuBLAS are summed in one order (keryx_plan.cu, ADR 0025).
// - mmq.cuh: quantised products run one block per whole tile, never a stream-k split of a tile.
// - gated_delta_net.cu: copies out the recurrent states the trace checkpoints, and writes its
//   output rounded to f16, as a record commits it: the node rebuilds the gated output from the
//   record's f16 o, which loses a head's tiny outputs (f16 subnormals), so llama must use them too.
// - mmvq.cu: a decode product reads the Q8_1 rows its X capture just wrote instead of quantising
//   its input again (keryx_trace_q8_1).
// - fattn.cu: tells the trace which kernel runs a FLASH_ATTN_EXT, so the softmax mass it commits
//   rounds the query the way that kernel does (Q8_1 in the vector kernel, f16 in the MMA one).
// - unary.cu: softplus as log1p(exp(x)), the node's formula. Under fast-math, log(1 + exp(x))
//   loses most of a small result (x below about -11), and the node then refuses the recurrent
//   decay a record commits (|g - a·softplus| over 1/32 of g).
const LLAMA_PATCHES: &[(&str, &str, &str, usize)] = &[
    (
        "src/llama-context.cpp",
        r#"#include <stdexcept>
#include <string>
"#,
        r#"#include <stdexcept>
#include <string>

// keryx: trace hook (the miner's build.rs). Called with every ubatch graph before it is
// allocated, with the scheduler that will place it; false fails the ubatch. The margin is room
// for the nodes it adds.
extern "C" {
bool (*keryx_llama_graph_hook)(const llama_context *, ggml_context *, ggml_cgraph *, uint32_t, ggml_backend_sched_t) = nullptr;
uint32_t keryx_llama_graph_margin = 0;
}
"#,
        1,
    ),
    (
        "src/llama-context.cpp",
        r#"        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
"#,
        r#"        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (keryx_llama_graph_hook && !keryx_llama_graph_hook(this, res->get_ctx(), gf, ubatch.n_tokens, sched.get())) {
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
"#,
        1,
    ),
    (
        "src/llama-context.cpp",
        r#"    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;
"#,
        r#"    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    if (keryx_llama_graph_hook && !keryx_llama_graph_hook(this, res->get_ctx(), gf, n_tokens, sched.get())) {
        return nullptr;
    }
"#,
        1,
    ),
    (
        "src/llama-context.cpp",
        r#"        return std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());"#,
        r#"        return keryx_llama_graph_margin + std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());"#,
        1,
    ),
    (
        "src/llama-context.cpp",
        r#"    uint32_t res = std::max<uint32_t>(1024u, 8u*model.n_tensors());"#,
        r#"    uint32_t res = keryx_llama_graph_margin + std::max<uint32_t>(1024u, 8u*model.n_tensors());"#,
        1,
    ),
    (
        "src/llama-kv-cache.cpp",
        r#"#include "llama-kv-cache.h"
"#,
        r#"#include "llama-kv-cache.h"

// keryx: the trace commits K and V as they are, so the miner can turn the rotation off.
extern "C" { bool keryx_llama_attn_rot = true; }
"#,
        1,
    ),
    (
        "src/llama-kv-cache.cpp",
        r#"        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;"#,
        r#"        const bool attn_rot_disable = !keryx_llama_attn_rot || (LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false);"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/ggml-cuda.cu",
        r#"#include "ggml-cuda/common.cuh"
"#,
        r#"#include "ggml-cuda/common.cuh"
#include "ggml-cuda/keryx_plan.cuh"
#include "ggml-cuda/keryx_trace_ops.cuh"
"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/ggml-cuda.cu",
        r#"        info.devices[id].nsm        = prop.multiProcessorCount;"#,
        r#"        info.devices[id].nsm        = keryx_plan_sms(prop.major, prop.minor, prop.multiProcessorCount);"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/ggml-cuda.cu",
        r#"static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
"#,
        r#"static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (keryx_plan_gemm(ctx, src0, src1, dst)) {
        return;
    }
"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/ggml-cuda.cu",
        r#"        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;"#,
        r#"        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_CUSTOM:
            if (!keryx_trace_compute(ctx, dst)) {
                return false;
            }
            break;"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/ggml-cuda.cu",
        r#"    switch (op->op) {
        case GGML_OP_UNARY:"#,
        r#"    switch (op->op) {
        case GGML_OP_CUSTOM:
            return keryx_trace_supports(op);
        case GGML_OP_UNARY:"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/gated_delta_net.cu",
        r#"#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"
"#,
        r#"#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"
#include "keryx_trace_abi.h"
"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/gated_delta_net.cu",
        r#"                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {"#,
        r#"                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     float *       ckpt,
                                     const int32_t * ckpt_pos0,
                                     int           ckpt_interval) {"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/gated_delta_net.cu",
        r#"    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;"#,
        r#"    // keryx: the trace checkpoints the state after every position p with (p + 1) % interval == 0.
    const int ckpt_p0 = ckpt ? *ckpt_pos0 : 0;
    const int ckpt_first = ckpt ? (ckpt_p0 + ckpt_interval) / ckpt_interval : 0;

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/gated_delta_net.cu",
        r#"        attn_data += S_v * H;
"#,
        r#"        attn_data += S_v * H;

        if (ckpt && (ckpt_p0 + t + 1) % ckpt_interval == 0) {
            float * c = ckpt + (((int64_t) ((ckpt_p0 + t + 1) / ckpt_interval - ckpt_first) * H + h_idx) * S_v + col) * S_v;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                c[r * warp_size + lane] = s_shard[r];
            }
        }
"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/gated_delta_net.cu",
        r#"        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {"#,
        r#"        float scale, int64_t state_slot_stride, int K,
        float * ckpt, const int32_t * ckpt_pos0, int ckpt_interval, cudaStream_t stream) {"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/gated_delta_net.cu",
        r#"                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);"#,
        r#"                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, ckpt, ckpt_pos0, ckpt_interval);"#,
        4,
    ),
    (
        "ggml/src/ggml-cuda/gated_delta_net.cu",
        r#"                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);"#,
        r#"                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, ckpt, ckpt_pos0, ckpt_interval, stream);"#,
        4,
    ),
    (
        "ggml/src/ggml-cuda/gated_delta_net.cu",
        r#"    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;
"#,
        r#"    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // keryx: where the trace wants the checkpointed states of this layer, if it does.
    float * ckpt = nullptr;
    const int32_t * ckpt_pos0 = nullptr;
    int ckpt_interval = 1;
    {
        const keryx_trace_gdn * kx = nullptr;
        memcpy(&kx, (const char *) dst->op_params + KERYX_TRACE_GDN_PARAM, sizeof(kx));
        if (kx && kx->magic == KERYX_TRACE_MAGIC) {
            for (const keryx_trace_seq & sq : kx->state->seq) {
                if (sq.device == ctx.device && sq.ckpt) {
                    ckpt = sq.ckpt;
                    ckpt_pos0 = sq.pos0;
                    ckpt_interval = (int) kx->interval;
                }
            }
            GGML_ASSERT(ckpt && !keep_rs && n_seqs == 1);
        }
    }
"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/mmvq.cu",
        r#"#include "mmvq.cuh"
"#,
        r#"#include "mmvq.cuh"
#include "keryx_trace_ops.cuh"
"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/mmvq.cu",
        r#"    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }
"#,
        r#"    // keryx: a decode trace capture may have just quantised src1 this way; read its rows.
    const void * keryx_q8_1 = keryx_trace_q8_1(ctx.device, src1, ne10, ne11*ne12*ne13);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    if (!keryx_q8_1) {
        src1_q8_1.alloc(ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }
    const void * src1_q = keryx_q8_1 ? keryx_q8_1 : (const void *) src1_q8_1.get();
"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/mmvq.cu",
        r#"        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,
"#,
        r#"        src0->data, src0->type, src1_q, ids_d, fusion_local, dst_d, ne00,
"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/unary.cu",
        r#"    return (x > 20.0f) ? x : logf(1.0f + expf(x));
"#,
        r#"    return (x > 20.0f) ? x : log1pf(expf(x));
"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/gated_delta_net.cu",
        r#"                attn_data[col] = attn_col * scale;
"#,
        r#"                attn_data[col] = __half2float(__float2half_rn(attn_col * scale));
"#,
        2,
    ),
    (
        "ggml/src/ggml-cuda/fattn.cu",
        r#"size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
"#,
        r#"// keryx: which kernel runs this FLASH_ATTN_EXT (keryx_trace_ops.cu rounds the query as it does).
int keryx_fattn_kernel(int device, const ggml_tensor * dst);
int keryx_fattn_kernel(int device, const ggml_tensor * dst) {
    return (int) ggml_cuda_get_best_fattn_kernel(device, dst);
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
"#,
        1,
    ),
    (
        "ggml/src/ggml-cuda/mmq.cuh",
        r#"    const dim3 block_nums_stream_k(GGML_CUDA_CC_IS_NVIDIA(cc) && tiles_efficiency_percent >= 90 ? ntiles_dst : nsm, 1, 1);"#,
        r#"    // keryx: one block per whole tile, so no tile's sum is split (ADR 0025). Measured on the
    // RTX 3060, 3070, 3080 and 5070 Ti, prefill as fast as llama's stream-k split per card.
    GGML_UNUSED(tiles_efficiency_percent);
    const dim3 block_nums_stream_k(ntiles_dst, 1, 1);"#,
        1,
    ),
];
