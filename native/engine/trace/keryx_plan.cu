// The work split of llama's kernels, made the same on every card of one architecture (ADR 0025).
//
// A float sum depends on its order. llama splits a product's or an attention's sums by the
// card's multiprocessor count (stream-k, flash-attention blocks, reductions), so two cards of
// one architecture with different counts sum in different orders and commit different trace
// bits. Here every card of an architecture splits for one count, the largest of its consumer
// cards: the split, and so every sum, depends on the architecture alone, and a smaller card runs
// the same blocks in more waves. Quantised products run one block per whole tile instead of
// stream-k (llama_patches.rs). cuBLAS picks its kernel per card, so the products llama would
// hand it are computed here in one order instead.
#include "common.cuh"

#include "keryx_plan.cuh"

#include <atomic>

extern "C" int keryx_plan_sms(int major, int minor, int sms) {
    switch (major * 10 + minor) {
        case 75:  return 72;   // Turing: TITAN RTX
        case 80:  return 108;  // A100
        case 86:  return 84;   // RTX 3090 Ti
        case 89:  return 128;  // RTX 4090
        case 120: return 170;  // RTX 5090
        default:  return sms;
    }
}

static std::atomic<uint64_t> keryx_vendor_products{0};

extern "C" uint64_t keryx_plan_vendor_products(void) {
    return keryx_vendor_products.load(std::memory_order_relaxed);
}

static constexpr int KERYX_GEMM_TILE = 64;  // outputs per block side
static constexpr int KERYX_GEMM_STEP = 16;  // k per shared-memory step
static constexpr int KERYX_GEMM_SIDE = 16;  // threads per block side; each owns 4x4 outputs

template <typename T> static __device__ __forceinline__ float keryx_to_f32(T x);
template <> __device__ __forceinline__ float keryx_to_f32<float>(float x) { return x; }
template <> __device__ __forceinline__ float keryx_to_f32<half>(half x) { return __half2float(x); }
template <> __device__ __forceinline__ float keryx_to_f32<nv_bfloat16>(nv_bfloat16 x) { return __bfloat162float(x); }

// d[n][m] = sum over k, ascending, of a[m][k]·b[n][k], one accumulator per output.
template <typename T>
static __global__ void __launch_bounds__(KERYX_GEMM_SIDE * KERYX_GEMM_SIDE)
keryx_gemm(const char * a, int64_t sa, const float * b, int64_t sb, float * d, int64_t sd, int M, int N, int K) {
    __shared__ float as[KERYX_GEMM_STEP][KERYX_GEMM_TILE + 1];
    __shared__ float bs[KERYX_GEMM_STEP][KERYX_GEMM_TILE + 1];
    const int tx = threadIdx.x % KERYX_GEMM_SIDE;
    const int ty = threadIdx.x / KERYX_GEMM_SIDE;
    const int m0 = blockIdx.x * KERYX_GEMM_TILE;
    const int n0 = blockIdx.y * KERYX_GEMM_TILE;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += KERYX_GEMM_STEP) {
        for (int e = threadIdx.x; e < KERYX_GEMM_TILE * KERYX_GEMM_STEP; e += blockDim.x) {
            const int r = e / KERYX_GEMM_STEP;
            const int c = e % KERYX_GEMM_STEP;
            const int k = k0 + c;
            const int m = m0 + r;
            const int n = n0 + r;
            as[c][r] = m < M && k < K ? keryx_to_f32(((const T *) (a + m * sa))[k]) : 0.0f;
            bs[c][r] = n < N && k < K ? b[n * sb + k] : 0.0f;
        }
        __syncthreads();
        const int steps = min(KERYX_GEMM_STEP, K - k0);  // past K nothing is added, not even a zero
        for (int kk = 0; kk < steps; ++kk) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = fmaf(as[kk][tx + KERYX_GEMM_SIDE * i], bs[kk][ty + KERYX_GEMM_SIDE * j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int m = m0 + tx + KERYX_GEMM_SIDE * i;
            const int n = n0 + ty + KERYX_GEMM_SIDE * j;
            if (m < M && n < N) {
                d[n * sd + m] = acc[i][j];
            }
        }
    }
}

bool keryx_plan_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const bool flat = src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    const bool types = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
                       src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    const bool rows = src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == sizeof(float) &&
                      dst->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0 && dst->nb[1] % sizeof(float) == 0;
    const bool fits = src0->ne[0] <= INT32_MAX && src0->ne[1] <= INT32_MAX && src1->ne[1] <= INT32_MAX;
    if (!flat || !types || !rows || !fits) {
        keryx_vendor_products.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const int K = (int) src0->ne[0];
    const int M = (int) src0->ne[1];
    const int N = (int) src1->ne[1];
    const dim3 grid((M + KERYX_GEMM_TILE - 1) / KERYX_GEMM_TILE, (N + KERYX_GEMM_TILE - 1) / KERYX_GEMM_TILE, 1);
    const dim3 block(KERYX_GEMM_SIDE * KERYX_GEMM_SIDE, 1, 1);
    const char * a = (const char *) src0->data;
    const float * b = (const float *) src1->data;
    float * d = (float *) dst->data;
    const int64_t sb = src1->nb[1] / sizeof(float);
    const int64_t sd = dst->nb[1] / sizeof(float);
    switch (src0->type) {
        case GGML_TYPE_F32:
            keryx_gemm<float><<<grid, block, 0, ctx.stream()>>>(a, src0->nb[1], b, sb, d, sd, M, N, K);
            break;
        case GGML_TYPE_F16:
            keryx_gemm<half><<<grid, block, 0, ctx.stream()>>>(a, src0->nb[1], b, sb, d, sd, M, N, K);
            break;
        default:
            keryx_gemm<nv_bfloat16><<<grid, block, 0, ctx.stream()>>>(a, src0->nb[1], b, sb, d, sd, M, N, K);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
