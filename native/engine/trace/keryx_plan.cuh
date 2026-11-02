// The work split of llama's kernels, made the same on every card of one architecture (ADR 0025).
// build.rs copies this file into ggml/src/ggml-cuda/ and patches ggml-cuda.cu to use it.
#pragma once

#include "common.cuh"

#include <cstdint>

// The multiprocessor count llama's kernels split their work for on a card of compute capability
// major.minor with `sms` multiprocessors: one count per known architecture, the card's own count
// on an architecture the table does not know.
extern "C" int keryx_plan_sms(int major, int minor, int sms);

// dst = src0·src1 for a product llama would hand to cuBLAS, whose choice of kernel follows the
// card. Each output sums its row in one fixed order. False when the shape is not one it takes;
// llama then uses cuBLAS, and keryx_plan_vendor_products() counts it.
bool keryx_plan_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// How many products went to cuBLAS in this process.
extern "C" uint64_t keryx_plan_vendor_products(void);
