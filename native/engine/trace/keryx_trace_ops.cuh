// The hooks the patched ggml-cuda.cu calls for GGML_OP_CUSTOM (see keryx_trace_ops.cu).
#pragma once

#include "common.cuh"

// Whether `t` is a trace capture node this backend can run.
bool keryx_trace_supports(const ggml_tensor * t);

// Runs a capture node; false (logged) when its sources do not have the shape it was planned for.
bool keryx_trace_compute(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// The MMVQ Q8_1 rows (rows of k values, padded) a decode X capture wrote on `device` for `src1`
// during the running traced ubatch, or null: the product reads them instead of quantising src1.
const void * keryx_trace_q8_1(int device, const ggml_tensor * src1, int64_t k, int64_t rows);
