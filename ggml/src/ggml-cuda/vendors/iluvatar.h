#pragma once

// CoreX CUDA Runtime API — API 名称与 NVIDIA CUDA 相同
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

// BF16/FP16 支持（MR100 支持 BF16）
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "iluvatar_warp.cuh"

// FP8 不支持 — 不 include cuda_fp8.h，不定义 FP8_AVAILABLE
// FP4 不支持 — 不 include cuda_fp4.h
// PTX 不支持 — 不定义任何 PTX 版本宏

#ifdef GGML_USE_NCCL
#include <nccl.h>
#endif // GGML_USE_NCCL

// CUDART_VERSION 兼容性
// CoreX SDK 定义自己的 CUDART_VERSION，确保与 CUDA 12.x 兼容
// 如果 CoreX 的 CUDART_VERSION 不满足要求，可在此覆盖：
// #undef CUDART_VERSION
// #define CUDART_VERSION 12000

// CoreX cuBLAS 兼容性
// CoreX cuBLAS 不提供 CUBLAS_TF32_TENSOR_OP_MATH，映射到 CUBLAS_TENSOR_OP_MATH
#ifndef CUBLAS_TF32_TENSOR_OP_MATH
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#endif

// CoreX cuBLAS 使用旧版 API（cudaDataType 作为 compute type 参数），
// 而 llama.cpp 使用新版 API（cublasComputeType_t）。
// 提供 cuBLAS 函数包装器，将 cublasComputeType_t 映射为正确的 cudaDataType。
// 注意：这些包装器和宏必须在 #include <cublas_v2.h> 之后、所有调用点之前生效。
// 此文件由 common.cuh 通过 #elif defined(GGML_USE_ILUVATAR) 条件引入，保证顺序正确。
static inline cudaDataType iluvatar_compute_type_to_cuda(cublasComputeType_t ct) {
    switch (ct) {
        case CUBLAS_COMPUTE_16F:          return CUDA_R_16F;
        case CUBLAS_COMPUTE_16F_PEDANTIC: return CUDA_R_16F;
        case CUBLAS_COMPUTE_32F:          return CUDA_R_32F;
        case CUBLAS_COMPUTE_32F_PEDANTIC: return CUDA_R_32F;
        case CUBLAS_COMPUTE_32F_FAST_16F: return CUDA_R_16F;
        case CUBLAS_COMPUTE_32F_FAST_16BF: return CUDA_R_16BF;
        case CUBLAS_COMPUTE_32F_FAST_TF32: return CUDA_R_32F;
        default:
            fprintf(stderr, "iluvatar_compute_type_to_cuda: unknown cublasComputeType_t %d, falling back to CUDA_R_32F\n", (int)ct);
            return CUDA_R_32F;
    }
}
static inline cublasStatus_t cublasGemmEx_compat(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void *alpha,
    const void *A, cudaDataType Atype, int lda,
    const void *B, cudaDataType Btype, int ldb,
    const void *beta, void *C, cudaDataType Ctype, int ldc,
    cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
    return cublasGemmEx(handle, transa, transb, m, n, k, alpha,
                        A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc,
                        iluvatar_compute_type_to_cuda(computeType), algo);
}
static inline cublasStatus_t cublasGemmStridedBatchedEx_compat(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void *alpha,
    const void *A, cudaDataType Atype, int lda, long long int strideA,
    const void *B, cudaDataType Btype, int ldb, long long int strideB,
    const void *beta, void *C, cudaDataType Ctype, int ldc, long long int strideC,
    int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
    return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha,
                        A, Atype, lda, strideA, B, Btype, ldb, strideB,
                        beta, C, Ctype, ldc, strideC, batchCount,
                        iluvatar_compute_type_to_cuda(computeType), algo);
}
static inline cublasStatus_t cublasGemmBatchedEx_compat(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void *alpha,
    const void *Aarray[], cudaDataType Atype, int lda,
    const void *Barray[], cudaDataType Btype, int ldb,
    const void *beta, void *Carray[], cudaDataType Ctype, int ldc,
    int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
    return cublasGemmBatchedEx(handle, transa, transb, m, n, k, alpha,
                        Aarray, Atype, lda, Barray, Btype, ldb,
                        beta, Carray, Ctype, ldc, batchCount,
                        iluvatar_compute_type_to_cuda(computeType), algo);
}
// 宏重定义 cuBLAS 公共 API：将 cublasComputeType_t 参数转换为 cudaDataType。
// 使用 push_macro/pop_macro 隔离，防止影响 compat 函数内部调用或其他头文件。
// 如果未来引入第三方库头文件也引用了 cublasGemmEx，需评估宏展开的影响。
#pragma push_macro("cublasGemmEx")
#pragma push_macro("cublasGemmStridedBatchedEx")
#pragma push_macro("cublasGemmBatchedEx")
#define cublasGemmEx cublasGemmEx_compat
#define cublasGemmStridedBatchedEx cublasGemmStridedBatchedEx_compat
#define cublasGemmBatchedEx cublasGemmBatchedEx_compat

// CoreX VMM 属性名不同
#ifndef CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#endif

// CoreX cudaStreamWaitEvent 需要 3 个参数（NVIDIA 允许省略第 3 个 flags 参数）
// 在 ggml-cuda.cu 中直接修复调用（添加第三个参数 0）
