#pragma once

// Iluvatar (CoreX) GPU 线程束操作宏
// 仅在天数后端编译时生效（GGML_USE_ILUVATAR 已定义）
//
// 天数 GPU warpSize=64，_sync 系列函数的 mask 参数被硬件忽略，
// 仅 width 控制子组大小。本文件提供天数专用的 64-bit 掩码宏。

#ifdef GGML_USE_ILUVATAR

using ggml_warp_mask_t = unsigned long long;
#define GGML_WARP_FULL_MASK 0xFFFFFFFFFFFFFFFFull

// mask 参数被天数硬件忽略，width 控制子组大小
#define GGML_SHFL_XOR_SYNC(mask, val, offset, width) \
    __shfl_xor_sync(GGML_WARP_FULL_MASK, val, offset, width)

#define GGML_SHFL_UP_SYNC(mask, val, delta, width) \
    __shfl_up_sync(GGML_WARP_FULL_MASK, val, delta, width)

#define GGML_SHFL_SYNC(mask, val, src_lane, width) \
    __shfl_sync(GGML_WARP_FULL_MASK, val, src_lane, width)

// vote: mask 被忽略，全 64 线程参与。仅在全 warp 参与时使用。
#define GGML_ALL_SYNC(mask, pred) \
    __all_sync(GGML_WARP_FULL_MASK, pred)

#define GGML_ANY_SYNC(mask, pred) \
    __any_sync(GGML_WARP_FULL_MASK, pred)

#define GGML_BALLOT_SYNC(mask, pred) \
    __ballot_sync(GGML_WARP_FULL_MASK, pred)

#define GGML_SYNCWARP(mask) \
    __syncwarp(GGML_WARP_FULL_MASK)

// 寄存器预算控制：限制 VGPR 使用量以提升 occupancy
// N = 每个线程的 VGPR 上限。低 N → 高 occupancy，但过低会导致溢出到 LMEM
#define GGML_ILUVATAR_NUM_VGPR(N) __attribute__((iluvatar_num_vgpr(N)))

#endif // GGML_USE_ILUVATAR

#ifndef GGML_ILUVATAR_NUM_VGPR
#define GGML_ILUVATAR_NUM_VGPR(N)
#endif
