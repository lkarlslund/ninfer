#include "ops/linear/fp8_block/fp8_block_launch.h"

#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/linear/fp8/fp8_a16_small_t_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using BlockMmaSchedule = Fp8A16SmallTMmaSchedule<8, 8, 1,
    Fp8A16SmallTMmaCache::Default, Fp8A16SmallTMmaCache::Streaming,
    Fp8A16SmallTMmaActivationStage::PaddedZero>;

template <int N, int K, int Tokens>
void launch_block_mma_exact(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    using Geometry = Fp8Geometry<N, K>;
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), N};
    fp8_a16_small_t_mma_kernel<Geometry, Tokens, BlockMmaSchedule, Fp8ContiguousOutput, true>
        <<<N / BlockMmaSchedule::kRowsPerCta, BlockMmaSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata), weight.scales, output);
}

template <int Tokens>
void launch_block_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.n == 12288 && weight.k == 2560) {
        launch_block_mma_exact<12288, 2560, Tokens>(x, weight, out, stream);
    } else if (weight.n == 512 && weight.k == 2560) {
        launch_block_mma_exact<512, 2560, Tokens>(x, weight, out, stream);
    } else if (weight.n == 10240 && weight.k == 2560) {
        launch_block_mma_exact<10240, 2560, Tokens>(x, weight, out, stream);
    } else if (weight.n == 6144 && weight.k == 2560) {
        launch_block_mma_exact<6144, 2560, Tokens>(x, weight, out, stream);
    } else if (weight.n == 2560 && weight.k == 6144) {
        launch_block_mma_exact<2560, 6144, Tokens>(x, weight, out, stream);
    } else {
        throw std::invalid_argument("block FP8 MMA: unsupported exact shape");
    }
}

__device__ __forceinline__ float decode_e4m3(std::uint8_t word) {
    __nv_fp8_e4m3 value;
    value.__x = word;
    return static_cast<float>(value);
}

__global__ void fp8_block_dequantize_kernel(const std::uint8_t* __restrict__ codes,
                                            const float* __restrict__ scales,
                                            __nv_bfloat16* __restrict__ output, int n, int k) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t elements = static_cast<std::int64_t>(n) * k;
    if (index >= elements) { return; }
    const int row = static_cast<int>(index / k);
    const int column = static_cast<int>(index - static_cast<std::int64_t>(row) * k);
    const float value = decode_e4m3(codes[index]) * scales[(row / 128) * (k / 128) + column / 128];
    output[index] = __float2bfloat16_rn(value);
}

} // namespace

void launch_fp8_block_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1: launch_block_mma<1>(x, weight, out, stream); break;
    case 2: launch_block_mma<2>(x, weight, out, stream); break;
    case 3: launch_block_mma<3>(x, weight, out, stream); break;
    case 4: launch_block_mma<4>(x, weight, out, stream); break;
    case 5: launch_block_mma<5>(x, weight, out, stream); break;
    case 6: launch_block_mma<6>(x, weight, out, stream); break;
    case 7: launch_block_mma<7>(x, weight, out, stream); break;
    case 8: launch_block_mma<8>(x, weight, out, stream); break;
    default: throw std::invalid_argument("block FP8 small-T requires T in [1,8]");
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_fp8_block_dequantize(const Weight& weight, void* bf16_weight, cudaStream_t stream) {
    constexpr int threads = 256;
    const std::int64_t elements = static_cast<std::int64_t>(weight.n) * weight.k;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    fp8_block_dequantize_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight.qdata), static_cast<const float*>(weight.scales),
        static_cast<__nv_bfloat16*>(bf16_weight), weight.n, weight.k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
