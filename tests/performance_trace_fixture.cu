// Small real Op execution for qualifying the offline trace join without loading a model.
// Run under Nsight Systems with graph-node tracing; it checks eager and cloned-graph output.
#include "core/device.h"
#include "core/nvtx.h"
#include "core/performance.h"
#include "ninfer/ops/hyperconnection.h"
#include "ops/flash_next_work.h"

#include <cuda_bf16.h>
#include <stdexcept>
#include <vector>

// Independent inventory totals for the modeled projections and packed expert traffic.
// Protect against counting all 512 banks at T=1, omitting K16 scales, or counting FLOPs
// once per unique expert instead of once per token assignment.
static_assert(ninfer::ops::flash_next_work::moe(1, true).bytes_min == 40115360);
static_assert(ninfer::ops::flash_next_work::moe(1, true).bytes_max == 40115360);
static_assert(ninfer::ops::flash_next_work::moe(1, true).bf16_flops == 12456960);
static_assert(ninfer::ops::flash_next_work::moe(8, true).nvfp4_flops == 786432000);
static_assert(ninfer::ops::flash_next_work::moe(1, false).bf16_flops == 110760960);
static_assert(ninfer::ops::flash_next_work::gdn(1, 1, false).bf16_flops == 115834880);

int main() {
    using namespace ninfer;
    constexpr int tokens = 8;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    DeviceBuffer input_storage(2 * 2560 * tokens), output_storage(2 * 10240 * tokens);
    Tensor input(input_storage.p, DType::BF16, {2560, tokens});
    Tensor output(output_storage.p, DType::BF16, {10240, tokens});
    std::vector<__nv_bfloat16> host(2560 * tokens, __float2bfloat16(1.0F));
    CUDA_CHECK(
        cudaMemcpyAsync(input.data, host.data(), input.bytes(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t graph, clone;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    ops::hyperconnection_repeat(input, output, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphClone(&clone, graph));
    cudaGraphExec_t first, second;
    CUDA_CHECK(cudaGraphInstantiate(&first, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphInstantiate(&second, clone, nullptr, nullptr, 0));
    {
        NINFER_PERF_SCOPE("ninfer.region/1|measured");
        nvtx::ScopedRange phase(nvtx::Name::Decode, nvtx::Category::Decode);
        ops::hyperconnection_repeat(input, output, stream);
        CUDA_CHECK(cudaGraphLaunch(first, stream));
        CUDA_CHECK(cudaGraphLaunch(second, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    host.resize(10240 * tokens);
    CUDA_CHECK(cudaMemcpy(host.data(), output.data, output.bytes(), cudaMemcpyDeviceToHost));
    for (auto value : host) {
        if (__bfloat162float(value) != 1.0F) { throw std::runtime_error("graph output mismatch"); }
    }
    CUDA_CHECK(cudaGraphExecDestroy(first));
    CUDA_CHECK(cudaGraphExecDestroy(second));
    CUDA_CHECK(cudaGraphDestroy(clone));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
