#include "ops/linear/fp8_block/fp8_block_dispatch.h"

#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/fp8_block/fp8_block_launch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

void validate(const Weight& weight, LinearPolicy policy) {
    if (policy != LinearPolicy::A16Only || weight.qtype != QType::FP8_E4M3FN_BLOCK128_F32S ||
        weight.layout != QuantLayout::BlockScaleK128M128 || weight.scale_dtype != DType::FP32 ||
        weight.n <= 0 || weight.k <= 0 || weight.n % 128 != 0 || weight.k % 128 != 0 ||
        weight.qdata == nullptr || weight.scales == nullptr || weight.group != 128 ||
        weight.scale_ne[0] != weight.k / 128 || weight.scale_ne[1] != weight.n / 128) {
        throw std::invalid_argument("block FP8 linear: invalid weight or policy");
    }
}

std::size_t dequant_bytes(std::int32_t n, std::int32_t k) {
    const auto elements = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k);
    if (elements > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::overflow_error("block FP8 linear workspace overflow");
    }
    return static_cast<std::size_t>(elements * 2);
}

} // namespace

std::size_t fp8_block_linear_workspace_capacity_bytes(
    std::int32_t output_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t min_tokens, std::int32_t max_tokens) {
    if (policy != LinearPolicy::A16Only || min_tokens <= 0 || max_tokens < min_tokens ||
        output_rows <= 0 || input_rows <= 0 || output_rows % 128 || input_rows % 128) {
        throw std::invalid_argument("block FP8 linear workspace: invalid profile");
    }
    (void)select_bf16_a16_launch(output_rows, input_rows, max_tokens);
    return max_tokens > 8 ? dequant_bytes(output_rows, input_rows) : 0;
}

void fp8_block_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                        WorkspaceArena* workspace, cudaStream_t stream) {
    validate(weight, policy);
    if (x.ne[1] <= 8) {
        launch_fp8_block_small_t(x, weight, out, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("block FP8 large-T linear requires caller workspace");
    }
    auto scope = workspace->scope();
    DeviceSpan storage = workspace->alloc_bytes(dequant_bytes(weight.n, weight.k));
    launch_fp8_block_dequantize(weight, storage.data, stream);
    Weight bf16{};
    bf16.payload = storage.data;
    bf16.payload_bytes = storage.bytes;
    bf16.qdata = storage.data;
    bf16.qtype = QType::BF16_CTRL;
    bf16.layout = QuantLayout::Contiguous;
    bf16.n = weight.n;
    bf16.k = weight.k;
    bf16.ndim = 2;
    bf16.shape[0] = weight.n;
    bf16.shape[1] = weight.k;
    bf16.padded_shape[0] = weight.n;
    bf16.padded_shape[1] = weight.k;
    bf16_dispatch(x, bf16, out, policy, stream);
}

} // namespace ninfer::ops::detail
