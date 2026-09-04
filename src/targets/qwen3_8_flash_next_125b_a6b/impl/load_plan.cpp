#include "targets/qwen3_8_flash_next_125b_a6b/impl/load_plan.h"

#include "artifact/typed_binding.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next_125b_a6b {
namespace {

using artifact::NumericFormat;
using artifact::ObjectHandle;

ObjectHandle device(artifact::Binder& binder, std::string_view name, NumericFormat format,
                    std::initializer_list<std::uint64_t> shape,
                    artifact::TensorPlacement placement = artifact::TensorPlacement::Device) {
    return artifact::bind_tensor(binder, name, format, shape, placement);
}

ObjectHandle expert_bank(artifact::Binder& binder, std::string_view name,
                         std::initializer_list<std::uint64_t> shape,
                         artifact::TensorPlacement placement = artifact::TensorPlacement::Device) {
    const ObjectHandle handle = binder.require_tensor(
        name, NumericFormat::NVFP4, artifact::StorageLayout::ExpertBlockScaleK16M128x4V1,
        std::span<const std::uint64_t>(shape.begin(), shape.size()));
    if (placement == artifact::TensorPlacement::Device) { binder.materialize_on_device(handle); }
    return handle;
}

HyperConnectionPlan
bind_hc(artifact::Binder& binder, const std::string& prefix,
        artifact::TensorPlacement placement = artifact::TensorPlacement::Device) {
    return {
        .block_inject = device(binder, prefix + "block_inject_weight.weight", NumericFormat::BF16,
                               {4, 10240}, placement),
        .norm = device(binder, prefix + "hc_norm.weight", NumericFormat::BF16, {10240}, placement),
        .down = device(binder, prefix + "input_mix_weight_down.weight", NumericFormat::BF16,
                       {320, 10240}, placement),
        .up   = device(binder, prefix + "input_mix_weight_up.weight", NumericFormat::BF16,
                       {10240, 320}, placement),
    };
}

FinalHyperConnectionPlan
bind_final_hc(artifact::Binder& binder, const std::string& prefix,
              artifact::TensorPlacement placement = artifact::TensorPlacement::Device) {
    return {
        .norm = device(binder, prefix + "hc_norm.weight", NumericFormat::BF16, {10240}, placement),
        .down = device(binder, prefix + "input_mix_weight_down.weight", NumericFormat::BF16,
                       {320, 10240}, placement),
        .up   = device(binder, prefix + "input_mix_weight_up.weight", NumericFormat::BF16,
                       {10240, 320}, placement),
    };
}

MoePlan bind_main_moe(artifact::Binder& binder, const std::string& prefix) {
    return {
        .router      = device(binder, prefix + "gate.weight", NumericFormat::BF16, {512, 2560}),
        .shared_gate = device(binder, prefix + "shared_expert.gate_proj.weight",
                              NumericFormat::BF16, {640, 2560}),
        .shared_up   = device(binder, prefix + "shared_expert.up_proj.weight", NumericFormat::BF16,
                              {640, 2560}),
        .shared_down = device(binder, prefix + "shared_expert.down_proj.weight",
                              NumericFormat::BF16, {2560, 640}),
        .shared_scale =
            device(binder, prefix + "shared_expert_gate.weight", NumericFormat::BF16, {1, 2560}),
        .routed_gate_up = expert_bank(binder, prefix + "experts.gate_up", {512, 1280, 2560}),
        .routed_gate_up_input_divisors =
            device(binder, prefix + "experts.gate_up_input_divisors", NumericFormat::FP32, {512}),
        .routed_down = expert_bank(binder, prefix + "experts.down", {512, 2560, 640}),
        .routed_down_input_divisors =
            device(binder, prefix + "experts.down_input_divisors", NumericFormat::FP32, {512}),
    };
}

MoePlan bind_mtp_moe(artifact::Binder& binder, const std::string& prefix,
                     artifact::TensorPlacement placement) {
    return {
        .router =
            device(binder, prefix + "gate.weight", NumericFormat::BF16, {512, 2560}, placement),
        .shared_gate  = device(binder, prefix + "shared_expert.gate_proj.weight",
                               NumericFormat::BF16, {640, 2560}, placement),
        .shared_up    = device(binder, prefix + "shared_expert.up_proj.weight", NumericFormat::BF16,
                               {640, 2560}, placement),
        .shared_down  = device(binder, prefix + "shared_expert.down_proj.weight",
                               NumericFormat::BF16, {2560, 640}, placement),
        .shared_scale = device(binder, prefix + "shared_expert_gate.weight", NumericFormat::BF16,
                               {1, 2560}, placement),
        .routed_gate_up = device(binder, prefix + "experts.gate_up_proj", NumericFormat::BF16,
                                 {512, 1280, 2560}, placement),
        .routed_down    = device(binder, prefix + "experts.down_proj", NumericFormat::BF16,
                                 {512, 2560, 640}, placement),
    };
}

FullAttentionPlan
bind_attention(artifact::Binder& binder, const std::string& prefix,
               NumericFormat projection_format,
               artifact::TensorPlacement placement = artifact::TensorPlacement::Device) {
    return {
        .query_gate =
            device(binder, prefix + "q_proj.weight", projection_format, {12288, 2560}, placement),
        .key =
            device(binder, prefix + "k_proj.weight", projection_format, {512, 2560}, placement),
        .value =
            device(binder, prefix + "v_proj.weight", projection_format, {512, 2560}, placement),
        .output =
            device(binder, prefix + "o_proj.weight", projection_format, {2560, 6144}, placement),
        .query_norm =
            device(binder, prefix + "q_norm.weight", NumericFormat::BF16, {256}, placement),
        .key_norm = device(binder, prefix + "k_norm.weight", NumericFormat::BF16, {256}, placement),
        .index_query_key  = device(binder, prefix + "indexer.index_qk_proj.weight",
                                   NumericFormat::BF16, {640, 2560}, placement),
        .index_query_norm = device(binder, prefix + "indexer.q_layernorm.weight",
                                   NumericFormat::BF16, {128}, placement),
        .index_key_norm = device(binder, prefix + "indexer.k_layernorm.weight", NumericFormat::BF16,
                                 {128}, placement),
    };
}

GdnPlan bind_gdn(artifact::Binder& binder, const std::string& prefix,
                 NumericFormat projection_format) {
    return {
        .a_log       = device(binder, prefix + "A_log", NumericFormat::FP32, {48}),
        .dt_bias     = device(binder, prefix + "dt_bias", NumericFormat::FP32, {48}),
        .convolution = device(binder, prefix + "conv1d.weight", NumericFormat::BF16, {4, 10240}),
        .a_projection =
            device(binder, prefix + "in_proj_a.weight", NumericFormat::BF16, {48, 2560}),
        .b_projection =
            device(binder, prefix + "in_proj_b.weight", NumericFormat::BF16, {48, 2560}),
        .query_key_value =
            device(binder, prefix + "in_proj_qkv.weight", projection_format, {10240, 2560}),
        .output_gate =
            device(binder, prefix + "in_proj_z.weight", projection_format, {6144, 2560}),
        .norm   = device(binder, prefix + "norm.weight", NumericFormat::BF16, {128}),
        .output = device(binder, prefix + "out_proj.weight", projection_format, {2560, 6144}),
    };
}

PlePlan bind_ple(artifact::Binder& binder, const std::string& prefix) {
    PlePlan out{
        .convolution = device(binder, prefix + "conv1d.weight", NumericFormat::BF16, {4, 10240}),
        .key_projection =
            device(binder, prefix + "key_proj.weight", NumericFormat::BF16, {10240, 2560}),
        .convolution_norm =
            device(binder, prefix + "norm_conv.weight", NumericFormat::BF16, {10240}),
        .key_norm   = device(binder, prefix + "norm_key.weight", NumericFormat::BF16, {10240}),
        .query_norm = device(binder, prefix + "norm_query.weight", NumericFormat::BF16, {10240}),
        .value_projection =
            device(binder, prefix + "value_proj.weight", NumericFormat::BF16, {2560, 2560}),
        .embedding_scale = device(binder, prefix + "ple_embedding.ngram_embedding.weight_scale",
                                  NumericFormat::BF16, {1}),
    };
    out.embedding_table =
        artifact::bind_file_backed_tensor(binder, prefix + "ple_embedding.ngram_embedding.weight",
                                          NumericFormat::FP8_E4M3FN, {320001536, 160});
    return out;
}

VisionPlan bind_vision(artifact::Binder& binder, artifact::TensorPlacement placement) {
    VisionPlan out;
    out.patch_embedding = device(binder, "vision/patch_embedding", NumericFormat::Q6G64_F16S,
                                 {1152, 1536}, placement);
    out.patch_embedding_bias =
        device(binder, "vision/patch_embedding_bias", NumericFormat::BF16, {1152}, placement);
    out.position_embedding =
        device(binder, "vision/position_embedding", NumericFormat::BF16, {2304, 1152}, placement);
    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const std::string p = "vision/layers/" + std::to_string(layer) + "/";
        auto& target        = out.layers[layer];
        target.qkv =
            device(binder, p + "attention/qkv", NumericFormat::Q4G64_F16S, {3456, 1152}, placement);
        target.qkv_bias =
            device(binder, p + "attention/qkv_bias", NumericFormat::BF16, {3456}, placement);
        target.output = device(binder, p + "attention/output", NumericFormat::Q5G64_F16S,
                               {1152, 1152}, placement);
        target.output_bias =
            device(binder, p + "attention/output_bias", NumericFormat::BF16, {1152}, placement);
        target.fc1 =
            device(binder, p + "mlp/fc1", NumericFormat::Q4G64_F16S, {4304, 1152}, placement);
        target.fc1_bias =
            device(binder, p + "mlp/fc1_bias", NumericFormat::BF16, {4304}, placement);
        target.fc2 =
            device(binder, p + "mlp/fc2", NumericFormat::Q5G64_F16S, {1152, 4304}, placement);
        target.fc2_bias =
            device(binder, p + "mlp/fc2_bias", NumericFormat::BF16, {1152}, placement);
        target.norm1_weight =
            device(binder, p + "norm1/weight", NumericFormat::BF16, {1152}, placement);
        target.norm1_bias =
            device(binder, p + "norm1/bias", NumericFormat::BF16, {1152}, placement);
        target.norm2_weight =
            device(binder, p + "norm2/weight", NumericFormat::BF16, {1152}, placement);
        target.norm2_bias =
            device(binder, p + "norm2/bias", NumericFormat::BF16, {1152}, placement);
    }
    out.merger_fc1 =
        device(binder, "vision/merger/fc1", NumericFormat::W8G32_F16S, {4608, 4608}, placement);
    out.merger_fc1_bias =
        device(binder, "vision/merger/fc1_bias", NumericFormat::BF16, {4608}, placement);
    out.merger_fc2 =
        device(binder, "vision/merger/fc2", NumericFormat::W8G32_F16S, {2560, 4608}, placement);
    out.merger_fc2_bias =
        device(binder, "vision/merger/fc2_bias", NumericFormat::BF16, {2560}, placement);
    out.merger_norm_weight =
        device(binder, "vision/merger/norm/weight", NumericFormat::BF16, {1152}, placement);
    out.merger_norm_bias =
        device(binder, "vision/merger/norm/bias", NumericFormat::BF16, {1152}, placement);
    return out;
}

} // namespace

ArtifactLoadPlan plan_artifact(artifact::Binder& binder,
                               qwen3_8_flash_next::StartupFeatures features,
                               detail::WeightsProfile profile) {
    static constexpr std::array<std::string_view, 6> resources = {
        "frontend/tokenizer.json",           "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",      "frontend/generation_config.json",
        "frontend/preprocessor_config.json", "frontend/video_preprocessor_config.json",
    };
    ArtifactLoadPlan out;
    out.bindings.features = features;
    out.bindings.projection_format =
        profile == detail::WeightsProfile::Nvfp4 ? NumericFormat::BF16
                                                 : NumericFormat::FP8_E4M3FN_BLOCK128_F32S;
    for (std::size_t i = 0; i < resources.size(); ++i) {
        out.bindings.frontend[i] = artifact::bind_raw_resource(binder, resources[i]);
    }
    out.bindings.token_embedding = device(binder, "model.language_model.embed_tokens.weight",
                                          NumericFormat::BF16, {248320, 2560});
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        auto& target             = out.bindings.text_layers[layer];
        const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
        target.attention_hc      = bind_hc(binder, prefix + "attn_hyper_connection.");
        target.is_full_attention = layer >= 3 && (layer - 3) % 4 == 0;
        if (target.is_full_attention) {
            target.attention = bind_attention(binder, prefix + "self_attn.",
                                              out.bindings.projection_format);
        } else {
            target.gdn = bind_gdn(binder, prefix + "linear_attn.",
                                  out.bindings.projection_format);
        }
        target.has_ple = layer == 1;
        if (target.has_ple) {
            target.ple    = bind_ple(binder, prefix + "ple.");
            out.ple_table = target.ple.embedding_table;
        }
        target.mlp_hc = bind_hc(binder, prefix + "mlp_hyper_connection.");
        target.moe    = bind_main_moe(binder, prefix + "mlp.");
    }
    out.bindings.final_hc = bind_final_hc(binder, "model.language_model.hyper_connection_mixer.");
    out.bindings.output_head =
        device(binder, "lm_head.weight", NumericFormat::BF16, {248320, 2560});
    const artifact::TensorPlacement proposal_placement =
        features.optimized_proposal() ? artifact::TensorPlacement::Device
                                      : artifact::TensorPlacement::ValidateOnly;
    out.bindings.optimized_proposal_head =
        device(binder, "ninfer.optimized_proposal_head.weight", NumericFormat::Q4G64_F16S,
               {147456, 2560}, proposal_placement);
    out.bindings.optimized_proposal_token_ids =
        device(binder, "ninfer.optimized_proposal_head.token_ids", NumericFormat::I32, {147456},
               proposal_placement);

    const artifact::TensorPlacement mtp_placement = features.mtp()
                                                        ? artifact::TensorPlacement::Device
                                                        : artifact::TensorPlacement::ValidateOnly;
    auto& mtp                                     = out.bindings.mtp;
    mtp.embedding_norm = device(binder, "mtp.pre_fc_norm_embedding.weight", NumericFormat::BF16,
                                {2560}, mtp_placement);
    mtp.hidden_norm = device(binder, "mtp.pre_fc_norm_hidden.weight", NumericFormat::BF16, {10240},
                             mtp_placement);
    mtp.embedding_projection =
        device(binder, "mtp.fc_embedding.weight", NumericFormat::BF16, {2560, 2560}, mtp_placement);
    mtp.hidden_projection =
        device(binder, "mtp.fc_hidden.weight", NumericFormat::BF16, {2560, 2560}, mtp_placement);
    mtp.attention_hc = bind_hc(binder, "mtp.layers.0.attn_hyper_connection.", mtp_placement);
    mtp.attention    = bind_attention(binder, "mtp.layers.0.self_attn.", NumericFormat::BF16,
                                     mtp_placement);
    mtp.mlp_hc       = bind_hc(binder, "mtp.layers.0.mlp_hyper_connection.", mtp_placement);
    mtp.moe          = bind_mtp_moe(binder, "mtp.layers.0.mlp.", mtp_placement);
    mtp.final_hc     = bind_final_hc(binder, "mtp.hyper_connection_mixer.", mtp_placement);
    const artifact::TensorPlacement vision_placement =
        features.vision ? artifact::TensorPlacement::Device
                        : artifact::TensorPlacement::ValidateOnly;
    out.bindings.vision = bind_vision(binder, vision_placement);

    out.materialization = binder.finish();
    if (out.materialization.object_count != 1633 ||
        out.materialization.file_backed_objects.size() != 1 ||
        out.materialization.host_objects.size() != resources.size() ||
        out.materialization.device_objects.size() !=
            static_cast<std::size_t>(1260 + (features.optimized_proposal() ? 2 : 0) +
                                     (features.mtp() ? 31 : 0) + (features.vision ? 333 : 0))) {
        throw std::logic_error("Flash-Next materialization partition is incomplete");
    }
    return out;
}

} // namespace ninfer::targets::qwen3_8_flash_next_125b_a6b
