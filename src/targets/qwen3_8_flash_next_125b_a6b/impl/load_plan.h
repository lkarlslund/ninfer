#pragma once

#include "artifact/binder.h"
#include <ninfer/targets/qwen3_8_flash_next/startup_features.h>

#include <array>
#include <cstddef>

namespace ninfer::targets::qwen3_8_flash_next_125b_a6b {

inline constexpr char kModelId[]   = "qwen3.8-flash-next-125b-a6b";
inline constexpr char kWeightsId[] = "nvfp4";
inline constexpr char kPleTableName[] =
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight";
inline constexpr std::size_t kTextLayers          = 48;
inline constexpr std::size_t kFullAttentionLayers = 12;
inline constexpr std::size_t kGdnLayers           = 36;

struct HyperConnectionPlan {
    artifact::ObjectHandle block_inject;
    artifact::ObjectHandle norm;
    artifact::ObjectHandle down;
    artifact::ObjectHandle up;
};

struct MoePlan {
    artifact::ObjectHandle router;
    artifact::ObjectHandle shared_gate;
    artifact::ObjectHandle shared_up;
    artifact::ObjectHandle shared_down;
    artifact::ObjectHandle shared_scale;
    artifact::ObjectHandle routed_gate_up;
    artifact::ObjectHandle routed_gate_up_input_divisors;
    artifact::ObjectHandle routed_down;
    artifact::ObjectHandle routed_down_input_divisors;
};

struct FullAttentionPlan {
    artifact::ObjectHandle query_gate;
    artifact::ObjectHandle key;
    artifact::ObjectHandle value;
    artifact::ObjectHandle output;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle index_query_key;
    artifact::ObjectHandle index_query_norm;
    artifact::ObjectHandle index_key_norm;
};

struct GdnPlan {
    artifact::ObjectHandle a_log;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle convolution;
    artifact::ObjectHandle a_projection;
    artifact::ObjectHandle b_projection;
    artifact::ObjectHandle query_key_value;
    artifact::ObjectHandle output_gate;
    artifact::ObjectHandle norm;
    artifact::ObjectHandle output;
};

struct PlePlan {
    artifact::ObjectHandle convolution;
    artifact::ObjectHandle key_projection;
    artifact::ObjectHandle convolution_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle value_projection;
    artifact::ObjectHandle embedding_scale;
    artifact::ObjectHandle embedding_table;
};

struct TextLayerPlan {
    HyperConnectionPlan attention_hc;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    bool has_ple           = false;
    PlePlan ple{};
    HyperConnectionPlan mlp_hc;
    MoePlan moe;
};

struct FinalHyperConnectionPlan {
    artifact::ObjectHandle norm;
    artifact::ObjectHandle down;
    artifact::ObjectHandle up;
};

struct MtpPlan {
    artifact::ObjectHandle embedding_norm;
    artifact::ObjectHandle hidden_norm;
    artifact::ObjectHandle embedding_projection;
    artifact::ObjectHandle hidden_projection;
    HyperConnectionPlan attention_hc;
    FullAttentionPlan attention;
    HyperConnectionPlan mlp_hc;
    MoePlan moe;
    FinalHyperConnectionPlan final_hc;
};

struct VisionPlan {
    artifact::ObjectHandle patch_embedding;
    artifact::ObjectHandle patch_embedding_bias;
    artifact::ObjectHandle position_embedding;

    struct Layer {
        artifact::ObjectHandle qkv;
        artifact::ObjectHandle qkv_bias;
        artifact::ObjectHandle output;
        artifact::ObjectHandle output_bias;
        artifact::ObjectHandle fc1;
        artifact::ObjectHandle fc1_bias;
        artifact::ObjectHandle fc2;
        artifact::ObjectHandle fc2_bias;
        artifact::ObjectHandle norm1_weight;
        artifact::ObjectHandle norm1_bias;
        artifact::ObjectHandle norm2_weight;
        artifact::ObjectHandle norm2_bias;
    };

    std::array<Layer, 27> layers;
    artifact::ObjectHandle merger_fc1;
    artifact::ObjectHandle merger_fc1_bias;
    artifact::ObjectHandle merger_fc2;
    artifact::ObjectHandle merger_fc2_bias;
    artifact::ObjectHandle merger_norm_weight;
    artifact::ObjectHandle merger_norm_bias;
};

struct BindingPlan {
    qwen3_8_flash_next::StartupFeatures features;
    std::array<artifact::ObjectHandle, 6> frontend;
    artifact::ObjectHandle token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    FinalHyperConnectionPlan final_hc;
    artifact::ObjectHandle output_head;
    artifact::ObjectHandle optimized_proposal_head;
    artifact::ObjectHandle optimized_proposal_token_ids;
    MtpPlan mtp;
    VisionPlan vision;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::ObjectHandle ple_table;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan plan_artifact(artifact::Binder& binder,
                               qwen3_8_flash_next::StartupFeatures features = {});

} // namespace ninfer::targets::qwen3_8_flash_next_125b_a6b
