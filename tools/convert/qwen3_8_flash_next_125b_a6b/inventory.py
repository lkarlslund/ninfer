"""Closed persistent-object contract for Qwen3.8 Flash-Next 125B-A6B NVFP4."""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import (
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    build_vision_specs,
)


MODEL_ID = "qwen3.8-flash-next-125b-a6b"
WEIGHTS_ID = "nvfp4"
MIXED_WEIGHTS_ID = "nvfp4-fp8-proj"
TARGET_KEY = "qwen3_8_flash_next_125b_a6b"

BF16 = "BF16"
FP32 = "FP32"
FP8 = "FP8_E4M3FN"
NVFP4 = "NVFP4"
FP8_BLOCK = "FP8_E4M3FN_BLOCK128_F32S"
Q4 = "Q4G64_F16S"
I32 = "I32"
CONTIGUOUS = "contiguous-le-v1"
ROW_SPLIT = "row-split-k128-v1"
EXPERT_NVFP4 = "expert-blockscale-k16-m128x4-v1"
BLOCK_FP8 = "blockscale-k128-m128-v1"

LAYERS = tuple(range(48))
FULL_ATTENTION_LAYERS = tuple(range(3, 48, 4))
GDN_LAYERS = tuple(layer for layer in LAYERS if layer not in FULL_ATTENTION_LAYERS)
EXPERTS = 512

RESOURCE_SPECS = tuple(
    ResourceSpec(name)
    for name in (
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
        "frontend/preprocessor_config.json",
        "frontend/video_preprocessor_config.json",
    )
)


def direct(name: str, shape: tuple[int, ...], numeric_format: str = BF16) -> TensorSpec:
    return TensorSpec(name, shape, numeric_format, CONTIGUOUS)


def _hyperconnection(prefix: str, *, inject: bool) -> tuple[TensorSpec, ...]:
    specs = [
        direct(prefix + "hc_norm.weight", (10240,)),
        direct(prefix + "input_mix_weight_down.weight", (320, 10240)),
        direct(prefix + "input_mix_weight_up.weight", (10240, 320)),
    ]
    if inject:
        specs.insert(0, direct(prefix + "block_inject_weight.weight", (4, 10240)))
    return tuple(specs)


def _moe(prefix: str) -> tuple[TensorSpec, ...]:
    return (
        direct(prefix + "gate.weight", (512, 2560)),
        direct(prefix + "shared_expert.gate_proj.weight", (640, 2560)),
        direct(prefix + "shared_expert.up_proj.weight", (640, 2560)),
        direct(prefix + "shared_expert.down_proj.weight", (2560, 640)),
        direct(prefix + "shared_expert_gate.weight", (1, 2560)),
        TensorSpec(prefix + "experts.gate_up", (512, 1280, 2560), NVFP4, EXPERT_NVFP4),
        direct(prefix + "experts.gate_up_input_divisors", (512,), FP32),
        TensorSpec(prefix + "experts.down", (512, 2560, 640), NVFP4, EXPERT_NVFP4),
        direct(prefix + "experts.down_input_divisors", (512,), FP32),
    )


def _text_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        direct("model.language_model.embed_tokens.weight", (248320, 2560)),
    ]
    for layer in LAYERS:
        prefix = f"model.language_model.layers.{layer}."
        specs.extend(_hyperconnection(prefix + "attn_hyper_connection.", inject=True))
        if layer in FULL_ATTENTION_LAYERS:
            attention = prefix + "self_attn."
            specs.extend(
                (
                    direct(attention + "q_proj.weight", (12288, 2560)),
                    direct(attention + "k_proj.weight", (512, 2560)),
                    direct(attention + "v_proj.weight", (512, 2560)),
                    direct(attention + "o_proj.weight", (2560, 6144)),
                    direct(attention + "q_norm.weight", (256,)),
                    direct(attention + "k_norm.weight", (256,)),
                    direct(attention + "indexer.index_qk_proj.weight", (640, 2560)),
                    direct(attention + "indexer.q_layernorm.weight", (128,)),
                    direct(attention + "indexer.k_layernorm.weight", (128,)),
                )
            )
        else:
            attention = prefix + "linear_attn."
            specs.extend(
                (
                    # GDN control values are represented as exact FP32 expansions of the
                    # checkpoint BF16 words so the persistent runtime state transition uses the
                    # same FP32 contract as the shared Gated DeltaNet Op.
                    direct(attention + "A_log", (48,), FP32),
                    direct(attention + "dt_bias", (48,), FP32),
                    direct(attention + "conv1d.weight", (4, 10240)),
                    direct(attention + "in_proj_a.weight", (48, 2560)),
                    direct(attention + "in_proj_b.weight", (48, 2560)),
                    direct(attention + "in_proj_qkv.weight", (10240, 2560)),
                    direct(attention + "in_proj_z.weight", (6144, 2560)),
                    direct(attention + "norm.weight", (128,)),
                    direct(attention + "out_proj.weight", (2560, 6144)),
                )
            )
        if layer == 1:
            ple = prefix + "ple."
            specs.extend(
                (
                    direct(ple + "conv1d.weight", (4, 10240)),
                    direct(ple + "key_proj.weight", (10240, 2560)),
                    direct(ple + "norm_conv.weight", (10240,)),
                    direct(ple + "norm_key.weight", (10240,)),
                    direct(ple + "norm_query.weight", (10240,)),
                    direct(ple + "value_proj.weight", (2560, 2560)),
                    direct(ple + "ple_embedding.ngram_embedding.weight_scale", (1,)),
                    direct(
                        ple + "ple_embedding.ngram_embedding.weight",
                        (320001536, 160),
                        FP8,
                    ),
                )
            )
        specs.extend(_hyperconnection(prefix + "mlp_hyper_connection.", inject=True))
        specs.extend(_moe(prefix + "mlp."))
    specs.extend(_hyperconnection("model.language_model.hyper_connection_mixer.", inject=False))
    specs.append(direct("lm_head.weight", (248320, 2560)))
    specs.extend(
        (
            TensorSpec("ninfer.optimized_proposal_head.weight", (147456, 2560), Q4,
                       ROW_SPLIT),
            direct("ninfer.optimized_proposal_head.token_ids", (147456,), I32),
        )
    )
    return tuple(specs)


def _mtp_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        direct("mtp.pre_fc_norm_embedding.weight", (2560,)),
        direct("mtp.pre_fc_norm_hidden.weight", (10240,)),
        direct("mtp.fc_embedding.weight", (2560, 2560)),
        direct("mtp.fc_hidden.weight", (2560, 2560)),
    ]
    specs.extend(_hyperconnection("mtp.layers.0.attn_hyper_connection.", inject=True))
    attention = "mtp.layers.0.self_attn."
    specs.extend(
        (
            direct(attention + "q_proj.weight", (12288, 2560)),
            direct(attention + "k_proj.weight", (512, 2560)),
            direct(attention + "v_proj.weight", (512, 2560)),
            direct(attention + "o_proj.weight", (2560, 6144)),
            direct(attention + "q_norm.weight", (256,)),
            direct(attention + "k_norm.weight", (256,)),
            direct(attention + "indexer.index_qk_proj.weight", (640, 2560)),
            direct(attention + "indexer.q_layernorm.weight", (128,)),
            direct(attention + "indexer.k_layernorm.weight", (128,)),
        )
    )
    specs.extend(_hyperconnection("mtp.layers.0.mlp_hyper_connection.", inject=True))
    moe = "mtp.layers.0.mlp."
    specs.extend(
        (
            direct(moe + "gate.weight", (512, 2560)),
            direct(moe + "shared_expert.gate_proj.weight", (640, 2560)),
            direct(moe + "shared_expert.up_proj.weight", (640, 2560)),
            direct(moe + "shared_expert.down_proj.weight", (2560, 640)),
            direct(moe + "shared_expert_gate.weight", (1, 2560)),
            direct(moe + "experts.gate_up_proj", (512, 1280, 2560)),
            direct(moe + "experts.down_proj", (512, 2560, 640)),
        )
    )
    specs.extend(_hyperconnection("mtp.hyper_connection_mixer.", inject=False))
    return tuple(specs)


TEXT_TENSOR_SPECS = _text_specs()
MTP_TENSOR_SPECS = _mtp_specs()
VISION_TENSOR_SPECS = build_vision_specs(2560)
TENSOR_SPECS = TEXT_TENSOR_SPECS + MTP_TENSOR_SPECS + VISION_TENSOR_SPECS
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS


def _is_mixed_projection(name: str) -> bool:
    if not name.startswith("model.language_model.layers."):
        return False
    return name.endswith(
        (
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.o_proj.weight",
            ".linear_attn.in_proj_qkv.weight",
            ".linear_attn.in_proj_z.weight",
            ".linear_attn.out_proj.weight",
        )
    )


MIXED_TENSOR_SPECS = tuple(
    TensorSpec(spec.name, spec.shape, FP8_BLOCK, BLOCK_FP8) if _is_mixed_projection(spec.name)
    else spec
    for spec in TENSOR_SPECS
)
MIXED_OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + MIXED_TENSOR_SPECS


__all__ = [
    "EXPERTS",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "LAYERS",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "MIXED_OBJECT_SPECS",
    "MIXED_TENSOR_SPECS",
    "MIXED_WEIGHTS_ID",
    "OBJECT_SPECS",
    "RESOURCE_SPECS",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_TENSOR_SPECS",
    "VISION_TENSOR_SPECS",
    "WEIGHTS_ID",
]
