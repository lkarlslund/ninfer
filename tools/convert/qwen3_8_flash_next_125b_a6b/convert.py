"""Build the Qwen3.8 Flash-Next 125B-A6B NVFP4 `.ninfer` artifact.

The checkpoint's numerical words are retained. BF16 projections and PLE FP8
tensors are copied directly, channel-wise convolution kernels are transposed
to NInfer's channel-fast runtime layout, and expert-major ModelOpt NVFP4
tensors are rearranged into the closed NInfer bank layout without dequantizing
or requantizing them.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Iterable, Iterator, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.artifact.layouts import encode_direct, encode_fp8_block_scaled, swizzle_nvfp4_scales
from tools.convert.common.safetensors import ShardReader, TensorMetadata
from tools.convert.common.quantize import pick_device
from tools.convert.qwen3_6.common import conversion
from tools.convert.qwen3_6.common import recipe as family_recipe
from . import draft_head, inventory


OUTPUT_BASENAME = "qwen3_8_flash_next_125b_a6b_nvfp4.ninfer"
RECIPE_ID = "qwen3_8_flash_next_125b_a6b_nvfp4-v1"
SOURCE_REPOSITORY = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
MIXED_OUTPUT_BASENAME = "qwen3_8_flash_next_125b_a6b_nvfp4_fp8_proj.ninfer"
MIXED_RECIPE_ID = "qwen3_8_flash_next_125b_a6b_nvfp4_fp8_proj-v1"
MIXED_SOURCE_REPOSITORY = "lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8"

_PLE_PREFIX = (
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
)
_PLE_TABLE = _PLE_PREFIX + "weight"
_PLE_SHARDS = tuple(_PLE_PREFIX + f"shard_{part}.weight" for part in range(128))
_PLE_METADATA = frozenset(
    {
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
    }
)
_DRAFT_HEAD = "ninfer.optimized_proposal_head.weight"
_DRAFT_HEAD_IDS = "ninfer.optimized_proposal_head.token_ids"
_VISION_RECIPES = family_recipe.build_vision_recipes(2560)
_VISION_BY_NAME = {item.object_name: item for item in _VISION_RECIPES}
_VISION_SOURCES = family_recipe.source_requirements(_VISION_RECIPES)
_CONVOLUTION_NAMES = frozenset(
    {
        *(f"model.language_model.layers.{layer}.linear_attn.conv1d.weight"
          for layer in inventory.GDN_LAYERS),
        "model.language_model.layers.1.ple.conv1d.weight",
    }
)


def _expert_source(layer: int, expert: int, projection: str, field: str) -> str:
    return (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}.{field}"
    )


def _bank_name(layer: int, role: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{role}"


def _load_resources(model_dir: Path) -> tuple[conversion.ResourcePayload, ...]:
    return conversion.load_resources(model_dir, inventory.RESOURCE_SPECS)


def _direct_source_specs(
    tensor_specs: Sequence[inventory.TensorSpec] = inventory.TENSOR_SPECS,
) -> tuple[inventory.TensorSpec, ...]:
    special = {
        _bank_name(layer, role)
        for layer in inventory.LAYERS
        for role in (
            "gate_up",
            "gate_up_input_divisors",
            "down",
            "down_input_divisors",
        )
    }
    return tuple(
        spec
        for spec in tensor_specs
        if spec.name not in special
        and spec.name != _PLE_TABLE
        and spec.name not in (_DRAFT_HEAD, _DRAFT_HEAD_IDS)
        and spec.name not in _VISION_BY_NAME
    )


def _expected_source_signatures(
    tensor_specs: Sequence[inventory.TensorSpec] = inventory.TENSOR_SPECS,
) -> dict[str, tuple[tuple[int, ...], str]]:
    expected = {
        spec.name: (spec.shape, "F8_E4M3" if spec.format == inventory.FP8_BLOCK else "BF16")
        for spec in _direct_source_specs(tensor_specs)
    }
    expected.update(
        {
            spec.name.removesuffix(".weight") + ".weight_scale_inv":
                ((spec.shape[0] // 128, spec.shape[1] // 128), "F32")
            for spec in tensor_specs
            if spec.format == inventory.FP8_BLOCK
        }
    )
    expected.update(
        {name: (source.shape, source.dtype) for name, source in _VISION_SOURCES.items()}
    )
    for name in _CONVOLUTION_NAMES:
        expected[name] = ((10240, 1, 4), "BF16")
    for name in _PLE_SHARDS:
        expected[name] = ((2_500_012, 160), "F8_E4M3")
    for layer in inventory.LAYERS:
        for expert in range(inventory.EXPERTS):
            for projection, n, k in (
                ("gate_proj", 640, 2560),
                ("up_proj", 640, 2560),
                ("down_proj", 2560, 640),
            ):
                expected[_expert_source(layer, expert, projection, "weight")] = (
                    (n, k // 2),
                    "U8",
                )
                expected[_expert_source(layer, expert, projection, "weight_scale")] = (
                    (n, k // 16),
                    "F8_E4M3",
                )
                expected[_expert_source(layer, expert, projection, "weight_scale_2")] = (
                    (),
                    "F32",
                )
                expected[_expert_source(layer, expert, projection, "input_scale")] = (
                    (),
                    "F32",
                )
    return expected


def _validate_config(model_dir: Path, *, mixed: bool = False) -> dict[str, object]:
    config = conversion.load_json(model_dir / "config.json")
    text = config.get("text_config")
    vision = config.get("vision_config")
    quant = config.get("quantization_config")
    if not isinstance(text, dict) or not isinstance(vision, dict) or not isinstance(quant, dict):
        raise ValueError("checkpoint config is missing text, vision, or quantization config")
    conversion.check_members(
        "config",
        config,
        {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "tie_word_embeddings": False,
        },
    )
    conversion.check_members(
        "text_config",
        text,
        {
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "full_attention_interval": 4,
            "hc_count": 4,
            "hc_lowrank": 320,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "max_position_embeddings": 262144,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "split_ngram_parts": 128,
            "ple_embedding_dtype": "float8_e4m3fn",
            "mtp_num_hidden_layers": 1,
        },
    )
    conversion.check_members(
        "vision_config",
        vision,
        {
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "out_hidden_size": 2560,
        },
    )
    conversion.check_members(
        "quantization_config",
        quant,
        {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION" if mixed else "NVFP4"},
    )
    return {
        "layers": 48,
        "hidden_size": 2560,
        "experts": 512,
        "active_experts": 10,
        "max_context": 262144,
        "ple_rows": 320001536,
    }


def _validate_source(
    reader: ShardReader,
    tensor_specs: Sequence[inventory.TensorSpec] = inventory.TENSOR_SPECS,
) -> tuple[dict[str, TensorMetadata], dict[str, int]]:
    expected = _expected_source_signatures(tensor_specs)
    actual_names = frozenset(reader.names)
    expected_names = frozenset(expected)
    unexpected = actual_names - expected_names - _PLE_METADATA
    missing = expected_names - actual_names
    if unexpected or missing:
        detail = sorted(unexpected)[0] if unexpected else sorted(missing)[0]
        kind = "unexpected" if unexpected else "missing"
        raise ValueError(f"checkpoint tensor allocation is not closed: {kind} {detail}")
    metadata = reader.metadata(expected_names)
    counts: Counter[str] = Counter()
    for name, signature in expected.items():
        item = metadata[name]
        if (item.shape, item.dtype) != signature:
            raise ValueError(
                f"{name}: source signature {(item.shape, item.dtype)} != {signature}"
            )
        counts[item.dtype] += 1
    return metadata, dict(sorted(counts.items()))


def _positive_reciprocal(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.dtype != torch.float32 or value.numel() != 1:
        raise ValueError(f"{name}: expected one FP32 scale")
    value = value.detach().reshape(()).cpu()
    if not bool(torch.isfinite(value)) or not bool(value > 0):
        raise ValueError(f"{name}: scale must be finite and positive")
    result = value.reciprocal()
    if not bool(torch.isfinite(result)) or not bool(result > 0):
        raise ValueError(f"{name}: reciprocal scale is not finite and positive")
    return result


def _matching_pair(
    reader: ShardReader,
    layer: int,
    expert: int,
    field: str,
) -> torch.Tensor:
    gate_name = _expert_source(layer, expert, "gate_proj", field)
    up_name = _expert_source(layer, expert, "up_proj", field)
    gate = reader.get(gate_name).detach().contiguous().cpu()
    up = reader.get(up_name).detach().contiguous().cpu()
    if gate.dtype != up.dtype or gate.shape != up.shape or not torch.equal(gate, up):
        raise ValueError(f"layer {layer} expert {expert}: gate/up {field} words differ")
    return gate


def _expert_bank_payload(
    reader: ShardReader,
    layer: int,
    role: str,
) -> Iterator[bytes]:
    if role not in ("gate_up", "down"):
        raise ValueError(f"invalid expert bank role: {role}")
    projections = ("gate_proj", "up_proj") if role == "gate_up" else ("down_proj",)
    n, k = (1280, 2560) if role == "gate_up" else (2560, 640)

    for expert in range(inventory.EXPERTS):
        pieces = [
            reader.get(_expert_source(layer, expert, projection, "weight"))
            for projection in projections
        ]
        if any(piece.dtype != torch.uint8 for piece in pieces):
            raise ValueError(f"layer {layer} expert {expert}: packed weight is not U8")
        packed = pieces[0].contiguous() if len(pieces) == 1 else torch.cat(pieces, dim=0)
        if tuple(packed.shape) != (n, k // 2):
            raise ValueError(f"layer {layer} expert {expert}: packed weight shape mismatch")
        yield packed.numpy().tobytes()

    for expert in range(inventory.EXPERTS):
        pieces = [
            reader.get(_expert_source(layer, expert, projection, "weight_scale"))
            for projection in projections
        ]
        if any(piece.dtype != torch.float8_e4m3fn for piece in pieces):
            raise ValueError(f"layer {layer} expert {expert}: weight scale is not E4M3FN")
        scales = pieces[0].contiguous() if len(pieces) == 1 else torch.cat(pieces, dim=0)
        if tuple(scales.shape) != (n, k // 16):
            raise ValueError(f"layer {layer} expert {expert}: weight scale shape mismatch")
        yield swizzle_nvfp4_scales(scales.view(torch.uint8), (n, k)).numpy().tobytes()

    divisors = torch.empty(inventory.EXPERTS, dtype=torch.float32)
    for expert in range(inventory.EXPERTS):
        if role == "gate_up":
            scale = _matching_pair(reader, layer, expert, "weight_scale_2")
            name = _expert_source(layer, expert, "gate_proj", "weight_scale_2")
        else:
            name = _expert_source(layer, expert, "down_proj", "weight_scale_2")
            scale = reader.get(name)
        divisors[expert] = _positive_reciprocal(scale, name)
    yield encode_direct(divisors, inventory.FP32)


def _input_divisors(reader: ShardReader, layer: int, role: str) -> bytes:
    projection = "gate_proj" if role == "gate_up" else "down_proj"
    values = torch.empty(inventory.EXPERTS, dtype=torch.float32)
    for expert in range(inventory.EXPERTS):
        if role == "gate_up":
            scale = _matching_pair(reader, layer, expert, "input_scale")
        else:
            scale = reader.get(_expert_source(layer, expert, projection, "input_scale"))
        name = _expert_source(layer, expert, projection, "input_scale")
        values[expert] = _positive_reciprocal(scale, name)
    return encode_direct(values, inventory.FP32)


def _ple_payload(reader: ShardReader) -> Iterator[bytes]:
    for name in _PLE_SHARDS:
        tensor = reader.get(name)
        if tensor.dtype != torch.float8_e4m3fn or tuple(tensor.shape) != (2_500_012, 160):
            raise ValueError(f"{name}: PLE shard signature mismatch")
        yield encode_direct(tensor, inventory.FP8)


def _payload(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    device: torch.device,
    draft: draft_head.DraftHeadContext,
) -> bytes | Iterable[bytes]:
    if spec.name == _DRAFT_HEAD_IDS:
        return encode_direct(draft_head.materialize_draft_head_token_ids(draft), inventory.I32)
    if spec.name == _DRAFT_HEAD:
        full_head = reader.get("lm_head.weight")
        selected = draft_head.materialize_draft_head(full_head, draft)
        return conversion.encode_tensor_payload(selected, spec, device)
    if spec.name == _PLE_TABLE:
        return _ple_payload(reader)
    for layer in inventory.LAYERS:
        prefix = _bank_name(layer, "")
        if spec.name == prefix + "gate_up":
            return _expert_bank_payload(reader, layer, "gate_up")
        if spec.name == prefix + "down":
            return _expert_bank_payload(reader, layer, "down")
        if spec.name == prefix + "gate_up_input_divisors":
            return _input_divisors(reader, layer, "gate_up")
        if spec.name == prefix + "down_input_divisors":
            return _input_divisors(reader, layer, "down")
    if spec.name in _VISION_BY_NAME:
        tensor = family_recipe.materialize_recipe(_VISION_BY_NAME[spec.name], reader)
        return conversion.encode_tensor_payload(tensor, spec, device)
    tensor = reader.get(spec.name)
    if spec.format == inventory.FP8_BLOCK:
        scale_name = spec.name.removesuffix(".weight") + ".weight_scale_inv"
        return encode_fp8_block_scaled(tensor.view(torch.uint8), reader.get(scale_name), spec.shape)
    expected_shape = (10240, 1, 4) if spec.name in _CONVOLUTION_NAMES else spec.shape
    if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.bfloat16:
        raise ValueError(f"{spec.name}: direct source signature mismatch")
    if spec.name in _CONVOLUTION_NAMES:
        tensor = tensor[:, 0, :].transpose(0, 1).contiguous()
    if spec.format == inventory.FP32:
        return encode_direct(tensor.float(), inventory.FP32)
    return encode_direct(tensor, inventory.BF16)


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    profile: str = "nvfp4",
) -> Path:
    source = Path(model_dir)
    output = Path(out_path)
    if profile not in ("nvfp4", "nvfp4-fp8-proj"):
        raise ValueError("profile must be 'nvfp4' or 'nvfp4-fp8-proj'")
    mixed = profile == "nvfp4-fp8-proj"
    output_basename = MIXED_OUTPUT_BASENAME if mixed else OUTPUT_BASENAME
    if output.name != output_basename:
        raise ValueError(f"output basename must be {output_basename!r}")
    object_specs = inventory.MIXED_OBJECT_SPECS if mixed else inventory.OBJECT_SPECS
    tensor_specs = inventory.MIXED_TENSOR_SPECS if mixed else inventory.TENSOR_SPECS
    started = time.perf_counter()
    resolved_device = pick_device(device)
    config_summary = _validate_config(source, mixed=mixed)
    resources = _load_resources(source)
    resource_map = {item.name: item.data for item in resources}
    object_plan = conversion.build_object_plan(object_specs, resource_map)

    with ShardReader(source) as reader:
        _, dtype_counts = _validate_source(reader, tensor_specs)
        draft = draft_head.compute_shortlist(
            Path(__file__).resolve().parents[3] / draft_head.DEFAULT_RANKING,
            source,
            reader.get("lm_head.weight"),
            n=147_456,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID,
                             inventory.MIXED_WEIGHTS_ID if mixed else inventory.WEIGHTS_ID),
            object_plan.specs,
        ) as writer:
            for index, spec in enumerate(object_specs, start=1):
                payload = (
                    resource_map[spec.name]
                    if isinstance(spec, inventory.ResourceSpec)
                    else _payload(spec, reader, resolved_device, draft)
                )
                writer.write(spec.name, payload)
                print(f"[{index}/{len(object_specs)}] {spec.name}", flush=True)

    elapsed = time.perf_counter() - started
    report = {
        "identity": {"model_id": inventory.MODEL_ID,
                     "weights_id": inventory.MIXED_WEIGHTS_ID if mixed else inventory.WEIGHTS_ID},
        "target_key": inventory.TARGET_KEY,
        "recipe_id": MIXED_RECIPE_ID if mixed else RECIPE_ID,
        "source": {"repository": MIXED_SOURCE_REPOSITORY if mixed else SOURCE_REPOSITORY,
                   "path": str(source.resolve())},
        "output": str(output.resolve()),
        "config_summary": config_summary,
        "source_dtype_counts": dtype_counts,
        "objects": conversion.object_statistics(object_plan.objects),
        "file_bytes": output.stat().st_size,
        "elapsed_seconds": elapsed,
        "conversion_device": str(resolved_device),
        "ple_materialization": "file-backed-read-only",
        "proposal_shortlist": "frequency-rank-plus-lm-head-norm-rank",
    }
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {output.stat().st_size} bytes in {elapsed:.1f}s", flush=True)
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile", choices=("nvfp4", "nvfp4-fp8-proj"), default="nvfp4")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device, profile=args.profile)


if __name__ == "__main__":
    main()
