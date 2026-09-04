# Qwen3.8 Flash-Next 125B-A6B artifact reference

This reference defines the sole registered storage profile for Qwen3.8 Flash-Next 125B-A6B. Generic
framing, layouts, and numeric formats remain governed by `artifact-container.md`,
`storage-layouts.md`, and `tensor-formats.md`; model mathematics are defined in
[`qwen3.8-flash-next-125b-a6b-model.md`](qwen3.8-flash-next-125b-a6b-model.md).

## Identity

```text
filename   = qwen3_8_flash_next_125b_a6b_nvfp4.ninfer
model_id   = qwen3.8-flash-next-125b-a6b
weights_id = nvfp4
target_key = qwen3_8_flash_next_125b_a6b
recipe_id  = qwen3_8_flash_next_125b_a6b_nvfp4-v1
```

The artifact is a complete version-2 image containing Text, MTP, Vision, and the six frontend
resources. The exact binder validates all 1,633 objects and selects optional MTP and Vision device
residency at startup; identity is never inferred from the filename or tensor shapes.

## Inventory and formats

The closed inventory contains 1,627 tensors and six raw resources. Main routed expert banks use
`NVFP4` with `expert-blockscale-k16-m128x4-v1`; their FP32 input divisors are separate bound
objects. MTP expert banks, projections, HyperConnection weights, norms, embeddings, output head,
and shared experts retain BF16. GDN control vectors and NVFP4 divisors use FP32. Vision retains the
existing Q4/Q5/Q6 groupwise and W8 merger profiles.

The 320,001,536-by-160 PLE embedding is one contiguous FP8 E4M3FN tensor plus a BF16 multiplier.
It is the artifact's only file-backed object. The reader validates its descriptor and payload
extent, then exposes a read-only mapping to the target. Demand paging and the operating-system file
cache own residency; the generic materializer does not allocate host or device storage for the
complete 51.2 GB table. Prompt preparation gathers only the sixteen rows selected for each token.

The measured artifact has 134,755,521,280 bytes. Its format allocation is 1,249 BF16, 168 FP32,
one FP8 table, 96 NVFP4 expert banks, 55 Q4, 54 Q5, one Q6, one I32 map, and two W8 tensors. The six embedded
resources are `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
`generation_config.json`, `preprocessor_config.json`, and `video_preprocessor_config.json` under
the `frontend/` namespace.

## Conversion

The converter accepts only the closed `RadixArk/Qwen3.8-Flash-Next-NVFP4` checkpoint allocation.
It preserves BF16 and FP8 words, transposes channel-wise convolution kernels into the target layout,
and rearranges ModelOpt expert-major NVFP4 codes and scales into NInfer's bank layout without
dequantizing or requantizing them. It concatenates the 128 PLE shards directly into the one table
payload and writes a conversion report beside the artifact.

```bash
python3 -m tools.convert.qwen3_8_flash_next_125b_a6b.convert \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --out out/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer \
  --device cuda
```

The output basename is fixed. Conversion rejects missing, unexpected, incorrectly shaped, or
incorrectly typed source tensors, mismatched paired gate/up scales, invalid divisors, incompatible
model configuration, and incomplete frontend resources.

## Runtime binding

The artifact binder partitions the inventory into 1,260 always-resident device objects, 31
additional MTP objects when enabled, 333 additional Vision objects when enabled, six host frontend
resources, and the one read-only file-backed PLE table. The mapping remains alive for the Program's
lifetime. Startup diagnostics report only bytes copied to the GPU; mapped PLE bytes are reported
separately and are not counted as upload traffic.
