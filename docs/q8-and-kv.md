# Q8 weights and KV-cache profiles

The Qwen3.6 converters retain the established mixed-quantization artifact by default. Pass
`--weight-profile q8` to encode every quantized weight matrix as signed W8G32 with an FP16 scale
per 32 values. Control tensors, norms, biases, and other native BF16/FP32/I32 objects retain their
declared formats.

```bash
python3 -m tools.convert.qwen3_6_27b.convert \
  --model /path/to/Qwen3.6-27B \
  --out qwen3_6_27b.q8.ninfer \
  --device cuda \
  --weight-profile q8

python3 -m tools.convert.qwen3_6_35b_a3b.convert \
  --model /path/to/Qwen3.6-35B-A3B \
  --dflash-model /path/to/Qwen3.6-35B-A3B-DFlash \
  --out qwen3_6_35b_a3b.q8.ninfer \
  --device cuda \
  --weight-profile q8
```

The runtime detects the profile from the artifact directory. Native and Q8 artifacts remain
self-describing and fail early if their tensor formats are mixed unexpectedly.

## KV cache

`--kv-dtype` selects storage independently of the weight profile:

| Value | Physical codes | Scale | Intended use |
|---|---|---|---|
| `bf16` | BF16 | none | fastest native 16-bit path and highest cache fidelity |
| `q8` | signed INT8 | FP16 per 64 values | balanced long-context storage |
| `q4` | two signed 4-bit codes per byte | FP16 per 64 values | minimum cache footprint |

`int8` and `int4` are accepted compatibility aliases. Quantization and dequantization are fused
into attention append, prefill, and decode kernels; Q4 is physically packed rather than stored as
one byte per value.

The model uses BF16 activations. BF16 is consequently the selected 16-bit cache path: an FP16
cache would have the same payload size while adding conversion at every append and attention
read.

For DFlash speculative decoding this option applies to the target model's persistent KV cache.
The DFlash drafter's private local, boundary, and full-attention scratch caches remain BF16; they
are short-lived implementation workspaces rather than the user-sized target context cache.

## Benchmark artifacts

`ninfer_bench --output json` retains every measured repetition. Normalize those reports and
generate CSV, JSON, and a self-contained HTML report:

```bash
python3 -m tools.bench.normalize_ninfer_bench \
  --input native:native-bf16.json \
  --input q8:q8-q4.json \
  --output performance-suite.json

python3 -m tools.bench.performance_report \
  performance-suite.json --output-dir performance-report
```
