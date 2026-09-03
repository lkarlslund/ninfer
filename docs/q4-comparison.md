# Original NInfer quantization versus llama.cpp Q4_K_M

This comparison uses NInfer's original mixed Q4G64/Q5G64/Q6G64 artifacts and llama.cpp
Q4_K_M GGUFs made from the same official Qwen model sources. The llama.cpp files were
requantized from the official-source Q8_0 GGUFs; no third-party or Unsloth model was used.

All measurements were collected on the RTX PRO 6000 with five measured repetitions after
warmup. The workloads are prompt processing at 128 and 512 tokens and generation of 128
tokens. Cache formats are paired by storage class: NInfer BF16 with llama.cpp F16, NInfer
Q8G64 with llama.cpp Q8_0, and NInfer packed Q4G64 with llama.cpp Q4_0.

## Artifact composition and size

| Model | NInfer original tensor formats | NInfer size | llama.cpp Q4_K_M size |
|---|---|---:|---:|
| Qwen3.6-27B | 183 Q4, 246 Q5, 3 Q6, 7 W8, plus unquantized tensors | 16.29 GiB | 15.65 GiB |
| Qwen3.6-35B-A3B | 95 Q4, 91 Q5, 5 Q6, 195 W8, plus unquantized tensors | 21.22 GiB | 20.21 GiB |

The formats are not numerically identical: NInfer uses group-64 signed formats with FP16
scales, while Q4_K_M is llama.cpp's mixed K-quant recipe.

## Throughput summary

The following values use the corresponding 16-bit cache mode. The complete report includes
Q8 and Q4 cache pairings and all raw repetitions.

| Model | Engine and weights | pp128 | pp512 | tg128 |
|---|---|---:|---:|---:|
| Qwen3.6-27B | NInfer mixed Q4/Q5/Q6 | 2,004 | 3,956 | 82.7 |
| Qwen3.6-27B | llama.cpp Q4_K_M | 2,809 | 4,096 | 78.3 |
| Qwen3.6-35B-A3B | NInfer mixed Q4/Q5/Q6 | 6,096 | 14,470 | 332.5 |
| Qwen3.6-35B-A3B | llama.cpp Q4_K_M | 4,200 | 9,441 | 288.9 |

For dense 27B, llama.cpp leads prompt processing by 40.2% at 128 tokens and 3.6% at 512,
while NInfer leads generation by 5.6%. For 35B-A3B, NInfer leads by 45.1%, 53.3%, and
15.1% respectively. The same qualitative result holds with Q8 and Q4 cache storage.

The machine-readable suite and self-contained report for this run are under
`out/report-q4-comparison/`.
