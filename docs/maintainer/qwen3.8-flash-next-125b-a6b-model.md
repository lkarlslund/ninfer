# Qwen3.8 Flash-Next 125B-A6B model reference

This reference records the exact Text, Vision, MTP, HyperConnection, PLE, QSA, sparse-MoE, and
persistent-state semantics implemented for the registered Qwen3.8 Flash-Next 125B-A6B target. The
artifact representation and conversion contract are defined in
[`qwen3.8-flash-next-125b-a6b-artifact.md`](qwen3.8-flash-next-125b-a6b-artifact.md).

The target instantiates the independent `qwen3_8_flash_next` family runtime. That family owns its
Frontend, prepared-prompt/output types, persistent state, Text/Vision/MTP schedules, workspace, and
CUDA Graph machinery; it does not specialize the `qwen3_6` runtime. Closed mathematical kernels
remain shared Ops where their semantic contracts genuinely coincide.

## Fixed dimensions

| Field | Value |
|---|---:|
| Text hidden size / layers | 2560 / 48 |
| vocabulary rows | 248320 |
| native context | 262144 |
| full-attention / GDN layers | 12 / 36 |
| QSA query heads / KV heads / head width | 24 / 2 / 256 |
| QSA rotary width / scale | 64 / `1/sqrt(256)` |
| GDN key heads / value heads / head width | 16 / 48 / 128 |
| GDN convolution channels / taps | 10240 / 4 |
| routed experts / selected experts | 512 / 10 |
| routed and shared expert width | 640 |
| HyperConnection streams / low-rank width | 4 / 320 |
| MTP layers / maximum draft tokens | 1 / 3 |
| Vision depth / hidden / intermediate | 27 / 1152 / 4304 |
| Vision merger output | 2560 |

Full attention occurs at zero-based layers `3, 7, ..., 47`; every other layer uses Gated
DeltaNet. Every Text layer has a sparse routed expert block and an independently sigmoid-gated
shared expert. The selected routed weights are normalized to sum to one. No expert capacity limit,
token dropping, or stochastic routing applies at inference.

## HyperConnection and layer schedule

The residual state is four BF16 streams of width 2560. Each attention/GDN and MoE sub-block first
forms its learned normalized input mix and injection weights, evaluates the block, and commits the
block result back into the four streams. The fused combine-and-mix implementation preserves the
same materialized BF16 combine boundary before grouped RMSNorm. The final learned mixer reduces the
four streams to the decoder output.

Layer 1 additionally applies PLE between its token mixer and MoE. PLE selects sixteen 160-element
FP8 rows per token: eight bigram heads and eight trigram heads. Hashes reset at EOS. The selected
2560 values are gathered on the host from the read-only table mapping, transferred for the current
chunk, and consumed by the PLE projections and nine-column persistent causal state. Only selected
rows enter device memory; the complete table is never uploaded.

## QSA and persistent state

QSA projects 24 gated query heads and two K/V heads. Its indexer constructs normalized 128-wide
query/key representations and selects causal four-token key groups before exact attention over the
selected paged K/V positions. Main K/V, raw index keys, and MRoPE positions are persistent cache
state. Main K/V may use BF16 or row-scaled FP8 E4M3; raw index keys remain BF16 and MRoPE positions
remain I32 in both profiles. FP8 K rows apply the shared normalized D256 Hadamard transform before
row quantization, and Q applies the same transform before the dot product. V is row-quantized
without that transform. Prefill uses a tensor-core selected-attention route; decode uses the
bounded split route.

Each GDN layer retains three previous BF16 convolution columns and 48 FP32 recurrent matrices of
shape `[128,128]`. PLE retains nine previous BF16 convolution columns. QSA KV/index state, GDN
state, PLE state, HyperConnection state, and MTP state participate together in prefix snapshots,
speculative replay/fold, commit, rollback, and restore. A generated token is public only after the
target transaction commits it.

## MTP and Vision

The one-layer MTP predictor uses the same QSA, HyperConnection, and MoE mathematics with its own
weights and state. Its expert tensors are BF16. Draft lengths 1 through 3 are supported; ordinary
MTP0 and MTP3 use the same target model and publication rules.

The Vision tower is the 27-layer Qwen multimodal backbone used by the registered Qwen3.6-family
targets, with a checkpoint-specific merger that emits width 2560. Image/video preprocessing,
MRoPE prompt construction, CLI input, and serving protocol translation remain the shared product
routes.

## Numerical boundaries

BF16 weights and activations retain their represented values. Routed main-model experts decode
signed NVFP4 codes with their stored block scales and per-expert input/weight divisors. GDN control
and recurrent state are FP32. PLE table values are FP8 E4M3FN multiplied by the stored BF16 table
scale. Production fusion may choose its reduction and staging precision, but each closed Op is
qualified directly against an independent mathematical oracle at its public output and persistent
state boundaries.
