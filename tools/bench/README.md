# tools/bench

Maintainer orchestration for the public `ninfer_bench` throughput tool and the external Serve TTFT
client. Correctness is owned by the affected suites under [`tests/`](../../tests/README.md).

## External Serve TTFT

[`ttft/README.md`](ttft/README.md) defines the black-box latency benchmark. The measurement runner
uses frozen text/media requests and public streaming protocols without calling Engine. A separate
controller manages the fixed Qwen3.8-27B NVFP4/FP8 Serve profiles and fresh-process isolation.

```bash
python3 tools/bench/run_serve_ttft_campaign.py --campaign resource --samples 5
```

The controller chooses the profile, starts and stops Serve for every sample, runs the external
client, stages the NVFP4 artifact once in `/dev/shm`, stores raw/progress/Serve artifacts below
`profiles/bench/ttft/`, records structured per-request Serve diagnostics, and writes Markdown,
JSON, and CSV summaries. The case catalog, exact profiles, TTFT boundary, and fixture qualification
are documented in the dedicated README.

## Flash-Next paired serving and numerical diagnostics

`run_flash_next_serving.py` measures fixed 8K/64K single requests, simultaneous pairs, a 64K
prefill admitted during an 8K decode, and optional eight-turn paired conversations. Each case
has one complete unmeasured warmup followed by five measured waves; the output limit defaults
to 512 tokens and natural stops remain enabled. The server must support two active requests. Configure NInfer cold runs with
`--no-prefix-reuse`; vLLM uses a fresh `cache_salt` per request. Run `--cases turns` separately
with prefix reuse enabled. Use NInfer `--preserve-thinking` and vLLM
`--reasoning-parser qwen3`; add vLLM `--enable-prompt-tokens-details` to retain per-request cache
counts. Conversation histories preserve reasoning separately from content,
and each conversation receives a distinct root and a stable salt.

The `--contexts` JSON maps token lengths to decoded text. Produce it from the committed
Flash-Next fixture using the matching local tokenizer (the tokenizer environment needs
`transformers`; the measurement scripts themselves use Python 3.11's standard library):

```python
import json
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/path/to/Flash-Next", local_files_only=True)
ids = list(map(int, Path("bench/fixtures/qwen3_8_flash_next_context.ids").read_text().split()))
Path("/tmp/flash-next-contexts.json").write_text(json.dumps({
    str(n): tokenizer.decode(ids[:n]) for n in (1980, 2049, 8192, 65536)
}))
```

```bash
python3 tools/bench/run_flash_next_serving.py --url http://127.0.0.1:18087 \
  --contexts /tmp/flash-next-contexts.json --output /tmp/baseline.jsonl
python3 tools/bench/compare_flash_next_serving.py /tmp/baseline.jsonl /tmp/candidate.jsonl \
  --output /tmp/comparison.json
python3 tools/bench/check_flash_next_isolation.py --url http://127.0.0.1:18087 \
  --contexts /tmp/flash-next-contexts.json --output /tmp/isolation.json
```

The comparator checks matched prompt/output lengths for cold workloads and output lengths for
rolling conversations. Unequal output lengths invalidate a completion-time comparison; use
actual token counts and cache/latency observations to describe such conversation runs. It
bootstraps paired wave completion-time ratios; conversation turns
remain grouped by repetition. Its timing gate requires a 2% gain with a positive 95% interval
on a concurrent workload, no single-request regression above 2%, and no mean TTFT/ITL regression
above 5%. Confidence intervals describe these repeated fixtures, not variation across arbitrary
prompts. Stream gaps are delivery gaps, not individual device-token latencies. The isolation
probe checks public-label recall across different request lengths and reversed admission order;
this behavioral probe complements numerical/state tests and is not a mathematical oracle.

`compare_flash_next_scores.py` compares `ninfer-perplexity --token-scores` output with vLLM
causal prompt scores using identical token IDs. See [perplexity](../../docs/perplexity.md) and
[Flash-Next numerical diagnostics](../../docs/maintainer/qwen3.8-flash-next-125b-a6b-model.md#numerical-diagnostics).

## Corpus baker

`ninfer_bench` benchmarks prefill at an exact length by slicing the first `P` token ids of a
committed corpus, so the corpus must be real, in-distribution text (not random tokens) and at
least as long as the largest prefill you want to run. `make_bench_corpus.py` bakes that corpus
offline with a local Hugging Face Qwen3.6 tokenizer.

Outputs (committed):

```text
bench/fixtures/bench_corpus.ids            whitespace-separated decimal token ids (exactly --tokens)
bench/fixtures/bench_corpus.manifest.json  tokenizer id, token count, and source description
```

Content sources:

- Built-in curated multi-domain prose (Chinese / English / code / math) — the default. It is
  encoded WITHOUT the chat template or special tokens, then tiled (paragraphs rotated each cycle)
  and truncated to exactly `--tokens`. Repetition only fills length; because prefill/decode
  throughput is token-count / bandwidth bound, it does not bias the numbers.
- `--source-text <file>` (repeatable) — tokenize your own long meaningful text instead, e.g. a
  downloaded public-domain book or a concatenated document set, for genuinely diverse very long
  content. The committed default is `~64k` tokens; raise `--tokens` and/or pass `--source-text`
  for more.

The binary slices `[0:P]`; the manifest is provenance only.

## Requirements

Install the tokenizer dependencies into the active Python environment:

```bash
pip install -r tools/bench/requirements.txt
```

The tokenizer is loaded locally only; the tool never downloads from the network. Pass
`--tokenizer-path` or set `NINFER_TOKENIZER_PATH`.

## Regenerate / check

```bash
# Regenerate the committed corpus from the built-in bank (default 65536 tokens).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 65536

# Bake from your own downloaded/assembled text instead (kept local; not committed).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 131072 --source-text /path/to/book.txt

# Check that the committed .ids and its descriptive manifest agree; no tokenizer or source needed.
python3 tools/bench/make_bench_corpus.py --check
```

`--tokens` is the exact committed corpus size and the ceiling on prefill length; increase it (and
optionally use `--source-text`) to benchmark longer prefills, memory permitting.

## NInfer performance matrix

`run_ninfer_bench_matrix.py` runs the layered public-Engine `ninfer_bench` matrix against the native
`.ninfer` artifact and stores its local reports under `profiles/bench/`. Its defaults are:

```text
artifact: out/qwen3_6_27b.ninfer
binary:   build/bench/ninfer_bench
corpus:   bench/fixtures/bench_corpus.ids
```

The matrix treats MTP `k=3` with the optimized proposal head as the primary path, keeps `k=0` and
`k=5` as controls, and sweeps `k=0..5` on representative context-decode cases. Decode-bearing cases
cover CUDA Graph and eager execution; prefill-only cases vary prompt length and prefill chunk.

```bash
# Configure the benchmark targets once; they are off in the default public build.
cmake -S . -B build -DNINFER_BUILD_BENCHMARKS=ON

# Inspect commands without running the model.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run

# Main run. Builds build/bench/ninfer_bench first, then writes JSON and summary.csv.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core

# Longer run that adds 32k/64k prompt and context-decode points.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full

# Run only the MTP draft-window sweep.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full --suite mtp_sweep
```

Default outputs:

```text
profiles/bench/ninfer-<preset>-<timestamp>/
  commands.sh
  manifest.json
  json/<suite>/<case>.json
  logs/<suite>.<case>.stderr.txt
  summary.csv
  summary.json
```

Use `--resume` to skip completed JSON reports in an existing `--output-dir`, and `--preset smoke`
for a minimal script/runner check. `--no-build` uses the binary supplied by `--bench` without
building it.

Each raw report must be `ninfer_bench_report` schema v13. The flattened summary and schema-v3 matrix
manifest carry native names from the report: selected target, canonical `weights_id`, artifact,
load/read/upload/staging values, Engine memory arenas including the non-additive Vision layout
inside the unified workspace and CUDA Graph allowance, per-test planned logical and
allocator-observed workspace peaks, KV capacity and
payload, configured proposal head and graph mode, phase timings and throughput, and speculative
rounds/drafts/acceptance/fallbacks. The matrix manifest is descriptive and records the commands and
selected local inputs; it does not make repository state part of report validity.

`run_serve_corpus.py` runs both registered targets and both published MTP0/MTP3 suites when both
artifacts are supplied. Pass one `--artifact` to select a single target and `--mode mtp0` or
`--mode mtp3` to run only that suite. The 35B-A3B-only `--mode dflash7` route runs the same
decode corpus with DFlash block=8 (`k=7`) and the optimized proposal head. Add
`--sampling greedy` to force exact argmax while retaining the same fixtures and repetition count.
Its schema-v6 result and flattened summaries retain the canonical `weights_id`, request Host
exposure, and decode Host/Device-wait time per round received from the schema-v20 serving records.
Request exposure is a latency distribution value and is never summed across concurrent requests;
worker aggregation uses the serving `throughput.host_work` interval deltas. The stochastic route pins its complete
temperature/top-p/top-k/min-p/presence/frequency profile explicitly, so model-default changes do
not alter the measurement method.

## Concurrent serving benchmark

`run_serve_concurrency.py` measures two separate concurrency properties through real loopback
Chat Completions requests:

- `decode-saturation` submits one long-decode wave and uses only complete one-second intervals in
  which every decode round has exactly the configured batch size. Ramp-up, prefill, and drain
  intervals are excluded.
- `corpus-makespan` shuffles the existing mode-specific corpus once with the fixed seed `20260811`,
  then runs that same order with exactly `N` persistent client workers. A worker submits the next
  request only after its current response completes, and makespan ends when the final response has
  been read. Request bodies are sent in shuffled-order sequence while response waits remain fully
  concurrent, removing client-thread arrival races without serializing inference.

Each concurrency point starts a fresh server because its execution graphs and memory plan are
startup-fixed. Prefix reuse is disabled, startup and warmup are outside both measurements, and the
runner writes per-point JSON, raw serving JSONL, and combined JSON/CSV/Markdown summaries.
The point report records the shuffle seed, dispatch method, shuffled position, and canonical corpus
position for every request.

```bash
python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 \
  --decode-tokens 8192 \
  --output profiles/bench/concurrent-decode

python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 \
  --output profiles/bench/concurrent-corpus
```

Use `--kv-capacity auto` when the fixed corpus needs more shared KV than the default 262,144-token
pool. A point is intentionally not resumable: combining fragments from separate server processes
would not preserve either a steady interval or one continuous makespan.
