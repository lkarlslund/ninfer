"""Model-aware proposal-head shortlist for Qwen3.8 Flash-Next 125B-A6B."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tools.convert.qwen3_6.common.draft_head import (
    DraftHeadContext,
    load_total_counts,
    materialize_draft_head,
    materialize_draft_head_token_ids,
    read_special_ids,
)


VOCAB_SIZE = 248_320
TOKENIZER_VOCAB_SIZE = 248_077
DEFAULT_RANKING = Path("tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64")


def _ordinal_rank_descending(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    return rank


def _row_squared_norms(weight: torch.Tensor, rows: int) -> np.ndarray:
    if weight.dim() != 2 or weight.shape[0] < rows:
        raise ValueError(f"lm_head.weight has invalid shape {tuple(weight.shape)}")
    norms = np.empty(rows, dtype=np.float32)
    chunk_rows = 4096
    for begin in range(0, rows, chunk_rows):
        end = min(begin + chunk_rows, rows)
        chunk = weight[begin:end].float()
        norms[begin:end] = chunk.square().sum(dim=1).cpu().numpy()
    return norms


def compute_shortlist(
    ranking_path: str | Path,
    tokenizer_dir: str | Path,
    lm_head: torch.Tensor,
    *,
    n: int,
) -> DraftHeadContext:
    """Rank rows by equal-weight frequency and LM-head-norm ordinal priors.

    Frequency alone systematically excludes plausible multilingual output rows for this target.
    LM-head norm estimates a row's ability to become an extreme logit independently of the corpus.
    Adding the two ordinal ranks retains the frequency prior while admitting high-leverage rows.
    """

    if n <= 0 or n > TOKENIZER_VOCAB_SIZE:
        raise ValueError(f"shortlist size {n} is outside 1..{TOKENIZER_VOCAB_SIZE}")
    ranking = Path(ranking_path)
    tokenizer = Path(tokenizer_dir)
    counts = load_total_counts(ranking, VOCAB_SIZE)[:TOKENIZER_VOCAB_SIZE]
    norms = _row_squared_norms(lm_head, TOKENIZER_VOCAB_SIZE)
    score = _ordinal_rank_descending(counts) + _ordinal_rank_descending(norms)

    forced = read_special_ids(tokenizer)
    forced_set = {token_id for token_id in forced if 0 <= token_id < TOKENIZER_VOCAB_SIZE}
    order = np.argsort(score, kind="stable")
    selected = [int(token_id) for token_id in order if int(token_id) not in forced_set]
    selected = selected[: n - len(forced_set)] + sorted(forced_set)
    selected_array = np.asarray(selected, dtype=np.int64)
    selected_array = selected_array[np.argsort(score[selected_array], kind="stable")]
    if selected_array.size != n or np.unique(selected_array).size != n:
        raise ValueError("hybrid proposal shortlist is not exact and unique")
    return DraftHeadContext(
        n=n,
        selected=np.ascontiguousarray(selected_array),
        ranking=ranking,
        tokenizer=tokenizer,
        force_include=tuple(sorted(forced_set)),
    )


__all__ = [
    "DEFAULT_RANKING",
    "DraftHeadContext",
    "compute_shortlist",
    "materialize_draft_head",
    "materialize_draft_head_token_ids",
]
