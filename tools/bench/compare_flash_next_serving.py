#!/usr/bin/env python3
"""Compare paired serving workloads; numerical qualification remains a separate requirement."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path


def load(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {(row['case'], row['repetition'], row['turn']): row for row in rows}


def mean_request_metric(row, name):
    return statistics.mean(request[name] for request in row['requests'])


def compare(before, after):
    if not before or before.keys() != after.keys():
        raise ValueError('paired workload keys differ or no workloads were selected')
    results = {}
    rng = random.Random(20260922)
    for case in sorted({key[0] for key in before}):
        keys = [key for key in before if key[0] == case]
        effects, ttft, gaps, itl = [], [], [], []
        for repetition in sorted({key[1] for key in keys}):
            group = [key for key in keys if key[1] == repetition]
            for key in group:
                baseline, candidate = before[key], after[key]
                baseline_usage = [request['usage'] for request in baseline['requests']]
                candidate_usage = [request['usage'] for request in candidate['requests']]
                if case != 'turns' and (
                    [usage['prompt_tokens'] for usage in baseline_usage]
                    != [usage['prompt_tokens'] for usage in candidate_usage]
                ):
                    raise ValueError('prompt lengths differ')
                if (
                    [usage['completion_tokens'] for usage in baseline_usage]
                    != [usage['completion_tokens'] for usage in candidate_usage]
                ):
                    raise ValueError('output lengths differ; do not compare completion times')
                ttft.append(mean_request_metric(candidate, 'ttft')
                            / mean_request_metric(baseline, 'ttft') - 1)
                itl.append(mean_request_metric(candidate, 'mean_itl')
                           / mean_request_metric(baseline, 'mean_itl') - 1)
                gaps.append(max(request['max_stream_gap'] for request in candidate['requests'])
                            / max(request['max_stream_gap'] for request in baseline['requests']) - 1)
            effects.append(math.log(
                sum(after[key]['wall_seconds'] for key in group)
                / sum(before[key]['wall_seconds'] for key in group)
            ))
        bootstrap = sorted(
            1 - math.exp(statistics.mean(rng.choices(effects, k=len(effects))))
            for _ in range(10000)
        )
        gain = 1 - math.exp(statistics.mean(effects))
        latency_guard = statistics.mean(ttft) <= .05 and statistics.mean(itl) <= .05
        single_guard = not case.startswith('single') or gain >= -.02
        results[case] = {
            'pairs': len(effects),
            'completion_time_improvement': gain,
            'bootstrap_95_interval': [bootstrap[250], bootstrap[9749]],
            'mean_ttft_change': statistics.mean(ttft),
            'mean_itl_change': statistics.mean(itl),
            'mean_max_stream_gap_change': statistics.mean(gaps),
            'completion_gain_gate': gain >= .02 and bootstrap[250] > 0,
            'latency_guard': latency_guard,
            'single_request_guard': single_guard,
        }
    return {
        'workloads': results,
        'meets_measured_timing_gates': (
            any(row['completion_gain_gate'] for case, row in results.items()
                if not case.startswith('single'))
            and all(row['latency_guard'] and row['single_request_guard']
                    for row in results.values())
        ),
        'interpretation': 'Timing gate only. Check numerical qualification and workload coverage.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', help='compare only these matched workloads')
    args = parser.parse_args()
    before, after = load(args.baseline), load(args.candidate)
    if args.cases:
        before = {key: row for key, row in before.items() if key[0] in args.cases}
        after = {key: row for key, row in after.items() if key[0] in args.cases}
    result = compare(before, after)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
