#!/usr/bin/env python3
"""Compare fixed-token causal scores; keep engine disagreement separate from correctness."""

import argparse
import json
import math
import statistics
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="ninfer-perplexity --token-scores report.json")
    parser.add_argument("--url", default="http://127.0.0.1:18086")
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--responses", type=Path,
                        help="cache vLLM responses by stream/window for offline comparison")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    scores = {}
    for line in (args.report.parent / "token_scores.jsonl").read_text().splitlines():
        row = json.loads(line)
        scores[row["stream"], row["window"], row["position"]] = row
    args.output.mkdir(parents=True, exist_ok=True)
    deltas = []
    ninfer_scores, reference_scores = [], []
    with (args.output / "positions.jsonl").open("w") as output:
        for stream in report["streams"]:
            for window in stream["windows"]:
                payload = {"model": args.model, "prompt": window["input_ids"],
                           "max_tokens": 1, "temperature": 0, "echo": True,
                           "logprobs": 1}
                cached = (args.responses / f'{stream["id"]}-{window["index"]}.json'
                          if args.responses else None)
                if cached and cached.exists():
                    saved = json.loads(cached.read_text())
                    if saved["input_ids"] != window["input_ids"]:
                        raise RuntimeError("cached reference has different token IDs")
                    reply = saved["response"]
                else:
                    request = urllib.request.Request(
                        args.url + "/v1/completions", data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=600) as response:
                        reply = json.load(response)
                    if cached:
                        cached.parent.mkdir(parents=True, exist_ok=True)
                        cached.write_text(json.dumps({"input_ids": window["input_ids"],
                                                      "response": reply}))
                values = reply["choices"][0]["logprobs"]["token_logprobs"]
                if len(values) < len(window["input_ids"]):
                    raise RuntimeError("vLLM did not return all prompt scores")
                for index in range(window["first_target"], len(window["input_ids"])):
                    position = window["input_begin"] + index
                    score = scores[stream["id"], window["index"], position]
                    if score["token"] != window["input_ids"][index]:
                        raise RuntimeError("NInfer score token does not match its input position")
                    ours = score["logprob"]
                    theirs = values[index]
                    if theirs is None or not math.isfinite(theirs) or not math.isfinite(ours):
                        raise RuntimeError("nonfinite/missing causal score")
                    delta = ours - theirs
                    deltas.append(delta)
                    ninfer_scores.append(ours)
                    reference_scores.append(theirs)
                    output.write(json.dumps({"stream": stream["id"], "position": position,
                                             "ninfer": ours, "vllm": theirs,
                                             "delta": delta}) + "\n")
    ordered = sorted(abs(x) for x in deltas)
    summary = {"positions": len(deltas),
               "ninfer_mean_nll": -statistics.mean(ninfer_scores),
               "reference_mean_nll": -statistics.mean(reference_scores), "mean_delta_nats": statistics.mean(deltas),
               "mean_absolute_delta_nats": statistics.mean(ordered),
               "p99_absolute_delta_nats": ordered[int((len(ordered) - 1) * .99)],
               "fraction_over_one_nat": sum(x > 1 for x in ordered) / len(ordered),
               "interpretation": "Cross-engine disagreement; neither engine is an oracle."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
