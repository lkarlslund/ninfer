"""Convert one or more ``ninfer_bench`` JSON reports into performance-suite samples.

Each input is tagged as ``WEIGHT_PROFILE:PATH``.  The resulting suite is accepted by
``tools/bench/performance_report.py`` and deliberately retains every measured repetition.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence


def _tagged(value: str) -> tuple[str, Path]:
    tag, separator, path = value.partition(":")
    if not separator or not tag or not path:
        raise argparse.ArgumentTypeError("expected WEIGHT_PROFILE:PATH")
    return tag, Path(path)


def _sample(
    report: dict[str, Any],
    profile: str,
    test: dict[str, Any],
    repetition: int,
    metric: str,
    value: float,
) -> dict[str, Any]:
    config = report["config"]
    speculative = test["speculative"]
    return {
        "model": (
            "qwen3.6-35b-a3b"
            if "35b" in report["artifact"]["path"].lower()
            else "qwen3.6-27b"
        ),
        "engine": "ninfer",
        "weight_profile": profile,
        "kv_cache": config["kv_cache"],
        "speculative_mode": (
            f"mtp{config['mtp_draft_tokens']}"
            if speculative["enabled"]
            else "off"
        ),
        "workload": test["label"],
        "context_tokens": int(test["n_prompt"] or config["max_context"]),
        "prompt_tokens": int(test["n_prompt"]),
        "generated_tokens": int(test["n_gen"]),
        "metric": metric,
        "unit": "token/s",
        "repetition": repetition,
        "value": value,
    }


def normalize(profile: str, path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    for test in report["tests"]:
        for repetition, rep in enumerate(test["reps"]):
            timings = rep["timings"]
            if test["kind"] in ("pp", "pp+tg"):
                samples.append(
                    _sample(
                        report,
                        profile,
                        test,
                        repetition,
                        "prefill_throughput",
                        test["n_prompt"] / timings["prefill_seconds"],
                    )
                )
            if test["kind"] in ("tg", "pp+tg"):
                decoded = rep["decode_output_tokens"]
                samples.append(
                    _sample(
                        report,
                        profile,
                        test,
                        repetition,
                        "decode_throughput",
                        decoded / timings["decode_seconds"],
                    )
                )
            if test["kind"] == "pp+tg":
                samples.append(
                    _sample(
                        report,
                        profile,
                        test,
                        repetition,
                        "combined_throughput",
                        (test["n_prompt"] + rep["decode_output_tokens"])
                        / timings["total_seconds"],
                    )
                )
    provenance = {
        "engine": "ninfer",
        "weight_profile": profile,
        "path": str(path.resolve()),
        "artifact": report["artifact"],
        "command": report["command"],
        "load": report["load"],
        "memory": report["memory"],
    }
    return samples, provenance


def normalize_llama(profile: str, path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    for row in rows:
        prompt = int(row["n_prompt"])
        generated = int(row["n_gen"])
        if prompt and generated:
            metric = "combined_throughput"
            workload = f"pp{prompt}+tg{generated}"
        elif prompt:
            metric = "prefill_throughput"
            workload = f"pp{prompt}"
        else:
            metric = "decode_throughput"
            workload = f"tg{generated}"
        kv_cache = str(row["type_k"])
        if row["type_v"] != row["type_k"]:
            kv_cache += f"/{row['type_v']}"
        for repetition, value in enumerate(row["samples_ts"]):
            samples.append(
                {
                    "model": (
                        "qwen3.6-27b"
                        if "27B" in row["model_type"]
                        else ("qwen3.6-35b-a3b" if "35B" in row["model_type"]
                              else row["model_type"])
                    ),
                    "engine": "llama.cpp",
                    "weight_profile": profile,
                    "kv_cache": kv_cache,
                    "speculative_mode": "off",
                    "workload": workload,
                    "context_tokens": prompt,
                    "prompt_tokens": prompt,
                    "generated_tokens": generated,
                    "metric": metric,
                    "unit": "token/s",
                    "repetition": repetition,
                    "value": float(value),
                }
            )
    first = rows[0]
    return samples, {
        "engine": "llama.cpp",
        "weight_profile": profile,
        "path": str(path.resolve()),
        "artifact": {
            "path": first["model_filename"],
            "file_size_bytes": first["model_size"],
        },
        "build_commit": first["build_commit"],
        "build_number": first["build_number"],
        "backends": first["backends"],
    }


def normalize_dflash(profile: str, path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        values = {row[0]: row[1:] for row in csv.reader(handle) if row}
    wall = [float(value) for value in values["round_wall_ms"]]
    licensed = [int(value) for value in values["round_licensed_tokens"]]
    if len(wall) != len(licensed):
        raise ValueError(f"{path}: DFlash wall-time and licensed-token samples differ")
    context = int(values["context_tokens"][0])
    draft = int(values["draft_tokens"][0])
    samples = [
        {
            "model": "qwen3.6-35b-a3b",
            "engine": "ninfer",
            "weight_profile": profile,
            "kv_cache": values["kv_cache"][0],
            "speculative_mode": f"dflash{draft}",
            "workload": f"dflash-round-{context}",
            "context_tokens": context,
            "prompt_tokens": context,
            "generated_tokens": draft + 1,
            "metric": "speculative_throughput",
            "unit": "token/s",
            "repetition": repetition,
            "value": tokens * 1000.0 / milliseconds,
        }
        for repetition, (milliseconds, tokens) in enumerate(zip(wall, licensed, strict=True))
    ]
    return samples, {
        "engine": "ninfer",
        "weight_profile": profile,
        "path": str(path.resolve()),
        "artifact": values["artifact"][0],
        "device": values["device"][0],
        "proposal_head": values["proposal_head"][0],
        "cuda_graph": values["cuda_graph"][0],
        "acceptance_rate": float(values["acceptance_rate"][0]),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=_tagged, required=True)
    parser.add_argument("--llama", action="append", type=_tagged, default=[])
    parser.add_argument("--dflash", action="append", type=_tagged, default=[])
    parser.add_argument(
        "--comparison-gates",
        action="store_true",
        help="require NInfer BF16-KV throughput to exceed matching llama.cpp F16-KV results",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="NInfer Qwen3.6 performance")
    args = parser.parse_args(argv)

    samples: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    environment: dict[str, Any] = {}
    for profile, path in args.input:
        normalized, provenance = normalize(profile, path)
        samples.extend(normalized)
        artifacts.append(provenance)
        if not environment:
            source = json.loads(path.read_text(encoding="utf-8"))["environment"]
            environment = source
    for profile, path in args.llama:
        normalized, provenance = normalize_llama(profile, path)
        samples.extend(normalized)
        artifacts.append(provenance)
    for profile, path in args.dflash:
        normalized, provenance = normalize_dflash(profile, path)
        samples.extend(normalized)
        artifacts.append(provenance)

    gates: list[dict[str, Any]] = []
    if args.comparison_gates:
        dimensions = (
            "model", "engine", "weight_profile", "kv_cache", "speculative_mode",
            "workload", "context_tokens", "prompt_tokens", "generated_tokens", "metric", "unit",
        )
        for numerator in samples:
            if (
                numerator["engine"] != "ninfer"
                or numerator["kv_cache"] not in ("bf16", "int8-group64", "int4-group64")
                or numerator["speculative_mode"] != "off"
                or numerator["repetition"] != 0
            ):
                continue
            llama_kv = {
                "bf16": "f16",
                "int8-group64": "q8_0",
                "int4-group64": "q4_0",
            }[numerator["kv_cache"]]
            denominator = next(
                (
                    candidate
                    for candidate in samples
                    if candidate["engine"] == "llama.cpp"
                    and candidate["weight_profile"] == numerator["weight_profile"]
                    and candidate["kv_cache"] == llama_kv
                    and candidate["speculative_mode"] == "off"
                    and candidate["model"] == numerator["model"]
                    and candidate["workload"] == numerator["workload"]
                    and candidate["metric"] == numerator["metric"]
                ),
                None,
            )
            if denominator is not None:
                gates.append(
                    {
                        "name": (
                            f"{numerator['model']} {numerator['weight_profile']} "
                            f"{numerator['workload']} "
                            f"KV {numerator['kv_cache']} vs {llama_kv}: "
                            "NInfer > llama.cpp"
                        ),
                        "minimum_ratio": 1.0,
                        "numerator": {name: numerator[name] for name in dimensions},
                        "denominator": {name: denominator[name] for name in dimensions},
                    }
                )

    suite = {
        "schema_version": 1,
        "artifact_type": "ninfer_performance_suite",
        "title": args.title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "artifacts": artifacts,
        "samples": samples,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
