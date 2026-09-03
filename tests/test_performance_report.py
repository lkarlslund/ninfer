from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from tools.bench import performance_report


def _sample(engine: str, repetition: int, value: float) -> dict[str, object]:
    return {
        "model": "qwen3.6-27b",
        "engine": engine,
        "weight_profile": "q8",
        "kv_cache": "q8",
        "speculative_mode": "none",
        "workload": "pp1024",
        "context_tokens": 0,
        "prompt_tokens": 1024,
        "generated_tokens": 1,
        "metric": "prefill_tok_s",
        "unit": "tok/s",
        "repetition": repetition,
        "value": value,
    }


class PerformanceReportTest(unittest.TestCase):
    def test_aggregate_and_offline_html(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            suite = {
                "schema_version": 1,
                "artifact_type": "ninfer_performance_suite",
                "title": "Qualification",
                "generated_at": "2026-07-28T00:00:00Z",
                "environment": {"gpu": "RTX PRO 6000"},
                "artifacts": [{"engine": "ninfer", "path": "model.ninfer"}],
                "samples": [
                    _sample("ninfer", 0, 120.0),
                    _sample("ninfer", 1, 122.0),
                    _sample("llama.cpp", 0, 100.0),
                    _sample("llama.cpp", 1, 101.0),
                ],
            }
            path = tmp_path / "suite.json"
            path.write_text(json.dumps(suite), encoding="utf-8")
            summary = performance_report.build_summary(
                performance_report.load_suite(path)
            )
            performance_report.write_summary(summary, tmp_path)
            performance_report.render_report(summary, tmp_path / "report.html")

            self.assertEqual(len(summary["results"]), 2)
            self.assertAlmostEqual(summary["results"][1]["mean"], 121.0)
            report = (tmp_path / "report.html").read_text(encoding="utf-8")
            self.assertIn("Qualification", report)
            self.assertIn("performance-data", report)
            self.assertNotIn("https://", report)

    def test_rejects_nonfinite_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_type": "ninfer_performance_suite",
                        "samples": [_sample("ninfer", 0, math.nan)],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "finite"):
                performance_report.load_suite(path)


if __name__ == "__main__":
    unittest.main()
