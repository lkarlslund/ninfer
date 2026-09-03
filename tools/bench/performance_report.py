#!/usr/bin/env python3
"""Normalize benchmark samples and render an offline performance report.

The input is a versioned JSON document. Raw benchmark adapters are deliberately
kept separate: this module owns aggregation and presentation, not process
execution or engine-specific output parsing.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "ninfer_performance_suite"
SUMMARY_TYPE = "ninfer_performance_summary"

_DIMENSIONS = (
    "model",
    "engine",
    "weight_profile",
    "kv_cache",
    "speculative_mode",
    "workload",
    "context_tokens",
    "prompt_tokens",
    "generated_tokens",
    "metric",
    "unit",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_suite(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "performance suite root must be an object")
    _require(value.get("schema_version") == SCHEMA_VERSION, "unsupported suite schema_version")
    _require(value.get("artifact_type") == ARTIFACT_TYPE, "unsupported suite artifact_type")
    samples = value.get("samples")
    _require(isinstance(samples, list) and samples, "suite samples must be a non-empty array")
    for index, sample in enumerate(samples):
        _require(isinstance(sample, dict), f"sample {index} must be an object")
        for name in _DIMENSIONS:
            _require(name in sample, f"sample {index} is missing {name}")
        _require(
            isinstance(sample.get("value"), (int, float))
            and math.isfinite(float(sample["value"])),
            f"sample {index} value must be finite",
        )
        _require(
            isinstance(sample.get("repetition"), int) and sample["repetition"] >= 0,
            f"sample {index} repetition must be a nonnegative integer",
        )
    return value


def _sample_key(sample: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(sample[name] for name in _DIMENSIONS)


def aggregate(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for sample in samples:
        grouped[_sample_key(sample)].append(float(sample["value"]))

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(part) for part in item)):
        values = grouped[key]
        mean = statistics.fmean(values)
        stddev = statistics.stdev(values) if len(values) > 1 else 0.0
        sem = stddev / math.sqrt(len(values)) if values else 0.0
        row = dict(zip(_DIMENSIONS, key, strict=True))
        row.update(
            {
                "samples": len(values),
                "mean": mean,
                "median": statistics.median(values),
                "stddev": stddev,
                "ci95_low": mean - 1.96 * sem,
                "ci95_high": mean + 1.96 * sem,
                "minimum": min(values),
                "maximum": max(values),
            }
        )
        rows.append(row)
    return rows


def evaluate_gates(
    rows: Sequence[dict[str, Any]], gates: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    indexed = {
        tuple(row[name] for name in _DIMENSIONS): row
        for row in rows
    }
    results: list[dict[str, Any]] = []
    for gate in gates:
        numerator_key = tuple(gate["numerator"][name] for name in _DIMENSIONS)
        denominator_key = tuple(gate["denominator"][name] for name in _DIMENSIONS)
        numerator = indexed.get(numerator_key)
        denominator = indexed.get(denominator_key)
        result = {
            "name": gate["name"],
            "minimum_ratio": float(gate.get("minimum_ratio", 1.0)),
            "status": "missing",
            "ratio": None,
            "conservative_ratio": None,
        }
        if numerator is not None and denominator is not None:
            denominator_mean = float(denominator["mean"])
            if denominator_mean <= 0:
                raise ValueError(f"gate {gate['name']}: denominator mean must be positive")
            ratio = float(numerator["mean"]) / denominator_mean
            conservative = float(numerator["ci95_low"]) / float(denominator["ci95_high"])
            result["ratio"] = ratio
            result["conservative_ratio"] = conservative
            result["status"] = (
                "pass" if conservative > result["minimum_ratio"] else "fail"
            )
        results.append(result)
    return results


def build_summary(suite: dict[str, Any]) -> dict[str, Any]:
    rows = aggregate(suite["samples"])
    gates = suite.get("gates", [])
    _require(isinstance(gates, list), "suite gates must be an array")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": SUMMARY_TYPE,
        "title": suite.get("title", "NInfer performance"),
        "generated_at": suite.get("generated_at"),
        "environment": suite.get("environment", {}),
        "artifacts": suite.get("artifacts", []),
        "results": rows,
        "gates": evaluate_gates(rows, gates),
    }


def write_summary(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = summary["results"]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _report_html(summary: dict[str, Any]) -> str:
    title = html.escape(str(summary["title"]))
    embedded = json.dumps(summary, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: dark; --bg:#09111f; --panel:#111d31; --line:#263957;
 --text:#eaf1ff; --muted:#9fb0ca; --accent:#6ee7ff; --good:#54e39e; --bad:#ff7b91; }}
* {{ box-sizing:border-box }} body {{ margin:0; background:radial-gradient(circle at 80% 0,#173259 0,#09111f 38%);
 color:var(--text); font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif; }}
main {{ max-width:1500px; margin:auto; padding:32px }} h1 {{ font-size:34px; margin:0 0 5px }}
.sub {{ color:var(--muted); margin-bottom:24px }} .toolbar,.cards {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0 }}
select {{ background:#0b1628; color:var(--text); border:1px solid var(--line); border-radius:8px; padding:8px 12px }}
.card,.panel {{ background:linear-gradient(145deg,#14223a,#0e192b); border:1px solid var(--line);
 border-radius:13px; box-shadow:0 14px 35px #0005 }} .card {{ padding:14px 18px; min-width:180px }}
.card b {{ display:block; font-size:22px }} .panel {{ padding:18px; margin:16px 0; overflow:auto }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(480px,1fr)); gap:16px }}
svg {{ width:100%; min-height:320px }} table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums }}
th,td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap }}
th:first-child,td:first-child {{ text-align:left }} th {{ color:var(--muted); position:sticky; top:0; background:#111d31 }}
.pass {{ color:var(--good) }} .fail {{ color:var(--bad) }} .missing {{ color:var(--muted) }}
details {{ margin-top:12px }} pre {{ white-space:pre-wrap; color:#cbd8ef }}
@media print {{ body {{ background:white; color:#111 }} .toolbar {{ display:none }} .card,.panel {{ box-shadow:none; background:white; border-color:#bbb }} }}
</style>
</head>
<body><main>
<h1>{title}</h1><div class="sub" id="subtitle"></div>
<div class="toolbar"><label>Model <select id="model"></select></label>
<label>Metric <select id="metric"></select></label></div>
<div class="cards" id="cards"></div>
<div class="grid"><section class="panel"><h2>Throughput by configuration</h2><svg id="bars"></svg></section>
<section class="panel"><h2>Performance across context</h2><svg id="lines"></svg></section></div>
<section class="panel"><h2>Acceptance gates</h2><table id="gates"></table></section>
<section class="panel"><h2>Normalized results</h2><table id="results"></table></section>
<section class="panel"><h2>Provenance</h2><details><summary>Environment and artifacts</summary><pre id="provenance"></pre></details></section>
</main>
<script id="performance-data" type="application/json">{embedded}</script>
<script>
const data=JSON.parse(document.getElementById('performance-data').textContent);
const results=data.results, model=document.getElementById('model'), metric=document.getElementById('metric');
function choices(el,values){{el.innerHTML=[...values].sort().map(v=>`<option>${{v}}</option>`).join('')}}
choices(model,new Set(results.map(r=>r.model))); choices(metric,new Set(results.map(r=>r.metric)));
function svg(name,attrs={{}}){{const e=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));return e}}
function text(root,x,y,value,anchor='middle'){{const e=svg('text',{{x,y,fill:'#9fb0ca','font-size':11,'text-anchor':anchor}});e.textContent=value;root.append(e)}}
function selected(){{return results.filter(r=>r.model===model.value&&r.metric===metric.value)}}
function bars(rows){{const root=document.getElementById('bars');root.innerHTML='';root.setAttribute('viewBox','0 0 760 340');
 const top=Math.max(...rows.map(r=>r.mean),1), shown=rows.slice().sort((a,b)=>b.mean-a.mean).slice(0,16), w=680/shown.length;
 shown.forEach((r,i)=>{{const h=260*r.mean/top,x=55+i*w,y=285-h;root.append(svg('rect',{{x,y,width:Math.max(5,w-5),height:h,rx:3,fill:i%2?'#54e39e':'#6ee7ff'}}));
 text(root,x+w/2,304,`${{r.engine}}/${{r.weight_profile}}/${{r.kv_cache}}`);text(root,x+w/2,y-5,r.mean.toFixed(1));}});text(root,8,20,shown[0]?.unit||'', 'start')}}
function lines(rows){{const root=document.getElementById('lines');root.innerHTML='';root.setAttribute('viewBox','0 0 760 340');
 const withCtx=rows.filter(r=>Number(r.context_tokens)>=0), xmax=Math.max(...withCtx.map(r=>Number(r.context_tokens)),1), ymax=Math.max(...withCtx.map(r=>r.mean),1);
 const groups=withCtx.reduce((out,r)=>{{const key=`${{r.engine}}/${{r.weight_profile}}/${{r.kv_cache}}`; (out[key]??=[]).push(r); return out;}},{{}});let gi=0;
 Object.entries(groups).forEach(([name,g])=>{{g.sort((a,b)=>a.context_tokens-b.context_tokens);const color=['#6ee7ff','#54e39e','#ffbf69','#c4a7ff'][gi++%4];
 const points=g.map(r=>`${{55+650*Number(r.context_tokens)/xmax}},${{285-250*r.mean/ymax}}`).join(' ');root.append(svg('polyline',{{points,fill:'none',stroke:color,'stroke-width':2}}));
 g.forEach(r=>root.append(svg('circle',{{cx:55+650*Number(r.context_tokens)/xmax,cy:285-250*r.mean/ymax,r:3,fill:color}})));text(root,720,20+gi*15,name,'end');}});
 text(root,55,305,'0');text(root,705,305,String(xmax));}}
function table(id,headers,rows){{document.getElementById(id).innerHTML='<thead><tr>'+headers.map(h=>`<th>${{h}}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>`<td>${{v??''}}</td>`).join('')+'</tr>').join('')+'</tbody>'}}
function render(){{const rows=selected();bars(rows);lines(rows);document.getElementById('subtitle').textContent=`${{data.generated_at||''}} · ${{data.environment.gpu||data.environment.gpu_name||'unknown GPU'}}`;
 document.getElementById('cards').innerHTML=`<div class=card>Configurations<b>${{rows.length}}</b></div><div class=card>Best result<b>${{Math.max(...rows.map(r=>r.mean),0).toFixed(1)}}</b></div><div class=card>Samples<b>${{rows.reduce((n,r)=>n+r.samples,0)}}</b></div>`;
 table('results',['Configuration','Workload','Context','Mean','95% CI','Unit'],rows.map(r=>[`${{r.engine}}/${{r.weight_profile}}/${{r.kv_cache}}`,r.workload,r.context_tokens,r.mean.toFixed(2),`${{r.ci95_low.toFixed(2)}}–${{r.ci95_high.toFixed(2)}}`,r.unit]));
 table('gates',['Gate','Status','Ratio','Conservative'],data.gates.map(g=>[g.name,`<span class=${{g.status}}>${{g.status}}</span>`,g.ratio?.toFixed(3),g.conservative_ratio?.toFixed(3)]));}}
model.onchange=metric.onchange=render;document.getElementById('provenance').textContent=JSON.stringify({{environment:data.environment,artifacts:data.artifacts}},null,2);render();
</script></body></html>"""


def render_report(summary: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(_report_html(summary), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build_summary(load_suite(args.suite))
    write_summary(summary, args.output_dir)
    render_report(summary, args.output_dir / "report.html")
    print(args.output_dir / "report.html")


if __name__ == "__main__":
    main()
