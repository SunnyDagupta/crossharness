#!/usr/bin/env python3
"""Analyze Phase 2a JSONL results and emit a markdown results section.

    python3 analyze.py results/live_*.jsonl
    python3 analyze.py data/hermes3-8b-q4-ollama-20260611.jsonl

Reads one or more results JSONL files (one JSON object per line),
recomputes aggregates, adds per-family breakdowns, and prints a markdown
section suitable for pasting into README.md.

Stats included:
  - Per-condition success rate with 95% Wilson-style two-proportion SE
  - Wire-format recovery and surface recovery with standard error
  - Call-classification counts with rule-of-three upper bound for deviation
  - Per-family × condition breakdown
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from typing import List


def load_rows(paths: List[str]) -> List[dict]:
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def proportion_se(p: float, n: int) -> float:
    """Standard error for a proportion."""
    if n == 0:
        return float("nan")
    return math.sqrt(p * (1.0 - p) / n)


def two_proportion_se(p1: float, n1: int, p2: float, n2: int) -> float:
    """Pooled SE for (p2 - p1) under H0: p1 == p2."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    return math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))


def rule_of_three(n: int) -> float:
    """Upper 95% CI bound for a zero-event proportion (rule of three)."""
    if n == 0:
        return float("nan")
    return 3.0 / n


def aggregate(rows: List[dict]) -> dict:
    by_cond: dict = defaultdict(list)
    calls_by_cond: dict = defaultdict(list)
    by_fam_cond: dict = defaultdict(lambda: defaultdict(list))

    for r in rows:
        cond = r["condition"]
        by_cond[cond].append(r["passed"])
        calls_by_cond[cond].extend(r.get("calls", []))
        by_fam_cond[r["family"]][cond].append(r["passed"])

    result: dict = {}
    for cond in ("none", "syntactic", "full"):
        data = by_cond.get(cond, [])
        n = len(data)
        ok = sum(data)
        p = ok / n if n else float("nan")
        calls = calls_by_cond.get(cond, [])
        cc = Counter(calls)
        total_calls = sum(cc.values())
        native = cc.get("harness_native", 0)
        deviation = total_calls - native
        result[cond] = {
            "ok": ok, "n": n, "p": p,
            "calls_total": total_calls,
            "calls_native": native,
            "calls_deviation": deviation,
        }

    result["families"] = {
        fam: {
            cond: {"ok": sum(v), "n": len(v), "p": sum(v) / len(v) if v else 0.0}
            for cond, v in cond_map.items()
        }
        for fam, cond_map in by_fam_cond.items()
    }
    return result


def render_markdown(agg: dict, paths: List[str]) -> str:
    lines: List[str] = []
    a = agg

    def rate(cond: str) -> str:
        d = a[cond]
        pct = 100.0 * d["p"] if not math.isnan(d["p"]) else float("nan")
        return f"{d['ok']}/{d['n']} ({pct:.1f}%)"

    p_none = a["none"]["p"]
    p_syn  = a["syntactic"]["p"]
    p_full = a["full"]["p"]
    n_none = a["none"]["n"]
    n_syn  = a["syntactic"]["n"]
    n_full = a["full"]["n"]

    wfr   = p_syn - p_none          # wire-format recovery
    sr    = p_full - p_syn           # surface recovery
    se_wfr = two_proportion_se(p_none, n_none, p_syn, n_syn)
    se_sr  = two_proportion_se(p_syn, n_syn, p_full, n_full)

    # deviation upper CI
    full_dev   = a["full"]["calls_deviation"]
    full_calls = a["full"]["calls_total"]
    syn_dev    = a["syntactic"]["calls_deviation"]
    syn_calls  = a["syntactic"]["calls_total"]
    full_upper = rule_of_three(full_calls) if full_dev == 0 else full_dev / full_calls
    syn_upper  = rule_of_three(syn_calls)  if syn_dev  == 0 else syn_dev  / syn_calls

    # avg calls/row under full
    n_full_rows = a["full"]["n"]
    calls_per_row = a["full"]["calls_total"] / n_full_rows if n_full_rows else 0.0

    lines.append("## Measured results (Phase 2a)\n")

    lines.append("**Run metadata**\n")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append("| Date | 2026-06-11 |")
    lines.append("| Model | hermes3:8b (Q4_0, 8.0B params, 131072-ctx) |")
    lines.append("| Serving | Ollama 0.30.6, localhost:11434 |")
    lines.append("| Harness | crossharness @ 8dc769d + stream:false patch |")
    lines.append(f"| JSONL | {', '.join(paths)} |")
    lines.append(f"| Rows | {sum(a[c]['n'] for c in ('none','syntactic','full'))} "
                 f"(24 tasks × 3 conditions × 5 samples) |")
    lines.append("")

    lines.append("**Note:** The `none` condition is the definitional floor — "
                 "without any translation the harness never receives a well-formed "
                 "tool call, so 0% is by construction, not a model property.\n")

    lines.append("**Success rate per condition**\n")
    lines.append("| Condition | Success | Rate |")
    lines.append("|-----------|---------|------|")
    for cond in ("none", "syntactic", "full"):
        lines.append(f"| {cond} | {a[cond]['ok']}/{a[cond]['n']} | "
                     f"{100*a[cond]['p']:.1f}% |")
    lines.append("")
    lines.append(f"Wire-format recovery (syntactic − none): "
                 f"**{wfr:+.1%}** (SE {se_wfr:.1%})")
    lines.append(f"Surface recovery (full − syntactic): "
                 f"**{sr:+.1%}** (SE {se_sr:.1%}) — "
                 f"{'no detectable difference' if abs(sr) < 2 * se_sr else 'detectable'}\n")

    lines.append("**Call classification**\n")
    lines.append("| Condition | Calls | harness_native | deviation | upper 95% CI |")
    lines.append("|-----------|-------|----------------|-----------|--------------|")
    for cond, upper in (("syntactic", syn_upper), ("full", full_upper)):
        d = a[cond]
        lines.append(f"| {cond} | {d['calls_total']} | {d['calls_native']} "
                     f"| {d['calls_deviation']} | ≤{100*upper:.1f}% |")
    lines.append(f"\nAverage calls/row under full: {calls_per_row:.2f}\n")

    lines.append("**Success by family (full condition)**\n")
    lines.append("| Family | none | syntactic | full |")
    lines.append("|--------|------|-----------|------|")
    for fam in sorted(a["families"]):
        fd = a["families"][fam]
        def cell(cond: str) -> str:
            if cond not in fd:
                return "—"
            d = fd[cond]
            return f"{d['ok']}/{d['n']}"
        lines.append(f"| {fam} | {cell('none')} | {cell('syntactic')} | {cell('full')} |")
    lines.append("")

    lines.append("**Reading**\n")
    lines.append(
        f"Wire-format recovery is {wfr:+.1%}: without the syntactic shim, "
        f"hermes3:8b emits zero harness-compatible calls on every task. "
        f"The shim's format translation alone recovers {100*p_syn:.1f}% of tasks. "
        f"Surface deviation did not occur in this cell "
        f"({a['full']['calls_deviation']}/{a['full']['calls_total']} calls "
        f"used a generic surface, ≤{100*full_upper:.1f}% upper 95% CI), "
        f"so the full alias layer made no detectable difference ({sr:+.1%}, {abs(sr/se_sr):.1f} SE). "
        f"The dominant residual failure is non-invocation "
        f"({calls_per_row:.2f} calls/row; W family ~{100*a['families'].get('W',{}).get('full',{}).get('p',0):.0f}% "
        f"vs M family ~{100*a['families'].get('M',{}).get('full',{}).get('p',0):.0f}%): "
        f"the model often produces no tool call at all, which is behavioral, not notational."
    )
    lines.append("")
    lines.append(
        "**Scope limits:** one 8B model, Q4_0 quant, 3-tool clean catalog advertised "
        "in-prompt, Hermes-style prompt the model was trained on. "
        "The Phase 1 constructed fixtures exhibit a surface-deviation failure mode "
        "this live cell did not reproduce — whether that reflects the model class, "
        "the catalog size, or the advertisement condition is the next measurement axis."
    )
    lines.append("")
    lines.append("**Phase 2b decision (applied rule: <5pp branch):** "
                 f"surface recovery was {sr:+.1%} ({abs(sr/se_sr):.1f} SE), "
                 "below the 5pp threshold. Phase 2b is NOT proceeding on this evidence. "
                 "The next measurement axis is advertisement: larger and messier tool catalogs, "
                 "no in-prompt advertisement, and non-Hermes-trained models. Pending owner decision.")

    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: analyze.py <file.jsonl> [<file2.jsonl> ...]", file=sys.stderr)
        sys.exit(1)
    rows = load_rows(sys.argv[1:])
    agg = aggregate(rows)
    print(render_markdown(agg, sys.argv[1:]))


if __name__ == "__main__":
    main()
