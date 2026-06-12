#!/usr/bin/env python3
"""Analyze Phase 2a JSONL results and emit a markdown results section.

    python3 analyze.py results/live_*.jsonl
    python3 analyze.py data/hermes3-8b-q4-ollama-20260611.jsonl
    python3 analyze.py data/cell1.jsonl data/cell-a.jsonl data/cell-b.jsonl

Reads one or more results JSONL files (one JSON object per line),
recomputes aggregates, adds per-family breakdowns, and prints a markdown
section suitable for pasting into README.md.

Stats included:
  - Per-condition success rate with 95% Wilson-style two-proportion SE
  - Wire-format recovery and surface recovery with standard error
  - Call-classification counts with rule-of-three upper bound for deviation
  - Invocation rate (share of rows with at least one parsed call)
  - First-call vs post-first-call deviation decomposition
  - advertised_distractor rate (semantic confusion, out of shim scope)
  - Per-family × condition breakdown
  - Cross-cell comparison table when multiple advertisement cells present
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


def _call_records(row: dict) -> List[dict]:
    """Normalize old list[str] and new list[dict] call formats to list[dict]."""
    out = []
    for i, c in enumerate(row.get("calls", [])):
        if isinstance(c, str):
            out.append({"class": c, "turn": 1})
        else:
            out.append(c)
    return out


def aggregate(rows: List[dict]) -> dict:
    by_cond: dict = defaultdict(list)
    calls_by_cond: dict = defaultdict(list)
    by_fam_cond: dict = defaultdict(lambda: defaultdict(list))

    for r in rows:
        cond = r["condition"]
        by_cond[cond].append(r["passed"])
        calls_by_cond[cond].extend(_call_records(r))
        by_fam_cond[r["family"]][cond].append(r["passed"])

    result: dict = {}
    for cond in ("none", "syntactic", "full"):
        data = by_cond.get(cond, [])
        n = len(data)
        ok = sum(data)
        p = ok / n if n else float("nan")
        records = calls_by_cond.get(cond, [])
        cc = Counter(rec["class"] for rec in records)
        total_calls = sum(cc.values())
        native = cc.get("harness_native", 0)
        distractor = cc.get("advertised_distractor", 0)
        deviation = total_calls - native - distractor  # semantic confusion excluded

        # invocation rate: rows with >= 1 call
        cond_rows = [r for r in rows if r["condition"] == cond]
        invoked = sum(1 for r in cond_rows if _call_records(r))
        n_invocable = len(cond_rows)

        # first-call vs post-first-call deviation
        first_calls = []
        post_calls = []
        for r in cond_rows:
            recs = _call_records(r)
            if recs:
                first_turn = recs[0]["turn"]
                for rec in recs:
                    if rec["turn"] == first_turn:
                        first_calls.append(rec["class"])
                    else:
                        post_calls.append(rec["class"])

        fc_total = len(first_calls)
        fc_deviation = sum(1 for c in first_calls if c not in ("harness_native", "advertised_distractor"))
        pc_total = len(post_calls)
        pc_deviation = sum(1 for c in post_calls if c not in ("harness_native", "advertised_distractor"))

        result[cond] = {
            "ok": ok, "n": n, "p": p,
            "calls_total": total_calls,
            "calls_native": native,
            "calls_distractor": distractor,
            "calls_deviation": deviation,
            "invoked": invoked,
            "n_invocable": n_invocable,
            "first_call_total": fc_total,
            "first_call_deviation": fc_deviation,
            "post_call_total": pc_total,
            "post_call_deviation": pc_deviation,
        }

    result["families"] = {
        fam: {
            cond: {"ok": sum(v), "n": len(v), "p": sum(v) / len(v) if v else 0.0}
            for cond, v in cond_map.items()
        }
        for fam, cond_map in by_fam_cond.items()
    }
    return result


def render_markdown(agg: dict, paths: List[str], advertisement: str = "full") -> str:
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

    wfr   = p_syn - p_none
    sr    = p_full - p_syn
    se_wfr = two_proportion_se(p_none, n_none, p_syn, n_syn)
    se_sr  = two_proportion_se(p_syn, n_syn, p_full, n_full)

    # deviation CI
    full_dev   = a["full"]["calls_deviation"]
    full_calls = a["full"]["calls_total"]
    syn_dev    = a["syntactic"]["calls_deviation"]
    syn_calls  = a["syntactic"]["calls_total"]
    full_upper = rule_of_three(full_calls) if full_dev == 0 else full_dev / full_calls
    syn_upper  = rule_of_three(syn_calls)  if syn_dev  == 0 else syn_dev  / syn_calls

    n_full_rows = a["full"]["n"]
    calls_per_row = a["full"]["calls_total"] / n_full_rows if n_full_rows else 0.0

    cell_label = {"full": "cell 1 (full advertisement)",
                  "none": "cell A (no advertisement)",
                  "large": "cell B (large catalog)"}.get(advertisement, advertisement)

    lines.append(f"## Measured results (Phase 2a — {cell_label})\n")

    lines.append("**Run metadata**\n")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append("| Advertisement | " + advertisement + " |")
    lines.append(f"| JSONL | {', '.join(paths)} |")
    lines.append(f"| Rows | {sum(a[c]['n'] for c in ('none','syntactic','full'))} |")
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

    lines.append("**Invocation rate (syntactic + full conditions)**\n")
    lines.append("| Condition | Rows with ≥1 call | Rate |")
    lines.append("|-----------|-------------------|------|")
    for cond in ("syntactic", "full"):
        d = a[cond]
        inv_p = d["invoked"] / d["n_invocable"] if d["n_invocable"] else 0.0
        lines.append(f"| {cond} | {d['invoked']}/{d['n_invocable']} | {100*inv_p:.1f}% |")
    lines.append("")

    lines.append("**Call classification (full condition)**\n")
    fd = a["full"]
    lines.append("| Bucket | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| harness_native | {fd['calls_native']} |")
    if fd["calls_distractor"]:
        lines.append(f"| advertised_distractor | {fd['calls_distractor']} (semantic confusion, out of shim scope) |")
    lines.append(f"| deviation (mappable_generic + unmapped) | {fd['calls_deviation']} |")
    lines.append(f"| total | {fd['calls_total']} |")
    lines.append(f"\nDeviation upper 95% CI: ≤{100*full_upper:.1f}%")
    lines.append(f"Average calls/row under full: {calls_per_row:.2f}\n")

    lines.append("**First-call vs post-first-call deviation (full condition)**\n")
    lines.append("| Scope | Calls | Deviation | Rate |")
    lines.append("|-------|-------|-----------|------|")
    fc_rate = fd["first_call_deviation"] / fd["first_call_total"] if fd["first_call_total"] else 0.0
    pc_rate = fd["post_call_deviation"] / fd["post_call_total"] if fd["post_call_total"] else float("nan")
    lines.append(f"| first call | {fd['first_call_total']} | {fd['first_call_deviation']} | {100*fc_rate:.1f}% |")
    lines.append(f"| post-first | {fd['post_call_total']} | {fd['post_call_deviation']} | "
                 f"{'—' if math.isnan(pc_rate) else f'{100*pc_rate:.1f}%'} |")
    lines.append("")

    lines.append("**Syntactic condition call classification**\n")
    lines.append("| Condition | Calls | harness_native | deviation | upper 95% CI |")
    lines.append("|-----------|-------|----------------|-----------|--------------|")
    sd = a["syntactic"]
    lines.append(f"| syntactic | {sd['calls_total']} | {sd['calls_native']} "
                 f"| {sd['calls_deviation']} | ≤{100*syn_upper:.1f}% |")
    lines.append("")

    lines.append("**Success by family (full condition)**\n")
    lines.append("| Family | none | syntactic | full |")
    lines.append("|--------|------|-----------|------|")
    for fam in sorted(a["families"]):
        fd2 = a["families"][fam]
        def cell(cond: str) -> str:
            if cond not in fd2:
                return "—"
            d = fd2[cond]
            return f"{d['ok']}/{d['n']}"
        lines.append(f"| {fam} | {cell('none')} | {cell('syntactic')} | {cell('full')} |")
    lines.append("")

    return "\n".join(lines)


def render_cross_cell(cells: dict) -> str:
    """Render a cross-cell comparison table.

    cells: {"full": agg_dict, "none": agg_dict, "large": agg_dict}
    """
    lines: List[str] = []
    lines.append("## Cross-cell comparison\n")
    lines.append("| Cell | none success | syntactic success | full success | "
                 "wire-format recovery | surface recovery | deviation ≤ |")
    lines.append("|------|-------------|------------------|-------------|"
                 "---------------------|-----------------|-------------|")

    cell_labels = {
        "full": "cell 1 (full adv.)",
        "none": "cell A (no adv.)",
        "large": "cell B (large catalog)",
    }

    for adv, agg in sorted(cells.items()):
        p_none = agg["none"]["p"]
        p_syn  = agg["syntactic"]["p"]
        p_full = agg["full"]["p"]
        n_none = agg["none"]["n"]
        n_syn  = agg["syntactic"]["n"]
        n_full = agg["full"]["n"]
        wfr = p_syn - p_none
        sr  = p_full - p_syn
        se_wfr = two_proportion_se(p_none, n_none, p_syn, n_syn)
        se_sr  = two_proportion_se(p_syn, n_syn, p_full, n_full)

        full_dev   = agg["full"]["calls_deviation"]
        full_calls = agg["full"]["calls_total"]
        upper = rule_of_three(full_calls) if full_dev == 0 else full_dev / full_calls

        label = cell_labels.get(adv, adv)
        lines.append(
            f"| {label} "
            f"| {100*p_none:.1f}% "
            f"| {100*p_syn:.1f}% "
            f"| {100*p_full:.1f}% "
            f"| {wfr:+.1%} (SE {se_wfr:.1%}) "
            f"| {sr:+.1%} (SE {se_sr:.1%}) "
            f"| ≤{100*upper:.1f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: analyze.py <file.jsonl> [<file2.jsonl> ...]", file=sys.stderr)
        sys.exit(1)
    rows = load_rows(sys.argv[1:])

    # Group by advertisement cell if present; fall back to treating all as "full"
    by_adv: dict = defaultdict(list)
    for r in rows:
        by_adv[r.get("advertisement", "full")].append(r)

    if len(by_adv) == 1:
        adv = next(iter(by_adv))
        agg = aggregate(by_adv[adv])
        print(render_markdown(agg, sys.argv[1:], advertisement=adv))
    else:
        cells = {}
        for adv, cell_rows in by_adv.items():
            cells[adv] = aggregate(cell_rows)
            print(render_markdown(cells[adv], sys.argv[1:], advertisement=adv))
            print()
        print(render_cross_cell(cells))


if __name__ == "__main__":
    main()
