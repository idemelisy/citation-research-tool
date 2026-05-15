"""Write evaluation artifacts: JSON, CSV, markdown, HTML."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_results_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"generated_at": _utc_now(), "results": rows}, indent=2, default=str),
        encoding="utf-8",
    )


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = ["query", *sorted(k for k in rows[0].get("metrics", {}) if k != "query")]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            m = dict(row.get("metrics") or {})
            w.writerow({"query": row.get("query"), **m})


def write_leaderboard_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda r: float((r.get("metrics") or {}).get("composite", 0)), reverse=True)
    lines = [
        "# SRG Lite evaluation leaderboard",
        "",
        f"Generated: {_utc_now()}",
        "",
        "| Rank | Query | Composite | Intent | Canonical | Foundational | Noise↓ |",
        "|------|-------|-----------|--------|-----------|--------------|--------|",
    ]
    for i, row in enumerate(ranked[:30], 1):
        m = row.get("metrics") or {}
        q = str(row.get("query") or "")[:48]
        lines.append(
            f"| {i} | {q} | {m.get('composite', 0):.3f} | {m.get('intent_precision', 0):.3f} | "
            f"{m.get('canonical_hit_rate', 0):.3f} | {m.get('foundational_quality', 0):.3f} | "
            f"{m.get('noise_ratio', 0):.3f} |"
        )
    if ranked:
        comps = [float((r.get("metrics") or {}).get("composite", 0)) for r in ranked]
        lines.extend(
            [
                "",
                "## Aggregate",
                "",
                f"- Queries: **{len(ranked)}**",
                f"- Mean composite: **{sum(comps) / len(comps):.3f}**",
                f"- Min / max composite: **{min(comps):.3f}** / **{max(comps):.3f}**",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_cases_md(path: Path, rows: list[dict[str, Any]], *, threshold: float = 0.45) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = [
        r
        for r in rows
        if float((r.get("metrics") or {}).get("composite", 1)) < threshold or r.get("error")
    ]
    lines = ["# Failure cases", "", f"Threshold composite < {threshold}", ""]
    for row in failures:
        q = row.get("query")
        lines.append(f"## {q}")
        if row.get("error"):
            lines.append(f"- **Error:** {row['error']}")
        m = row.get("metrics") or {}
        lines.append(f"- Composite: {m.get('composite', 'n/a')}")
        lines.append(f"- Noise ratio: {m.get('noise_ratio', 'n/a')}")
        found = row.get("foundational_papers") or []
        if found:
            lines.append("- Foundational:")
            for frow in found[:5]:
                if isinstance(frow, dict):
                    lines.append(f"  - {frow.get('title', '?')}")
        lines.append("")
    path.write_text("\n".join(lines) if failures else "# Failure cases\n\nNo failures under threshold.\n", encoding="utf-8")


def write_html_dashboard(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda r: float((r.get("metrics") or {}).get("composite", 0)), reverse=True)
    rows_html = []
    for row in ranked:
        m = row.get("metrics") or {}
        err = row.get("error")
        status = f'<span class="err">{err}</span>' if err else "ok"
        rows_html.append(
            f"<tr><td>{row.get('query','')}</td>"
            f"<td>{m.get('composite',0):.3f}</td>"
            f"<td>{m.get('intent_precision',0):.3f}</td>"
            f"<td>{m.get('canonical_hit_rate',0):.3f}</td>"
            f"<td>{m.get('noise_ratio',0):.3f}</td>"
            f"<td>{status}</td></tr>"
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>SRG Lite Evaluation</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #334155; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #1e293b; }}
    tr:nth-child(even) {{ background: #1e293b55; }}
    .err {{ color: #f87171; }}
    h1 {{ font-size: 1.5rem; }}
  </style>
</head>
<body>
  <h1>SRG Lite evaluation</h1>
  <p>Generated {_utc_now()} · {len(rows)} queries</p>
  <table>
    <thead><tr><th>Query</th><th>Composite</th><th>Intent</th><th>Canonical</th><th>Noise</th><th>Status</th></tr></thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
