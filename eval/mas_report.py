"""
MAS report generator — produces a standalone interactive HTML dashboard.

Reads results.jsonl + summary.json from a run directory and renders
6 Plotly charts into a single self-contained HTML file (no server needed).

Charts:
  1. Convergence rate by persona          (horizontal bar)
  2. Turn distribution                    (histogram, converged vs. failed)
  3. Clarification funnel by persona      (stacked bar: 0 / 1 / 2 / forced)
  4. Spec accuracy by persona             (horizontal bar)
  5. Rolling convergence rate             (line, 50-run window)
  6. Summary scorecard                    (table)

Usage:
    from eval.mas_report import generate_report
    generate_report(Path("eval_results/mas_run_20260315_120000"))
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PERSONA_COLORS = [
    "#2E86AB",  # blue
    "#A23B72",  # purple
    "#F18F01",  # amber
    "#C73E1D",  # red
    "#3B1F2B",  # dark
]

_PASS_COLOR = "#2ecc71"
_FAIL_COLOR = "#e74c3c"


def _load(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    records = []
    with open(run_dir / "results.jsonl", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)

    with open(run_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    return df, summary


def _chart_convergence_by_persona(df: pd.DataFrame) -> go.Figure:
    personas = df["persona_name"].unique().tolist()
    rates = [df[df["persona_name"] == p]["converged"].mean() for p in personas]
    counts = [df[df["persona_name"] == p]["converged"].sum() for p in personas]
    totals = [len(df[df["persona_name"] == p]) for p in personas]

    fig = go.Figure(go.Bar(
        x=rates,
        y=personas,
        orientation="h",
        marker_color=_PERSONA_COLORS[:len(personas)],
        text=[f"{r:.1%}  ({c}/{t})" for r, c, t in zip(rates, counts, totals)],
        textposition="outside",
    ))
    fig.update_layout(
        title="Convergence Rate by Persona",
        xaxis=dict(title="Convergence Rate", tickformat=".0%", range=[0, 1.1]),
        yaxis=dict(title=""),
        height=350,
        margin=dict(l=160, r=80, t=50, b=40),
    )
    return fig


def _chart_turn_distribution(df: pd.DataFrame) -> go.Figure:
    converged = df[df["converged"]]["turns"]
    failed = df[~df["converged"]]["turns"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=converged, name="Converged", marker_color=_PASS_COLOR,
        opacity=0.75, xbins=dict(start=0.5, end=6.5, size=1),
    ))
    fig.add_trace(go.Histogram(
        x=failed, name="Failed", marker_color=_FAIL_COLOR,
        opacity=0.75, xbins=dict(start=0.5, end=6.5, size=1),
    ))
    fig.update_layout(
        title="Turn Distribution (Converged vs. Failed)",
        barmode="overlay",
        xaxis=dict(title="Turns to Close", dtick=1),
        yaxis=dict(title="Run Count"),
        height=350,
        margin=dict(l=60, r=40, t=50, b=40),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    return fig


def _chart_clarification_funnel(df: pd.DataFrame) -> go.Figure:
    personas = df["persona_name"].unique().tolist()

    def _pct(p: str, n: int) -> float:
        sub = df[df["persona_name"] == p]
        return len(sub[sub["clarification_count"] == n]) / len(sub)

    def _pct_forced(p: str) -> float:
        sub = df[df["persona_name"] == p]
        return sub["forced_forward"].mean()

    fig = go.Figure()
    colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
    labels = ["0 clarifications", "1 clarification", "2 clarifications", "Forced forward"]

    for i, (label, color) in enumerate(zip(labels, colors)):
        if label == "Forced forward":
            values = [_pct_forced(p) for p in personas]
        else:
            values = [_pct(p, i) for p in personas]
        fig.add_trace(go.Bar(
            name=label,
            x=personas,
            y=values,
            marker_color=color,
            text=[f"{v:.0%}" for v in values],
            textposition="inside",
        ))

    fig.update_layout(
        title="Clarification Funnel by Persona",
        barmode="stack",
        yaxis=dict(title="Proportion of Runs", tickformat=".0%"),
        height=380,
        margin=dict(l=60, r=40, t=50, b=80),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def _chart_spec_accuracy(df: pd.DataFrame) -> go.Figure:
    personas = df["persona_name"].unique().tolist()
    rates = [df[df["persona_name"] == p]["spec_match"].mean() for p in personas]

    fig = go.Figure(go.Bar(
        x=rates,
        y=personas,
        orientation="h",
        marker_color=_PERSONA_COLORS[:len(personas)],
        text=[f"{r:.1%}" for r in rates],
        textposition="outside",
    ))
    fig.update_layout(
        title="Spec Accuracy by Persona",
        xaxis=dict(title="Spec Match Rate", tickformat=".0%", range=[0, 1.1]),
        yaxis=dict(title=""),
        height=350,
        margin=dict(l=160, r=80, t=50, b=40),
    )
    return fig


def _chart_rolling_convergence(df: pd.DataFrame, window: int = 50) -> go.Figure:
    sorted_df = df.sort_values("run_id").reset_index(drop=True)
    rolling = sorted_df["converged"].rolling(window=window, min_periods=1).mean()

    fig = go.Figure(go.Scatter(
        x=sorted_df["run_id"],
        y=rolling,
        mode="lines",
        line=dict(color="#2E86AB", width=2),
        fill="tozeroy",
        fillcolor="rgba(46,134,171,0.15)",
    ))
    fig.add_hline(
        y=sorted_df["converged"].mean(),
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"Overall: {sorted_df['converged'].mean():.1%}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=f"Rolling Convergence Rate (window={window} runs)",
        xaxis=dict(title="Run Index"),
        yaxis=dict(title="Convergence Rate", tickformat=".0%", range=[0, 1.05]),
        height=350,
        margin=dict(l=60, r=60, t=50, b=40),
    )
    return fig


def _chart_scorecard(summary: dict) -> go.Figure:
    personas = [k for k in summary if k not in ("overall", "meta")]
    rows = personas + ["overall"]

    def _fmt(v, pct: bool = True) -> str:
        if v is None:
            return "—"
        if pct:
            return f"{float(v):.1%}"
        return f"{float(v):.2f}"

    headers = [
        "Persona", "N", "Convergence", "Avg Turns",
        "Clarif. Rate", "Forced Fwd", "Spec Acc.", "Intent Acc."
    ]

    cell_values = [[], [], [], [], [], [], [], []]
    for row in rows:
        s = summary[row]
        cell_values[0].append("<b>overall</b>" if row == "overall" else row)
        cell_values[1].append(str(s.get("n", "—")))
        cell_values[2].append(_fmt(s.get("convergence_rate")))
        cell_values[3].append(_fmt(s.get("mean_turns_to_close"), pct=False))
        cell_values[4].append(_fmt(s.get("clarification_rate")))
        cell_values[5].append(_fmt(s.get("forced_forward_rate")))
        cell_values[6].append(_fmt(s.get("spec_accuracy")))
        cell_values[7].append(_fmt(s.get("intent_accuracy")))

    fill_colors = [
        ["#f0f4f8"] * len(personas) + ["#2E86AB"],
    ] + [["white"] * len(personas) + ["#d6eaf8"]] * 7

    font_colors = [
        ["#2c3e50"] * len(personas) + ["white"],
    ] + [["#2c3e50"] * (len(personas) + 1)] * 7

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color="#2E86AB",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=cell_values,
            fill_color=fill_colors,
            font=dict(color=font_colors, size=11),
            align="left",
            height=28,
        ),
    ))
    fig.update_layout(
        title="Summary Scorecard",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _chart_judge_scores(df: pd.DataFrame) -> go.Figure | None:
    """Horizontal bar chart of judge composite scores by persona. Returns None if no scores."""
    if "judge_composite" not in df.columns or df["judge_composite"].isna().all():
        return None

    personas = df["persona_name"].unique().tolist()
    composites = []
    personas_with_data = []
    for p in personas:
        sub = df[df["persona_name"] == p]["judge_composite"].dropna()
        if not sub.empty:
            composites.append(sub.mean())
            personas_with_data.append(p)

    if not personas_with_data:
        return None

    fig = go.Figure(go.Bar(
        x=composites,
        y=personas_with_data,
        orientation="h",
        marker_color=_PERSONA_COLORS[:len(personas_with_data)],
        text=[f"{v:.3f}" for v in composites],
        textposition="outside",
    ))
    fig.update_layout(
        title="Judge Composite Score by Persona (Claude Opus 4.6)",
        xaxis=dict(title="Composite Score", range=[0, 1.1]),
        yaxis=dict(title=""),
        height=350,
        margin=dict(l=160, r=80, t=50, b=40),
    )
    return fig


def generate_report(run_dir: Path) -> Path:
    """
    Generate a standalone HTML report from a MAS run directory.
    Returns the path to the generated report.html.
    """
    df, summary = _load(run_dir)

    meta = summary.get("meta", {})
    overall = summary.get("overall", {})
    timestamp = meta.get("timestamp", "")
    n_runs = meta.get("total_runs", len(df))
    conv_rate = overall.get("convergence_rate", 0)
    judge_enabled = meta.get("judge_enabled", False)

    charts = [
        _chart_convergence_by_persona(df),
        _chart_turn_distribution(df),
        _chart_clarification_funnel(df),
        _chart_spec_accuracy(df),
        _chart_rolling_convergence(df),
        _chart_scorecard(summary),
    ]

    # Add judge chart if scores were collected
    if judge_enabled:
        judge_chart = _chart_judge_scores(df)
        if judge_chart is not None:
            charts.insert(4, judge_chart)  # insert before rolling convergence

    # Assemble HTML
    chart_divs = "\n".join(
        f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'
        for fig in charts
    )

    # Load plotly CDN script from first chart
    plotlyjs = charts[0].to_html(full_html=False, include_plotlyjs="cdn")
    plotlyjs_tag = plotlyjs.split("\n")[0] if "<script" in plotlyjs else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Greenvest MAS Report — {timestamp[:10]}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #f5f7fa; margin: 0; padding: 20px; color: #2c3e50; }}
    .header {{ background: #2E86AB; color: white; padding: 24px 32px; border-radius: 8px;
               margin-bottom: 24px; }}
    .header h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .header .subtitle {{ opacity: 0.85; font-size: 14px; }}
    .kpi-row {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
    .kpi {{ background: white; border-radius: 8px; padding: 20px 28px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08); flex: 1; min-width: 140px; }}
    .kpi .value {{ font-size: 32px; font-weight: 700; color: #2E86AB; }}
    .kpi .label {{ font-size: 12px; color: #7f8c8d; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
    .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .chart {{ background: white; border-radius: 8px; padding: 16px;
              box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    .chart.full-width {{ grid-column: 1 / -1; }}
    @media (max-width: 900px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Greenvest — Multi-Agent Simulation Report</h1>
    <div class="subtitle">
      {n_runs} runs · {len(df["persona_name"].unique())} personas · {timestamp}
    </div>
  </div>

  <div class="kpi-row">
    <div class="kpi">
      <div class="value">{conv_rate:.1%}</div>
      <div class="label">Overall Convergence Rate</div>
    </div>
    <div class="kpi">
      <div class="value">{overall.get("mean_turns_to_close") or "—"}</div>
      <div class="label">Avg Turns to Close</div>
    </div>
    <div class="kpi">
      <div class="value">{overall.get("spec_accuracy", 0):.1%}</div>
      <div class="label">Spec Accuracy</div>
    </div>
    <div class="kpi">
      <div class="value">{overall.get("clarification_rate", 0):.1%}</div>
      <div class="label">Clarification Rate</div>
    </div>
    <div class="kpi">
      <div class="value">{overall.get("forced_forward_rate", 0):.1%}</div>
      <div class="label">Forced Forward Rate</div>
    </div>
    <div class="kpi">
      <div class="value">{overall.get("intent_accuracy", 0):.1%}</div>
      <div class="label">Intent Accuracy</div>
    </div>
    {"" if not judge_enabled or not overall.get("judge_composite") else f'''
    <div class="kpi">
      <div class="value">{overall.get("judge_composite", 0):.3f}</div>
      <div class="label">Judge Composite</div>
    </div>'''}
  </div>

  <div class="chart-grid">
    {chart_divs}
  </div>
</body>
</html>"""

    report_path = run_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"Report written to: {report_path}")
    return report_path
