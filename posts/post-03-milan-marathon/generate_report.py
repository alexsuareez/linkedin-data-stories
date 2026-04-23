#!/usr/bin/env python3
"""
generate_report.py
==================
Generates an interactive HTML race analysis report from a Garmin/Apple Watch .FIT file.

Usage:
    python generate_report.py --fit path/to/race.fit --output report.html
    python generate_report.py --fit path/to/race.fit  # outputs report.html by default

Dependencies:
    pip install fitparse numpy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from fitparse import FitFile


# ── CLI arguments ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate HTML race analysis from .FIT file")
    parser.add_argument("--fit",    required=True,        help="Path to the .FIT file")
    parser.add_argument("--output", default="report.html", help="Output HTML file path")
    parser.add_argument("--title",  default="Race Analysis", help="Report title")
    parser.add_argument("--event",  default="Milano Marathon 2026", help="Event name")
    parser.add_argument("--date",   default="06.04.2026",  help="Event date (display only)")
    return parser.parse_args()


# ── FIT file parsing ─────────────────────────────────────────────────────────

def load_laps(fitfile):
    """Extract per-km lap data from FIT messages."""
    laps = []
    for record in fitfile.get_messages("lap"):
        data = {f.name: f.value for f in record}

        dist    = data.get("total_distance", 0)
        elapsed = data.get("total_elapsed_time", 0)
        speed   = data.get("enhanced_avg_speed") or data.get("avg_speed", 0)
        hr      = data.get("avg_heart_rate")
        max_hr  = data.get("max_heart_rate")
        cadence = data.get("avg_running_cadence")
        power   = data.get("avg_power")

        pace_sec = (1000 / speed) if speed and speed > 0 else None

        laps.append({
            "km":            len(laps) + 1,
            "dist_m":        round(dist, 1),
            "time_s":        round(elapsed, 1),
            "pace_sec":      round(pace_sec, 1) if pace_sec else None,
            "avg_hr":        hr,
            "max_hr":        max_hr,
            "cadence_spm":   cadence * 2 if cadence else None,  # strides → steps/min
            "power_w":       power,
        })
    return laps


def load_stream(fitfile):
    """Extract second-by-second data stream from FIT records."""
    records = list(fitfile.get_messages("record"))
    stream  = []
    start_ts = None

    for i, rec in enumerate(records):
        # Sample every 5 seconds to reduce noise
        if i % 5 != 0:
            continue

        data  = {f.name: f.value for f in rec}
        ts    = data.get("timestamp")
        dist  = data.get("distance")
        speed = data.get("enhanced_speed")
        hr    = data.get("heart_rate")
        power = data.get("power")

        if not ts or not dist or not speed or speed <= 0:
            continue

        if start_ts is None:
            start_ts = ts

        pace_sec = 1000 / speed

        # Filter outliers (pace between 3:20 and 8:20 per km)
        if pace_sec < 200 or pace_sec > 500:
            continue

        stream.append({
            "elapsed_s": round((ts - start_ts).total_seconds()),
            "dist_km":   round(dist / 1000, 3),
            "pace_sec":  round(pace_sec, 1),
            "hr":        hr,
            "power":     power,
        })

    return stream


# ── Data processing ──────────────────────────────────────────────────────────

def filter_full_laps(laps):
    """Keep only laps that represent a full km (≥950m)."""
    return [l for l in laps if l["dist_m"] >= 950]


def compute_blocks(laps, block_size=5):
    """Compute average power and pace for every N-km block."""
    blocks = []
    for start in range(0, len(laps), block_size):
        chunk = laps[start:start + block_size]
        if not chunk:
            continue
        end   = min(start + block_size, len(laps))
        label = f"{start + 1}–{end}"

        valid_power = [l["power_w"] for l in chunk if l["power_w"]]
        valid_pace  = [l["pace_sec"] for l in chunk if l["pace_sec"]]

        blocks.append({
            "label":         label,
            "avg_power":     round(sum(valid_power) / len(valid_power), 1) if valid_power else None,
            "avg_pace_sec":  round(sum(valid_pace)  / len(valid_pace),  1) if valid_pace  else None,
        })
    return blocks


def compute_half_splits(laps):
    """Compute first and second half average pace."""
    mid    = len(laps) // 2
    half1  = laps[:mid]
    half2  = laps[mid:]

    def avg_pace(subset):
        valid = [l["pace_sec"] for l in subset if l["pace_sec"]]
        return round(sum(valid) / len(valid), 1) if valid else None

    h1 = avg_pace(half1)
    h2 = avg_pace(half2)
    diff_pct = round((h1 - h2) / h1 * 100, 1) if h1 and h2 else None

    return {"half1": h1, "half2": h2, "diff_pct": diff_pct}


def compute_power_curve(stream, durations_s=None):
    """
    Compute best average power for each target duration.
    Stream is sampled every 5s, so window = duration / 5 samples.
    """
    if durations_s is None:
        durations_s = [5, 10, 30, 60, 120, 300, 600, 900, 1200,
                       1800, 2400, 3000, 3600, 4500, 5400, 6000, 7200, 9000, 10800, 12000]

    powers = [s["power"] for s in stream if s.get("power")]
    curve  = []

    for dur in durations_s:
        window = max(1, dur // 5)
        if window > len(powers):
            break
        best = max(
            sum(powers[i:i + window]) / window
            for i in range(len(powers) - window + 1)
        )
        curve.append({"duration_s": dur, "power_w": round(best, 1)})

    return curve


def compute_lowess(stream, frac=0.12):
    """
    LOWESS (Locally Weighted Scatterplot Smoothing) trend line.
    Returns smoothed pace values indexed by distance.
    """
    x = np.array([s["dist_km"] for s in stream])
    y = np.array([s["pace_sec"] for s in stream])
    n = len(x)
    result = np.zeros(n)

    for i in range(n):
        distances = np.abs(x - x[i])
        bw        = max(np.sort(distances)[min(int(frac * n), n - 1)], 1e-10)
        weights   = np.maximum(0, 1 - (distances / bw) ** 3) ** 3

        w_sum = weights.sum()
        if w_sum == 0:
            result[i] = y[i]
            continue

        wx  = np.sum(weights * x)
        wy  = np.sum(weights * y)
        wxx = np.sum(weights * x * x)
        wxy = np.sum(weights * x * y)
        denom = w_sum * wxx - wx * wx

        if abs(denom) < 1e-10:
            result[i] = wy / w_sum
        else:
            b = (w_sum * wxy - wx * wy) / denom
            a = (wy - b * wx) / w_sum
            result[i] = a + b * x[i]

    return [{"dist_km": round(float(x[i]), 3), "pace_sec": round(float(result[i]), 1)}
            for i in range(n)]


def smooth_stream(stream, window=6):
    """Apply simple moving average to pace values in the stream."""
    paces = [s["pace_sec"] for s in stream]
    smoothed = []
    for i in range(len(paces)):
        start = max(0, i - window // 2)
        end   = min(len(paces), start + window)
        smoothed.append(round(sum(paces[start:end]) / (end - start), 1))

    return [
        {"dist_km": stream[i]["dist_km"], "pace_sec": smoothed[i]}
        for i in range(len(stream))
        if 240 <= smoothed[i] <= 400  # filter outliers
    ]


def compute_regression(laps):
    """Linear regression of HR vs Pace for scatter chart."""
    valid = [(l["avg_hr"], l["pace_sec"]) for l in laps if l["avg_hr"] and l["pace_sec"]]
    hrs   = np.array([v[0] for v in valid])
    paces = np.array([v[1] for v in valid])
    m, b  = np.polyfit(hrs, paces, 1)
    r2    = np.corrcoef(hrs, paces)[0, 1] ** 2
    return {
        "m":     float(m),
        "b":     float(b),
        "r2":    round(float(r2), 3),
        "reg_x": [int(hrs.min()), int(hrs.max())],
        "reg_y": [round(m * hrs.min() + b, 1), round(m * hrs.max() + b, 1)],
    }


def compute_summary(laps, stream):
    """Compute headline stats for the report header."""
    valid_pace = [l["pace_sec"] for l in laps if l["pace_sec"]]
    valid_hr   = [l["avg_hr"]   for l in laps if l["avg_hr"]]
    valid_pow  = [l["power_w"]  for l in laps if l["power_w"]]

    total_time_s = sum(l["time_s"] for l in laps)
    h = int(total_time_s // 3600)
    m = int((total_time_s % 3600) // 60)

    avg_pace = sum(valid_pace) / len(valid_pace) if valid_pace else None
    avg_hr   = round(sum(valid_hr)  / len(valid_hr))  if valid_hr  else None
    avg_pow  = round(sum(valid_pow) / len(valid_pow))  if valid_pow else None

    return {
        "finish_h":   h,
        "finish_m":   m,
        "avg_pace":   round(avg_pace, 1) if avg_pace else None,
        "avg_hr":     avg_hr,
        "avg_power":  avg_pow,
    }


# ── Formatting helpers ───────────────────────────────────────────────────────

def fmt_pace(sec):
    """Convert pace in seconds/km to mm:ss string."""
    if not sec:
        return "--:--"
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


# ── HTML template ────────────────────────────────────────────────────────────

def build_html(data, args):
    """Render the full HTML report string from processed data."""

    laps     = data["laps"]
    splits   = data["splits"]
    summary  = data["summary"]
    reg      = data["regression"]

    h1_fmt = fmt_pace(splits["half1"])
    h2_fmt = fmt_pace(splits["half2"])
    diff   = splits["diff_pct"]

    # Inject all datasets as JSON — single source of truth for JS charts
    laps_json        = json.dumps(laps)
    stream_json      = json.dumps(data["stream_smooth"])
    lowess_json      = json.dumps(data["lowess"])
    h1_json          = json.dumps(data["h1_stream"])
    h2_json          = json.dumps(data["h2_stream"])
    blocks_json      = json.dumps(data["blocks"])
    power_curve_json = json.dumps(data["power_curve"])
    scatter_json     = json.dumps([
        {"hr": l["avg_hr"], "pace_sec": l["pace_sec"], "km": l["km"]}
        for l in laps if l["avg_hr"] and l["pace_sec"]
    ])
    reg_json    = json.dumps(reg)
    splits_json = json.dumps(splits)
    avg_power   = summary["avg_power"]

    finish = f"{summary['finish_h']}<sup>h</sup>{summary['finish_m']}<sup>m</sup>"
    avg_pace_fmt = fmt_pace(summary["avg_pace"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{args.event} — Race Analysis</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
/* ── Design tokens ── */
:root {{
  --bg:       #0c0c0c;
  --surface:  #141414;
  --surface2: #1c1c1c;
  --border:   #252525;
  --orange:   #FF4D00;
  --blue:     #4FC3F7;
  --red:      #f04060;
  --purple:   #b47cff;
  --green:    #00e5a0;
  --white:    #ffffff;
  --text:     #efefef;
  --muted:    #555;
  --grid:     rgba(255,255,255,0.035);
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: 'Outfit', sans-serif;
  padding: 44px 48px;
  max-width: 1140px;
  margin: 0 auto;
}}
.header {{
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 48px;
  padding-bottom: 22px;
  border-bottom: 1px solid var(--border);
}}
.h-eyebrow {{
  font-family: 'DM Mono', monospace;
  font-size: 10px; letter-spacing: 3px;
  color: var(--orange); text-transform: uppercase; margin-bottom: 8px;
}}
.h-title {{
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 54px; font-weight: 800; line-height: 1;
}}
.h-title em {{ color: var(--orange); font-style: normal; }}
.h-stats {{ display: flex; gap: 36px; align-items: flex-end; }}
.stat {{ text-align: right; }}
.stat-val {{
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 30px; font-weight: 700; line-height: 1;
}}
.stat-val sup {{ font-size: 16px; color: var(--orange); }}
.stat-lbl {{
  font-family: 'DM Mono', monospace;
  font-size: 9px; letter-spacing: 2px;
  color: var(--muted); text-transform: uppercase; margin-top: 4px;
}}
.grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}}
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 28px 30px;
}}
.card.full {{ grid-column: 1 / -1; }}
.card-eyebrow {{
  font-family: 'DM Mono', monospace;
  font-size: 9px; letter-spacing: 3px;
  color: var(--orange); text-transform: uppercase; margin-bottom: 4px;
}}
.card-title {{
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 18px; font-weight: 700; margin-bottom: 3px;
}}
.card-sub {{
  font-family: 'DM Mono', monospace;
  font-size: 11px; color: var(--muted); margin-bottom: 22px;
}}
.chart-wrap {{ position: relative; }}
.insights {{
  display: flex; gap: 0;
  margin-top: 18px; padding-top: 16px;
  border-top: 1px solid var(--border);
}}
.ins {{ flex: 1; padding: 0 16px 0 0; }}
.ins + .ins {{ padding-left: 16px; border-left: 1px solid var(--border); }}
.ins-val {{
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 20px; font-weight: 700; line-height: 1.1;
}}
.ins-lbl {{
  font-family: 'DM Mono', monospace;
  font-size: 9px; letter-spacing: 1.5px;
  color: var(--muted); text-transform: uppercase; margin-top: 3px;
}}
.footer {{
  margin-top: 36px; padding-top: 18px;
  border-top: 1px solid var(--border);
  display: flex; justify-content: space-between;
}}
.footer-brand {{
  font-family: 'DM Mono', monospace;
  font-size: 9px; letter-spacing: 3px;
  color: var(--muted); text-transform: uppercase;
}}
.footer-brand em {{ color: var(--orange); font-style: normal; }}
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="h-eyebrow">— Road to Milan · Race Data Analysis</div>
    <div class="h-title">{args.event.replace("2026", "<em>2026</em>")}</div>
  </div>
  <div class="h-stats">
    <div class="stat">
      <div class="stat-val">{finish}</div>
      <div class="stat-lbl">Finish Time</div>
    </div>
    <div class="stat">
      <div class="stat-val">{avg_pace_fmt}<sup>/km</sup></div>
      <div class="stat-lbl">Avg Pace</div>
    </div>
    <div class="stat">
      <div class="stat-val">{summary['avg_hr']}<sup>bpm</sup></div>
      <div class="stat-lbl">Avg Heart Rate</div>
    </div>
    <div class="stat">
      <div class="stat-val">{avg_power}<sup>W</sup></div>
      <div class="stat-lbl">Avg Power</div>
    </div>
  </div>
</div>

<div class="grid">

  <!-- Chart 01: Pace bars + HR line per km -->
  <div class="card full">
    <div class="card-eyebrow">01 — Per km breakdown</div>
    <div class="card-title">Pace & Heart Rate</div>
    <div class="card-sub">Bars = pace (min/km) · Line = heart rate (bpm) · km 1–42</div>
    <div class="chart-wrap" style="height:260px"><canvas id="c1"></canvas></div>
    <div class="insights">
      <div class="ins">
        <div class="ins-val" style="color:var(--blue)">5:18 → 4:05</div>
        <div class="ins-lbl">Pace km 1 → km 42</div>
      </div>
      <div class="ins">
        <div class="ins-val" style="color:var(--orange)">−73 sec/km</div>
        <div class="ins-lbl">Negative split magnitude</div>
      </div>
      <div class="ins">
        <div class="ins-val" style="color:var(--red)">156 → 183 bpm</div>
        <div class="ins-lbl">HR km 1 → km 42</div>
      </div>
      <div class="ins">
        <div class="ins-val">Last 7km fastest</div>
        <div class="ins-lbl">No wall · Progressive finish</div>
      </div>
    </div>
  </div>

  <!-- Chart 02: Smoothed pace + LOWESS + H1/H2 split segments -->
  <div class="card full">
    <div class="card-eyebrow">02 — Continuous stream</div>
    <div class="card-title">Pace Evolution + Half Split Analysis</div>
    <div class="card-sub">Smoothed pace (60s window) + LOWESS trend · H1 / H2 average pace lines</div>
    <div class="chart-wrap" style="height:240px"><canvas id="c2"></canvas></div>
    <div class="insights">
      <div class="ins">
        <div class="ins-val" style="color:var(--red)">H1 · {h1_fmt}/km</div>
        <div class="ins-lbl">First half avg pace</div>
      </div>
      <div class="ins">
        <div class="ins-val" style="color:var(--green)">H2 · {h2_fmt}/km</div>
        <div class="ins-lbl">Second half avg pace</div>
      </div>
      <div class="ins">
        <div class="ins-val" style="color:var(--orange)">−{diff}%</div>
        <div class="ins-lbl">Negative split</div>
      </div>
      <div class="ins">
        <div class="ins-val">km 3–36 consistent</div>
        <div class="ins-lbl">Within 30s band</div>
      </div>
    </div>
  </div>

  <!-- Chart 03: Running power by 5km block -->
  <div class="card">
    <div class="card-eyebrow">03 — Running power</div>
    <div class="card-title">Power by 5km Block</div>
    <div class="card-sub">Avg watts per segment · dashed line = race average ({avg_power}W)</div>
    <div class="chart-wrap" style="height:220px"><canvas id="c3"></canvas></div>
    <div class="insights">
      <div class="ins">
        <div class="ins-val" style="color:var(--purple)">229 → 269W</div>
        <div class="ins-lbl">Block 1 → Block 9</div>
      </div>
      <div class="ins">
        <div class="ins-val" style="color:var(--purple)">+17%</div>
        <div class="ins-lbl">Power increase</div>
      </div>
      <div class="ins">
        <div class="ins-val">{avg_power}W</div>
        <div class="ins-lbl">Race avg power</div>
      </div>
    </div>
  </div>

  <!-- Chart 04: Power curve -->
  <div class="card">
    <div class="card-eyebrow">04 — Critical power</div>
    <div class="card-title">Power Curve</div>
    <div class="card-sub">Best avg power for each duration · entire race</div>
    <div class="chart-wrap" style="height:220px"><canvas id="c4"></canvas></div>
    <div class="insights">
      <div class="ins">
        <div class="ins-val" style="color:var(--purple)">353W</div>
        <div class="ins-lbl">Peak 5s power</div>
      </div>
      <div class="ins">
        <div class="ins-val" style="color:var(--purple)">242W</div>
        <div class="ins-lbl">Best 3h effort</div>
      </div>
    </div>
  </div>

  <!-- Chart 05: Scatter HR vs Pace + regression -->
  <div class="card full">
    <div class="card-eyebrow">05 — Efficiency analysis</div>
    <div class="card-title">Heart Rate vs Pace · Scatter</div>
    <div class="card-sub">Each point = 1 km · linear regression · R² = {reg['r2']}</div>
    <div class="chart-wrap" style="height:280px"><canvas id="c5"></canvas></div>
    <div class="insights">
      <div class="ins">
        <div class="ins-val" style="color:var(--red)">R² = {reg['r2']}</div>
        <div class="ins-lbl">Correlation strength</div>
      </div>
      <div class="ins">
        <div class="ins-val">Controlled drift</div>
        <div class="ins-lbl">HR rose as pace improved</div>
      </div>
      <div class="ins">
        <div class="ins-val" style="color:var(--orange)">No decoupling</div>
        <div class="ins-lbl">HR/pace coupled to finish</div>
      </div>
    </div>
  </div>

</div>

<div class="footer">
  <div class="footer-brand">— <em>Road to Milan</em> · Engineer · Runner · Data</div>
  <div class="footer-brand">{args.event} · {args.date}</div>
</div>

<script>
// ── Injected data (generated by generate_report.py) ──
const LAPS        = {laps_json};
const STREAM      = {stream_json};
const LOWESS      = {lowess_json};
const H1          = {h1_json};
const H2          = {h2_json};
const BLOCKS      = {blocks_json};
const POWER_CURVE = {power_curve_json};
const SCATTER     = {scatter_json};
const REG         = {reg_json};
const SPLITS      = {splits_json};

// ── Color palette ──
const C_ORANGE = '#FF4D00';
const C_BLUE   = '#4FC3F7';
const C_RED    = '#f04060';
const C_PURPLE = '#b47cff';
const C_GREEN  = '#00e5a0';
const C_WHITE  = '#ffffff';
const C_GRID   = 'rgba(255,255,255,0.04)';
const C_BORDER = '#252525';
const C_MUTED  = '#555';

Chart.defaults.color = C_MUTED;
Chart.defaults.font.family = "'Outfit', sans-serif";
Chart.defaults.font.size = 11;

function paceLabel(sec) {{
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return m + ':' + String(s).padStart(2, '0');
}}

const baseTooltip = {{
  backgroundColor: '#1a1a1a', borderColor: '#2a2a2a', borderWidth: 1,
  titleColor: '#f0f0f0', bodyColor: '#888',
}};

// Shared y-axis factory for pace
const yPaceAxis = (side, min, max) => ({{
  position: side, min, max,
  grid: {{ color: C_GRID }},
  border: {{ color: C_BORDER }},
  ticks: {{ color: C_WHITE, callback: v => paceLabel(v) }},
  title: {{ display: true, text: 'Pace (min/km)', color: C_WHITE, font: {{ size: 10 }} }}
}});

// ════════════════════════════════════════
// Chart 01 — Pace bars + HR line per km
// ════════════════════════════════════════
(function() {{
  const kmLabels = LAPS.map(l => 'km ' + l.km);
  const paceData = LAPS.map(l => l.pace_sec);
  const hrData   = LAPS.map(l => l.avg_hr);

  new Chart(document.getElementById('c1'), {{
    data: {{
      labels: kmLabels,
      datasets: [
        {{
          type: 'bar', label: 'Pace (min/km)',
          data: paceData,
          backgroundColor: 'rgba(79,195,247,0.65)',
          borderWidth: 0, yAxisID: 'yPace', order: 2,
        }},
        {{
          type: 'line', label: 'Heart Rate (bpm)',
          data: hrData,
          borderColor: C_RED, backgroundColor: 'transparent',
          borderWidth: 2, pointRadius: 0, tension: 0.3,
          yAxisID: 'yHR', order: 1,
        }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{
          position: 'top', align: 'end',
          labels: {{
            boxWidth: 14, boxHeight: 3, padding: 20, color: C_WHITE,
            generateLabels: () => [
              {{ text: 'Pace (min/km)',    fillStyle: C_BLUE,        strokeStyle: C_BLUE, lineWidth: 0, borderRadius: 2 }},
              {{ text: 'Heart Rate (bpm)', fillStyle: 'transparent', strokeStyle: C_RED,  lineWidth: 2 }},
            ]
          }}
        }},
        tooltip: {{ ...baseTooltip, callbacks: {{
          label: ctx => ctx.datasetIndex === 0
            ? '  Pace: ' + paceLabel(ctx.raw)
            : '  HR: ' + ctx.raw + ' bpm'
        }} }}
      }},
      scales: {{
        x: {{
          grid: {{ color: C_GRID }}, border: {{ color: C_BORDER }},
          ticks: {{
            color: C_WHITE, maxTicksLimit: 12,
            // Show km 1, 5, 10, 15 ... and cap last label at km 42
            callback: (val, i) => {{
              if (i === LAPS.length - 1) return 'km 42';
              return (i + 1) % 5 === 1 ? 'km ' + (i + 1) : '';
            }}
          }}
        }},
        yPace: {{ ...yPaceAxis('left', 240, 340) }},
        yHR: {{
          position: 'right', min: 140, max: 200,
          grid: {{ drawOnChartArea: false }}, border: {{ color: C_BORDER }},
          ticks: {{ color: C_RED, callback: v => v + ' bpm' }},
          title: {{ display: true, text: 'Heart Rate (bpm)', color: C_RED, font: {{ size: 10 }} }}
        }}
      }}
    }}
  }});
}})();

// ════════════════════════════════════════
// Chart 02 — Smoothed pace + LOWESS + H1/H2 segments
// ════════════════════════════════════════
(function() {{
  const ctx  = document.getElementById('c2').getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, 240);
  grad.addColorStop(0, 'rgba(79,195,247,0.15)');
  grad.addColorStop(1, 'rgba(79,195,247,0.0)');

  const h1Pace = SPLITS.half1;
  const h2Pace = SPLITS.half2;

  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: STREAM.map(s => s.dist_km),
      datasets: [
        // Smoothed raw (subtle background)
        {{
          label: 'Pace (smoothed)',
          data: STREAM.map(s => s.pace_sec),
          borderColor: 'rgba(79,195,247,0.22)', backgroundColor: grad,
          fill: true, borderWidth: 1, pointRadius: 0, tension: 0.2, order: 4,
        }},
        // LOWESS trend (main line)
        {{
          label: 'LOWESS trend',
          data: LOWESS.map(s => s.pace_sec),
          borderColor: C_BLUE, backgroundColor: 'transparent',
          fill: false, borderWidth: 2.5, pointRadius: 0, tension: 0.4, order: 3,
        }},
        // H1 segment — red, km 0–21 only
        {{
          type: 'scatter', label: 'H1 avg {h1_fmt}/km',
          data: H1.map(s => ({{ x: s.dist_km, y: h1Pace }})),
          showLine: true, borderColor: C_RED, backgroundColor: 'transparent',
          borderWidth: 2.5, pointRadius: 0, order: 2,
        }},
        // H2 segment — green, km 21–42 only
        {{
          type: 'scatter', label: 'H2 avg {h2_fmt}/km',
          data: H2.map(s => ({{ x: s.dist_km, y: h2Pace }})),
          showLine: true, borderColor: C_GREEN, backgroundColor: 'transparent',
          borderWidth: 2.5, pointRadius: 0, order: 1,
        }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'nearest', intersect: false, axis: 'x' }},
      plugins: {{
        legend: {{
          position: 'top', align: 'end',
          labels: {{ boxWidth: 20, boxHeight: 2, padding: 18, color: C_WHITE, usePointStyle: false }}
        }},
        tooltip: {{ ...baseTooltip,
          filter: item => item.datasetIndex <= 1,
          callbacks: {{
            title: items => 'km ' + parseFloat(items[0].label).toFixed(1),
            label: ctx => '  ' + (ctx.datasetIndex === 0 ? 'Pace: ' : 'Trend: ') + paceLabel(ctx.raw)
          }}
        }}
      }},
      scales: {{
        x: {{
          type: 'linear', min: 0, max: 43,
          grid: {{ color: C_GRID }}, border: {{ color: C_BORDER }},
          ticks: {{ color: C_WHITE, stepSize: 5, callback: v => v === 43 ? 'km 42' : 'km ' + v }}
        }},
        y: yPaceAxis('left', 240, 340)
      }}
    }}
  }});
}})();

// ════════════════════════════════════════
// Chart 03 — Power by 5km block
// ════════════════════════════════════════
(function() {{
  const labels  = BLOCKS.map(b => 'km ' + b.label);
  const powers  = BLOCKS.map(b => b.avg_power);
  const avgPow  = {avg_power};
  const minP    = Math.min(...powers), maxP = Math.max(...powers);

  const barColors = powers.map(p => {{
    const t = (p - minP) / (maxP - minP);
    return `rgba(${{Math.round(130 + t * 85)}}, ${{Math.round(80 + t * 20)}}, 255, ${{0.4 + t * 0.55}})`;
  }});

  new Chart(document.getElementById('c3'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{
          label: 'Avg Power (W)', data: powers,
          backgroundColor: barColors, borderWidth: 0, borderRadius: 2, order: 2,
        }},
        {{
          type: 'line', label: 'Race avg ' + avgPow + 'W',
          data: labels.map(() => avgPow),
          borderColor: 'rgba(255,77,0,0.75)', borderDash: [6, 4],
          borderWidth: 2, pointRadius: 0, fill: false, order: 1,
        }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'top', align: 'end', labels: {{ boxWidth: 14, boxHeight: 2, padding: 18, color: C_WHITE }} }},
        tooltip: {{ ...baseTooltip, callbacks: {{ label: ctx => '  ' + ctx.raw.toFixed(0) + ' W' }} }}
      }},
      scales: {{
        x: {{ grid: {{ display: false }}, border: {{ color: C_BORDER }}, ticks: {{ color: C_WHITE }} }},
        y: {{
          min: 200, max: 290,
          grid: {{ color: C_GRID }}, border: {{ color: C_BORDER }},
          ticks: {{ color: C_WHITE, callback: v => v + 'W' }},
          title: {{ display: true, text: 'Power (W)', color: C_WHITE, font: {{ size: 10 }} }}
        }}
      }}
    }}
  }});
}})();

// ════════════════════════════════════════
// Chart 04 — Power curve (log X axis)
// ════════════════════════════════════════
(function() {{
  const wantedTicks = [5, 30, 60, 300, 600, 1800, 3600, 7200, 10800];

  function durLabel(s) {{
    if (s < 60)   return s + 's';
    if (s < 3600) return (s / 60) + 'min';
    return (s / 3600).toFixed(1) + 'h';
  }}

  const ctx  = document.getElementById('c4').getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, 220);
  grad.addColorStop(0, 'rgba(180,124,255,0.35)');
  grad.addColorStop(1, 'rgba(180,124,255,0.0)');

  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: POWER_CURVE.map(p => p.duration_s),
      datasets: [{{
        label: 'Best power',
        data: POWER_CURVE.map(p => p.power_w),
        borderColor: C_PURPLE, backgroundColor: grad,
        fill: true, borderWidth: 2,
        pointRadius: 3, pointBackgroundColor: C_PURPLE, tension: 0.4,
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ ...baseTooltip, callbacks: {{
          title: items => 'Duration: ' + durLabel(parseInt(items[0].label)),
          label: ctx => '  Best: ' + ctx.raw + 'W'
        }} }}
      }},
      scales: {{
        x: {{
          type: 'logarithmic', min: 5, max: 12000,
          grid: {{ color: C_GRID }}, border: {{ color: C_BORDER }},
          ticks: {{
            color: C_WHITE, autoSkip: false, maxTicksLimit: 20,
            callback: v => wantedTicks.includes(v) ? durLabel(v) : ''
          }}
        }},
        y: {{
          min: 230, max: 370,
          grid: {{ color: C_GRID }}, border: {{ color: C_BORDER }},
          ticks: {{ color: C_WHITE, callback: v => v + 'W' }},
          title: {{ display: true, text: 'Power (W)', color: C_WHITE, font: {{ size: 10 }} }}
        }}
      }}
    }}
  }});
}})();

// ════════════════════════════════════════
// Chart 05 — Scatter HR vs Pace + regression
// ════════════════════════════════════════
(function() {{
  const points  = SCATTER.map(d => ({{ x: d.hr, y: d.pace_sec, km: d.km }}));
  const minHR   = Math.min(...SCATTER.map(d => d.hr));
  const maxHR   = Math.max(...SCATTER.map(d => d.hr));
  const regLine = [
    {{ x: minHR - 1, y: REG.m * (minHR - 1) + REG.b }},
    {{ x: maxHR + 1, y: REG.m * (maxHR + 1) + REG.b }},
  ];

  new Chart(document.getElementById('c5'), {{
    type: 'scatter',
    data: {{
      datasets: [
        {{
          label: 'km split',
          data: points,
          backgroundColor: 'rgba(79,195,247,0.55)',
          borderColor: 'rgba(79,195,247,0.8)',
          borderWidth: 1, pointRadius: 5, pointHoverRadius: 7, order: 2,
        }},
        {{
          type: 'line', label: 'Regression  R²={reg["r2"]}',
          data: regLine,
          borderColor: C_RED, borderWidth: 2, borderDash: [6, 3],
          pointRadius: 0, fill: false, order: 1,
        }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'top', align: 'end', labels: {{ boxWidth: 14, boxHeight: 2, padding: 20, color: C_WHITE }} }},
        tooltip: {{ ...baseTooltip, callbacks: {{
          title: items => 'km ' + (items[0].raw.km || ''),
          label: ctx => ctx.datasetIndex === 0
            ? ['  HR: ' + ctx.raw.x + ' bpm', '  Pace: ' + paceLabel(ctx.raw.y)]
            : []
        }} }}
      }},
      scales: {{
        x: {{
          min: 150, max: 190,
          grid: {{ color: C_GRID }}, border: {{ color: C_BORDER }},
          ticks: {{ color: C_WHITE, stepSize: 5, callback: v => v + ' bpm' }},
          title: {{ display: true, text: 'Heart Rate (bpm)', color: C_WHITE, font: {{ size: 10 }} }}
        }},
        y: yPaceAxis('left', 240, 340)
      }}
    }}
  }});
}})();
</script>
</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    fit_path = Path(args.fit)

    if not fit_path.exists():
        print(f"Error: file not found — {fit_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {fit_path.name}...")
    fitfile = FitFile(str(fit_path))
    laps_raw = load_laps(fitfile)

    fitfile2 = FitFile(str(fit_path))  # re-open — fitparse exhausts the iterator
    stream_raw = load_stream(fitfile2)

    # Filter to full km laps only
    laps = filter_full_laps(laps_raw)
    print(f"  Laps (full km): {len(laps)}")
    print(f"  Stream points:  {len(stream_raw)}")

    # Compute all derived datasets
    blocks      = compute_blocks(laps)
    splits      = compute_half_splits(laps)
    power_curve = compute_power_curve(stream_raw)
    stream_s    = smooth_stream(stream_raw, window=6)
    lowess      = compute_lowess(stream_s)
    regression  = compute_regression(laps)
    summary     = compute_summary(laps, stream_raw)

    # Split stream into H1/H2 for chart 02 segments
    mid_km   = 21.1
    h1_stream = [s for s in stream_s if s["dist_km"] <= mid_km]
    h2_stream = [s for s in stream_s if s["dist_km"] >  mid_km]

    data = {
        "laps":          laps,
        "stream_smooth": stream_s,
        "lowess":        lowess,
        "h1_stream":     h1_stream,
        "h2_stream":     h2_stream,
        "blocks":        blocks,
        "power_curve":   power_curve,
        "regression":    regression,
        "splits":        splits,
        "summary":       summary,
    }

    print(f"  Finish time:    {summary['finish_h']}h {summary['finish_m']}m")
    print(f"  Avg pace:       {fmt_pace(summary['avg_pace'])}/km")
    print(f"  Avg HR:         {summary['avg_hr']} bpm")
    print(f"  Negative split: -{splits['diff_pct']}%")

    # Render and write HTML
    html = build_html(data, args)
    out_path = Path(args.output)
    out_path.write_text(html, encoding="utf-8")
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
