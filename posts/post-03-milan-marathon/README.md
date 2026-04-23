# post-05-milan-marathon

Race data analysis for the Milano Marathon 2026 — interactive HTML report generated from a `.FIT` file.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python generate_report.py --fit data/race.fit --output output/report.html
```

### Optional arguments

| Argument  | Default             | Description                     |
|-----------|---------------------|---------------------------------|
| `--fit`   | *(required)*        | Path to the `.FIT` file         |
| `--output`| `report.html`       | Output HTML path                |
| `--event` | `Race Analysis`     | Event name shown in the header  |
| `--date`  | `06.04.2026`        | Event date (display only)       |

## Charts generated

1. **Pace & Heart Rate** — per km bars + HR line
2. **Pace Evolution** — smoothed stream + LOWESS trend + H1/H2 split lines
3. **Power by 5km Block** — avg watts per segment
4. **Power Curve** — best power for each duration (log scale)
5. **HR vs Pace Scatter** — efficiency analysis with linear regression

## Output

A self-contained HTML file with all data embedded. Open in any browser or deploy to Netlify as a static site.
