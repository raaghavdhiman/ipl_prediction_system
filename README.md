# 🏏 IPL Playoff Predictor 2026

A cricket intelligence system that predicts IPL playoff match outcomes 
using ball-by-ball data from the 2025/26 season.

**Not a machine learning classifier.** A domain-driven analytics engine 
— batting strength, bowling control, form momentum, and venue 
compatibility, combined into a weighted prediction with full reasoning.

Note: data/raw/ is excluded from this repo (too large). 
Download IPL JSON files from cricsheet.org and place them 
in data/raw/ before running parse_json.py

## Live Demo
[🔮 Try it here](https://iplpredictionsystem-dyrm7iy2gnah9clkn2bffc.streamlit.app/)

## What it predicts
- Predicted winner with win probability (realistic 52–72% range)
- Stability score — how predictable the match is
- Chaos potential — likelihood of sudden momentum swings
- Human-readable reasons why a team wins or could collapse

## How it works
1. **Data pipeline** — Parses 293,000+ ball-by-ball deliveries from 
   Cricsheet IPL JSON files (2025/26 season)
2. **Batting engine** — Powerplay runs, middle-overs run rate, 
   death SR, CR4 top-order dependency
3. **Bowling engine** — Death economy, PP wickets, dot ball %, 
   overall wicket rate
4. **Form engine** — Last 5 matches with time-decay weights + NRR
5. **Venue engine** — Avg first innings, chase rate, spin/pace split, 
   volatility from all historical IPL data
6. **Prediction engine** — Weighted strength score + sigmoid probability

## Project structure

ipl_predictor/
├── parse_json.py          # Converts Cricsheet JSON → CSVs
├── make_matches.py        # Builds match-level table
├── engines/
│   ├── batting.py         # Batting strength engine
│   ├── bowling.py         # Bowling strength engine
│   ├── form.py            # Recent form engine
│   └── venue.py           # Venue profile engine
├── prediction/
│   └── predictor.py       # Core prediction logic
├── app/
│   └── app.py             # Streamlit UI
└── data/processed/        # Pre-computed score CSVs


## How to Run Locally

### Prerequisites
- Python 3.10 or higher
- IPL JSON files from [cricsheet.org](https://cricsheet.org/downloads/) 
  (download the IPL pack in JSON format)

### Setup

**1. Clone the repository**
```bash
git clone https://github.com/raaghavdhiman/ipl_prediction_system.git
cd ipl_prediction_system
```

**2. Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add raw data**
Place all downloaded Cricsheet JSON files into:
data/raw/

### Build the data pipeline

Run these five scripts in order. Each one depends on the previous.

**Step 1 — Parse raw JSON into deliveries table**
```bash
python parse_json.py
```
Output: `data/processed/deliveries_all.csv` and `data/processed/deliveries_current.csv`

**Step 2 — Build match-level table**
```bash
python make_matches.py
```
Output: `data/processed/matches_current.csv`

**Step 3 — Compute batting scores**
```bash
python engines/batting.py
```
Output: `data/processed/batting_scores.csv`

**Step 4 — Compute bowling scores**
```bash
python engines/bowling.py
```
Output: `data/processed/bowling_scores.csv`

**Step 5 — Compute form scores**
```bash
python engines/form.py
```
⚠️ Before running: open `engines/form.py` and update the 
`nrr_lookup` dictionary with current NRR values from 
[iplt20.com](https://www.iplt20.com/points-table/men)

Output: `data/processed/form_scores.csv`

**Step 6 — Build venue profiles**
```bash
python engines/venue.py
```
Output: `data/processed/venue_profiles.csv`

### Launch the app

```bash
streamlit run app/app.py
```

Opens at `http://localhost:8501`

Select Team A, Team B, Venue and click **Predict Match**.

### Updating for new matches

When new IPL matches are played and Cricsheet updates their data:

```bash
# 1. Re-download IPL JSON from cricsheet.org and replace data/raw/
# 2. Re-run the pipeline in order:

python parse_json.py
python make_matches.py
python engines/batting.py
python engines/bowling.py
python engines/form.py
python engines/venue.py
streamlit run app/app.py
```

Form scores are the most time-sensitive — 
recompute those first if short on time.

## Tech stack
Python · pandas · numpy · Streamlit

## Data source
[Cricsheet.org](https://cricsheet.org) — ball-by-ball IPL data

## Built in 5 days
May 21–25, 2026. First real data project.
