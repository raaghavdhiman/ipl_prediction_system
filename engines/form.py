import pandas as pd
import numpy as np

# ── LOAD MATCHES TABLE ────────────────────────────────────
# One row per match — we need results, dates, teams
matches = pd.read_csv('data/processed/matches_current.csv',
                      dtype={'season': str},
                      low_memory=False)

# Convert date to datetime so we can sort chronologically
# Without this, sorting by date treats it as text: "2026-04-01" > "2026-03-15" works
# but "2026-04-01" > "2026-10-01" would break — datetime is safer
matches['date'] = pd.to_datetime(matches['date'])

print("Matches loaded:", len(matches))
print("Date range:", matches['date'].min(), "to", matches['date'].max())
print("Sample:\n", matches.head(3).to_string())

# ── ALL TEAMS ─────────────────────────────────────────────
# Get unique teams from both team1 and team2 columns
all_teams = pd.unique(matches[['team1', 'team2']].values.ravel())
all_teams = sorted([t for t in all_teams if pd.notna(t)])
print("\nTeams found:", all_teams)

# ── TIME-DECAY WEIGHTS ────────────────────────────────────
# Most recent match counts most, oldest counts least
# [oldest → newest]: 0.10, 0.15, 0.20, 0.25, 0.30
# These sum to 1.0 — they're proportional importance weights
WEIGHTS = [0.10, 0.15, 0.20, 0.25, 0.30]

# ── RESULT VALUES ─────────────────────────────────────────
# Win = full credit, Loss = minimal credit
# No Result = neutral (treated as half credit)
WIN_VAL = 1.0
LOSS_VAL = 0.3
NR_VAL = 0.5

# ── NRR LOOKUP ────────────────────────────────────────────
# Manually enter current NRR from iplt20.com points table
# Check right now: iplt20.com → Points Table → note NRR column
# Update these values with today's actual NRR
nrr_lookup = {
    'Chennai Super Kings':        -0.345,   # ← replace with actual
    'Delhi Capitals':             -0.651,
    'Gujarat Titans':             0.695,
    'Kolkata Knight Riders':      -0.147,
    'Lucknow Super Giants':       -0.74,
    'Mumbai Indians':             -0.584,
    'Punjab Kings':               0.309,
    'Rajasthan Royals':           0.189,
    'Royal Challengers Bengaluru':0.783,
    'Sunrisers Hyderabad':        0.524,
}
# IMPORTANT: go to iplt20.com RIGHT NOW and fill these in
# NRR looks like +0.423 or -0.156

# ── COMPUTE FORM SCORE PER TEAM ───────────────────────────
form_rows = []

for team in all_teams:
    # Get all matches this team played, sorted oldest → newest
    # A team appears as either team1 or team2
    team_matches = matches[
        (matches['team1'] == team) | (matches['team2'] == team)
    ].sort_values('date', ascending=True)

    # Take last 5 matches only
    last5 = team_matches.tail(5)

    if len(last5) == 0:
        print(f"WARNING: No matches found for {team}")
        continue

    # Assign result value for each match
    result_values = []
    result_labels = []   # for display

    for _, row in last5.iterrows():
        w = row['winner']

        if pd.isna(w) or w == 'No Result':
            result_values.append(NR_VAL)
            result_labels.append('NR')
        elif w == team:
            result_values.append(WIN_VAL)
            result_labels.append('W')
        else:
            result_values.append(LOSS_VAL)
            result_labels.append('L')

    # Pad weights if team has fewer than 5 matches
    # (unlikely but handles edge cases)
    n = len(result_values)
    if n < 5:
        # Use last n weights, renormalized
        weights_used = WEIGHTS[-n:]
        total_w = sum(weights_used)
        weights_used = [w / total_w for w in weights_used]
    else:
        weights_used = WEIGHTS

    # Weighted form score
    # Example: WWLWW with weights [.10,.15,.20,.25,.30]
    # = 1.0×.10 + 1.0×.15 + 0.3×.20 + 1.0×.25 + 1.0×.30 = 0.86
    form_raw = sum(v * w for v, w in zip(result_values, weights_used))

    # NRR adjustment
    # Good NRR = team is winning by big margins = real strength
    # Bad NRR = winning close or losing big
    nrr = nrr_lookup.get(team, 0.0)
    if   nrr >  0.5: nrr_bonus =  0.07
    elif nrr >  0.2: nrr_bonus =  0.03
    elif nrr < -0.5: nrr_bonus = -0.07
    elif nrr < -0.2: nrr_bonus = -0.03
    else:            nrr_bonus =  0.0

    form_adjusted = min(1.0, max(0.0, form_raw + nrr_bonus))

    last5_str = ''.join(result_labels)

    form_rows.append({
        'team':           team,
        'form_raw':       round(form_raw, 4),
        'nrr':            nrr,
        'nrr_bonus':      nrr_bonus,
        'form_adjusted':  round(form_adjusted, 4),
        'last5':          last5_str,
        'matches_used':   n,
    })

form_df = pd.DataFrame(form_rows)

# ── NORMALIZE TO 0-100 ────────────────────────────────────
def normalize(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0 + 50
    return (series - mn) / (mx - mn) * 100

form_df['form_score'] = normalize(form_df['form_adjusted'])

# ── PRINT RESULTS ─────────────────────────────────────────
print("\n" + "=" * 55)
print("FORM SCORES (most recent 5 matches, time-weighted)")
print("=" * 55)
print(
    form_df[['team', 'last5', 'form_raw', 'nrr', 'form_score']]
    .sort_values('form_score', ascending=False)
    .to_string(index=False)
)
print("\nKey: W=Win  L=Loss  NR=No Result  (left=oldest, right=most recent)")

# ── SAVE ──────────────────────────────────────────────────
form_df.to_csv('data/processed/form_scores.csv', index=False)
print("\n✅ form_scores.csv saved")