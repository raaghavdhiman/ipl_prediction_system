import pandas as pd
import numpy as np

# ── LOAD DATA ─────────────────────────────────────────────
# dtype={'season':str} forces season column to be text, not number
# low_memory=False reads whole file at once so types are consistent
df = pd.read_csv('data/processed/deliveries_current.csv',
                 dtype={'season': str},
                 low_memory=False)

# Fill NaN in wicket columns with empty string
# WHY: when CSV is saved and reloaded, empty strings become NaN
# This causes bugs in comparisons like wicket_type != ''
df['wicket_type']      = df['wicket_type'].fillna('')
df['player_dismissed'] = df['player_dismissed'].fillna('')

print("Rows loaded:", len(df))
print("Teams:", sorted(df['batting_team'].unique()))

# ── ADD PHASE COLUMN ──────────────────────────────────────
# Every delivery gets labelled: powerplay, middle, or death
# over column is 0-indexed: over 0 = first over, over 19 = last
df['phase'] = 'middle'                        # default everyone to middle
df.loc[df['over'] <= 5,  'phase'] = 'powerplay'  # overs 0-5
df.loc[df['over'] >= 15, 'phase'] = 'death'      # overs 15-19

print("\nPhase distribution:")
print(df['phase'].value_counts())

# ── POWERPLAY BATTING ─────────────────────────────────────
pp = df[df['phase'] == 'powerplay']

# Average PP runs per innings per team
# Step 1: sum runs in PP for each (match, team) pair
# Step 2: average those sums across all matches
pp_per_match = (
    pp.groupby(['match_id', 'batting_team'])['runs_off_bat']
    .sum()
    .reset_index()
    .rename(columns={'runs_off_bat': 'pp_runs'})
)
pp_avg = (
    pp_per_match
    .groupby('batting_team')['pp_runs']
    .mean()
    .reset_index()
    .rename(columns={'pp_runs': 'avg_pp_runs'})
)

# Average PP wickets lost per innings per team
# wicket_type != '' means a wicket fell on that delivery
# We count how many such deliveries per (match, team) in PP
pp_wickets_per_match = (
    pp.groupby(['match_id', 'batting_team'])
    .apply(
        lambda x: (x['wicket_type'] != '').sum(),
        include_groups=False
    )
    .reset_index(name='pp_wickets')
)
pp_wickets_avg = (
    pp_wickets_per_match
    .groupby('batting_team')['pp_wickets']
    .mean()
    .reset_index()
    .rename(columns={'pp_wickets': 'avg_pp_wickets'})
)

# Merge PP runs and PP wickets into one table
pp_avg = pp_avg.merge(pp_wickets_avg, on='batting_team')

print("\nPP runs and wickets (avg per innings):")
print(pp_avg.sort_values('avg_pp_runs', ascending=False).to_string())

# ── MIDDLE OVERS RUN RATE ─────────────────────────────────
mo = df[df['phase'] == 'middle']

# agg computes multiple things at once:
# runs = sum of runs_off_bat, balls = count of deliveries
mo_rr = (
    mo.groupby(['match_id', 'batting_team'])
    .agg(runs=('runs_off_bat', 'sum'), balls=('runs_off_bat', 'count'))
    .reset_index()
)
# Run rate = runs per over = (runs/balls) × 6
mo_rr['mo_run_rate'] = (mo_rr['runs'] / mo_rr['balls']) * 6

mo_avg = (
    mo_rr.groupby('batting_team')['mo_run_rate']
    .mean()
    .reset_index()
    .rename(columns={'mo_run_rate': 'avg_mo_rr'})
)

print("\nMiddle overs run rate:")
print(mo_avg.sort_values('avg_mo_rr', ascending=False).to_string())

# ── DEATH OVERS STRIKE RATE ───────────────────────────────
death = df[df['phase'] == 'death']

death_sr = (
    death.groupby(['match_id', 'batting_team'])
    .agg(runs=('runs_off_bat', 'sum'), balls=('runs_off_bat', 'count'))
    .reset_index()
)
# Strike rate = runs per 100 balls = (runs/balls) × 100
death_sr['death_sr'] = (death_sr['runs'] / death_sr['balls']) * 100

death_avg = (
    death_sr.groupby('batting_team')['death_sr']
    .mean()
    .reset_index()
    .rename(columns={'death_sr': 'avg_death_sr'})
)

print("\nDeath overs strike rate:")
print(death_avg.sort_values('avg_death_sr', ascending=False).to_string())

# ── CR4: BATTING DEPENDENCY ───────────────────────────────
# For each team: top 4 batters' runs / total team runs
# High ratio = team depends heavily on those 4 players

batter_runs = (
    df.groupby(['batting_team', 'batter'])['runs_off_bat']
    .sum()
    .reset_index()
    .rename(columns={'runs_off_bat': 'total_runs'})
)

cr4_rows = []
for team in df['batting_team'].unique():
    team_data = batter_runs[batter_runs['batting_team'] == team]
    total     = team_data['total_runs'].sum()
    top4_runs = team_data.nlargest(4, 'total_runs')['total_runs'].sum()
    cr4       = round(top4_runs / total, 3) if total > 0 else 0

    if   cr4 >= 0.72: risk = 'HIGH'
    elif cr4 >= 0.55: risk = 'MEDIUM'
    else:             risk = 'LOW'

    cr4_rows.append({'batting_team': team, 'cr4': cr4, 'cr4_risk': risk})

cr4_df = pd.DataFrame(cr4_rows)

print("\nCR4 Dependency:")
print(cr4_df.sort_values('cr4', ascending=False).to_string())

# ── COMBINE ALL METRICS ───────────────────────────────────
batting = (
    pp_avg
    .merge(mo_avg,    on='batting_team')
    .merge(death_avg, on='batting_team')
    .merge(cr4_df,    on='batting_team')
)

# ── NORMALIZE to 0-100 ────────────────────────────────────
# Converts any range of values to 0-100 relative scale
# Best team = 100, worst team = 0, others in between
def normalize(series):
    mn, mx = series.min(), series.max()
    if mx == mn:           # all teams identical → give everyone 50
        return series * 0 + 50
    return (series - mn) / (mx - mn) * 100

batting['pp_score']    = normalize(batting['avg_pp_runs'])

# FLIP wickets: fewer wickets lost = better preservation
# So we subtract from max: worst team gets 0, best gets 100
batting['pp_preserve'] = normalize(batting['avg_pp_wickets'].max()
                                   - batting['avg_pp_wickets'])

batting['mo_score']    = normalize(batting['avg_mo_rr'])
batting['death_score'] = normalize(batting['avg_death_sr'])

# CR4 depth score: LOWER dependency = BETTER batting depth
# So (1 - cr4) means: low dependency teams score high here
batting['depth_score'] = normalize(1 - batting['cr4'])

# ── FINAL BATTING SCORE ───────────────────────────────────
# Weights explain how much each phase contributes:
# Death (0.30) > PP runs (0.25) > MO (0.20) > depth (0.15) > PP preserve (0.10)
batting['batting_score'] = (
    batting['pp_score']    * 0.25 +
    batting['pp_preserve'] * 0.10 +
    batting['mo_score']    * 0.20 +
    batting['death_score'] * 0.30 +
    batting['depth_score'] * 0.15   # replaces the duplicate pp_score
)
batting['batting_score'] = normalize(batting['batting_score'])

print("\n" + "=" * 50)
print("FINAL BATTING SCORES")
print("=" * 50)
print(
    batting[['batting_team', 'batting_score', 'cr4', 'cr4_risk']]
    .sort_values('batting_score', ascending=False)
    .to_string(index=False)
)

# ── SAVE ──────────────────────────────────────────────────
# Note: batting_scores.csv (no typo this time)
batting.to_csv('data/processed/batting_scores.csv', index=False)
print("\n✅ batting_scores.csv saved")