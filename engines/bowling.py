import pandas as pd
import numpy as np

# ── LOAD DATA ─────────────────────────────────────────────
# IMPORTANT: we use bowling_team column here, not batting_team
# Everything in this file asks "what does this team do when THEY bowl?"
df = pd.read_csv('data/processed/deliveries_current.csv',
                 dtype={'season': str},
                 low_memory=False)

df['wicket_type']      = df['wicket_type'].fillna('')
df['player_dismissed'] = df['player_dismissed'].fillna('')

# Add phase labels — same logic as batting engine
df['phase'] = 'middle'
df.loc[df['over'] <= 5,  'phase'] = 'powerplay'
df.loc[df['over'] >= 15, 'phase'] = 'death'

print("Rows loaded:", len(df))
print("Teams:", sorted(df['bowling_team'].unique()))

# ── POWERPLAY WICKETS TAKEN ───────────────────────────────
# How many wickets does each team take in PP when BOWLING?
# More PP wickets = more pressure on opposition early = better
pp = df[df['phase'] == 'powerplay']

pp_wickets_taken = (
    pp[pp['wicket_type'] != '']               # only wicket deliveries
    .groupby(['match_id', 'bowling_team'])
    .size()
    .reset_index(name='pp_wkts_taken')
)

# Teams that took 0 PP wickets in a match won't appear above
# So we need all match-team combinations, then fill missing with 0
all_pp_matches = (
    pp.groupby(['match_id', 'bowling_team'])
    .size()
    .reset_index()[['match_id', 'bowling_team']]
)

pp_wickets_taken = (
    all_pp_matches
    .merge(pp_wickets_taken, on=['match_id', 'bowling_team'], how='left')
    .fillna(0)
    .groupby('bowling_team')['pp_wkts_taken']
    .mean()
    .reset_index()
    .rename(columns={'pp_wkts_taken': 'avg_pp_wkts_taken'})
)

print("\nPP wickets taken per innings (bowling):")
print(pp_wickets_taken.sort_values('avg_pp_wkts_taken', ascending=False).to_string())

# ── DEATH ECONOMY ─────────────────────────────────────────
# Runs conceded per over in death overs (overs 15-19)
# LOWER is better — great death bowlers concede 8-9, poor ones 12+
death = df[df['phase'] == 'death']

death_eco = (
    death.groupby(['match_id', 'bowling_team'])
    .agg(
        runs_given = ('total_runs', 'sum'),   # total runs incl extras
        balls      = ('total_runs', 'count')  # number of legal balls
    )
    .reset_index()
)
# Economy = runs per over = (runs / balls) × 6
death_eco['death_economy'] = (death_eco['runs_given'] / death_eco['balls']) * 6

death_eco_avg = (
    death_eco.groupby('bowling_team')['death_economy']
    .mean()
    .reset_index()
    .rename(columns={'death_economy': 'avg_death_eco'})
)

print("\nDeath economy (lower = better):")
print(death_eco_avg.sort_values('avg_death_eco').to_string())

# ── DOT BALL PERCENTAGE ───────────────────────────────────
# % of balls bowled where 0 runs were scored off the bat
# High dot ball % = pressure, frustration, wicket opportunities
# Dot ball = runs_off_bat == 0 AND no extras (true dot)
df['is_dot'] = (df['runs_off_bat'] == 0) & (df['extras'] == 0)

dot_pct = (
    df.groupby(['match_id', 'bowling_team'])
    .agg(
        dots  = ('is_dot', 'sum'),
        balls = ('is_dot', 'count')
    )
    .reset_index()
)
dot_pct['dot_pct'] = (dot_pct['dots'] / dot_pct['balls']) * 100

dot_avg = (
    dot_pct.groupby('bowling_team')['dot_pct']
    .mean()
    .reset_index()
    .rename(columns={'dot_pct': 'avg_dot_pct'})
)

print("\nDot ball percentage (higher = better):")
print(dot_avg.sort_values('avg_dot_pct', ascending=False).to_string())

# ── WICKET RATE (overall) ─────────────────────────────────
# Average wickets taken per match across all overs
# Reflects overall bowling potency of the attack
all_wickets = (
    df[df['wicket_type'] != '']
    .groupby(['match_id', 'bowling_team'])
    .size()
    .reset_index(name='wickets')
)

all_matches = (
    df.groupby(['match_id', 'bowling_team'])
    .size()
    .reset_index()[['match_id', 'bowling_team']]
)

wicket_rate = (
    all_matches
    .merge(all_wickets, on=['match_id', 'bowling_team'], how='left')
    .fillna(0)
    .groupby('bowling_team')['wickets']
    .mean()
    .reset_index()
    .rename(columns={'wickets': 'avg_wickets_per_match'})
)

print("\nAverage wickets per match:")
print(wicket_rate.sort_values('avg_wickets_per_match', ascending=False).to_string())

# ── OVERALL ECONOMY ───────────────────────────────────────
# Runs conceded per over across ALL overs
# Gives a general picture of how expensive the bowling attack is
overall_eco = (
    df.groupby(['match_id', 'bowling_team'])
    .agg(
        runs_given = ('total_runs', 'sum'),
        balls      = ('total_runs', 'count')
    )
    .reset_index()
)
overall_eco['economy'] = (overall_eco['runs_given'] / overall_eco['balls']) * 6

overall_eco_avg = (
    overall_eco.groupby('bowling_team')['economy']
    .mean()
    .reset_index()
    .rename(columns={'economy': 'avg_economy'})
)

print("\nOverall economy (lower = better):")
print(overall_eco_avg.sort_values('avg_economy').to_string())

# ── COMBINE ALL BOWLING METRICS ───────────────────────────
bowling = (
    pp_wickets_taken
    .merge(death_eco_avg,   on='bowling_team')
    .merge(dot_avg,         on='bowling_team')
    .merge(wicket_rate,     on='bowling_team')
    .merge(overall_eco_avg, on='bowling_team')
)

# ── NORMALIZE ─────────────────────────────────────────────
def normalize(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0 + 50
    return (series - mn) / (mx - mn) * 100

# Higher PP wickets = better → normalize normally
bowling['pp_wkt_score']   = normalize(bowling['avg_pp_wkts_taken'])

# Lower death economy = better → flip before normalizing
bowling['death_eco_score'] = normalize(
    bowling['avg_death_eco'].max() - bowling['avg_death_eco']
)

# Higher dot% = better → normalize normally
bowling['dot_score']       = normalize(bowling['avg_dot_pct'])

# More wickets per match = better → normalize normally
bowling['wicket_score']    = normalize(bowling['avg_wickets_per_match'])

# Lower overall economy = better → flip
bowling['economy_score']   = normalize(
    bowling['avg_economy'].max() - bowling['avg_economy']
)

# ── FINAL BOWLING SCORE ───────────────────────────────────
# Death economy weighted highest (35%) — most decisive in T20 playoffs
# PP wickets second (25%) — early breakthroughs set up matches
# Dot balls (20%) — pressure and control
# Wicket rate (15%) — overall threat
# Economy (5%) — general, less specific than the above
bowling['bowling_score'] = (
    bowling['pp_wkt_score']    * 0.25 +
    bowling['death_eco_score'] * 0.35 +
    bowling['dot_score']       * 0.20 +
    bowling['wicket_score']    * 0.15 +
    bowling['economy_score']   * 0.05
)
bowling['bowling_score'] = normalize(bowling['bowling_score'])

print("\n" + "=" * 50)
print("FINAL BOWLING SCORES")
print("=" * 50)
print(
    bowling[['bowling_team', 'bowling_score',
             'avg_death_eco', 'avg_pp_wkts_taken', 'avg_dot_pct']]
    .sort_values('bowling_score', ascending=False)
    .to_string(index=False)
)

# Rename for consistency with rest of project
bowling = bowling.rename(columns={'bowling_team': 'team'})

bowling.to_csv('data/processed/bowling_scores.csv', index=False)
print("\n✅ bowling_scores.csv saved")