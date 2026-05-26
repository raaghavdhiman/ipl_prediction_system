import pandas as pd
import numpy as np

# ── LOAD ALL-SEASONS DATA ─────────────────────────────────
# Venue characteristics are about the GROUND not the team
# So we use all historical seasons, not just 2025/26
df = pd.read_csv('data/processed/deliveries_all.csv',
                 dtype={'season': str},
                 low_memory=False)

df['wicket_type']      = df['wicket_type'].fillna('')
df['player_dismissed'] = df['player_dismissed'].fillna('')

print("Total deliveries (all seasons):", len(df))
print("Unique venues:", df['venue'].nunique())

# ── STEP 1: AVERAGE FIRST INNINGS SCORE PER VENUE ─────────
# First innings score tells you how batting-friendly a venue is
# High avg = batting paradise (Chinnaswamy, Wankhede)
# Low avg = bowler's ground (Chepauk, Abu Dhabi)

first_innings = df[df['innings'] == 1]

first_inn_totals = (
    first_innings
    .groupby(['match_id', 'venue'])['total_runs']
    .sum()
    .reset_index()
    .rename(columns={'total_runs': 'first_inn_score'})
)

venue_avg_score = (
    first_inn_totals
    .groupby('venue')
    .agg(
        avg_first_inn  = ('first_inn_score', 'mean'),
        matches_played = ('first_inn_score', 'count')
    )
    .reset_index()
)

# Only keep venues with at least 5 matches (enough data)
venue_avg_score = venue_avg_score[venue_avg_score['matches_played'] >= 5]

print("\nTop 15 venues by avg first innings score:")
print(
    venue_avg_score
    .sort_values('avg_first_inn', ascending=False)
    .head(15)
    .to_string(index=False)
)

# ── STEP 2: CHASING WIN RATE PER VENUE ────────────────────
# Does the team batting second win more here?
# High chase rate = good for chasing teams
# Low chase rate = first innings score is defended more often

# Get match-level: who won, which innings did they bat?
match_info = (
    df.groupby(['match_id', 'venue'])
    .agg(winner=('winner', 'first'))
    .reset_index()
)

# Get which team batted second (innings == 2)
second_innings_teams = (
    df[df['innings'] == 2]
    .groupby('match_id')['batting_team']
    .first()
    .reset_index()
    .rename(columns={'batting_team': 'team_batting_second'})
)

match_info = match_info.merge(second_innings_teams, on='match_id', how='left')

# Chase win = winner is the team that batted second
match_info['chase_win'] = (
    match_info['winner'] == match_info['team_batting_second']
).astype(int)

# Remove no-result matches
match_info = match_info[match_info['winner'].notna()]
match_info = match_info[match_info['winner'] != 'No Result']

chase_rate = (
    match_info.groupby('venue')
    .agg(
        chase_wins    = ('chase_win', 'sum'),
        total_matches = ('chase_win', 'count')
    )
    .reset_index()
)
chase_rate['chase_win_rate'] = (
    chase_rate['chase_wins'] / chase_rate['total_matches']
).round(3)

# ── STEP 3: PACE VS SPIN WICKET SPLIT ─────────────────────
# What % of wickets at this venue are taken by spinners?
# High spin% = spin-friendly surface (Chepauk, Pune)
# High pace% = pace-friendly (Wankhede, Mohali)

# Build a spin bowler identifier
# Strategy: if a bowler's name appears in known spinners list OR
# their bowling style can be inferred — we use a name-based list
# This is manual cricket knowledge — no shortcut here

SPIN_BOWLERS = {
    # IPL 2025 + 2026 Spinners (verified list)
    'Ravichandran Ashwin', 'R Ashwin',
    'Ravindra Jadeja', 'RA Jadeja',
    'Noor Ahmad',
    'Rahul Chahar', 'R Chahar',
    'Shreyas Gopal',
    'Maheesh Theekshana', 'M Theekshana',
    'Yuzvendra Chahal', 'YS Chahal',
    'Wanindu Hasaranga', 'PWH de Silva',
    'Ravi Bishnoi',
    'Digvesh Rathi', 'DS Rathi', 'Digvesh Singh Rathi',
    'Krunal Pandya', 'KH Pandya',
    'Shahbaz Ahmed',
    'Sunil Narine', 'SP Narine',
    'Varun Chakravarthy', 'CV Varun',
    'Anukul Roy', 'AS Roy',
    'Mayank Markande',
    'Suyash Sharma',
    'Kuldeep Yadav',
    'Axar Patel', 'AR Patel',
    'Vipraj Nigam', 'V Nigam',
    'Rashid Khan',
    'R Sai Kishore',
    'Washington Sundar',
    'Manav Suthar',
    'Jayant Yadav',
    'Karn Sharma', 'KV Sharma',
    'Mitchell Santner', 'MJ Santner',
    'Mujeeb Ur Rahman', 'Mujeeb',
    'Harpreet Brar',
    'Glenn Maxwell', 'GJ Maxwell',
    'Abhishek Sharma',
    'Adam Zampa',
    'Kumar Kartikeya',
    'Murugan Ashwin',
    'Amit Mishra',
    'Piyush Chawla',
    'Moeen Ali', 'MM Ali',
    # 2026 additions
    'Akeal Hosein', 'AJ Hosein',
    'Prashant Veer',
    'Matthew Short',
    'Will Jacks', 'WG Jacks',
    'Naman Dhir',
    'Liam Livingstone',
    'Swapnil Singh',
    'Mayank Dagar',
    'Prashant Solanki',
    'Rahul Tewatia',
    'Ajay Mandal',
    'Riyan Parag', 'R Parag',
    'Praveen Dubey',
    'M Siddharth',
}

# Label each delivery as spin or pace based on bowler name
df['is_spin'] = df['bowler'].isin(SPIN_BOWLERS)

# Only look at wicket deliveries
wicket_deliveries = df[df['wicket_type'] != '']

spin_split = (
    wicket_deliveries
    .groupby('venue')
    .agg(
        spin_wickets  = ('is_spin', 'sum'),
        total_wickets = ('is_spin', 'count')
    )
    .reset_index()
)
spin_split['spin_pct']  = (
    spin_split['spin_wickets'] / spin_split['total_wickets'] * 100
).round(1)
spin_split['pace_pct'] = (100 - spin_split['spin_pct']).round(1)

# ── STEP 4: VENUE VOLATILITY ──────────────────────────────
# Standard deviation of first innings scores
# High std dev = unpredictable venue (could be 140 or 220)
# Low std dev = consistent scoring conditions

venue_volatility = (
    first_inn_totals
    .groupby('venue')['first_inn_score']
    .std()
    .reset_index()
    .rename(columns={'first_inn_score': 'score_std_dev'})
)

# ── STEP 5: COMBINE ALL VENUE METRICS ─────────────────────
venue_profiles = (
    venue_avg_score
    .merge(chase_rate[['venue', 'chase_win_rate']], on='venue', how='left')
    .merge(spin_split[['venue', 'spin_pct', 'pace_pct']], on='venue', how='left')
    .merge(venue_volatility, on='venue', how='left')
)

venue_profiles['spin_pct']     = venue_profiles['spin_pct'].fillna(50)
venue_profiles['pace_pct']     = venue_profiles['pace_pct'].fillna(50)
venue_profiles['chase_win_rate'] = venue_profiles['chase_win_rate'].fillna(0.5)
venue_profiles['score_std_dev']  = venue_profiles['score_std_dev'].fillna(20)

# Volatility score 0-100 (higher = more volatile/chaotic)
def normalize(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0 + 50
    return (series - mn) / (mx - mn) * 100

venue_profiles['volatility_score'] = normalize(
    venue_profiles['score_std_dev']
)

# ── PRINT PLAYOFF-RELEVANT VENUES ─────────────────────────
# Filter to venues likely used in IPL 2025/26 playoffs
playoff_keywords = [
    'Eden', 'Wankhede', 'Chepauk', 'Chidambaram',
    'Narendra Modi', 'Chinnaswamy', 'Hyderabad', 'Uppal',
    'Mohali', 'Kotla', 'Arun Jaitley', 'Brabourne',
    'DY Patil', 'Sawai', 'Jaipur', 'Lucknow', 'Ekana',
    'Pune', 'Dharamsala', 'HPCA'
]

mask = venue_profiles['venue'].apply(
    lambda v: any(kw.lower() in v.lower() for kw in playoff_keywords)
)
playoff_venues = venue_profiles[mask].copy()

print("\n" + "=" * 70)
print("VENUE PROFILES (playoff-relevant venues)")
print("=" * 70)
print(
    playoff_venues[['venue', 'avg_first_inn', 'chase_win_rate',
                    'spin_pct', 'pace_pct', 'volatility_score',
                    'matches_played']]
    .sort_values('avg_first_inn', ascending=False)
    .to_string(index=False)
)

# ── SAVE FULL PROFILE ──────────────────────────────────────
venue_profiles.to_csv('data/processed/venue_profiles.csv', index=False)
print("\n✅ venue_profiles.csv saved")
print(f"   Total venues profiled: {len(venue_profiles)}")