import pandas as pd
import numpy as np
df = pd.read_csv('data/processed/deliveries_current.csv',
                 dtype={'season': str}, low_memory=False)

# Get every bowler who bowled in 2025/26 with their stats
bowler_stats = (
    df.groupby('bowler')
    .agg(
        balls        = ('runs_off_bat', 'count'),
        runs_conceded= ('total_runs',   'sum'),
        wickets      = ('wicket_type',  lambda x: (x != '').sum()),
        teams_faced  = ('batting_team', 'nunique'),
    )
    .reset_index()
)

# Only bowlers with at least 60 balls (10 overs) — filters out batters
# who bowled 1 over
bowler_stats = bowler_stats[bowler_stats['balls'] >= 60].copy()
bowler_stats['economy'] = (bowler_stats['runs_conceded'] /
                            bowler_stats['balls'] * 6).round(2)
bowler_stats['sr']      = (bowler_stats['balls'] /
                            bowler_stats['wickets'].replace(0, np.nan)).round(1)


bowler_stats['sr'] = bowler_stats['sr'].fillna(999)

# Sort by economy — spinners generally have lower economy in IPL
print("All bowlers with 60+ balls in 2025/26 sorted by economy:")
print(f"Total bowlers: {len(bowler_stats)}\n")
print(bowler_stats.sort_values('economy')
      [['bowler','balls','economy','wickets']]
      .to_string(index=False))

bowler_stats.to_csv('data/processed/bowler_stats_2526.csv', index=False)
print("\n✅ Saved to bowler_stats_2526.csv")