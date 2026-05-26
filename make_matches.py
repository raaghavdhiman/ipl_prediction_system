import pandas as pd

# Fix: read CURRENT not ALL, and force season as string
df = pd.read_csv('data/processed/deliveries_current.csv', 
                 dtype={'season': str},
                 low_memory=False)

matches = (
    df.groupby('match_id')
    .agg(
        date   = ('date',         'first'),
        venue  = ('venue',        'first'),
        season = ('season',       'first'),
        team1  = ('batting_team', 'first'),
        team2  = ('bowling_team', 'first'),
        winner = ('winner',       'first'),
    )
    .reset_index()
)

print(f"Total matches: {len(matches)}")
print(f"\nPer season:")
print(matches['season'].value_counts())

matches.to_csv('data/processed/matches_current.csv', index=False)
print("\n✅ matches_current.csv saved")