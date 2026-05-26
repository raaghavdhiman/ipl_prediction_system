import json
import glob
import pandas as pd
import os

# ── STEP 1: Find all JSON files ───────────────────────────
json_files = glob.glob('data/raw/*.json')
print(f"Found {len(json_files)} JSON files")

# ── STEP 2: Empty basket to collect all rows ──────────────
all_rows = []   # THIS must be OUTSIDE the loop

# ── STEP 3: Loop through every match file ─────────────────
for filepath in json_files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        continue

    # Match-level info
    match_id   = os.path.basename(filepath).replace('.json', '')
    info       = data['info']
    venue      = info.get('venue', 'Unknown')
    date       = info.get('dates', ['Unknown'])[0]
    season     = str(info.get('season', 'Unknown'))   # force string
    teams      = info.get('teams', [])
    outcome    = info.get('outcome', {})
    winner     = outcome.get('winner', 'No Result')

    # ── STEP 4: Loop innings ───────────────────────────────
    for inn_num, innings in enumerate(data.get('innings', []), 1):
        batting_team = innings.get('team', 'Unknown')

        # Bowling team = the other team
        other = [t for t in teams if t != batting_team]
        bowling_team = other[0] if other else 'Unknown'

        # ── STEP 5: Loop overs ────────────────────────────
        for over_data in innings.get('overs', []):
            over_num = over_data.get('over', 0)

            # ── STEP 6: Loop deliveries ───────────────────
            for delivery in over_data.get('deliveries', []):
                runs = delivery.get('runs', {})

                # Wicket info (only exists if wicket fell)
                wickets          = delivery.get('wickets', [])
                wicket_type      = wickets[0].get('kind', '')        if wickets else ''
                player_dismissed = wickets[0].get('player_out', '')  if wickets else ''

                # ── STEP 7: One dict = one row ────────────
                row = {
                    'match_id'         : match_id,
                    'season'           : season,
                    'date'             : date,
                    'venue'            : venue,
                    'innings'          : inn_num,
                    'over'             : over_num,
                    'batting_team'     : batting_team,
                    'bowling_team'     : bowling_team,
                    'batter'           : delivery.get('batter', ''),
                    'bowler'           : delivery.get('bowler', ''),
                    'runs_off_bat'     : runs.get('batter', 0),
                    'extras'           : runs.get('extras', 0),
                    'total_runs'       : runs.get('total', 0),
                    'wicket_type'      : wicket_type,
                    'player_dismissed' : player_dismissed,
                    'winner'           : winner,
                }
                all_rows.append(row)   # THIS must be inside all 3 loops

    # Progress every 100 files so you know it's working
    if json_files.index(filepath) % 100 == 0:
        print(f"  Processed {json_files.index(filepath)}/{len(json_files)} files...")

# ── STEP 8: Convert and save ──────────────────────────────
print(f"\nBuilding DataFrame from {len(all_rows)} rows...")
df = pd.DataFrame(all_rows)

print(f"Total rows : {len(df)}")
print(f"Columns    : {list(df.columns)}")
print(f"Seasons    : {sorted(df['season'].unique())}")
print()
print(df.head(3))

# Save full dataset (all seasons — for venue profiles)
df.to_csv('data/processed/deliveries_all.csv', index=False)

# Save current season only (for team engines)
df_current = df[df['season'].isin(['2025', '2026'])]
print(f"\n2025/26 rows: {len(df_current)}")
df_current.to_csv('data/processed/deliveries_current.csv', index=False)

print("\n✅ Done. Both CSVs saved to data/processed/")