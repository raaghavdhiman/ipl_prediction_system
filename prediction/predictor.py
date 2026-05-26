import pandas as pd
import numpy as np
import math

# ═══════════════════════════════════════════════════════════
# MANUAL FORM OVERRIDES
# Cricsheet sometimes lags behind real matches by 1-2 days.
# Update these manually when playoffs happen tonight.
# Format: 'Team Name': 'WWLWW' (left=oldest, right=most recent)
# Leave empty dict {} if form_scores.csv is up to date.
# ═══════════════════════════════════════════════════════════
FORM_OVERRIDES = {
    # Example — update based on today's actual results:
    # 'Punjab Kings'               : 'LLLLW',
    # 'Royal Challengers Bengaluru': 'LWWWW',
    'Chennai Super Kings'         : 'WWLLL',
    'Delhi Capitals'              : 'LLWWW',
    'Gujarat Titans'              : 'WWWLW',
    'Kolkata Knight Riders'       : 'WLWWL',
    'Lucknow Super Giants'        : 'WLWLL',
    'Mumbai Indians'              : 'WLWLL',
    'Punjab Kings'                : 'LLLLW',
    'Rajasthan Royals'            : 'LLLWW',
    'Royal Challengers Bengaluru' : 'LWWWL',
    'Sunrisers Hyderabad'         : 'LWLWW',
}

# ═══════════════════════════════════════════════════════════
# WEIGHTS — tweak to calibrate predictions
# ═══════════════════════════════════════════════════════════
WEIGHTS = {
    'batting' : 0.30,
    'bowling' : 0.30,
    'form'    : 0.25,
    'venue'   : 0.15,
}

# ═══════════════════════════════════════════════════════════
# SPIN BOWLERS — verified IPL 2025 + 2026 list
# ═══════════════════════════════════════════════════════════
SPIN_BOWLERS = {
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

# ═══════════════════════════════════════════════════════════
# SCORE COMPRESSION
# normalize() creates 0-100 extremes. compress() shrinks
# that to 35-75 so no team is truly hopeless or perfect.
# ═══════════════════════════════════════════════════════════
def compress(score, low=35, high=75):
    return low + (score / 100) * (high - low)

# ═══════════════════════════════════════════════════════════
# FORM SCORE HELPER
# Converts a last5 string like 'WWLWW' into a 0-100 score.
# Used when FORM_OVERRIDES is set for a team.
# ═══════════════════════════════════════════════════════════
def last5_to_score(last5_str):
    WEIGHTS_FORM = [0.10, 0.15, 0.20, 0.25, 0.30]
    results = list(last5_str.upper())[-5:]
    values  = []
    for r in results:
        if   r == 'W':  values.append(1.0)
        elif r == 'L':  values.append(0.3)
        else:           values.append(0.5)
    n = len(values)
    w = WEIGHTS_FORM[-n:]
    total_w = sum(w)
    raw = sum(v * (wt/total_w) for v, wt in zip(values, w))
    # Convert raw (0.3-1.0 range) to 0-100 scale
    return round((raw - 0.3) / 0.7 * 100, 1)

# ═══════════════════════════════════════════════════════════
# LOAD ALL DATA
# ═══════════════════════════════════════════════════════════
def load_all_data():
    batting  = pd.read_csv('data/processed/batting_scores.csv')
    bowling  = pd.read_csv('data/processed/bowling_scores.csv')
    form     = pd.read_csv('data/processed/form_scores.csv')
    venues   = pd.read_csv('data/processed/venue_profiles.csv')
    delivery = pd.read_csv('data/processed/deliveries_current.csv',
                           dtype={'season': str}, low_memory=False)
    batting  = batting.rename(columns={'batting_team': 'team'})
    return {
        'batting' : batting,
        'bowling' : bowling,
        'form'    : form,
        'venues'  : venues,
        'delivery': delivery,
    }

# ═══════════════════════════════════════════════════════════
# VENUE LOOKUP
# ═══════════════════════════════════════════════════════════
def get_venue_profile(venue_name, venues_df):
    keywords = [w for w in venue_name.split() if len(w) > 3]
    mask = venues_df['venue'].apply(
        lambda v: any(kw.lower() in v.lower() for kw in keywords)
    )
    matches = venues_df[mask]
    if len(matches) == 0:
        print(f"  ⚠ Venue '{venue_name}' not found. Using neutral defaults.")
        return {
            'avg_first_inn'   : 175.0,
            'chase_win_rate'  : 0.5,
            'spin_pct'        : 20.0,
            'pace_pct'        : 80.0,
            'volatility_score': 50.0,
        }
    return {
        'avg_first_inn'   : matches['avg_first_inn'].mean(),
        'chase_win_rate'  : matches['chase_win_rate'].mean(),
        'spin_pct'        : matches['spin_pct'].mean(),
        'pace_pct'        : matches['pace_pct'].mean(),
        'volatility_score': matches['volatility_score'].mean(),
    }

# ═══════════════════════════════════════════════════════════
# TEAM SPIN USAGE
# ═══════════════════════════════════════════════════════════
def get_team_spin_usage(team, delivery_df):
    team_bowling = delivery_df[delivery_df['bowling_team'] == team]
    if len(team_bowling) == 0:
        return 20.0
    spin_balls  = team_bowling['bowler'].isin(SPIN_BOWLERS).sum()
    total_balls = len(team_bowling)
    return round((spin_balls / total_balls) * 100, 1)

# ═══════════════════════════════════════════════════════════
# VENUE COMPATIBILITY
# ═══════════════════════════════════════════════════════════
def venue_compatibility(team, venue_profile, team_spin_pct,
                        batting_row, bowling_row):
    modifier  = 0.0
    reasons   = []
    spin_pct  = venue_profile['spin_pct']
    chase_rate = venue_profile['chase_win_rate']
    avg_score  = venue_profile['avg_first_inn']

    if spin_pct > 25 and team_spin_pct > 25:
        modifier += 8
        reasons.append(f"spin-heavy attack suits {int(spin_pct)}% spin venue")
    elif spin_pct < 15 and team_spin_pct < 15:
        modifier += 5
        reasons.append("pace attack matches pace-dominated surface")
    elif spin_pct > 25 and team_spin_pct < 15:
        modifier -= 5
        reasons.append("pace-only attack at spin-friendly venue")

    death_sr = batting_row.get('avg_death_sr', 160)
    if chase_rate > 0.55 and death_sr > 165:
        modifier += 6
        reasons.append("strong death batting suits chase-friendly venue")
    elif chase_rate < 0.45 and death_sr < 150:
        modifier -= 4
        reasons.append("weak death batting at defend-friendly venue")

    batting_score = batting_row.get('batting_score', 50)
    if avg_score > 185 and batting_score > 70:
        modifier += 5
        reasons.append("strong batting unit thrives at high-scoring venue")
    elif avg_score > 185 and batting_score < 30:
        modifier -= 4
        reasons.append("weak batting exposed at high-scoring venue")

    return round(modifier, 2), reasons

# ═══════════════════════════════════════════════════════════
# GET TEAM SCORES
# compress() applied here — shrinks 0-100 → 35-75
# FORM_OVERRIDES applied here — manual last5 overrides CSV
# ═══════════════════════════════════════════════════════════
def get_team_scores(team, data):
    bat = data['batting'][data['batting']['team'] == team]
    bow = data['bowling'][data['bowling']['team'] == team]
    frm = data['form'][data['form']['team'] == team]

    bat_row = bat.iloc[0].to_dict() if len(bat) > 0 else {}
    bow_row = bow.iloc[0].to_dict() if len(bow) > 0 else {}
    frm_row = frm.iloc[0].to_dict() if len(frm) > 0 else {}

    # Check if this team has a manual form override
    if team in FORM_OVERRIDES:
        override_last5 = FORM_OVERRIDES[team]
        form_score = last5_to_score(override_last5)
        last5      = override_last5
        print(f"  ℹ Form override active for {team}: {override_last5} → score {form_score}")
    else:
        form_score = frm_row.get('form_score', 50)
        last5      = frm_row.get('last5', '?????')

    return {
        # compress() shrinks extremes: 0→35, 50→55, 100→75
        'batting_score' : compress(bat_row.get('batting_score', 50)),
        'bowling_score' : compress(bow_row.get('bowling_score', 50)),
        'form_score'    : compress(form_score),
        # raw metrics kept uncompressed for narrative logic
        'cr4'           : bat_row.get('cr4', 0.6),
        'cr4_risk'      : bat_row.get('cr4_risk', 'MEDIUM'),
        'last5'         : last5,
        'avg_death_sr'  : bat_row.get('avg_death_sr', 160),
        'avg_death_eco' : bow_row.get('avg_death_eco', 10),
        'avg_pp_wkts'   : bow_row.get('avg_pp_wkts_taken', 1.5),
        'avg_dot_pct'   : bow_row.get('avg_dot_pct', 30),
        'nrr'           : frm_row.get('nrr', 0),
    }

# ═══════════════════════════════════════════════════════════
# TEAM STRENGTH
# ═══════════════════════════════════════════════════════════
def compute_strength(team, venue_profile, data):
    scores   = get_team_scores(team, data)
    spin_pct = get_team_spin_usage(team, data['delivery'])

    bat_row = data['batting'][data['batting']['team'] == team]
    bat_row = bat_row.iloc[0].to_dict() if len(bat_row) > 0 else {}
    bow_row = data['bowling'][data['bowling']['team'] == team]
    bow_row = bow_row.iloc[0].to_dict() if len(bow_row) > 0 else {}

    v_mod, v_reasons = venue_compatibility(
        team, venue_profile, spin_pct, bat_row, bow_row
    )

    base = (
        scores['batting_score'] * WEIGHTS['batting'] +
        scores['bowling_score'] * WEIGHTS['bowling'] +
        scores['form_score']    * WEIGHTS['form']
    )
    venue_contribution = v_mod * WEIGHTS['venue']
    strength = max(0, min(100, base + venue_contribution))

    return strength, v_mod, v_reasons, spin_pct, scores

# ═══════════════════════════════════════════════════════════
# WIN PROBABILITY — sigmoid with divisor 45
# divisor 45 keeps probabilities in 52-72% realistic range
# ═══════════════════════════════════════════════════════════
def win_probability(strength_a, strength_b):
    delta  = (strength_a - strength_b) / 45.0
    prob_a = 1 / (1 + math.exp(-delta))
    return round(prob_a * 100, 1), round((1 - prob_a) * 100, 1)

# ═══════════════════════════════════════════════════════════
# STABILITY SCORE
# ═══════════════════════════════════════════════════════════
def compute_stability(scores_a, scores_b, venue_profile):
    stability = 100.0
    if scores_a['cr4_risk'] == 'HIGH':    stability -= 12
    if scores_b['cr4_risk'] == 'HIGH':    stability -= 12
    if scores_a['avg_death_eco'] > 11:    stability -= 8
    if scores_b['avg_death_eco'] > 11:    stability -= 8
    vol = venue_profile.get('volatility_score', 50)
    stability -= (vol / 100) * 15
    form_gap = abs(scores_a['form_score'] - scores_b['form_score'])
    if form_gap > 40:
        stability -= 8
    return round(max(0, min(100, stability)), 1)

# ═══════════════════════════════════════════════════════════
# CHAOS SCORE
# ═══════════════════════════════════════════════════════════
def compute_chaos(scores_a, scores_b, venue_profile):
    chaos = 0.0
    if scores_a['cr4_risk'] == 'HIGH':    chaos += 18
    elif scores_a['cr4_risk'] == 'MEDIUM': chaos += 8
    if scores_b['cr4_risk'] == 'HIGH':    chaos += 18
    elif scores_b['cr4_risk'] == 'MEDIUM': chaos += 8
    if scores_a['avg_death_eco'] > 10.5:  chaos += 10
    if scores_b['avg_death_eco'] > 10.5:  chaos += 10
    vol = venue_profile.get('volatility_score', 50)
    chaos += (vol / 100) * 20
    if 'LLL' in scores_a.get('last5', ''): chaos += 8
    if 'LLL' in scores_b.get('last5', ''): chaos += 8
    return round(min(100, chaos), 1)

# ═══════════════════════════════════════════════════════════
# NARRATIVE GENERATOR
# ═══════════════════════════════════════════════════════════
def generate_narrative(team_a, team_b, scores_a, scores_b,
                       venue_profile, spin_a, spin_b, winner):
    w_scores = scores_a if winner == team_a else scores_b
    l_scores = scores_b if winner == team_a else scores_a
    w_name   = team_a   if winner == team_a else team_b
    l_name   = team_b   if winner == team_a else team_a
    w_spin   = spin_a   if winner == team_a else spin_b
    l_spin   = spin_b   if winner == team_a else spin_a

    win_reasons  = []
    risk_reasons = []

    bat_gap  = w_scores['batting_score'] - l_scores['batting_score']
    bowl_gap = w_scores['bowling_score'] - l_scores['bowling_score']
    form_gap = w_scores['form_score']    - l_scores['form_score']

    if bat_gap > 8:
        win_reasons.append(
            f"{w_name} has a stronger batting unit "
            f"({w_scores['avg_death_sr']:.0f} vs "
            f"{l_scores['avg_death_sr']:.0f} death SR)"
        )
    if w_scores['avg_death_sr'] > l_scores['avg_death_sr'] + 10:
        win_reasons.append(
            f"Superior death batting "
            f"(SR {w_scores['avg_death_sr']:.0f} vs {l_scores['avg_death_sr']:.0f})"
        )
    if bowl_gap > 8:
        win_reasons.append(
            f"{w_name}'s bowling is stronger — "
            f"death eco {w_scores['avg_death_eco']:.1f} "
            f"vs {l_scores['avg_death_eco']:.1f}"
        )
    if l_scores['avg_death_eco'] - w_scores['avg_death_eco'] > 0.8:
        win_reasons.append(
            f"Tighter death bowling "
            f"({w_scores['avg_death_eco']:.1f} vs "
            f"{l_scores['avg_death_eco']:.1f} economy)"
        )
    if form_gap > 8:
        win_reasons.append(
            f"Better recent form "
            f"({w_scores['last5']} vs {l_scores['last5']})"
        )
    if w_scores['avg_dot_pct'] > l_scores['avg_dot_pct'] + 2:
        win_reasons.append(
            f"More dot ball pressure "
            f"({w_scores['avg_dot_pct']:.1f}% vs "
            f"{l_scores['avg_dot_pct']:.1f}%)"
        )
    spin_pct = venue_profile['spin_pct']
    if spin_pct > 25 and w_spin > l_spin + 5:
        win_reasons.append(
            f"Spin-heavier attack ({w_spin:.0f}% vs {l_spin:.0f}%) "
            f"suits {int(spin_pct)}% spin venue"
        )

    # Risk reasons
    if w_scores['cr4_risk'] == 'HIGH':
        risk_reasons.append(
            f"{w_name} top-order dependent (CR4={w_scores['cr4']:.2f}) "
            f"— early wickets could trigger collapse"
        )
    if l_scores['avg_pp_wkts'] > 1.6:
        risk_reasons.append(
            f"{l_name} takes {l_scores['avg_pp_wkts']:.1f} PP wickets/match "
            f"— could derail {w_name} early"
        )
    if venue_profile['volatility_score'] > 65:
        risk_reasons.append(
            f"Volatile venue — scores here vary wildly, "
            f"conditions can completely change match dynamics"
        )
    if 'LLL' in l_scores.get('last5', ''):
        risk_reasons.append(
            f"{l_name} is desperate — knockout pressure can "
            f"produce unexpected performances"
        )
    if abs(bat_gap) < 6 and abs(bowl_gap) < 6:
        risk_reasons.append(
            "Teams are very evenly matched — "
            "one brilliant over or one dropped catch decides this"
        )

    if not win_reasons:
        win_reasons.append(f"{w_name} holds a slight overall edge")
    if not risk_reasons:
        risk_reasons.append("Competitive match — either team can win on the day")

    return win_reasons, risk_reasons

# ═══════════════════════════════════════════════════════════
# MAIN PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════
def predict_match(team_a, team_b, venue_name, data=None):
    if data is None:
        data = load_all_data()

    venue_profile = get_venue_profile(venue_name, data['venues'])

    str_a, vmod_a, vreasons_a, spin_a, scores_a = compute_strength(
        team_a, venue_profile, data)
    str_b, vmod_b, vreasons_b, spin_b, scores_b = compute_strength(
        team_b, venue_profile, data)

    prob_a, prob_b = win_probability(str_a, str_b)
    winner         = team_a if prob_a >= prob_b else team_b
    winner_prob    = max(prob_a, prob_b)
    stability      = compute_stability(scores_a, scores_b, venue_profile)
    chaos          = compute_chaos(scores_a, scores_b, venue_profile)

    win_reasons, risk_reasons = generate_narrative(
        team_a, team_b, scores_a, scores_b,
        venue_profile, spin_a, spin_b, winner
    )

    return {
        'team_a'       : team_a,
        'team_b'       : team_b,
        'venue'        : venue_name,
        'winner'       : winner,
        'prob_a'       : prob_a,
        'prob_b'       : prob_b,
        'winner_prob'  : winner_prob,
        'strength_a'   : round(str_a, 1),
        'strength_b'   : round(str_b, 1),
        'stability'    : stability,
        'chaos'        : chaos,
        'win_reasons'  : win_reasons,
        'risk_reasons' : risk_reasons,
        'scores_a'     : scores_a,
        'scores_b'     : scores_b,
        'venue_profile': venue_profile,
        'spin_a'       : spin_a,
        'spin_b'       : spin_b,
        'vmod_a'       : vmod_a,
        'vmod_b'       : vmod_b,
    }

# ═══════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Loading data...")
    data = load_all_data()

    tests = [
        ('Royal Challengers Bengaluru', 'Punjab Kings',      'Eden Gardens'),
        ('Gujarat Titans',              'Kolkata Knight Riders', 'Narendra Modi Stadium'),
        ('Royal Challengers Bengaluru', 'Gujarat Titans',    'Eden Gardens'),
    ]

    for team_a, team_b, venue in tests:
        r = predict_match(team_a, team_b, venue, data)
        print(f"\n{'─'*55}")
        print(f"  {team_a} vs {team_b}")
        print(f"  Venue: {venue}")
        print(f"  🏆 {r['winner']} — {r['winner_prob']}%")
        print(f"  Strengths: {r['strength_a']} vs {r['strength_b']}")
        print(f"  Stability: {r['stability']} | Chaos: {r['chaos']}")
        print(f"  Why wins:")
        for reason in r['win_reasons']:
            print(f"    • {reason}")
        print(f"  Why flips:")
        for reason in r['risk_reasons']:
            print(f"    • {reason}")