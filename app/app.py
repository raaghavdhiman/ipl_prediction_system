import streamlit as st
import sys
import os

# Add parent folder to path so we can import predictor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction.predictor import predict_match, load_all_data

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="IPL Playoff Predictor",
    page_icon="🏏",
    layout="wide"
)

# ── LOAD DATA ONCE ────────────────────────────────────────
# st.cache_resource keeps data in memory across interactions
# Without this, every button click reloads all CSVs (slow)
@st.cache_resource
def get_data():
    return load_all_data()

data = get_data()

# ── TEAMS AND VENUES ──────────────────────────────────────
TEAMS = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad',
]

# Playoff-relevant venues with clean display names
# Format: "Display Name": "Exact match keyword for venue lookup"
VENUES = {
    # ── Home grounds ──────────────────────────────────────
    'M Chinnaswamy Stadium, Bengaluru'         : 'Chinnaswamy',
    'Wankhede Stadium, Mumbai'                 : 'Wankhede',
    'MA Chidambaram Stadium, Chennai'          : 'Chidambaram',
    'Eden Gardens, Kolkata'                    : 'Eden Gardens',
    'Narendra Modi Stadium, Ahmedabad'         : 'Narendra Modi Stadium',
    'Arun Jaitley Stadium, Delhi'              : 'Arun Jaitley',
    'Rajiv Gandhi Intl Stadium, Hyderabad'     : 'Rajiv Gandhi International Stadium, Uppal',
    'Ekana Cricket Stadium, Lucknow'           : 'Ekana',
    'MYSI Stadium, Mullanpur (New Chandigarh)' : 'New Chandigarh',
    'Sawai Mansingh Stadium, Jaipur'           : 'Sawai Mansingh',
    # ── Alternate venues ──────────────────────────────────
    'Barsapara Cricket Stadium, Guwahati'      : 'Barsapara',
    'Shaheed VNSSI Stadium, Naya Raipur'       : 'Raipur',
    'HPCA Stadium, Dharamsala'                 : 'Dharamsala',
    # ── Neutral / playoff venues ──────────────────────────
    'DY Patil Sports Academy, Mumbai'          : 'DY Patil',
    'Brabourne Stadium, Mumbai'                : 'Brabourne',
    'MCA Stadium, Pune'                        : 'Pune',
}

# ── HEADER ────────────────────────────────────────────────
st.title("🏏 IPL Playoff Predictor 2026")
st.markdown(
    "Powered by cricket intelligence — batting, bowling, form "
    "and venue analysis from 2025/26 season data."
)
st.divider()

# ── INPUT SECTION ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    team_a = st.selectbox(
        "🔵 Team A",
        TEAMS,
        index=8  # defaults to RCB
    )

with col2:
    # Filter out Team A so you can't pick same team twice
    team_b_options = [t for t in TEAMS if t != team_a]
    team_b = st.selectbox(
        "🔴 Team B",
        team_b_options,
        index=0
    )

with col3:
    venue_display = st.selectbox(
        "🏟 Venue",
        list(VENUES.keys())
    )

# ── PREDICT BUTTON ────────────────────────────────────────
st.markdown("")  # small spacer
predict_clicked = st.button("🔮 Predict Match", type="primary", use_container_width=True)

# ── RESULTS ───────────────────────────────────────────────
if predict_clicked:
    venue_lookup = VENUES[venue_display]

    with st.spinner("Analysing..."):
        result = predict_match(team_a, team_b, venue_lookup, data)

    st.divider()

    # ── WINNER BANNER ─────────────────────────────────────
    winner     = result['winner']
    loser      = result['team_b'] if winner == result['team_a'] else result['team_a']
    winner_prob = result['winner_prob']
    loser_prob  = round(100 - winner_prob, 1)

    if winner_prob >= 65:
        st.success(f"## 🏆 Predicted Winner: {winner}")
    elif winner_prob >= 57:
        st.info(f"## 🏆 Predicted Winner: {winner} (slight favourite)")
    else:
        st.warning(f"## 🏆 Predicted Winner: {winner} (very close match)")

    st.markdown(f"*Match: **{team_a}** vs **{team_b}** at {venue_display}*")
    st.markdown("")

    # ── PROBABILITY BAR ───────────────────────────────────
    st.markdown("### Win Probability")
    prob_col1, prob_col2 = st.columns(2)
    with prob_col1:
        st.metric(team_a, f"{result['prob_a']}%")
        st.progress(result['prob_a'] / 100)
    with prob_col2:
        st.metric(team_b, f"{result['prob_b']}%")
        st.progress(result['prob_b'] / 100)

    st.markdown("")

    # ── STRENGTH + STABILITY + CHAOS ──────────────────────
    st.markdown("### Match Metrics")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            f"{team_a} Strength",
            f"{result['strength_a']:.1f}",
        )
    with m2:
        st.metric(
            f"{team_b} Strength",
            f"{result['strength_b']:.1f}",
        )
    with m3:
        stability = result['stability']
        st.metric(
            "Stability Score",
            f"{stability:.0f} / 100",
            help="Higher = more predictable match"
        )
    with m4:
        chaos = result['chaos']
        st.metric(
            "Chaos Potential",
            f"{chaos:.0f} / 100",
            help="Higher = more likely to have sudden swings"
        )

    st.markdown("")

    # ── TEAM BREAKDOWN ────────────────────────────────────
    st.markdown("### Team Breakdown")
    bc1, bc2 = st.columns(2)

    sa = result['scores_a']
    sb = result['scores_b']

    with bc1:
        st.markdown(f"**{team_a}**")
        st.markdown(f"- Last 5 matches: `{sa['last5']}`")
        st.markdown(f"- Batting dependency (CR4): `{sa['cr4']:.2f}` — {sa['cr4_risk']} risk")
        st.markdown(f"- Death batting SR: `{sa['avg_death_sr']:.0f}`")
        st.markdown(f"- Death bowling economy: `{sa['avg_death_eco']:.2f}`")
        st.markdown(f"- Spin usage: `{result['spin_a']:.0f}%`")

    with bc2:
        st.markdown(f"**{team_b}**")
        st.markdown(f"- Last 5 matches: `{sb['last5']}`")
        st.markdown(f"- Batting dependency (CR4): `{sb['cr4']:.2f}` — {sb['cr4_risk']} risk")
        st.markdown(f"- Death batting SR: `{sb['avg_death_sr']:.0f}`")
        st.markdown(f"- Death bowling economy: `{sb['avg_death_eco']:.2f}`")
        st.markdown(f"- Spin usage: `{result['spin_b']:.0f}%`")

    st.markdown("")

    # ── NARRATIVE ─────────────────────────────────────────
    n1, n2 = st.columns(2)

    with n1:
        st.markdown(f"### ✅ Why {winner} wins")
        for reason in result['win_reasons']:
            st.markdown(f"- {reason}")

    with n2:
        st.markdown("### ⚠️ Why it could flip")
        for reason in result['risk_reasons']:
            st.markdown(f"- {reason}")

    st.markdown("")

    # ── VENUE PROFILE ─────────────────────────────────────
    with st.expander("🏟 Venue Profile"):
        vp = result['venue_profile']
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Avg 1st Innings", f"{vp['avg_first_inn']:.0f}")
        v2.metric("Chase Win Rate",  f"{vp['chase_win_rate']*100:.0f}%")
        v3.metric("Spin Wickets %",  f"{vp['spin_pct']:.0f}%")
        v4.metric("Volatility",      f"{vp['volatility_score']:.0f}/100")

# ── FOOTER ────────────────────────────────────────────────
st.divider()
st.caption(
    "Built on IPL 2025/26 ball-by-ball data from Cricsheet · "
    "Predictions are probabilistic, not guarantees · Cricket is glorious chaos 🏏"
)