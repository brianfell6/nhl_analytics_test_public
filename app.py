import streamlit as st
import pandas as pd
import altair as alt

# ── SETUP ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NHL Analytics Pipeline", layout="wide")

@st.cache_data
def load_local_data(filename: str) -> pd.DataFrame:
    """Load pre-exported query results from the local repository."""
    df = pd.read_csv(filename)
    df.columns = [col.upper() for col in df.columns]
    return df

counts_df = load_local_data("counts.csv")
trend_all_seasons_df = load_local_data("trend_trends.csv")
all_seasons_normalized_df = load_local_data("composite_normalized.csv")
all_seasons_breakout_df = load_local_data("breakout_candidates.csv")

# Official NHL HEX Color Mapping from https://teamcolorcodes.com
NHL_TEAM_COLORS = {
    "ANA": "#F47A38", "ARI": "#8C2633", "BOS": "#FFB81C", "BUF": "#003087",
    "CGY": "#C8102E", "CAR": "#CE1126", "CHI": "#CF1126", "COL": "#6F263D",
    "CBJ": "#002654", "DAL": "#006847", "DET": "#CE1126", "EDM": "#041E42",
    "FLA": "#041E42", "LAK": "#111111", "MIN": "#154734", "MTL": "#AF1E2D",
    "NSH": "#FFB81C", "NJD": "#CE1126", "NYI": "#00539C", "NYR": "#0038A8",
    "OTT": "#C8102E", "PHI": "#F74902", "PIT": "#FCB514", "STL": "#002F87",
    "SJS": "#006D75", "SEA": "#001628", "TBL": "#002868", "TOR": "#00205B",
    "VAN": "#00843D", "VGK": "#B4975A", "WSH": "#C8102E", "WPG": "#041E42"
}

# ── HEADER ──────────────────────────────────────────────────────────────────
st.title("🏒 NHL Analytics Pipeline")
st.caption(
    "End-to-end data pipeline: NHL API + MoneyPuck CSVs → Python ETL → "
    "PostgreSQL → Snowflake. 11 seasons, 6 tables, fully self-built."
)

with st.expander("📐 Pipeline architecture", expanded=False):
    st.markdown("""
    **Data sources**
    - NHL Stats API — skater scoring, physical play, goaltending
    - MoneyPuck CSVs — Corsi, Fenwick, expected goals (xG)

    **Pipeline**
    1. Python scripts (`requests`, `pandas`, `psycopg2`) extract and transform data from both sources
    2. Data lands in **PostgreSQL** as the primary analytical database — 6 normalized tables joined via a unified view
    3. A custom Python migration script (using `snowflake-connector-python`) replicates the full schema into **Snowflake**
    4. This dashboard queries **historical Snowflake data** hosted statically on GitHub — zero compute costs

    **Engineering practices applied:** environment variables for credentials, structured logging, per-batch error handling, reusable/parameterized ETL functions instead of duplicated scripts.
    """)

st.divider()

# ── KPI ROW ─────────────────────────────────────────────────────────────────
cols = st.columns(len(counts_df))
for col, (_, row) in zip(cols, counts_df.iterrows()):
    col.metric(row["METRIC"], f"{int(row['VALUE']):,}")

st.divider()

# ── LEAGUE SCORING TREND ───────────────────────────────────────────────────
st.subheader("League-Wide Scoring Trend (11 Seasons)")

trend_clean = trend_all_seasons_df.copy()
if "SEASON" not in trend_clean.columns:
    trend_clean = trend_clean.reset_index()

def format_season_label(val):
    try:
        raw_str = str(val).strip().split('.')[0]
        start_year = int(raw_str[:4])
        next_year_short = str(start_year + 1)[2:]
        return f"{start_year}-{next_year_short}"
    except Exception:
        return str(val)

trend_clean["SEASON_LABEL"] = trend_clean["SEASON"].apply(format_season_label)

trend_melted = trend_clean.melt(
    id_vars=["SEASON_LABEL"], 
    value_vars=["AVG_POINTS", "AVG_GOALS"], 
    var_name="Metric", 
    value_name="Average"
)

trend_chart = alt.Chart(trend_melted).mark_line(point=True).encode(
    x=alt.X("SEASON_LABEL:N", title="Season", sort=None, axis=alt.Axis(labelAngle=0)), 
    y=alt.Y("Average:Q", title="Value"),
    color=alt.Color("Metric:N", scale=alt.Scale(range=["#00205B", "#F47A38"])) 
).properties(
    height=500 
)

st.altair_chart(trend_chart, use_container_width=True)

st.divider()

# ── COMPOSITE SCORE LEADERBOARD ────────────────────────────────────────────
st.subheader("Composite Player Score — Normalized Multi-Stat Model")
st.caption(
    "Blends scoring, expected goals, possession (Corsi), and physical play into one "
    "normalized score (0–1 scale) so no single stat dominates due to raw magnitude. "
    "Adjust the weight percentages below to see rankings shift live."
)

with st.expander("📊 How the Composite Score is Calculated", expanded=False):
    st.markdown("""
    Because raw stats utilize completely different baseline scales (e.g., a player might have 90 **Points** but only 5 **Expected Goals**), comparing them directly would cause the highest numerical stat to completely override the others. 
    
    **The Math Behind the Model:**
    1. **Min-Max Normalization**: For each season, every player's raw metrics (Points, xG, Corsi %, Hits + Blocks) are scaled to a strict **0.0 to 1.0 range** using the formula:  
       $$\\text{Normalized Stat} = \\frac{\\text{Value} - \\text{Min}}{\\text{Max} - \\text{Min}}$$
    2. **Weighted Combination**: When you adjust the sliders below, your custom weights are added together to create a true percentage allocation ($W_{Total}$). 
    3. **Final Metric Execution**: The system computes the dot-product sum of each player's individual normalized metrics multiplied by your relative weight values:
       $$\\text{Composite Score} = \\left(\\text{Pts}_{\\text{norm}} \\times \\frac{W_{\\text{pts}}}{W_{\\text{total}}}\\right) + \\left(\\text{xG}_{\\text{norm}} \\times \\frac{W_{\\text{xg}}}{W_{\\text{total}}}\\right) + \\left(\\text{Corsi}_{\\text{norm}} \\times \\frac{W_{\\text{corsi}}}{W_{\\text{total}}}\\right) + \\left(\\text{Phys}_{\\text{norm}} \\times \\frac{W_{\\text{phys}}}{W_{\\text{total}}}\\right)$$
    """)

season_options = sorted(all_seasons_normalized_df["SEASON"].unique().tolist(), reverse=True)
selected_season = st.selectbox("Season", season_options, index=0)

w1, w2, w3, w4 = st.columns(4)
weight_points = w1.slider("Points weight", 0.0, 1.0, 0.40, step=0.05, format="%.2f")
weight_xg = w2.slider("xG weight", 0.0, 1.0, 0.30, step=0.05, format="%.2f")
weight_corsi = w3.slider("Corsi weight", 0.0, 1.0, 0.20, step=0.05, format="%.2f")
weight_phys = w4.slider("Physical weight", 0.0, 1.0, 0.10, step=0.05, format="%.2f")
weight_total = max(weight_points + weight_xg + weight_corsi + weight_phys, 0.01)

normalized_df = all_seasons_normalized_df[all_seasons_normalized_df["SEASON"] == selected_season].copy()

norm_cols = ["PTS_NORM", "XG_NORM", "CORSI_NORM", "PHYSICAL_NORM", "CORSI", "XG"]
normalized_df[norm_cols] = normalized_df[norm_cols].astype(float)

normalized_df["Composite Score"] = (
    normalized_df["PTS_NORM"] * (weight_points / weight_total) +
    normalized_df["XG_NORM"] * (weight_xg / weight_total) +
    normalized_df["CORSI_NORM"] * (weight_corsi / weight_total) +
    normalized_df["PHYSICAL_NORM"] * (weight_phys / weight_total)
)

composite_df = normalized_df.sort_values("Composite Score", ascending=False).head(15).copy()

display_df = composite_df[["PLAYER_NAME", "PRIMARY_TEAM", "GOALS", "ASSISTS", "POINTS"]].copy()
display_df.columns = ["Player", "Team", "Goals", "Assists", "Points"]

display_df["Corsi %"] = composite_df["CORSI"].apply(lambda v: f"{float(str(v).strip()) * 100:.2f}%" if float(str(v).strip()) <= 1.0 else f"{float(str(v).strip()):.2f}%")
display_df["xG"] = composite_df["XG"].map(lambda v: f"{v:.1f}")

composite_df["Composite_Score_Num"] = composite_df["Composite Score"]
display_df["Composite Score"] = composite_df["Composite Score"].map(lambda v: f"{v:.2f}")

c1, c2 = st.columns(2)
with c1:
    st.dataframe(display_df, hide_index=True, use_container_width=True)
with c2:
    present_teams = composite_df["PRIMARY_TEAM"].unique().tolist()
    color_range = [NHL_TEAM_COLORS.get(team, "#A7A9AC") for team in present_teams]
    
    leaderboard_chart = alt.Chart(composite_df).mark_bar().encode(
        x=alt.X("Composite_Score_Num:Q", title="Composite Score"),
        y=alt.Y("PLAYER_NAME:N", title="Player", sort="-x"), 
        color=alt.Color("PRIMARY_TEAM:N", title="Team", scale=alt.Scale(domain=present_teams, range=color_range)) 
    ).properties(
        height=440
    )
    st.altair_chart(leaderboard_chart, use_container_width=True)

st.divider()

# ── BREAKOUT CANDIDATE FINDER ──────────────────────────────────────────────
st.subheader("Breakout Candidate Finder")
st.caption(
    "Surfaces young, possession-positive players who are efficient in limited "
    "ice time — the profile of an undervalued asset before the market catches on."
)

f1, f2, f3 = st.columns(3)
max_age = f1.slider("Max age", 18, 30, 23)
min_corsi = f2.slider("Min Corsi %", 0.40, 0.60, 0.52, step=0.01)
max_toi = f3.slider("Max minutes/game", 8.0, 20.0, 14.0, step=0.5)

breakout_df_clean = all_seasons_breakout_df.copy()

def extract_start_year(val):
    try:
        raw_str = str(val).strip().split('.')[0]
        return int(raw_str[:4])
    except Exception:
        return int(val)

def extract_dropdown_year(val):
    try:
        raw_str = str(val).strip().split('.')[0]
        return int(raw_str[:4])
    except Exception:
        return int(val)

breakout_df_clean["CLEAN_SEASON"] = breakout_df_clean["SEASON"].apply(extract_start_year)
target_year = extract_dropdown_year(selected_season)

breakout_filtered = breakout_df_clean[
    (breakout_df_clean["CLEAN_SEASON"] == target_year) &
    (breakout_df_clean["AGE"] <= max_age) &
    (breakout_df_clean["CORSI_PCT"] >= min_corsi) &
    (breakout_df_clean["TOI_PER_GAME"] <= max_toi)
].sort_values("XG_PER_GAME", ascending=False).head(15).copy()

breakout_display = pd.DataFrame()
if not breakout_filtered.empty:
    breakout_display["Player"] = breakout_filtered["PLAYER_NAME"]
    breakout_display["Team"] = breakout_filtered["PRIMARY_TEAM"]
    breakout_display["Age"] = breakout_filtered["AGE"]
    breakout_display["Min/GP"] = breakout_filtered["TOI_PER_GAME"]
