import streamlit as st
import pandas as pd
import altair as alt

# ── SETUP ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NHL Analytics Pipeline", layout="wide")

@st.cache_data
def load_local_data(filename: str) -> pd.DataFrame:
    """Load pre-exported query results from the local repository."""
    df = pd.read_csv(filename)
    # Standardize column casing to uppercase to match Snowflake's default output structure
    df.columns = [col.upper() for col in df.columns]
    return df

# Load datasets locally instead of querying Snowflake live
counts_df = load_local_data("counts.csv")
trend_all_seasons_df = load_local_data("trend_trends.csv")
all_seasons_normalized_df = load_local_data("composite_normalized.csv")
all_seasons_breakout_df = load_local_data("breakout_candidates.csv")

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

# Reset index to extract SEASON as an accessible data column
trend_clean = trend_all_seasons_df.reset_index()

# Robust string conversion to convert numeric years (e.g., 2013) to clean string tags (e.g., "2013-14")
def format_season_label(val):
    try:
        year_num = int(float(str(val).strip()))
        next_year_short = str(year_num + 1)[2:]
        return f"{year_num}-{next_year_short}"
    except Exception:
        return str(val)

trend_clean["SEASON_LABEL"] = trend_clean["SEASON"].apply(format_season_label)

# Unpivot data for unified multi-line charting
trend_melted = trend_clean.melt(
    id_vars=["SEASON_LABEL"], 
    value_vars=["AVG_POINTS", "AVG_GOALS"], 
    var_name="Metric", 
    value_name="Average"
)

# Build a strictly non-interactive, tall, cleanly labeled timeline graph
trend_chart = alt.Chart(trend_melted).mark_line(point=True).encode(
    x=alt.X("SEASON_LABEL:N", title="Season", sort=None), # :N enforces explicit string sorting order
    y=alt.Y("Average:Q", title="Value"),
    color=alt.Color("Metric:N", scale=alt.Scale(range=["#00205B", "#F47A38"])) # Custom NHL Team Colors (Navy & Orange)
).properties(
    height=500 # Explicitly set a taller custom canvas framework 
)

# Render using use_container_width=True. Omitting '.interactive()' keeps the canvas completely static.
st.altair_chart(trend_chart, use_container_width=True)

st.divider()

# ── COMPOSITE SCORE LEADERBOARD ────────────────────────────────────────────
st.subheader("Composite Player Score — Normalized Multi-Stat Model")
st.caption(
    "Blends scoring, expected goals, possession (Corsi), and physical play into one "
    "normalized score (0–1 scale) so no single stat dominates due to raw magnitude. "
    "Drag the weights below to see rankings shift live."
)

# Pull options from the pre-loaded static dataset
season_options = sorted(all_seasons_normalized_df["SEASON"].unique().tolist(), reverse=True)
selected_season = st.selectbox("Season", season_options, index=0)

w1, w2, w3, w4 = st.columns(4)
weight_points = w1.slider("Points weight", 0, 100, 40)
weight_xg = w2.slider("xG weight", 0, 100, 30)
weight_corsi = w3.slider("Corsi weight", 0, 100, 20)
weight_phys = w4.slider("Physical weight", 0, 100, 10)
weight_total = max(weight_points + weight_xg + weight_corsi + weight_phys, 1)

# Filter the global dataset down to the single selected season using Pandas
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

# Dynamically evaluates if Corsi is a fraction (0.523) or standard percentage integer to formatting decimals correctly
display_df["Corsi %"] = composite_df["CORSI"].apply(lambda v: f"{float(v) * 100:.1f}%" if float(v) <= 1.0 else f"{float(v):.1f}%")
display_df["xG"] = composite_df["XG"].map(lambda v: f"{v:.1f}")
display_df["Composite Score"] = composite_df["Composite Score"].map(lambda v: f"{v:.2f}")

c1, c2 = st.columns([2, 1])
with c1:
    st.dataframe(display_df, hide_index=True, use_container_width=True)
with c2:
    chart_df = composite_df.set_index("PLAYER_NAME")["Composite Score"]
    st.bar_chart(chart_df, height=420)

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

# Filter your dataset using Python instead of sending a new query to Snowflake
breakout_filtered = all_seasons_breakout_df[
    (all_seasons_breakout_df["SEASON"] == selected_season) &
    (all_seasons_breakout_df["AGE"] <= max_age) &
    (all_seasons_breakout_df["CORSI_PCT"] >= min_corsi) &
    (all_seasons_breakout_df["TOI_PER_GAME"] <= max_toi)
].sort_values("XG_PER_GAME", ascending=False).head(15).copy()

breakout_display = pd.DataFrame()
breakout_display["Player"] = breakout_filtered["PLAYER_NAME"]
breakout_display["Team"] = breakout_filtered["PRIMARY_TEAM"]
breakout_display["Age"] = breakout_filtered["AGE"]
breakout_display["Min/GP"] = breakout_filtered["TOI_PER_GAME"]
breakout_display["Points"] = breakout_filtered["POINTS"]
breakout_display["PPG"] = breakout_filtered["PPG"].astype(float).map(lambda v: f"{v:.2f}")
breakout_display["xG/GP"] = breakout_filtered["XG_PER_GAME"].astype(float).map(lambda v: f"{v:.2f}")

# Accurate float parsing ensures the true fractional trailing decimal fields display correctly
breakout_display["Corsi %"] = breakout_filtered["CORSI_PCT"].apply(lambda v: f"{float(v) * 100:.1f}%" if float(v) <= 1.0 else f"{float(v):.1f}%")

st.dataframe(breakout_display, hide_index=True, use_container_width=True)

st.divider()
st.caption("Built entirely with Python, PostgreSQL, and Snowflake — full pipeline self-designed and self-debugged.")
