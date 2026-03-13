import streamlit as st
from src.data_loader import load_data, get_data_summary
from src.filters import apply_sidebar_filters
from src.charts import (
    plot_tip_distribution,
    plot_total_bill_vs_tip,
    plot_avg_tip_by_day,
    plot_tip_boxplot_by_time,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tips Dashboard",
    page_icon="💰",
    layout="wide",
)

# ── Load data ──────────────────────────────────────────────────────────────────
df_raw = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────────
df = apply_sidebar_filters(df_raw)

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("💰 Restaurant Tips Dashboard")
st.markdown(
    "Explore tipping patterns from the classic **Tips** dataset. "
    "Use the sidebar to filter the data interactively."
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
st.header("📊 Dataset Overview")

summary = get_data_summary(df_raw)
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", summary["n_rows"])
col2.metric("Columns", summary["n_cols"])
col3.metric("Records after filters", len(df))

st.subheader("Raw Data Preview")
st.dataframe(df, use_container_width=True)

with st.expander("Column types & missing values"):
    col_a, col_b = st.columns(2)
    col_a.write("**Data types**")
    col_a.json(summary["dtypes"])
    col_b.write("**Missing values**")
    col_b.json(summary["missing_values"])

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Basic Statistics
# ══════════════════════════════════════════════════════════════════════════════
st.header("🔍 Filtered Data — Basic Statistics")
st.dataframe(df.describe(), use_container_width=True)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Visualizations
# ══════════════════════════════════════════════════════════════════════════════
st.header("📈 Visualizations")

if df.empty:
    st.warning("No data matches the current filters. Please adjust the sidebar.")
else:
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.plotly_chart(plot_tip_distribution(df), use_container_width=True)

    with row1_col2:
        st.plotly_chart(plot_total_bill_vs_tip(df), use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.plotly_chart(plot_avg_tip_by_day(df), use_container_width=True)

    with row2_col2:
        st.plotly_chart(plot_tip_boxplot_by_time(df), use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Key Insights
# ══════════════════════════════════════════════════════════════════════════════
st.header("💡 Key Insights")

if not df.empty:
    avg_tip = df["tip"].mean()
    avg_pct = (df["tip"] / df["total_bill"] * 100).mean()
    best_day = df.groupby("day", observed=True)["tip"].mean().idxmax()
    top_tipper = df.loc[df["tip"].idxmax()]

    ins1, ins2, ins3, ins4 = st.columns(4)
    ins1.metric("Average Tip", f"${avg_tip:.2f}")
    ins2.metric("Average Tip %", f"{avg_pct:.1f}%")
    ins3.metric("Best Tipping Day", str(best_day))
    ins4.metric("Highest Single Tip", f"${top_tipper['tip']:.2f}")
else:
    st.info("Apply filters to see insights.")
