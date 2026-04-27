import streamlit as st
import plotly.express as px
from utils.data_loader import load_blocks_summary, load_weekly_summary

st.set_page_config(
    page_title="FINRA OTC Market Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 FINRA OTC Market Data Dashboard")
st.markdown(
    "Interactive analytics for FINRA block trading summaries and 2026 weekly OTC/ATS trading statistics."
)

blocks = load_blocks_summary()
weekly = load_weekly_summary()

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Block Records", f"{len(blocks):,}")
c2.metric("Weekly Records", f"{len(weekly):,}")
c3.metric("Unique Firms (Blocks)", f"{blocks['MPID'].nunique()}")
c4.metric("Unique Tickers (2026)", f"{weekly['issueSymbolIdentifier'].nunique():,}")
c5.metric("Total Block Shares", f"{blocks['totalShareQuantity'].sum() / 1e9:.1f}B")

st.divider()

# ── Dataset coverage ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("Blocks Summary")
    st.info(
        f"**Date range:** {blocks['summaryStartDate'].min().date()} → "
        f"{blocks['summaryStartDate'].max().date()}\n\n"
        f"**Size categories:** {', '.join(sorted(blocks['summaryTypeCode'].dropna().unique()))}\n\n"
        f"**Firms:** {blocks['MPID'].nunique()} unique MPIDs"
    )

with col2:
    st.subheader("Weekly Summary 2026")
    st.info(
        f"**Date range:** {weekly['weekStartDate'].min().date()} → "
        f"{weekly['weekStartDate'].max().date()}\n\n"
        f"**Tiers:** {', '.join(sorted(weekly['tierIdentifier'].dropna().unique()))}\n\n"
        f"**Firms:** {weekly['MPID'].dropna().nunique()} unique MPIDs"
    )

st.divider()

# ── Overview charts ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Block Trading: Monthly Share Volume")
    monthly = (
        blocks.groupby("summaryStartDate")["totalShareQuantity"]
        .sum()
        .reset_index()
        .rename(columns={"summaryStartDate": "Month", "totalShareQuantity": "Total Shares"})
    )
    fig = px.area(monthly, x="Month", y="Total Shares")
    fig.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("2026 Weekly Volume by Tier")
    tier_data = (
        weekly[weekly["summaryTypeCode"].isin(["ATS_W_VOL_STATS", "OTC_W_VOL_STATS"])]
        .groupby("tierIdentifier")["totalWeeklyShareQuantity"]
        .sum()
        .reset_index()
    )
    fig2 = px.pie(
        tier_data,
        values="totalWeeklyShareQuantity",
        names="tierIdentifier",
    )
    fig2.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Navigation guide ──────────────────────────────────────────────────────────
st.subheader("Pages")
n1, n2, n3, n4 = st.columns(4)
with n1:
    st.markdown("**📦 Blocks Summary**\nBlock trading stats by firm, size category, and time period (2016–2026).")
with n2:
    st.markdown("**📅 Weekly Summary**\n2026 ATS/OTC weekly volumes across tiers and venues.")
with n3:
    st.markdown("**🏢 Firm Analysis**\nDrill into any market participant's trading activity.")
with n4:
    st.markdown("**📈 Stock Analysis**\nSearch a ticker to see which firms trade it and at what volume.")
