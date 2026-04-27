import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Block Trade Pricer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  /* Dark financial terminal palette */
  :root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #00d4aa;
    --accent2:   #ff6b35;
    --muted:     #8b949e;
    --text:      #e6edf3;
  }

  .stApp { background: var(--bg); }

  .metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
  }
  .metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
  }
  .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 600;
    color: var(--accent);
    line-height: 1;
  }
  .metric-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.3rem;
  }
  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
  }
  .warn-card {
    background: rgba(255,107,53,0.08);
    border: 1px solid rgba(255,107,53,0.35);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #ff9966;
    font-family: 'IBM Plex Mono', monospace;
  }
  .ok-card {
    background: rgba(0,212,170,0.07);
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: var(--accent);
    font-family: 'IBM Plex Mono', monospace;
  }
  /* Sidebar tweaks */
  section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
  }
  div[data-testid="stMetric"] label { color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — INPUTS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📊 Block Trade Pricer")
    st.markdown('<div class="section-header">Stock & Market</div>', unsafe_allow_html=True)

    ticker       = st.text_input("Ticker", value="AAPL")
    side         = st.radio("Side", ["Sell", "Buy"], horizontal=True)
    stock_price  = st.number_input("Mid Price ($)", value=185.0, step=1.0, format="%.2f")
    adv_shares   = st.number_input("ADV (shares, 000s)", value=50_000, step=1_000)
    adv_shares  *= 1_000   # convert to actual shares
    daily_vol    = st.number_input("Daily Volatility (σ, %)", value=1.8, step=0.1, format="%.2f") / 100
    bid_ask_bps  = st.number_input("Bid-Ask Spread (bps)", value=5.0, step=1.0)

    st.markdown('<div class="section-header">Block Parameters</div>', unsafe_allow_html=True)

    block_shares = st.number_input("Block Size (shares, 000s)", value=5_000, step=500) * 1_000
    pov_rate     = st.slider("POV Rate (% of ADV traded per day)", min_value=5, max_value=40, value=20, step=1)
    exec_style   = st.selectbox("Execution Style", ["Principal Bid (Firm)", "Accelerated Bookbuild (ABB)", "VWAP Agency"])

    st.markdown('<div class="section-header">Risk Factors</div>', unsafe_allow_html=True)

    beta          = st.number_input("Stock Beta", value=1.2, step=0.1, format="%.2f")
    short_int_pct = st.slider("Short Interest (% of float)", 0, 30, 5)
    upcoming_cat  = st.checkbox("Upcoming Catalyst (earnings / FDA)", value=False)
    natural_buyer = st.checkbox("Known Natural Buyer Exists", value=False)
    capital_cost  = st.number_input("Cost of Capital / Financing (% p.a.)", value=5.5, step=0.1, format="%.2f") / 100

# ══════════════════════════════════════════════════════════════════════════════
# CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════

adv_notional     = adv_shares * stock_price
block_notional   = block_shares * stock_price
pct_of_adv       = block_shares / adv_shares          # fraction
days_to_distrib  = pct_of_adv / (pov_rate / 100)      # days

# 1. Square-root market impact model
#    impact = σ × √(Q/ADV)  — in price-return terms
market_impact_pct = daily_vol * np.sqrt(pct_of_adv)

# 2. Inventory / vol risk over distribution window
#    Uncertainty of the stock over the holding period
inventory_risk_pct = daily_vol * np.sqrt(days_to_distrib) * beta

# 3. Financing cost over the holding period
financing_cost_pct = capital_cost * (days_to_distrib / 252)

# 4. Bid-ask spread component (half spread, round-trip capture)
bidask_cost_pct = (bid_ask_bps / 10_000) / 2

# 5. Adjustments for risk qualitative factors
adj = 0.0
if short_int_pct > 15:
    adj += 0.005
elif short_int_pct > 8:
    adj += 0.002
if upcoming_cat:
    adj += 0.008
if natural_buyer:
    adj -= 0.003

# Style-specific dealer spread
style_spread = {"Principal Bid (Firm)": 0.008,
                "Accelerated Bookbuild (ABB)": 0.005,
                "VWAP Agency": 0.002}[exec_style]

# Total discount
total_discount_pct = (market_impact_pct
                      + inventory_risk_pct
                      + financing_cost_pct
                      + bidask_cost_pct
                      + style_spread
                      + adj)

# Block price — discount below mid for sell, premium above mid for buy
if side == "Sell":
    block_price     = stock_price * (1 - total_discount_pct)
else:
    block_price     = stock_price * (1 + total_discount_pct)

gross_proceeds  = block_price * block_shares
discount_dollars= abs(stock_price - block_price) * block_shares
bid_price = block_price  # alias kept for heatmap/scenario reuse

# Rule-of-thumb band
lo_discount = total_discount_pct * 0.75
hi_discount = total_discount_pct * 1.30

# Component breakdown for chart
components = {
    "Market Impact":    market_impact_pct * 100,
    "Inventory Risk":   inventory_risk_pct * 100,
    "Financing Cost":   financing_cost_pct * 100,
    "Bid-Ask":          bidask_cost_pct * 100,
    "Dealer Spread":    style_spread * 100,
    "Qualitative Adj":  adj * 100,
}

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
side_color = "#ff6b35" if side == "Sell" else "#00d4aa"
st.markdown(f"""
<div style='display:flex; align-items:baseline; gap:1rem; margin-bottom:0.3rem;'>
  <span style='font-family:"IBM Plex Mono",monospace; font-size:1.6rem; font-weight:600; color:#e6edf3;'>
    {ticker.upper()} — Block Trade Analysis
  </span>
  <span style='font-family:"IBM Plex Mono",monospace; font-size:0.9rem; font-weight:600; color:{side_color};'>
    {side.upper()}
  </span>
  <span style='font-family:"IBM Plex Mono",monospace; font-size:0.8rem; color:#8b949e;'>
    {exec_style}
  </span>
</div>
<div style='font-family:"IBM Plex Mono",monospace; font-size:0.72rem; color:#8b949e; margin-bottom:1.5rem;'>
  {block_shares:,.0f} shares  ·  ${block_notional/1e6:.1f}M notional  ·
  {pct_of_adv*100:.1f}% of ADV  ·  Est. {days_to_distrib:.1f} days @ {pov_rate}% POV
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TOP METRICS ROW
# ══════════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)

def metric_card(label, value, sub=""):
    return f"""
    <div class='metric-card'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value'>{value}</div>
      <div class='metric-sub'>{sub}</div>
    </div>"""

price_label = "Block Bid" if side == "Sell" else "Block Offer"
proceeds_label = "Gross Proceeds" if side == "Sell" else "Total Cost"
discount_label = "Discount to Mid" if side == "Sell" else "Premium to Mid"
per_share_diff = block_price - stock_price if side == "Buy" else stock_price - block_price

c1.markdown(metric_card(price_label, f"${block_price:.2f}", f"vs ${stock_price:.2f} mid"), unsafe_allow_html=True)
c2.markdown(metric_card(discount_label, f"{total_discount_pct*100:.2f}%", f"${per_share_diff:.2f}/share"), unsafe_allow_html=True)
c3.markdown(metric_card(proceeds_label, f"${gross_proceeds/1e6:.1f}M", f"{'Discount' if side == 'Sell' else 'Premium'} ${discount_dollars/1e6:.2f}M"), unsafe_allow_html=True)
c4.markdown(metric_card("Distrib. Days", f"{days_to_distrib:.1f}d", f"@ {pov_rate}% POV"), unsafe_allow_html=True)
c5.markdown(metric_card("Price Range", f"{lo_discount*100:.2f}–{hi_discount*100:.2f}%", "Discount band (±)"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS ROW
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1.1, 1])

# ── Waterfall: discount components ──────────────────────────────────────────
with left:
    st.markdown('<div class="section-header">Discount Component Waterfall</div>', unsafe_allow_html=True)

    labels = list(components.keys()) + ["TOTAL DISCOUNT"]
    vals   = list(components.values())
    total  = sum(vals)

    # build waterfall measures
    measures = ["relative"] * len(vals) + ["total"]
    x_vals   = vals + [total]

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=labels,
        y=x_vals,
        connector={"line": {"color": "#30363d", "width": 1}},
        increasing={"marker": {"color": "#ff6b35"}},
        decreasing={"marker": {"color": "#00d4aa"}},
        totals={"marker": {"color": "#58a6ff", "line": {"color": "#58a6ff", "width": 1}}},
        text=[f"{v:.3f}%" for v in x_vals],
        textposition="outside",
        textfont={"family": "IBM Plex Mono", "size": 11, "color": "#e6edf3"},
    ))
    fig_wf.update_layout(
        plot_bgcolor="#161b22", paper_bgcolor="#161b22",
        font={"family": "IBM Plex Mono", "color": "#8b949e", "size": 11},
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(ticksuffix="%", gridcolor="#21262d", zerolinecolor="#30363d"),
        xaxis=dict(tickfont={"size": 10}),
        showlegend=False,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

# ── POV sensitivity heat-map ─────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-header">Discount vs POV Rate × Block Size</div>', unsafe_allow_html=True)

    pov_range   = np.arange(5, 41, 5)          # 5% … 40%
    block_range = np.linspace(0.05, 0.60, 8)   # 5–60% of ADV

    heatmap_data = np.zeros((len(pov_range), len(block_range)))
    for i, pov in enumerate(pov_range):
        for j, blk_frac in enumerate(block_range):
            d_to_d = blk_frac / (pov / 100)
            mi  = daily_vol * np.sqrt(blk_frac)
            ir  = daily_vol * np.sqrt(d_to_d) * beta
            fc  = capital_cost * (d_to_d / 252)
            heatmap_data[i, j] = (mi + ir + fc + bidask_cost_pct + style_spread + adj) * 100

    fig_hm = go.Figure(go.Heatmap(
        z=heatmap_data,
        x=[f"{int(b*100)}%" for b in block_range],
        y=[f"{int(p)}%" for p in pov_range],
        colorscale=[[0,"#0d3b33"],[0.4,"#00d4aa"],[0.7,"#ff9944"],[1,"#ff3333"]],
        text=np.round(heatmap_data, 2),
        texttemplate="%{text}%",
        textfont={"size": 9, "family": "IBM Plex Mono"},
        showscale=True,
        colorbar=dict(ticksuffix="%", tickfont={"family":"IBM Plex Mono","size":10}, thickness=12),
    ))
    fig_hm.update_layout(
        plot_bgcolor="#161b22", paper_bgcolor="#161b22",
        font={"family": "IBM Plex Mono", "color": "#8b949e", "size": 11},
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Block Size (% ADV)", title_font={"size":10}),
        yaxis=dict(title="POV Rate", title_font={"size":10}),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECOND ROW — Distribution timeline + scenario table
# ══════════════════════════════════════════════════════════════════════════════
col_tl, col_sc = st.columns([1, 1.1])

# ── Distribution timeline ─────────────────────────────────────────────────────
with col_tl:
    st.markdown('<div class="section-header">Intraday Distribution Timeline</div>', unsafe_allow_html=True)

    pov_frac   = pov_rate / 100
    daily_exec = adv_shares * pov_frac            # shares executed per day
    n_days_full = int(np.ceil(days_to_distrib))
    days_axis  = np.linspace(0, n_days_full, 200)
    remaining  = np.maximum(block_shares - daily_exec * days_axis, 0)
    executed   = block_shares - remaining

    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(
        x=days_axis, y=remaining / 1e6,
        name="Remaining", fill="tozeroy",
        line=dict(color="#ff6b35", width=2),
        fillcolor="rgba(255,107,53,0.15)",
    ))
    fig_tl.add_trace(go.Scatter(
        x=days_axis, y=executed / 1e6,
        name="Executed", fill="tozeroy",
        line=dict(color="#00d4aa", width=2),
        fillcolor="rgba(0,212,170,0.10)",
    ))
    fig_tl.add_vline(x=days_to_distrib, line_dash="dot", line_color="#58a6ff",
                     annotation_text=f" Done: {days_to_distrib:.1f}d", annotation_font_size=10)
    fig_tl.update_layout(
        plot_bgcolor="#161b22", paper_bgcolor="#161b22",
        font={"family": "IBM Plex Mono", "color": "#8b949e", "size": 11},
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.08, font={"size":10}),
        xaxis=dict(title="Trading Days", gridcolor="#21262d"),
        yaxis=dict(title="Shares (M)", gridcolor="#21262d", ticksuffix="M"),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

# ── Scenario comparison table ─────────────────────────────────────────────────
with col_sc:
    st.markdown('<div class="section-header">Scenario Comparison</div>', unsafe_allow_html=True)

    scenarios = []
    scen_price_col = "Block Bid" if side == "Sell" else "Block Offer"
    scen_proceeds_col = "Proceeds ($M)" if side == "Sell" else "Cost ($M)"
    for pov_s in [10, 15, 20, 25, 30]:
        d = block_shares / adv_shares / (pov_s / 100)
        mi = daily_vol * np.sqrt(pct_of_adv)
        ir = daily_vol * np.sqrt(d) * beta
        fc = capital_cost * (d / 252)
        tot = mi + ir + fc + bidask_cost_pct + style_spread + adj
        bp = stock_price * (1 - tot) if side == "Sell" else stock_price * (1 + tot)
        scenarios.append({
            "POV Rate":           f"{pov_s}%",
            "Distrib. Days":      f"{d:.1f}",
            "Market Impact":      f"{mi*100:.3f}%",
            "Inventory Risk":     f"{ir*100:.3f}%",
            "Total Disc/Prem":    f"{tot*100:.3f}%",
            scen_price_col:       f"${bp:.2f}",
            scen_proceeds_col:    f"${bp*block_shares/1e6:.1f}M",
        })

    df = pd.DataFrame(scenarios)
    # highlight current POV row
    def highlight_current(row):
        color = "background-color: rgba(0,212,170,0.12); color: #00d4aa" if row["POV Rate"] == f"{pov_rate}%" else ""
        return [color] * len(row)

    styled = (df.style
               .apply(highlight_current, axis=1)
               .set_properties(**{"font-family":"IBM Plex Mono","font-size":"12px"})
               .hide(axis="index"))

    st.dataframe(styled, use_container_width=True, height=230)

# ══════════════════════════════════════════════════════════════════════════════
# RISK FLAGS + SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Risk & Execution Assessment</div>', unsafe_allow_html=True)

flags = []
if pct_of_adv > 0.30:
    flags.append(f"⚠  Block is {pct_of_adv*100:.0f}% of ADV — elevated market impact expected")
if days_to_distrib > 5:
    flags.append(f"⚠  {days_to_distrib:.1f}-day distribution window creates significant overnight inventory risk")
if upcoming_cat:
    flags.append("⚠  Upcoming catalyst — options hedge strongly recommended; widen discount")
if short_int_pct > 15:
    flags.append(f"⚠  High short interest ({short_int_pct}%) — distribution may face covering pressure")
if daily_vol > 0.025:
    flags.append(f"⚠  Elevated daily vol ({daily_vol*100:.1f}%) — inventory risk component is material")
if natural_buyer:
    flags.append("✅  Natural buyer identified — can tighten discount and reduce distribution risk")
if total_discount_pct < 0.01:
    flags.append("✅  Liquid, low-impact trade — competitive principal bid expected")

if not flags:
    flags.append("✅  No significant risk flags detected under current parameters")

f_cols = st.columns(min(len(flags), 3))
for i, flag in enumerate(flags):
    card_class = "ok-card" if flag.startswith("✅") else "warn-card"
    f_cols[i % 3].markdown(f"<div class='{card_class}'>{flag}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER FORMULA NOTE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='font-family:"IBM Plex Mono",monospace; font-size:0.65rem; color:#484f58; border-top:1px solid #21262d; padding-top:0.8rem;'>
  Discount = σ·√(Q/ADV) [market impact]  +  σ·√(T)·β [inventory risk]  +  r·(T/252) [financing]  +  ½·spread  +  dealer margin  +  qualitative adj
  &nbsp;·&nbsp; T = (Q/ADV) / POV  &nbsp;·&nbsp; For indicative purposes only. Not investment advice.
</div>
""", unsafe_allow_html=True)