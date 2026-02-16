"""
STREAMLIT DASHBOARD v3 - Premium Hackathon Edition
===================================================
Agentic AI: Demand Blocking & Smart Rebooking
Features: Glass-morphism UI, ChatGPT-powered AI Insights, 7-Agent Pipeline
Run:  streamlit run dashboard.py --server.port 8502
"""
import os, sys, pickle, logging, subprocess, io, base64, html as html_mod
from fpdf import FPDF
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from utils.db_utils import get_sqlalchemy_engine, read_table, test_connection
from config.thresholds import (
    MIN_BLOCKING_SCORE, SELLOUT_PROBABILITY_THRESHOLD,
    DEMAND_SPIKE_THRESHOLD, SELLOUT_CAPTURE_TARGET,
    MIN_SAVINGS_INR, MAX_RISK_SCORE, MIN_EQUIVALENCE_SCORE,
    BLOCKING_WEIGHT_DEMAND, BLOCKING_WEIGHT_SELLOUT,
    BLOCKING_WEIGHT_REVENUE, BLOCKING_WEIGHT_SUPPLIER,
    BLOCKING_WEIGHT_PRICE,
    RISK_WEIGHT_CANCELLATION, RISK_WEIGHT_SUPPLIER,
    RISK_WEIGHT_EQUIVALENCE, RISK_WEIGHT_TIMING,
    MIN_NET_PROFIT_INR, MIN_DAYS_BEFORE_DEADLINE,
    MAX_PENALTY_FRACTION, MIN_REBOOKING_CONFIDENCE,
    REVENUE_SCORE_THRESHOLD, PRICE_INFLATION_THRESHOLD,
    MAX_BLOCK_FRACTION,
)

st.set_page_config(page_title="Agentic AI", page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><rect width='64' height='64' rx='14' fill='%236366f1'/><text x='32' y='44' font-family='Arial,sans-serif' font-size='32' font-weight='bold' fill='white' text-anchor='middle'>AI</text></svg>", layout="wide", initial_sidebar_state="expanded")

def fmt_inr(n):
    """Format number in Indian numbering system (e.g., 71,59,722)."""
    n = int(round(n))
    is_neg = n < 0
    n = abs(n)
    s = str(n)
    if len(s) <= 3:
        result = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while rest:
            parts.append(rest[-2:] if len(rest) >= 2 else rest)
            rest = rest[:-2]
        result = ",".join(reversed(parts)) + "," + last3
    return ("-" + result) if is_neg else result

INR_COLS = {"old_cost_inr","new_cost_inr","savings_inr","estimated_penalty_inr",
            "net_profit_inr","expected_revenue_uplift_inr","base_adr_inr",
            "expected_uplift_inr","margin_recovered_inr","avg_savings","total_savings"}

def fmt_df_inr(df):
    """Return a copy of df with INR columns formatted in Indian numbering."""
    df = df.copy()
    for c in df.columns:
        if c in INR_COLS:
            df[c] = df[c].apply(lambda v: fmt_inr(v) if pd.notna(v) else v)
    return df

COL_ALIASES = {
    "property_id": "Property Id", "property_name": "Property Name",
    "city": "City", "star_rating": "Star Rating",
    "base_adr_inr": "Base ADR (INR)", "base_inventory_rooms": "Base Inventory",
    "popularity_index": "Popularity Index",
    "supplier_id": "Supplier Id", "supplier_name": "Supplier Name",
    "rooms_blocked_per_night": "Rooms Blocked/Night", "rooms_to_block": "Rooms To Block",
    "block_reason": "Block Reason", "block_start_date": "Block Start",
    "block_end_date": "Block End",
    "expected_revenue_uplift_inr": "Expected Uplift (INR)",
    "expected_uplift_inr": "Expected Uplift (INR)",
    "week_start": "Week Start", "eval_week": "Eval Week",
    "rebooking_eval_id": "Eval Id",
    "room_type": "Room Type", "meal_plan": "Meal Plan",
    "old_supplier": "Old Supplier", "new_supplier": "New Supplier",
    "old_cost_inr": "Old Cost (INR)", "new_cost_inr": "New Cost (INR)",
    "savings_inr": "Savings (INR)",
    "estimated_penalty_inr": "Penalty (INR)", "net_profit_inr": "Net Profit (INR)",
    "risk_score": "Risk Score", "profit_score": "Profit Score",
    "confidence": "Confidence", "decision": "Decision",
    "decision_reasons": "Decision Reasons",
    "booking_failure_rate": "Failure Rate",
    "supplier_cancellation_rate": "Cancellation Rate",
    "dispute_rate": "Dispute Rate",
    "preferred_supplier_flag": "Preferred",
    "avg_savings": "Avg Savings", "total_savings": "Total Savings",
    "avg_risk": "Avg Risk", "avg_conf": "Avg Confidence",
    "date": "Date", "start_date": "Start Date", "end_date": "End Date",
    "event_name": "Event Name", "event_type": "Event Type",
    "demand_intensity": "Demand Intensity",
    "city_demand_multiplier": "Demand Multiplier",
    "seasonality": "Seasonality", "weekend_multiplier": "Weekend Multiplier",
    "event_multiplier": "Event Multiplier",
    "occupancy_rate": "Occupancy Rate", "booking_requests": "Booking Requests",
    "sold_out_flag": "Sold Out", "rooms_sold": "Rooms Sold",
    "total_bookings": "Total Bookings",
    "cancellation_type": "Cancellation Type",
    "cancel_penalty_pct": "Penalty %",
    "equivalence_score": "Equivalence Score",
    "blocking_score": "Blocking Score", "total_score": "Total Score",
    "demand_score": "Demand Score", "sellout_score": "Sellout Score",
    "revenue_score": "Revenue Score", "supplier_score": "Supplier Score",
    "price_score": "Price Score",
    "properties_blocked": "Properties Blocked",
    "total_rooms_blocked": "Total Rooms Blocked",
    "margin_recovered_inr": "Margin Recovered (INR)",
    "rebook_evaluations": "Rebook Evaluations",
    "rebook_count": "Rebook Count",
    "rebooking_rate_pct": "Rebooking Rate %",
    "missed_sellouts": "Missed Sellouts",
    "sellout_capture_pct": "Sellout Capture %",
    "net_rate_inr": "Net Rate (INR)",
    "is_available": "Available",
    "snapshot_date": "Snapshot Date",
}

def friendly_cols(df):
    """Rename columns to friendly names and build column_config with original name as tooltip."""
    rename_map = {c: COL_ALIASES.get(c, c.replace("_", " ").title()) for c in df.columns}
    col_config = {}
    for orig, alias in rename_map.items():
        col_config[alias] = st.column_config.Column(label=alias, help=f"Column: {orig}")
    return df.rename(columns=rename_map), col_config

def st_df(df, **kwargs):
    """Display dataframe as styled HTML table with sticky header and click-to-sort."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(2)
    renamed, _ = friendly_cols(df)
    tip_map = {}
    for orig in df.columns:
        alias = COL_ALIASES.get(orig, orig.replace("_"," ").title())
        tip_map[alias] = orig
    max_h = kwargs.get("height", 420)
    tid = f"tbl_{id(df)}"
    # Build header
    hdr = ""
    for i, col in enumerate(renamed.columns):
        orig = html_mod.escape(tip_map.get(col, col))
        col_esc = html_mod.escape(str(col))
        hdr += (f'<th onclick="sortTbl(\'{tid}\',{i})" title="{orig}" '
                f'style="padding:10px 14px;background:#6366f1;color:#fff;font-weight:600;'
                f'font-size:12px;letter-spacing:0.3px;text-align:left;border-bottom:2px solid #4f46e5;'
                f'white-space:nowrap;cursor:pointer;position:sticky;top:0;z-index:2;'
                f'user-select:none">{col_esc} <span style="opacity:0.5;font-size:10px">⇅</span></th>')
    # Build rows
    rows = ""
    for i, (_, row) in enumerate(renamed.iterrows()):
        bg = '#f8fafc' if i % 2 == 0 else '#ffffff'
        cells = ""
        for col in renamed.columns:
            val = row[col]
            val_str = html_mod.escape(str(val)) if pd.notna(val) else ""
            cells += f'<td style="padding:8px 14px;border-bottom:1px solid #f1f5f9;white-space:nowrap">{val_str}</td>'
        rows += f'<tr style="background:{bg}" onmouseover="this.style.background=\'#eef2ff\'" onmouseout="this.style.background=\'{bg}\'">{cells}</tr>'
    html_out = f'''<div style="max-height:{max_h}px;overflow:auto;border-radius:12px;border:1px solid #e2e8f0;box-shadow:0 2px 8px rgba(0,0,0,0.06)">
<table id="{tid}" style="width:100%;border-collapse:collapse;font-size:13px;font-family:Inter,sans-serif">
<thead><tr>{hdr}</tr></thead><tbody>{rows}</tbody></table></div>
<script>
function sortTbl(id,col){{var t=document.getElementById(id),tb=t.tBodies[0],
rows=Array.from(tb.rows),asc=t.dataset['s'+col]!='1';t.dataset['s'+col]=asc?'1':'0';
rows.sort(function(a,b){{var x=a.cells[col].textContent.trim().replace(/,/g,''),
y=b.cells[col].textContent.trim().replace(/,/g,'');
var nx=parseFloat(x),ny=parseFloat(y);
if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx;
return asc?x.localeCompare(y):y.localeCompare(x);}});
rows.forEach(function(r,i){{tb.appendChild(r);r.style.background=i%2==0?'#f8fafc':'#ffffff';}})}}
</script>'''
    components.html(html_out, height=min(max_h + 10, len(renamed) * 38 + 50), scrolling=False)

# ── ChatGPT Client ──
@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        return OpenAI(api_key=api_key)
    return None

def ask_gpt(prompt, system_msg="You are an expert hotel revenue management AI analyst. Be concise, data-driven, and actionable. Use bullet points. Max 150 words.", max_tokens=300):
    client = get_openai_client()
    if not client:
        return "OpenAI API key not configured. Add OPENAI_API_KEY to .env file."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system_msg},{"role":"user","content":prompt}],
            max_completion_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI Error: {str(e)}"

# ── Premium CSS ──
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

@keyframes gradientShift{
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}
@keyframes subtleFloat{
    0%,100%{transform:translateY(0)}
    50%{transform:translateY(-3px)}
}
@keyframes glowPulse{
    0%,100%{box-shadow:0 0 15px rgba(99,102,241,0.15)}
    50%{box-shadow:0 0 25px rgba(139,92,246,0.25)}
}
/* ═══ Entrance Animations ═══ */
@keyframes revealUp{
    from{opacity:0;transform:translateY(30px)}
    to{opacity:1;transform:translateY(0)}
}
@keyframes revealFade{
    from{opacity:0;transform:scale(0.97)}
    to{opacity:1;transform:scale(1)}
}
@keyframes revealLeft{
    from{opacity:0;transform:translateX(-20px)}
    to{opacity:1;transform:translateX(0)}
}

/* Hide Deploy button, hamburger menu & collapse header space */
[data-testid="stToolbar"]{display:none!important;}
#MainMenu{visibility:hidden!important;}
header[data-testid="stHeader"]{display:none!important;}
.block-container{padding-top:0.5rem!important;}

/* Premium vertical rhythm */
[data-testid="stVerticalBlock"]{gap:0.6rem!important}
[data-testid="stHorizontalBlock"]{gap:0.6rem!important}
[data-testid="stExpander"]{margin-bottom:0.5rem!important}
[data-testid="stMetric"]{margin-bottom:4px!important}
[data-testid="stPlotlyChart"]{margin-top:6px!important;margin-bottom:6px!important}
[data-testid="stDataFrame"]{margin-top:6px!important;margin-bottom:6px!important}


:root{
    --pr:#6366f1;--pr-d:#4f46e5;--pr-l:#818cf8;
    --ok:#10b981;--ok-d:#059669;--ok-l:#34d399;
    --err:#ef4444;--err-d:#dc2626;
    --warn:#f59e0b;--warn-d:#d97706;
    --pur:#8b5cf6;--pur-d:#7c3aed;
    --cyan:#06b6d4;--pink:#ec4899;
    --g50:#f8fafc;--g100:#f1f5f9;--g200:#e2e8f0;--g300:#cbd5e1;
    --g400:#94a3b8;--g500:#64748b;--g600:#475569;--g700:#334155;--g800:#1e293b;--g900:#0f172a;
    --glass:rgba(255,255,255,0.55);--glass-strong:rgba(255,255,255,0.72);
    --glass-border:rgba(255,255,255,0.35);--glass-border-hover:rgba(99,102,241,0.4);
    --shadow-sm:0 1px 3px rgba(0,0,0,0.04);
    --shadow:0 4px 6px -1px rgba(0,0,0,0.06),0 2px 4px -2px rgba(0,0,0,0.04);
    --shadow-lg:0 10px 25px -5px rgba(0,0,0,0.08),0 4px 10px -4px rgba(0,0,0,0.04);
    --shadow-xl:0 20px 40px -8px rgba(0,0,0,0.1),0 8px 16px -6px rgba(0,0,0,0.06);
    --shadow-glow:0 0 20px rgba(99,102,241,0.15),0 0 40px rgba(139,92,246,0.08);
}

html,body,[class*="st-"]{font-family:'Inter',sans-serif!important}

/* Main area - subtle mesh gradient background */
.main{
    background:
        radial-gradient(ellipse at 15% 10%, rgba(99,102,241,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 20%, rgba(139,92,246,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(236,72,153,0.04) 0%, transparent 50%),
        linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
}
.main .block-container{padding:0.5rem 2.5rem 2rem;max-width:1500px}

/* ═══ Sidebar - Frosted Glass Dark ═══ */
@keyframes sidebarOrb{
    0%,100%{transform:translate(0,0) scale(1);opacity:0.12}
    33%{transform:translate(30px,40px) scale(1.2);opacity:0.18}
    66%{transform:translate(-20px,20px) scale(0.9);opacity:0.10}
}
@keyframes borderGlow{
    0%,100%{border-right-color:rgba(99,102,241,0.2)}
    50%{border-right-color:rgba(139,92,246,0.4)}
}
@keyframes logoShimmer{
    0%{background-position:200% center}
    100%{background-position:-200% center}
}
section[data-testid="stSidebar"]{
    background:
        linear-gradient(180deg,
            rgba(15,23,42,0.97) 0%,
            rgba(30,27,75,0.95) 35%,
            rgba(49,46,129,0.93) 65%,
            rgba(15,23,42,0.97) 100%) !important;
    backdrop-filter:blur(20px) saturate(1.8) !important;
    -webkit-backdrop-filter:blur(20px) saturate(1.8) !important;
    border-right:2px solid rgba(99,102,241,0.2) !important;
    box-shadow:4px 0 30px rgba(0,0,0,0.2) !important;
    animation:borderGlow 4s ease-in-out infinite!important;
    overflow:hidden!important;
}
section[data-testid="stSidebar"]::before{
    content:'';position:absolute;top:0;left:0;right:0;bottom:0;
    background:
        radial-gradient(circle at 20% 20%, rgba(99,102,241,0.12) 0%, transparent 50%),
        radial-gradient(circle at 80% 60%, rgba(139,92,246,0.08) 0%, transparent 50%),
        radial-gradient(circle at 40% 90%, rgba(236,72,153,0.06) 0%, transparent 50%);
    pointer-events:none;z-index:0;
}
section[data-testid="stSidebar"]::after{
    content:'';position:absolute;top:10%;left:10%;width:120px;height:120px;
    background:radial-gradient(circle,rgba(99,102,241,0.25),transparent 70%);
    border-radius:50%;filter:blur(30px);pointer-events:none;z-index:0;
    animation:sidebarOrb 8s ease-in-out infinite;
}
section[data-testid="stSidebar"] *{color:#e2e8f0!important;position:relative;z-index:1}
section[data-testid="stSidebar"] [data-testid="stDateInput"] *,
section[data-testid="stSidebar"] [data-testid="stDateInput"] div,
section[data-testid="stSidebar"] [data-testid="stDateInput"] button,
section[data-testid="stSidebar"] [data-testid="stDateInput"] svg,
section[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb]{
    border:none!important;box-shadow:none!important;outline:none!important;
    border-width:0!important;border-style:none!important;
}
section[data-testid="stSidebar"] [data-testid="stDateInput"] button{
    background:transparent!important;animation:none!important;
}
section[data-testid="stSidebar"] [data-testid="stDateInput"] > div > div{
    border:1px solid rgba(99,102,241,0.3)!important;border-radius:10px!important;
    background:rgba(255,255,255,0.95)!important;overflow:hidden!important;
    transition:all 0.3s ease!important;
}
section[data-testid="stSidebar"] [data-testid="stDateInput"] > div > div:focus-within{
    border-color:rgba(139,92,246,0.6)!important;
    box-shadow:0 0 12px rgba(139,92,246,0.2)!important;
}
section[data-testid="stSidebar"] [data-testid="stDateInput"] input{
    color:#0f172a!important;font-weight:600!important;
    background:transparent!important;
}
section[data-testid="stSidebar"] .stRadio label{
    transition:all 0.3s cubic-bezier(0.4,0,0.2,1)!important;border-radius:10px!important;
    padding:4px 10px!important;margin:1px 0!important;
    border:1px solid transparent!important;
}
section[data-testid="stSidebar"] .stRadio label:hover{
    color:#c4b5fd!important;background:rgba(99,102,241,0.12)!important;
    border-color:rgba(99,102,241,0.2)!important;
    transform:translateX(4px)!important;
    box-shadow:0 2px 12px rgba(99,102,241,0.1)!important;
}
section[data-testid="stSidebar"] button{
    background:linear-gradient(135deg,#6366f1,#8b5cf6,#a855f7)!important;
    background-size:200% 200%!important;
    animation:gradientShift 3s ease infinite!important;
    border:none!important;border-radius:12px!important;
    font-weight:700!important;letter-spacing:0.5px!important;
    transition:all 0.3s ease!important;
    box-shadow:0 4px 15px rgba(99,102,241,0.3)!important;
}
section[data-testid="stSidebar"] button:hover{
    transform:translateY(-2px)!important;
    box-shadow:0 6px 20px rgba(99,102,241,0.4)!important;
}
section[data-testid="stSidebar"] button:active{
    transform:translateY(0)!important;
}

/* Hide sidebar collapse/expand buttons */
[data-testid="stSidebarCollapseButton"]{display:none!important}
[data-testid="stExpandSidebarButton"]{display:none!important}
[data-testid="stSidebarCollapsedControl"]{display:none!important}
[data-testid="collapsedControl"]{display:none!important}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3{
    background:linear-gradient(135deg,#a5b4fc,#c4b5fd,#f0abfc);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    font-weight:900!important;letter-spacing:0.8px;font-size:1.1rem!important;
    text-shadow:0 0 30px rgba(165,180,252,0.3);
}

/* ═══ Metric Cards - Premium Glass ═══ */
div[data-testid="stMetric"]{
    background:var(--glass-strong)!important;
    backdrop-filter:blur(16px) saturate(1.5);
    -webkit-backdrop-filter:blur(16px) saturate(1.5);
    padding:20px 22px!important;border-radius:18px!important;
    border:1px solid var(--glass-border)!important;
    box-shadow:var(--shadow-lg),inset 0 1px 0 rgba(255,255,255,0.5)!important;
    transition:all 0.35s cubic-bezier(0.25,0.46,0.45,0.94);
    position:relative;overflow:hidden;
}
div[data-testid="stMetric"]::before{
    content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,var(--pr),var(--pur),var(--pink));
    background-size:200% 100%;animation:gradientShift 4s ease infinite;
    border-radius:18px 18px 0 0;opacity:0;transition:opacity 0.3s ease;
}
div[data-testid="stMetric"]:hover{
    transform:translateY(-4px);
    box-shadow:var(--shadow-xl),var(--shadow-glow),inset 0 1px 0 rgba(255,255,255,0.6)!important;
    border-color:var(--glass-border-hover)!important;
}
div[data-testid="stMetric"]:hover::before{opacity:1}
div[data-testid="stMetric"] label{
    font-size:9px!important;text-transform:uppercase;letter-spacing:1.2px;
    color:var(--g500)!important;font-weight:700!important;
    white-space:nowrap;overflow:visible!important;text-overflow:unset!important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{
    font-size:22px!important;font-weight:900!important;
    background:linear-gradient(135deg,var(--pr),var(--pur));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    white-space:nowrap!important;overflow:visible!important;text-overflow:unset!important;
}
div[data-testid="stMetric"]:hover [data-testid="stMetricValue"]{
    font-size:24px!important;
    transition:font-size 0.3s ease;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] div{
    overflow:visible!important;text-overflow:unset!important;white-space:nowrap!important;
}

/* ═══ Headings ═══ */
h1{
    font-weight:900!important;font-size:1.8rem!important;margin-top:0.6rem!important;margin-bottom:0.4rem!important;
    background:linear-gradient(135deg,var(--pr) 0%,var(--pur) 40%,var(--pink) 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    letter-spacing:-0.5px;position:relative;
}
h2{
    font-weight:900!important;color:var(--g800)!important;font-size:1.6rem!important;
    letter-spacing:-0.3px;margin-top:0.5rem!important;margin-bottom:0.3rem!important;
}
h3{font-weight:700!important;color:var(--g700)!important;margin-top:0.3rem!important;margin-bottom:0.2rem!important}

/* ═══ Info Boxes - Glass ═══ */
.ibox{
    padding:14px 18px;border-radius:14px;margin-top:4px;margin-bottom:10px;font-size:13.5px;line-height:1.7;
    backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
    box-shadow:var(--shadow-sm),inset 0 1px 0 rgba(255,255,255,0.4);
    border:1px solid rgba(255,255,255,0.2);
    transition:all 0.3s ease;
}
.ibox:hover{box-shadow:var(--shadow)}
.ib{background:linear-gradient(135deg,rgba(239,246,255,0.85),rgba(224,231,255,0.85));border-left:4px solid var(--pr)}
.ig{background:linear-gradient(135deg,rgba(236,253,245,0.85),rgba(209,250,229,0.85));border-left:4px solid var(--ok)}
.ia{background:linear-gradient(135deg,rgba(255,251,235,0.85),rgba(254,243,199,0.85));border-left:4px solid var(--warn)}
.ir{background:linear-gradient(135deg,rgba(254,242,242,0.85),rgba(254,226,226,0.85));border-left:4px solid var(--err)}
.ip{background:linear-gradient(135deg,rgba(245,243,255,0.85),rgba(237,233,254,0.85));border-left:4px solid var(--pur)}

/* ═══ Agent Pipeline Cards ═══ */
.agent-card{
    background:var(--glass-strong);
    backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
    border-radius:18px;padding:22px 18px;text-align:center;
    border:1px solid var(--glass-border);
    box-shadow:var(--shadow),inset 0 1px 0 rgba(255,255,255,0.5);
    transition:all 0.4s cubic-bezier(0.25,0.46,0.45,0.94);
    position:relative;overflow:hidden;
}
.agent-card::after{
    content:'';position:absolute;bottom:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,var(--pr),var(--pur),var(--pink));
    background-size:200% 100%;animation:gradientShift 4s ease infinite;
    opacity:0;transition:opacity 0.3s ease;
}
.agent-card:hover{
    transform:translateY(-6px);
    box-shadow:var(--shadow-xl),var(--shadow-glow);
    border-color:var(--glass-border-hover);
}
.agent-card:hover::after{opacity:1}
.agent-step{
    font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:1.5px;
    margin-bottom:8px;
    background:linear-gradient(135deg,var(--pr),var(--pur));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.agent-name{font-weight:800;font-size:14px;color:var(--g900);margin-bottom:8px}
.agent-desc{font-size:11px;color:var(--g500);line-height:1.6}

/* ═══ AI Insight Box ═══ */
.ai-box{
    background:linear-gradient(135deg,rgba(15,23,42,0.95),rgba(30,27,75,0.95));
    backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
    border-radius:18px;padding:22px 26px;margin:14px 0;
    border:1px solid rgba(139,92,246,0.25);
    box-shadow:0 0 30px rgba(139,92,246,0.12),0 10px 30px rgba(0,0,0,0.15);
    position:relative;overflow:hidden;
}
.ai-box::before{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,var(--pr),var(--pur),var(--pink),var(--cyan));
    background-size:300% 100%;animation:gradientShift 6s ease infinite;
}
.ai-box .ai-title{
    font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:2px;
    background:linear-gradient(135deg,#a5b4fc,#c4b5fd,#f0abfc);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    margin-bottom:12px;
}
.ai-box .ai-content{color:#cbd5e1;font-size:13.5px;line-height:1.9}
.ai-box .ai-content strong{color:#e2e8f0}
.ai-box .ai-content li{margin-bottom:5px}

/* ═══ Divider ═══ */
.sdiv{
    border:none;margin:18px 0;padding:0;height:1px;
    background:linear-gradient(90deg,transparent,var(--g200),transparent);
}

/* ═══ Hero Banner ═══ */
.hero-banner{
    background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 35%,#312e81 65%,#4c1d95 100%);
    border-radius:22px;padding:32px 40px;margin-bottom:20px;
    border:1px solid rgba(139,92,246,0.2);
    box-shadow:0 0 50px rgba(99,102,241,0.15),0 20px 40px rgba(0,0,0,0.12);
    position:relative;overflow:hidden;
}
.hero-banner::before{
    content:'';position:absolute;top:-50%;right:-20%;width:60%;height:200%;
    background:radial-gradient(circle,rgba(139,92,246,0.12) 0%,transparent 60%);
    pointer-events:none;
}
.hero-banner::after{
    content:'';position:absolute;bottom:-30%;left:-10%;width:50%;height:150%;
    background:radial-gradient(circle,rgba(99,102,241,0.08) 0%,transparent 60%);
    pointer-events:none;
}
.hero-banner h1{
    font-size:2.1rem!important;margin-bottom:10px;position:relative;z-index:1;
    background:linear-gradient(135deg,#e0e7ff,#c4b5fd,#f0abfc)!important;
    -webkit-background-clip:text!important;-webkit-text-fill-color:transparent!important;
}
.hero-banner p{color:#94a3b8;font-size:14.5px;line-height:1.8;margin:4px 0 0;position:relative;z-index:1}
.hero-banner .hero-tag{
    display:inline-block;
    background:rgba(99,102,241,0.15);color:#a5b4fc;
    padding:5px 16px;border-radius:24px;font-size:11px;font-weight:600;
    letter-spacing:0.5px;margin-top:14px;
    border:1px solid rgba(99,102,241,0.25);
    backdrop-filter:blur(8px);
    position:relative;z-index:1;
}

/* ═══ Buttons ═══ */
.stButton>button{
    background:linear-gradient(135deg,var(--pr),var(--pur))!important;
    color:white!important;border:none!important;border-radius:12px!important;
    font-weight:600!important;letter-spacing:0.3px;
    transition:all 0.35s cubic-bezier(0.25,0.46,0.45,0.94)!important;
    box-shadow:0 2px 8px rgba(99,102,241,0.25)!important;
    position:relative;overflow:hidden;
}
.stButton>button::before{
    content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.15),transparent);
    transition:left 0.5s ease;
}
.stButton>button:hover{
    transform:translateY(-2px);
    box-shadow:0 6px 20px rgba(99,102,241,0.4)!important;
}
.stButton>button:hover::before{left:100%}

/* ═══ Expanders ═══ */
.streamlit-expanderHeader{
    font-weight:700!important;color:var(--g800)!important;
    background:var(--glass)!important;border-radius:12px!important;
}

/* ═══ Dataframes ═══ */
[data-testid="stDataFrame"]{
    border-radius:14px;overflow:hidden;
    box-shadow:var(--shadow-lg);
    border:1px solid var(--glass-border);
    transition:box-shadow 0.3s ease;
}
[data-testid="stDataFrame"]:hover{
    box-shadow:var(--shadow-xl);
}

/* ═══ Select boxes / inputs ═══ */
[data-testid="stSelectbox"] > div > div{
    border-radius:12px!important;
    border:1px solid var(--g200)!important;
    transition:all 0.3s ease!important;
}
[data-testid="stSelectbox"] > div > div:focus-within{
    border-color:var(--pr-l)!important;
    box-shadow:0 0 0 3px rgba(99,102,241,0.12)!important;
}
[data-testid="stMultiSelect"] > div > div{
    border-radius:12px!important;
    border:1px solid var(--g200)!important;
    transition:all 0.3s ease!important;
}
[data-testid="stMultiSelect"] > div > div:focus-within{
    border-color:var(--pr-l)!important;
    box-shadow:0 0 0 3px rgba(99,102,241,0.12)!important;
}
[data-testid="stTextArea"] textarea{
    border-radius:12px!important;
    border:1px solid var(--g200)!important;
    transition:all 0.3s ease!important;
}
[data-testid="stTextArea"] textarea:focus{
    border-color:var(--pr-l)!important;
    box-shadow:0 0 0 3px rgba(99,102,241,0.12)!important;
}
pre,code{
    border-radius:10px!important;
    font-family:'JetBrains Mono',monospace!important;
}

/* ═══ Tabs ═══ */
.stTabs [data-baseweb="tab-list"]{
    background:var(--glass);backdrop-filter:blur(8px);
    border-radius:12px;padding:4px;gap:4px;
    border:1px solid var(--glass-border);
}
.stTabs [data-baseweb="tab"]{
    border-radius:8px!important;font-weight:600!important;
    transition:all 0.3s ease!important;
}
.stTabs [aria-selected="true"]{
    background:linear-gradient(135deg,var(--pr),var(--pur))!important;
    color:white!important;
}

/* ═══ Page Navigation Buttons ═══ */
.page-nav-container{display:flex;justify-content:space-between;align-items:center;margin-top:40px;padding:20px 0;border-top:2px solid var(--g200)}
.page-nav-btn{
    display:inline-flex;align-items:center;gap:8px;
    padding:14px 28px;border-radius:14px;font-weight:700;font-size:14px;
    text-decoration:none;cursor:pointer;
    transition:all 0.4s cubic-bezier(0.25,0.46,0.45,0.94);
    border:1px solid rgba(99,102,241,0.2);
    position:relative;overflow:hidden;
    backdrop-filter:blur(8px);
}
.page-nav-btn::before{
    content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.2),transparent);
    transition:left 0.5s ease;
}
.page-nav-btn:hover::before{left:100%}
.nav-prev{
    background:linear-gradient(135deg,rgba(30,41,59,0.9),rgba(51,65,85,0.9));color:#e2e8f0;
}
.nav-prev:hover{transform:translateX(-4px);box-shadow:0 8px 25px rgba(30,41,59,0.35);border-color:var(--pr-l)}
.nav-prev .nav-arrow{transition:transform 0.3s ease}
.nav-prev:hover .nav-arrow{transform:translateX(-4px)}
.nav-next{
    background:linear-gradient(135deg,var(--pr),var(--pur));color:white;
    box-shadow:0 4px 12px rgba(99,102,241,0.25);
}
.nav-next:hover{transform:translateX(4px);box-shadow:0 8px 25px rgba(99,102,241,0.4);border-color:var(--pr-l)}
.nav-next .nav-arrow{transition:transform 0.3s ease}
.nav-next:hover .nav-arrow{transform:translateX(4px)}
.nav-label{font-size:10px;text-transform:uppercase;letter-spacing:1px;opacity:0.7;display:block}
.nav-title{display:block;margin-top:2px}
.nav-spacer{flex:1}

/* ═══ Scrollbar ═══ */
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:linear-gradient(180deg,var(--pr-l),var(--pur));border-radius:10px}
::-webkit-scrollbar-thumb:hover{background:linear-gradient(180deg,var(--pr),var(--pur-d))}

/* ═══ Toast ═══ */
[data-testid="stToast"]{
    backdrop-filter:blur(12px)!important;
    border-radius:14px!important;
    border:1px solid var(--glass-border)!important;
}

/* ═══ Plotly charts ═══ */
[data-testid="stPlotlyChart"]{
    border-radius:14px;overflow:hidden;
    box-shadow:var(--shadow);
    border:1px solid var(--glass-border);
    transition:box-shadow 0.3s ease,transform 0.3s ease;
}
[data-testid="stPlotlyChart"]:hover{
    box-shadow:var(--shadow-lg);
    transform:translateY(-2px);
}
</style>""", unsafe_allow_html=True)


def ibox(txt, c="b"):
    st.markdown(f'<div class="ibox i{c}">{txt}</div>', unsafe_allow_html=True)

def sdiv():
    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)

def spc(px=12):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)

def ai_insight_box(content):
    st.markdown(f'''<div class="ai-box">
        <div class="ai-title">AI-Powered Insight</div>
        <div class="ai-content">{content}</div>
    </div>''', unsafe_allow_html=True)

def hero_banner(title, subtitle, tag=""):
    tag_html = f'<span class="hero-tag">{tag}</span>' if tag else ""
    st.markdown(f'''<div class="hero-banner">
        <h1>{title}</h1>
        <p>{subtitle}</p>
        {tag_html}
    </div>''', unsafe_allow_html=True)

PAGE_ORDER = [
    "Executive Overview", "How It Works", "ML Models & Data",
    "Sense & Predict Agent", "Decide & Reserve Agent",
    "Monitor & Optimize Agent", "Report Agent",
    "Property Explorer", "Revenue AI Insights",
]

def page_nav(current_page):
    idx = PAGE_ORDER.index(current_page) if current_page in PAGE_ORDER else 0
    prev_page = PAGE_ORDER[idx - 1] if idx > 0 else None
    next_page = PAGE_ORDER[idx + 1] if idx < len(PAGE_ORDER) - 1 else None
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_l:
        if prev_page:
            if st.button(f"<< {prev_page}", key=f"nav_prev_{current_page}", use_container_width=True):
                st.session_state["nav_page"] = prev_page
                st.rerun()
    with col_r:
        if next_page:
            if st.button(f"{next_page} >>", key=f"nav_next_{current_page}", use_container_width=True):
                st.session_state["nav_page"] = next_page
                st.rerun()

# ── Data ──
@st.cache_resource
def get_engine():
    return get_sqlalchemy_engine()

@st.cache_data(ttl=3600, show_spinner=False)
def load_table(k):
    try: return read_table(k, engine=get_engine())
    except: return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_all():
    d = {}
    for t in ["property_master","supplier_reliability","events_calendar","city_demand_signals",
              "property_daily","room_mapping","rate_snapshots","confirmed_bookings",
              "demand_block_actions","rebooking_evaluations","weekly_demand_bycity","weekly_kpi_summary"]:
        d[t] = load_table(t)
    return d

# ── Sidebar ──
def sidebar():
    with st.sidebar:
        st.markdown("""<div style='text-align:center;padding:14px 0 8px'>
            <div style='font-size:30px;font-weight:900;letter-spacing:2px;
                background:linear-gradient(90deg,#a5b4fc,#c4b5fd,#f0abfc,#a5b4fc);
                background-size:300% 100%;
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                animation:logoShimmer 4s linear infinite;
                margin-bottom:6px'>AGENTIC AI</div>
            <div style='display:flex;align-items:center;justify-content:center;gap:8px;margin:4px 0'>
                <span style='flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(165,180,252,0.4))'></span>
                <span style='font-size:9px;font-weight:700;letter-spacing:2.5px;color:#94a3b8!important;
                    text-transform:uppercase;white-space:nowrap'>Demand Blocking & Smart<br>Rebooking Intelligence</span>
                <span style='flex:1;height:1px;background:linear-gradient(90deg,rgba(165,180,252,0.4),transparent)'></span>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:2px;background:linear-gradient(90deg,transparent,rgba(99,102,241,0.4),rgba(139,92,246,0.4),transparent);margin:8px 0;border-radius:2px'></div>", unsafe_allow_html=True)
        ws = st.date_input("Week Start", value=datetime(2026,2,16), format="DD/MM/YYYY")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Refresh Data", use_container_width=True):
            current_page = st.session_state.get("nav_radio", PAGE_ORDER[0])
            st.cache_data.clear()
            st.session_state["nav_radio"] = current_page
            st.rerun()
        st.markdown("<div style='height:2px;background:linear-gradient(90deg,transparent,rgba(99,102,241,0.4),rgba(139,92,246,0.4),transparent);margin:8px 0;border-radius:2px'></div>", unsafe_allow_html=True)
        try: ok = test_connection(get_engine())
        except: ok = False
        
        if "nav_page" in st.session_state:
            st.session_state["nav_radio"] = st.session_state.pop("nav_page")
        st.markdown("""<div style='font-size:10px;font-weight:800;letter-spacing:3px;color:#a5b4fc!important;
            text-transform:uppercase;margin-bottom:6px;text-align:center'>Navigation</div>""", unsafe_allow_html=True)
        pg = st.radio("NAVIGATION", PAGE_ORDER, key="nav_radio", label_visibility="collapsed")
        st.markdown("<div style='height:2px;background:linear-gradient(90deg,transparent,rgba(99,102,241,0.4),rgba(139,92,246,0.4),transparent);margin:12px 0;border-radius:2px'></div>", unsafe_allow_html=True)
        return pg, str(ws)

# ═══════════════════════════════════════════════════════
# PAGE 0: BUSINESS CONTEXT
# ═══════════════════════════════════════════════════════
def pg_business_context(D, ws):
    hero_banner("Business Context",
                "Why This Matters — Understanding the revenue leakage problem in a travel company",
                "Data-Driven Insights")

    pd_df = D["property_daily"]
    cb = D["confirmed_bookings"]
    rs = D["rate_snapshots"]
    total_records = len(pd_df) + len(cb) + len(rs)
    ibox(f"<b>Data Source:</b> Confirmed_Bookings + Property_Daily + Rate_Snapshots (MS SQL) "
         f"&nbsp;|&nbsp; <b>{total_records:,}</b> records", "g")
    spc(8)

    # ── The Problem ──
    st.markdown("## The Problem")
    ibox("When travel companies book hotels for corporate clients or guests, revenue is silently lost through "
         "<b>two critical leakage patterns</b> — one before booking and one after. Our AI agents tackle both.", "r")

    cols = st.columns(2)
    for i, (step, name, desc) in enumerate([
        ("Problem 1", "Demand Spike → Sell Out",
         "When high-demand hotels sell out before we act → Lost bookings, missed revenue, customer dissatisfaction"),
        ("Problem 2", "Price Drop After Booking",
         "When booked prices drop after confirmation → Margin leakage, opportunity cost, competitive disadvantage"),
    ]):
        with cols[i]:
            st.markdown(f'<div class="agent-card" style="border:1.5px solid rgba(99,102,241,0.25)">'
                        f'<div class="agent-step" style="font-size:12px;letter-spacing:2px;margin-bottom:10px">{step}</div>'
                        f'<div class="agent-name" style="font-size:18px;margin-bottom:12px">{name}</div>'
                        f'<div class="agent-desc" style="font-size:16px;font-weight:600;line-height:1.8">{desc}</div></div>',
                        unsafe_allow_html=True)
    spc(6)
    sdiv()

    # ── Visual Explanation ──
    st.markdown("## How Revenue Leaks")
    ibox("<b>Two visual timelines</b> showing exactly how each problem causes financial loss. "
         "Left: demand exceeds supply. Right: market rate drops below your booked price.", "b")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Demand Spike → Sell Out")
        days = [f"Day {i}" for i in range(1, 8)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=[3,4,5,7,9,10,12], mode="lines+markers",
                                 name="Demand", line=dict(color="#ef4444", width=2.5),
                                 marker=dict(size=7), fill="tozeroy",
                                 fillcolor="rgba(239,68,68,0.08)"))
        fig.add_trace(go.Scatter(x=days, y=[10]*7, mode="lines",
                                 name="Available Inventory", line=dict(color="#6366f1", width=2, dash="dash")))
        fig.add_annotation(x="Day 6", y=10, text="Capacity hit!", showarrow=True,
                          arrowhead=2, arrowcolor="#ef4444", font=dict(color="#ef4444", size=11))
        fig.update_layout(height=360, template="plotly_white", title="The Sell-Out Timeline",
                         xaxis_title="Days", yaxis_title="Rooms",
                         legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5),
                         margin=dict(t=40, b=80))
        st.plotly_chart(fig, use_container_width=True)
        ibox("<b>Lost Booking:</b> Customer request arrives on Day 7 → "
             "No rooms available → Revenue lost", "a")

    with c2:
        st.markdown("### Price Drop After Booking")
        days2 = ["Booking Day", "Day+1", "Day+3", "Day+5", "Day+7", "Day+10", "Day+14"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=days2, y=[5000]*7, mode="lines",
                                  name="Your Booked Price", line=dict(color="#f59e0b", width=2.5)))
        fig2.add_trace(go.Scatter(x=days2, y=[5000,4800,4500,4200,3900,3800,3850],
                                  mode="lines+markers", name="Market Rate",
                                  line=dict(color="#10b981", width=2.5), marker=dict(size=7),
                                  fill="tozeroy", fillcolor="rgba(16,185,129,0.06)"))
        fig2.update_layout(height=360, template="plotly_white", title="Price Drop Timeline",
                          xaxis_title="Days After Booking", yaxis_title="Rate (INR)",
                          legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5),
                          margin=dict(t=40, b=80))
        st.plotly_chart(fig2, use_container_width=True)
        ibox("<b>Margin Leakage:</b> You paid INR 5,000, but market dropped to INR 3,850 = "
             "<b>INR 1,150 lost per night</b>", "a")

    spc(6)
    sdiv()

    # ── Our Solution ──
    st.markdown("## Our Solution")
    ibox("<b>Agentic AI Pipeline</b> — a fully autonomous 7-agent system that addresses "
         "both revenue leakage problems with zero human intervention.", "b")

    solutions = [
        ("Sell-out Prevention", "SENSE & PREDICT",
         "Detect demand spikes early using 3 ML models. DECIDE & RESERVE agents proactively block rooms before they sell out."),
        ("Smart Rebooking", "MONITOR & OPTIMIZE",
         "Continuously watch rates after booking. Trigger rebooking when cheaper alternatives appear, recovering lost margin."),
    ]
    cols = st.columns(2)
    for i, (step, name, desc) in enumerate(solutions):
        with cols[i]:
            st.markdown(f'<div class="agent-card" style="border:1.5px solid rgba(99,102,241,0.25)">'
                        f'<div class="agent-step" style="font-size:12px;letter-spacing:2px;margin-bottom:10px">{step}</div>'
                        f'<div class="agent-name" style="font-size:18px;margin-bottom:12px">{name}</div>'
                        f'<div class="agent-desc" style="font-size:16px;font-weight:600;line-height:1.8">{desc}</div></div>',
                        unsafe_allow_html=True)
    spc(8)
    page_nav("Business Context")

# ═══════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════
def pg_overview(D, ws):
    # Agent pipeline flow inside hero banner
    steps = [
        ("SENSE",   "#38bdf8","rgba(56,189,248,0.55)","rgba(56,189,248,0.7)"),
        ("PREDICT", "#a78bfa","rgba(167,139,250,0.55)","rgba(167,139,250,0.7)"),
        ("DECIDE",  "#f472b6","rgba(244,114,182,0.55)","rgba(244,114,182,0.7)"),
        ("RESERVE", "#fb923c","rgba(251,146,60,0.55)","rgba(251,146,60,0.7)"),
        ("MONITOR", "#34d399","rgba(52,211,153,0.55)","rgba(52,211,153,0.7)"),
        ("OPTIMIZE","#fbbf24","rgba(251,191,36,0.55)","rgba(251,191,36,0.7)"),
        ("REPORT",  "#60a5fa","rgba(96,165,250,0.55)","rgba(96,165,250,0.7)"),
    ]
    badges = []
    for j, (s, clr, bg, bdr) in enumerate(steps):
        badges.append(
            f'<span style="display:inline-block;padding:6px 14px;border-radius:8px;'
            f'background:{bg};border:1px solid {bdr};color:#ffffff;font-weight:700;'
            f'font-size:11px;letter-spacing:1.2px;'
            f'box-shadow:0 2px 8px {bg}">{s}</span>'
        )
        if j < len(steps) - 1:
            badges.append('<span style="color:#ffffff;font-size:14px;margin:0 4px">›</span>')
    flow_html = "".join(badges)
    st.markdown(f'''<div class="hero-banner">
        <h1>Executive Overview <span style="font-size:18px;font-weight:600;color:#a5b4fc;margin-left:12px">| Week: {ws}</span></h1>
        <p>Real-time intelligence from the 7-agent autonomous pipeline. 
        ML-driven demand blocking & smart rebooking intelligence with zero human intervention.</p>
        <div style="margin-top:16px;display:flex;align-items:center;flex-wrap:wrap;gap:6px">
            {flow_html}</div>
    </div>''', unsafe_allow_html=True)

    kpi = D["weekly_kpi_summary"]
    if not kpi.empty:
        kpi["week_start"] = pd.to_datetime(kpi["week_start"])
        wk = kpi[kpi["week_start"]==pd.to_datetime(ws)]
        r = wk.iloc[-1] if not wk.empty else kpi.iloc[-1]
        c1,c2,c3 = st.columns(3)
        c1.metric("Properties Blocked", int(r.get("properties_blocked",0)), help="Unique properties where rooms were proactively reserved")
        c2.metric("Rooms Blocked", int(r.get("total_rooms_blocked",0)), help="Total room-nights blocked across all properties")
        c3.metric("Expected Uplift", f"INR {fmt_inr(r.get('expected_uplift_inr',0))}", help="Projected revenue from selling blocked rooms at peak")
        spc(4)
        c4,c5,c6 = st.columns(3)
        c4.metric("Rebook Evaluations", int(r.get("rebook_evaluations",0)), help="Bookings evaluated for cheaper alternatives")
        c5.metric("Rebookings Done", int(r.get("rebook_count",0)), help="Bookings successfully switched to cheaper supplier")
        c6.metric("Margin Recovered", f"INR {fmt_inr(r.get('margin_recovered_inr',0))}", help="Net savings after penalties from rebooking")

    else:
        st.info("No KPI data yet. Run: `python main.py --week-start 2026-02-16 --train`"); return

    sdiv()

    cl,cr = st.columns(2)
    with cl:
        st.markdown("## Blocking by City")
        ibox("Each bar = total expected revenue uplift from blocking in that city. Color = rooms blocked.","g")
        bl = D["demand_block_actions"]
        if not bl.empty:
            bl["week_start"]=pd.to_datetime(bl["week_start"])
            cb=bl.groupby("city").agg(rooms=("rooms_blocked_per_night","sum"),uplift=("expected_revenue_uplift_inr","sum")).reset_index().sort_values("uplift",ascending=False)
            fig=px.bar(cb,x="city",y="uplift",color="rooms",color_continuous_scale="Blues",labels={"uplift":"Uplift (INR)","city":"City"})
            fig.update_layout(height=360,template="plotly_white",margin=dict(t=20))
            st.plotly_chart(fig,use_container_width=True)
    with cr:
        st.markdown("## Rebooking Decisions")
        ibox("<b>Green</b> = rebooked (margin saved). <b>Red</b> = skipped (too risky). Healthy target: >30% rebook rate.","g")
        ev = D["rebooking_evaluations"]
        if not ev.empty:
            dc=ev["decision"].value_counts().reset_index(); dc.columns=["Decision","Count"]
            fig=px.pie(dc,names="Decision",values="Count",color="Decision",color_discrete_map={"Rebook":"#10b981","Skip":"#ef4444"},hole=.45)
            fig.update_layout(height=360,template="plotly_white",margin=dict(t=20))
            st.plotly_chart(fig,use_container_width=True)
    page_nav("Executive Overview")

# ═══════════════════════════════════════════════════════
# PAGE 2: HOW IT WORKS
# ═══════════════════════════════════════════════════════
def pg_how(D, ws):
    st.markdown("# How It Works")
    ibox("This page explains <b>every component</b> of the Agentic AI system -- the problem, the 7 agents, "
         "the 3 ML models, blocking strategy, and rebooking logic. Read this to understand <i>why</i> each decision is made.","p")
    spc(6)
    sdiv()

    st.markdown("## The Problem")
    ibox("When travel companies book hotels for corporate clients or guests, revenue is silently lost through "
         "<b>two critical leakage patterns</b> — one before booking and one after. Our AI agents tackle both.", "r")

    cols = st.columns(2)
    for i, (step, name, desc) in enumerate([
        ("Problem 1", "Demand Spike → Sell Out",
         "When high-demand hotels sell out before we act → Lost bookings, missed revenue, customer dissatisfaction"),
        ("Problem 2", "Price Drop After Booking",
         "When booked prices drop after confirmation → Margin leakage, opportunity cost, competitive disadvantage"),
    ]):
        with cols[i]:
            st.markdown(f'<div class="agent-card" style="border:1.5px solid rgba(99,102,241,0.25)">'
                        f'<div class="agent-step" style="font-size:15px;letter-spacing:2px;margin-bottom:10px;font-weight:800">{step}</div>'
                        f'<div class="agent-name" style="font-size:20px;font-weight:800;margin-bottom:12px">{name}</div>'
                        f'<div class="agent-desc" style="font-size:16px;font-weight:600;line-height:1.8">{desc}</div></div>',
                        unsafe_allow_html=True)
    spc(6)
    sdiv()

    # ─── Problem 1: DEMAND BLOCKING ───
    st.markdown("# Problem 1: Demand Blocking Workflow")
    ibox("<b>Goal:</b> Proactively reserve hotel rooms <i>before</i> demand spikes, "
         "so the company has inventory when competitors sell out. "
         "The system predicts which properties will spike, scores them, and blocks rooms automatically.", "p")
    sdiv()

    st.markdown("## Step 1: SENSE Agent -- Data Ingestion")
    ibox("""
<b>What happens:</b> The Sense Agent loads <b>12 database tables</b> and engineers 50+ ML features.<br><br>
<b>Key tables loaded:</b><br>
- <code>City_Demand_Signals</code> -- daily demand multiplier per city (1.0 = normal)<br>
- <code>Property_Daily</code> -- occupancy, bookings, sell-out flags per property<br>
- <code>Rate_Snapshots</code> -- supplier rates and availability<br>
- <code>Events_Calendar</code> -- festivals, conferences, holidays<br>
- <code>Supplier_Reliability</code> -- failure rates, cancellation rates<br>
- <code>Property_Master</code> -- hotel details (city, tier, stars, ADR, popularity)<br><br>
<b>Features engineered:</b> Rolling averages (7/14/30 day), lag features, temporal (day of week, weekend),
event proximity (days to nearest event + intensity), occupancy signals, price volatility.
""", "b")

    tbl_counts = {k: len(v) for k, v in D.items() if not v.empty}
    if tbl_counts:
        st.markdown("**Live data loaded:**")
        count_df = pd.DataFrame([{"Table": k.replace("_"," ").title(), "Rows": f"{v:,}"} for k, v in tbl_counts.items()])
        st.dataframe(count_df, use_container_width=True, hide_index=True, height=200)
    sdiv()

    st.markdown("## Step 2: PREDICT Agent -- 3 ML Models")
    ibox("""
<b>What happens:</b> Three ML models are trained on historical data and generate predictions for the next 14 days.<br><br>
<b>Model 1: Demand Spike</b> (XGBoost Regressor)<br>
- Input: city demand lags, seasonality, events, weekends<br>
- Output: <code>predicted_demand_index</code> (1.0 = normal, >1.3 = spike)<br><br>
<b>Model 2: Sell-out Probability</b> (XGBoost Classifier)<br>
- Input: occupancy trends, booking velocity, star rating, events<br>
- Output: <code>sellout_probability</code> (0.0 to 1.0)<br><br>
<b>Model 3: Price Movement</b> (XGBoost Regressor)<br>
- Input: rate history, demand signals, supplier info<br>
- Output: <code>predicted_rate_7d</code> (expected rate in 7 days)
""", "p")

    bl = D["demand_block_actions"]
    if not bl.empty:
        bl["week_start"] = pd.to_datetime(bl["week_start"])
        bl_wk = bl[bl["week_start"] == pd.to_datetime(ws)]
        sample = bl_wk if not bl_wk.empty else bl
        ex = sample.iloc[0]
        prop_id = ex.get("property_id", "PROP_001")
        city = ex.get("city", "Mumbai")

        st.markdown(f"**Example from Data:** Property `{prop_id}` in `{city}`")
        ibox(f"""
Predictions generated for <code>{prop_id}</code> ({city}):<br>
- Demand multiplier for {city}: e.g. <b>1.25</b> (above normal)<br>
- Sell-out probability: e.g. <b>0.15</b> (15% chance of selling out)<br>
- Current rate: e.g. <b>INR 4,500</b> | Predicted 7-day rate: e.g. <b>INR 5,200</b> (price rising)
""", "g")
    sdiv()

    st.markdown("## Step 3: DECIDE Agent -- Composite Blocking Score")
    ibox(f"""
<b>What happens:</b> All 3 predictions are combined into a single <b>composite blocking score</b> (0 to 1).<br><br>
<b>Formula:</b><br>
<code>blocking_score = {BLOCKING_WEIGHT_DEMAND} x demand_score + {BLOCKING_WEIGHT_SELLOUT} x sellout_score + {BLOCKING_WEIGHT_REVENUE} x revenue_score + {BLOCKING_WEIGHT_SUPPLIER} x supplier_score + {BLOCKING_WEIGHT_PRICE} x price_score</code><br><br>
<b>Threshold:</b> Block only if <code>blocking_score >= {MIN_BLOCKING_SCORE}</code>
""", "a")

    st.markdown("### Component Score Calculations")

    st.markdown("**1. Demand Score** (weight: 35%)")
    ibox(f"""
<code>demand_score = min((predicted_demand - 1.0) / ({DEMAND_SPIKE_THRESHOLD} - 1.0), 1.0)</code><br><br>
<b>Example:</b> If predicted_demand = 1.25:<br>
<code>demand_score = min((1.25 - 1.0) / (1.3 - 1.0), 1.0) = min(0.25 / 0.30, 1.0) = min(0.833, 1.0) = <b>0.833</b></code><br><br>
Interpretation: Demand at 83.3% of spike threshold. If demand <= 1.0, score = 0.
""", "b")

    st.markdown("**2. Sell-out Score** (weight: 30%)")
    ibox(f"""
<code>sellout_score = min(sellout_prob / {SELLOUT_PROBABILITY_THRESHOLD}, 1.0) ^ 0.5</code><br><br>
<b>Example:</b> If sellout_prob = 0.15:<br>
<code>scaled = min(0.15 / {SELLOUT_PROBABILITY_THRESHOLD}, 1.0) = min(0.375, 1.0) = 0.375</code><br>
<code>sellout_score = 0.375 ^ 0.5 = <b>0.612</b></code><br><br>
Note: Square root amplifies small probabilities. Even prob = 0.05 gives scaled = 0.125, score = 0.125 ^ 0.5 = <b>0.354</b>.
""", "b")

    st.markdown("**3. Revenue Score** (weight: 20%)")
    ibox(f"""
<code>revenue_score = min((base_adr x popularity_index) / ({REVENUE_SCORE_THRESHOLD} x 2), 1.0)</code><br><br>
<b>Example:</b> If base_adr = 4500, popularity = 1.8:<br>
<code>raw = 4500 x 1.8 = 8100</code><br>
<code>revenue_score = min(8100 / 10000, 1.0) = <b>0.810</b></code><br><br>
Higher ADR + popular properties get higher scores.
""", "b")

    st.markdown("**4. Supplier Score** (weight: 10%)")
    ibox("""
<code>supplier_score = min((1 - failure_rate) + preferred_bonus, 1.0)</code><br>
where <code>preferred_bonus = 0.2</code> if preferred supplier, else 0.<br><br>
<b>Example:</b> If failure_rate = 0.05, preferred = 1:<br>
<code>supplier_score = min((1 - 0.05) + 0.2, 1.0) = min(1.15, 1.0) = <b>1.000</b></code>
""", "b")

    st.markdown("**5. Price Score** (weight: 5%)")
    ibox(f"""
<code>inflation = (predicted_rate - current_rate) / current_rate</code><br>
<code>price_score = min(inflation / ({PRICE_INFLATION_THRESHOLD} x 2), 1.0)</code><br><br>
<b>Example:</b> If current = 4500, predicted = 5200:<br>
<code>inflation = (5200 - 4500) / 4500 = 0.1556</code><br>
<code>price_score = min(0.1556 / 0.16, 1.0) = <b>0.972</b></code><br><br>
If prices are dropping (inflation <= 0), score = 0.
""", "b")

    st.markdown("### Final Composite Score Calculation")
    ibox(f"""
<b>Plugging in the example values:</b><br><br>
<code>blocking_score = {BLOCKING_WEIGHT_DEMAND} x 0.833 + {BLOCKING_WEIGHT_SELLOUT} x 0.612 + {BLOCKING_WEIGHT_REVENUE} x 0.810 + {BLOCKING_WEIGHT_SUPPLIER} x 1.000 + {BLOCKING_WEIGHT_PRICE} x 0.972</code><br><br>
<code>= 0.292 + 0.184 + 0.162 + 0.100 + 0.049</code><br><br>
<code>= <b>0.787</b></code><br><br>
Since <b>0.787 >= {MIN_BLOCKING_SCORE}</b> (threshold), this property <b>WILL BE BLOCKED</b>.
""", "g")
    sdiv()

    st.markdown("## Step 4: RESERVE Agent -- Execute Block")
    ibox(f"""
<b>What happens:</b> For each property that passes the score threshold:<br><br>
<b>4a. Determine rooms to block:</b><br>
<code>rooms = max(MIN_ROOMS, min(base_inventory x {MAX_BLOCK_FRACTION}, demand_driven_qty))</code><br>
Example: inventory = 20 rooms, demand factor = 0.6 -> <code>max(2, min(20 x 0.4, 12)) = max(2, 8) = <b>8 rooms</b></code><br><br>
<b>4b. Find best supplier:</b><br>
- Query <code>Rate_Snapshots</code> for availability in the block period<br>
- Score each supplier: preferred flag + refundable terms + lowest rate<br>
- Select the best option with sufficient date coverage<br><br>
<b>4c. Calculate expected revenue uplift:</b><br>
<code>uplift = rooms x block_days x base_adr x demand_premium</code><br>
Example: 8 rooms x 7 days x INR 4,500 x 0.25 = <b>INR 63,000</b><br><br>
<b>4d. Write to database:</b> Insert record into <code>Demand_Block_Actions</code> table.
""", "g")

    if not bl.empty:
        st.markdown("### Real Blocking Decisions from Data")
        show_cols = ["property_id","city","supplier_id","rooms_blocked_per_night",
                     "block_reason","expected_revenue_uplift_inr","week_start"]
        avail = [c for c in show_cols if c in sample.columns]
        st_df(fmt_df_inr(sample[avail].sort_values("expected_revenue_uplift_inr", ascending=False).head(10)))
    sdiv()

    # ─── Problem 2: SMART REBOOKING ───
    st.markdown("# Problem 2: Smart Rebooking Workflow")
    ibox("<b>Goal:</b> After a booking is confirmed, monitor the market for cheaper rates on a <b>6-hour cycle</b>. "
         "Supplier rates are refreshed every 6 hours via API calls. After each refresh, the Monitor Agent scans "
         "100% of active bookings against the latest rates. If a better deal appears from another supplier "
         "for the same room type, cancel and rebook -- "
         "but <i>only</i> if the savings exceed the cancellation penalty and all safety checks pass.", "p")
    sdiv()

    st.markdown("## Production Monitoring Cycle")
    ibox("""
<b>Every 6 Hours:</b><br>
<code>00:00</code> → Supplier API call → <code>Rate_Snapshots</code> table updated with latest rates<br>
<code>00:01</code> → Monitor Agent scans 100% of active bookings against fresh rates<br>
<code>00:02</code> → Optimize Agent evaluates candidates → executes rebookings<br><br>
<code>06:00</code> → Supplier API call → Rate refresh → Monitor → Optimize<br>
<code>12:00</code> → Supplier API call → Rate refresh → Monitor → Optimize<br>
<code>18:00</code> → Supplier API call → Rate refresh → Monitor → Optimize<br><br>
<b>Result:</b> 4 scans per day × 100% of bookings = every booking checked against the latest supplier rates 4 times daily.
""", "g")
    sdiv()

    st.markdown("## Step 5: MONITOR Agent -- Find Cheaper Rates")
    ibox(f"""
<b>What happens:</b> After each 6-hour supplier rate refresh, scans every confirmed booking where:<br>
- Check-in date is in the future<br>
- Cancellation deadline hasn't passed<br>
- Booking is NOT NonRefundable<br><br>
<b>For each booking:</b><br>
1. Look up the property + room type in <code>Room_Mapping</code> to find equivalent rooms<br>
2. Filter equivalence score >= <b>{MIN_EQUIVALENCE_SCORE}</b><br>
3. Exclude the same supplier (must be a <i>different</i> supplier)<br>
4. Exclude NonRefundable rate snapshots<br>
5. Exclude upgrade-only mappings<br>
6. Find rates that are <b>cheaper</b> than the current booking cost<br><br>
<b>Output:</b> A list of rebooking candidates with old cost, new cost, savings, and supplier details.
""", "b")

    ev = D["rebooking_evaluations"]
    if not ev.empty:
        ev["decision"] = ev["decision"].str.strip().str.title()
        ev_wk = ev.copy()
        if "eval_week" in ev.columns:
            ev_wk["eval_week"] = pd.to_datetime(ev_wk["eval_week"])
            ev_filt = ev_wk[ev_wk["eval_week"] == pd.to_datetime(ws)]
            if not ev_filt.empty:
                ev_wk = ev_filt

        ex_rb = ev_wk.iloc[0]
        old_cost = ex_rb.get("old_cost_inr", 12000)
        new_cost = ex_rb.get("new_cost_inr", 9500)
        savings = ex_rb.get("savings_inr", 2500)
        old_sup = ex_rb.get("old_supplier", "Supplier A")
        new_sup = ex_rb.get("new_supplier", "Supplier B")
        room = ex_rb.get("room_type", "Deluxe")
        city_rb = ex_rb.get("city", "Mumbai")

        st.markdown(f"**Example from Data:** Booking in `{city_rb}`, room `{room}`")
        ibox(f"""
<b>Monitor Agent found a cheaper rate:</b><br>
- Old supplier: <code>{old_sup}</code> at <b>INR {fmt_inr(old_cost)}</b><br>
- New supplier: <code>{new_sup}</code> at <b>INR {fmt_inr(new_cost)}</b><br>
- Potential savings: <b>INR {fmt_inr(savings)}</b>
""", "g")
    sdiv()

    st.markdown("## Step 6: OPTIMIZE Agent -- Risk & Profit Scoring")
    ibox(f"""
<b>What happens:</b> Each rebooking candidate is evaluated on <b>4 risk dimensions</b> and a <b>profit score</b>.<br><br>
<b>Risk Score Formula (0-100, lower = safer):</b><br>
<code>risk_score = {RISK_WEIGHT_CANCELLATION} x cancellation_risk + {RISK_WEIGHT_SUPPLIER} x supplier_risk + {RISK_WEIGHT_EQUIVALENCE} x equivalence_risk + {RISK_WEIGHT_TIMING} x timing_risk</code>
""", "a")

    st.markdown("### Risk Component Calculations")

    st.markdown("**1. Cancellation Risk** (weight: 40%)")
    ibox("""
<code>cancellation_risk = type_risk x 0.4 + penalty_risk x 0.3 + timing_risk x 0.3</code><br><br>
<b>Type risk:</b> Refundable = 0, PartialRefund = 30, NonRefundable = 100<br>
<b>Penalty risk:</b> <code>min(penalty_pct x 100, 100)</code><br>
<b>Timing risk:</b> 0 days = 100, <= 2 days = 70, <= 7 days = 30, > 7 days = 10<br><br>
<b>Example:</b> Refundable, 10% penalty, 10 days to deadline:<br>
<code>cancellation_risk = 0 x 0.4 + 10 x 0.3 + 10 x 0.3 = 0 + 3.0 + 3.0 = <b>6.0</b></code>
""", "b")

    st.markdown("**2. Supplier Risk** (weight: 30%)")
    ibox("""
Looks up new supplier from <code>Supplier_Reliability</code> table using 3 factors:<br>
<code>supplier_risk = (failure_rate x 40 + cancellation_rate x 30 + dispute_rate x 20) x 100</code><br>
<code>If preferred supplier: risk x 0.6</code><br><br>
<b>Example:</b> failure = 8%, cancellation = 5%, dispute = 3%, not preferred:<br>
<code>supplier_risk = (0.08 x 40 + 0.05 x 30 + 0.03 x 20) x 100 = (3.2 + 1.5 + 0.6) x 100 = <b>5.3</b></code>
""", "b")

    st.markdown("**3. Equivalence Risk** (weight: 20%)")
    ibox(f"""
<code>equivalence_risk = (1 - equivalence_score) x 100</code><br><br>
<b>Example:</b> Room mapping equivalence = 0.92:<br>
<code>equivalence_risk = (1 - 0.92) x 100 = <b>8.0</b></code><br><br>
Perfect match (1.0) = 0 risk. Below {MIN_EQUIVALENCE_SCORE} = auto-skip.
""", "b")

    st.markdown("**4. Timing Risk** (weight: 10%)")
    ibox("""
Based on days until cancellation deadline:<br>
- <= 1 day: <b>90</b> (very dangerous)<br>
- 2-3 days: <b>60</b><br>
- 4-7 days: <b>30</b><br>
- 8-14 days: <b>15</b><br>
- > 14 days: <b>5</b> (very safe)<br><br>
<b>Example:</b> 10 days to deadline -> <code>timing_risk = <b>15.0</b></code>
""", "b")

    st.markdown("### Composite Risk Score")
    ibox(f"""
<b>Plugging in example values:</b><br><br>
<code>risk_score = {RISK_WEIGHT_CANCELLATION} x 6.0 + {RISK_WEIGHT_SUPPLIER} x 5.3 + {RISK_WEIGHT_EQUIVALENCE} x 8.0 + {RISK_WEIGHT_TIMING} x 15.0</code><br><br>
<code>= 2.4 + 1.59 + 1.6 + 1.5</code><br><br>
<code>= <b>7.09</b></code><br><br>
Since <b>7.09 <= {MAX_RISK_SCORE}</b> (max allowed), risk check <b>PASSES</b>.
""", "g")
    sdiv()

    st.markdown("### Profit Score Calculation")
    ibox(f"""
<code>net_profit = savings - estimated_penalty</code><br>
<code>profit_score = (net_profit / old_cost) x 100</code><br><br>
<b>Example:</b> savings = INR 2,500, penalty = INR 1,200, old_cost = INR 12,000:<br>
<code>net_profit = 2500 - 1200 = <b>INR 1,300</b></code><br>
<code>profit_score = (1300 / 12000) x 100 = <b>10.83</b></code>
""", "b")

    st.markdown("### Confidence Score")
    ibox(f"""
<code>confidence = clip(1.0 - (risk_score/100 x 0.5) + (profit_score/100 x 0.3), 0.0, 1.0)</code><br><br>
<b>Example:</b><br>
<code>confidence = clip(1.0 - (7.09/100 x 0.5) + (10.83/100 x 0.3), 0, 1)</code><br>
<code>= clip(1.0 - 0.0355 + 0.0325, 0, 1)</code><br>
<code>= clip(0.997, 0, 1) = <b>0.997</b></code>
""", "b")
    sdiv()

    st.markdown("## Step 7: Decision -- Rebook or Skip?")
    ibox(f"""
<b>ALL of these criteria must pass for a REBOOK decision:</b><br><br>
<table style="width:100%;border-collapse:collapse;font-size:13px">
<tr style="border-bottom:2px solid #e2e8f0;font-weight:700">
    <td style="padding:8px">Check</td><td style="padding:8px">Threshold</td><td style="padding:8px">Example Value</td><td style="padding:8px">Result</td>
</tr>
<tr style="border-bottom:1px solid #f1f5f9">
    <td style="padding:8px">Savings >= minimum</td><td style="padding:8px">INR {MIN_SAVINGS_INR:.0f}</td><td style="padding:8px">INR 2,500</td><td style="padding:8px;color:#10b981;font-weight:700">PASS</td>
</tr>
<tr style="border-bottom:1px solid #f1f5f9">
    <td style="padding:8px">Net profit >= minimum</td><td style="padding:8px">INR {MIN_NET_PROFIT_INR:.0f}</td><td style="padding:8px">INR 1,300</td><td style="padding:8px;color:#10b981;font-weight:700">PASS</td>
</tr>
<tr style="border-bottom:1px solid #f1f5f9">
    <td style="padding:8px">Equivalence >= minimum</td><td style="padding:8px">{MIN_EQUIVALENCE_SCORE}</td><td style="padding:8px">0.92</td><td style="padding:8px;color:#10b981;font-weight:700">PASS</td>
</tr>
<tr style="border-bottom:1px solid #f1f5f9">
    <td style="padding:8px">Risk score <= maximum</td><td style="padding:8px">{MAX_RISK_SCORE}</td><td style="padding:8px">7.09</td><td style="padding:8px;color:#10b981;font-weight:700">PASS</td>
</tr>
<tr style="border-bottom:1px solid #f1f5f9">
    <td style="padding:8px">Days to deadline >= minimum</td><td style="padding:8px">{MIN_DAYS_BEFORE_DEADLINE}</td><td style="padding:8px">10</td><td style="padding:8px;color:#10b981;font-weight:700">PASS</td>
</tr>
<tr style="border-bottom:1px solid #f1f5f9">
    <td style="padding:8px">Penalty fraction <= maximum</td><td style="padding:8px">{MAX_PENALTY_FRACTION:.0%}</td><td style="padding:8px">10%</td><td style="padding:8px;color:#10b981;font-weight:700">PASS</td>
</tr>
<tr style="border-bottom:1px solid #f1f5f9">
    <td style="padding:8px">Confidence >= minimum</td><td style="padding:8px">{MIN_REBOOKING_CONFIDENCE}</td><td style="padding:8px">0.997</td><td style="padding:8px;color:#10b981;font-weight:700">PASS</td>
</tr>
</table><br>
<b>All 7 checks passed -> Decision: REBOOK</b><br>
If <i>any single check</i> fails -> Decision: <b>SKIP</b> (with reason logged).
""", "g")

    if not ev.empty:
        sdiv()
        st.markdown("### Real Rebooking Evaluations from Data")
        show_cols = ["rebooking_eval_id","city","room_type","old_supplier","new_supplier",
                     "old_cost_inr","new_cost_inr","savings_inr","risk_score","profit_score",
                     "confidence","decision"]
        avail = [c for c in show_cols if c in ev_wk.columns]
        st_df(fmt_df_inr(ev_wk[avail].sort_values("savings_inr", ascending=False).head(15)))

        rb_count = (ev_wk["decision"] == "Rebook").sum()
        sk_count = (ev_wk["decision"] == "Skip").sum()
        rate = rb_count / max(len(ev_wk), 1) * 100
        st.markdown(f"**Summary:** {rb_count} Rebooked, {sk_count} Skipped ({rate:.1f}% rebook rate)")
    sdiv()

    st.markdown("## Step 8: REPORT Agent -- Weekly KPI Summary")
    ibox("""
<b>What happens:</b> After all blocking and rebooking decisions are executed, the Report Agent:<br><br>
1. Counts total properties blocked, rooms blocked, expected uplift<br>
2. Counts rebooking evaluations, successful rebookings, margin recovered<br>
3. Identifies missed opportunities (properties that sold out but weren't blocked)<br>
4. Writes all KPIs to <code>Weekly_KPI_Summary</code> table<br>
5. Generates an HTML report with charts and recommendations
""", "b")

    kpi = D["weekly_kpi_summary"]
    if not kpi.empty:
        kpi["week_start"] = pd.to_datetime(kpi["week_start"])
        kpi_sorted = kpi.sort_values("week_start")
        st_df(fmt_df_inr(kpi_sorted))

    page_nav("How It Works")

# ═══════════════════════════════════════════════════════
# PAGE 3: DEMAND INTELLIGENCE
# ═══════════════════════════════════════════════════════
def pg_demand(D, ws):
    st.markdown("# 1. Sense & Predict Agent")
    ibox("<b>What is this?</b> Shows raw demand signals and how the PREDICT Agent uses them. The "
         f"<b>city_demand_multiplier</b> is key: 1.0 = normal, above {DEMAND_SPIKE_THRESHOLD} = spike. "
         "ML model learns from seasonality, weekends, and events to predict 14 days ahead.","b")

    cds=D["city_demand_signals"]
    if cds.empty: st.warning("No data."); return
    cds["date"]=pd.to_datetime(cds["date"])
    cities=sorted(cds["city"].unique())
    sel=st.multiselect("Filter Cities",cities,default=cities[:5])
    if not sel: return
    f=cds[cds["city"].isin(sel)]
    spc(6)
    sdiv()

    st.markdown("## Demand Multiplier Over Time")
    ibox(f"The red dashed line at <b>{DEMAND_SPIKE_THRESHOLD}</b> is the spike threshold. When multiplier "
         "crosses it, the DECIDE Agent considers blocking. Spikes come from festivals, conferences, seasons.","a")
    fig=px.line(f,x="date",y="city_demand_multiplier",color="city",labels={"city_demand_multiplier":"Multiplier"})
    fig.add_hline(y=DEMAND_SPIKE_THRESHOLD,line_dash="dash",line_color="red",annotation_text=f"Threshold ({DEMAND_SPIKE_THRESHOLD})")
    fig.update_layout(height=400,template="plotly_white"); st.plotly_chart(fig,use_container_width=True)
    sdiv()

    st.markdown("## Signal Decomposition")
    ibox("<b>Seasonality</b> = long-term patterns. <b>Weekend</b> = Fri-Sun leisure boost. "
         "<b>Event</b> = festivals/sports cause localized spikes. All 3 are separate ML features.","g")
    c1,c2,c3=st.columns(3)
    for col,y,t in [(c1,"seasonality","Seasonality"),(c2,"weekend_multiplier","Weekend Effect"),(c3,"event_multiplier","Event Impact")]:
        with col:
            fig=px.line(f,x="date",y=y,color="city",title=t)
            fig.update_layout(height=260,template="plotly_white",showlegend=False); st.plotly_chart(fig,use_container_width=True)
    sdiv()

    st.markdown("## Events Calendar")
    ibox("<b>demand_intensity</b> (1-3): 1=mild, 2=moderate, 3=extreme spike. The SENSE Agent computes "
         "<b>days_to_event</b> for each property-date and feeds it to ML models.","a")
    ev=D["events_calendar"]
    if not ev.empty:
        ef=ev[ev["city"].isin(sel)].copy()
        if not ef.empty:
            ef["start_date"]=pd.to_datetime(ef["start_date"])
            st_df(ef.sort_values("start_date"))
    page_nav("Sense & Predict Agent")

# ═══════════════════════════════════════════════════════
# PAGE 4: BLOCKING STRATEGY
# ═══════════════════════════════════════════════════════
def pg_blocking(D, ws):
    st.markdown("# 2. Decide & Reserve Agent")
    ibox("<b>What is this?</b> Demand blocking = proactively reserving rooms <i>before</i> spikes so they "
         "can be sold at peak prices. Shows every action by DECIDE + RESERVE agents: which properties, "
         "how many rooms, why chosen, expected revenue uplift.","b")

    bl_all=D["demand_block_actions"]
    if bl_all.empty: st.info("No blocks yet. Run pipeline."); return
    bl_all["week_start"]=pd.to_datetime(bl_all["week_start"]); bl_all["block_start_date"]=pd.to_datetime(bl_all["block_start_date"])
    bl=bl_all[bl_all["week_start"]==pd.to_datetime(ws)]
    if bl.empty:
        st.info(f"No blocking data for week {ws}. Showing all available weeks.")
        bl=bl_all
    sdiv()

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total Blocks",len(bl),help="Blocking actions for selected week")
    c2.metric("Unique Properties",bl["property_id"].nunique(),help="Distinct properties blocked")
    c3.metric("Total Rooms",int(bl["rooms_blocked_per_night"].sum()),help="Sum of rooms_blocked_per_night")
    c4.metric("Total Uplift",f"INR {fmt_inr(bl['expected_revenue_uplift_inr'].sum())}",help="Sum of expected uplift")
    spc(6)
    sdiv()

    st.markdown("## Weekly Blocking Trend")
    ibox("Consistent blocking = actively protecting against sell-outs. Dips may indicate lower predicted demand or limited supplier availability.","g")
    wk=bl_all.groupby("week_start").agg(props=("property_id","nunique"),rooms=("rooms_blocked_per_night","sum"),uplift=("expected_revenue_uplift_inr","sum")).reset_index()
    fig=make_subplots(rows=1,cols=2,subplot_titles=("Properties & Rooms per Week","Weekly Uplift (INR)"))
    fig.add_trace(go.Bar(x=wk["week_start"],y=wk["props"],name="Properties",marker_color="#3b82f6"),row=1,col=1)
    fig.add_trace(go.Bar(x=wk["week_start"],y=wk["rooms"],name="Rooms",marker_color="#93c5fd"),row=1,col=1)
    fig.add_trace(go.Bar(x=wk["week_start"],y=wk["uplift"],name="Uplift",marker_color="#10b981"),row=1,col=2)
    fig.update_layout(height=360,template="plotly_white"); st.plotly_chart(fig,use_container_width=True)
    sdiv()

    st.markdown("## Why Properties Were Blocked")
    ibox("Every block includes a <b>block_reason</b>: high sell-out risk, demand spike, price inflation, event activity. Multiple reasons can combine.","a")
    c1,c2=st.columns(2)
    with c1:
        reasons_col = bl["block_reason"].dropna().astype(str)
        reasons_col = reasons_col[reasons_col.str.strip() != ""]
        if reasons_col.empty:
            st.info("No block reasons found. Click **Refresh Data** in the sidebar.")
        else:
            rr=[]
            for r in reasons_col:
                rr.extend([x.strip() for x in r.split("+")])
            rc=pd.Series(rr).value_counts().head(10).reset_index(); rc.columns=["Reason","Count"]
            fig=px.bar(rc,x="Count",y="Reason",orientation="h",color="Count",color_continuous_scale="Viridis",title="Top Reasons")
            fig.update_layout(height=340,template="plotly_white"); st.plotly_chart(fig,use_container_width=True)
    with c2:
        cd=bl.groupby("city")["block_id"].count().reset_index(); cd.columns=["City","Blocks"]
        fig=px.pie(cd,names="City",values="Blocks",hole=.35,title="By City")
        fig.update_layout(height=340,template="plotly_white"); st.plotly_chart(fig,use_container_width=True)
    sdiv()

    st.markdown("## Full Block Log")
    ibox("Complete log from <code>Demand_Block_Actions</code> table. Each row = one block decision.","b")
    st_df(fmt_df_inr(bl.sort_values("expected_revenue_uplift_inr",ascending=False).head(100)))
    spc(6)
    page_nav("Decide & Reserve Agent")

# ═══════════════════════════════════════════════════════
# PAGE 5: REBOOKING ENGINE
# ═══════════════════════════════════════════════════════
def pg_rebooking(D, ws):
    st.markdown("# 3. Monitor & Optimize Agent")
    ibox("<b>What is this?</b> After booking confirmation, supplier rates are refreshed every <b>6 hours</b> via API. "
         "MONITOR Agent then scans 100% of active bookings against fresh rates. "
         "OPTIMIZE Agent evaluates safety (cancellation risk, supplier reliability, room match) and "
         "profitability (savings - penalty). Only rebooks when ALL criteria pass.","b")

    ev_all=D["rebooking_evaluations"]
    if ev_all.empty: st.info("No evaluations yet."); return
    ev_all["eval_week"]=pd.to_datetime(ev_all["eval_week"])
    ev_all["decision"]=ev_all["decision"].str.strip().str.title()
    ev=ev_all[ev_all["eval_week"]==pd.to_datetime(ws)]
    if ev.empty:
        st.warning(f"No rebooking data for week {ws}. Showing all weeks.")
        ev=ev_all
    ev=ev.copy()
    ev["net_profit_inr"] = ev["savings_inr"] - ev["estimated_penalty_inr"]
    rb=ev[ev["decision"]=="Rebook"]; sk=ev[ev["decision"]=="Skip"]
    sdiv()

    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.metric("Evaluations",len(ev),help="Total booking-supplier pairs evaluated")
    c2.metric("Rebooked",len(rb),help="Successfully switched to cheaper supplier")
    c3.metric("Skipped",len(sk),help="Too risky or insufficient savings")
    c4.metric("Total Savings",f"INR {fmt_inr(rb['savings_inr'].sum())}",help="Gross savings from rebookings")
    c5.metric("Net Margin",f"INR {fmt_inr(ev.loc[ev['decision']=='Rebook','net_profit_inr'].sum())}",help="Savings minus penalties for rebooked")
    c6.metric("Avg Confidence",f"{rb['confidence'].mean():.2f}" if not rb.empty else "N/A")
    spc(6)
    sdiv()

    st.markdown("## Risk vs Profit Analysis")
    ibox(f"<b>X-axis</b>: risk score (0-100, lower=safer). <b>Y-axis</b>: profit score (savings/cost %). "
         f"Green=rebooked, Red=skipped. System rebooks only when risk <= {MAX_RISK_SCORE} AND all criteria pass. "
         "Ideal zone: bottom-right (low risk, high profit).","g")
    c1,c2=st.columns(2)
    with c1:
        sd=ev.copy(); sd["abs_sav"]=sd["savings_inr"].abs().clip(lower=1)
        fig=px.scatter(sd,x="risk_score",y="profit_score",color="decision",
                       color_discrete_map={"Rebook":"#10b981","Skip":"#ef4444"},
                       size="abs_sav",size_max=22,hover_data=["booking_id","city","savings_inr","confidence"],
                       title="Risk vs Profit (size = savings)")
        fig.add_vline(x=MAX_RISK_SCORE,line_dash="dash",line_color="rgba(239,68,68,0.5)",
                      annotation_text=f"Max Risk ({MAX_RISK_SCORE})",annotation_position="top left")
        fig.update_traces(marker=dict(opacity=0.8,line=dict(width=1,color="white")))
        fig.update_layout(height=440,template="plotly_white",
                          xaxis=dict(title="Risk Score (0=safe, 100=risky)",gridcolor="rgba(0,0,0,0.06)",range=[-2,105]),
                          yaxis=dict(title="Profit Score (%)",gridcolor="rgba(0,0,0,0.06)"),
                          legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5,font=dict(size=12)),
                          margin=dict(t=60,b=40))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=px.histogram(ev,x="savings_inr",color="decision",color_discrete_map={"Rebook":"#10b981","Skip":"#ef4444"},
                         title="Savings Distribution (INR)",nbins=30,barmode="overlay",opacity=.7)
        fig.update_layout(height=380,template="plotly_white"); st.plotly_chart(fig,use_container_width=True)
    sdiv()

    st.markdown("## Weekly Trend")
    ibox("Growing green = more profitable opportunities found. Dominant red = suppliers not competitive or strict cancellation terms.","a")
    ev_trend=ev_all[ev_all["eval_week"]>=pd.Timestamp("2026-01-01")]
    wr=ev_trend.groupby(["eval_week","decision"]).size().reset_index(name="count")
    fig=px.bar(wr,x="eval_week",y="count",color="decision",color_discrete_map={"Rebook":"#10b981","Skip":"#ef4444"},barmode="stack")
    fig.update_layout(height=320,template="plotly_white"); st.plotly_chart(fig,use_container_width=True)
    sdiv()

    if not rb.empty:
        st.markdown("## Supplier Performance")
        ibox("Which suppliers offer best rebooking savings? High avg_savings + many rebookings = best alternative.","g")
        sp=rb.groupby("new_supplier").agg(rebookings=("rebooking_eval_id","count"),avg_savings=("savings_inr","mean"),
            total_savings=("savings_inr","sum"),avg_risk=("risk_score","mean"),avg_conf=("confidence","mean")).reset_index()
        sp[["avg_savings","total_savings"]]=sp[["avg_savings","total_savings"]].round(0)
        sp[["avg_risk","avg_conf"]]=sp[["avg_risk","avg_conf"]].round(2)
        st_df(fmt_df_inr(sp.sort_values("total_savings",ascending=False)))
    sdiv()

    st.markdown("## Full Evaluation Log")
    ibox("From <code>Rebooking_Evaluations</code> table. Each row = one evaluation with scores, penalty, net profit and decision.","b")
    dc=["rebooking_eval_id","booking_id","city","room_type","old_supplier","new_supplier",
        "old_cost_inr","new_cost_inr","savings_inr","estimated_penalty_inr","net_profit_inr",
        "risk_score","profit_score","confidence","decision"]
    ac=[c for c in dc if c in ev.columns]
    st_df(fmt_df_inr(ev[ac].sort_values("savings_inr",ascending=False).head(100)))
    spc(6)

    page_nav("Monitor & Optimize Agent")

# ═══════════════════════════════════════════════════════
# PAGE 6: KPI TRENDS
# ═══════════════════════════════════════════════════════
def pg_kpi(D, ws):
    st.markdown("# 4. Report Agent")
    ibox("Weekly_KPI_Summary tracks system performance over time. Use these charts to evaluate if "
         "blocking & rebooking are improving and whether thresholds need adjustment.","b")
    kpi=D["weekly_kpi_summary"]
    if kpi.empty: st.info("No KPI data."); return
    kpi["week_start"]=pd.to_datetime(kpi["week_start"]); kpi=kpi.sort_values("week_start")
    kpi=kpi[kpi["week_start"]>=pd.Timestamp("2026-01-01")]
    if kpi.empty: st.info("No KPI data from Jan 2026 onwards."); return
    sdiv()

    fig=make_subplots(rows=2,cols=2,subplot_titles=("Properties Blocked","Rooms Blocked","Expected Uplift (INR)","Margin Recovered (INR)"))
    fig.add_trace(go.Scatter(x=kpi["week_start"],y=kpi["properties_blocked"],mode="lines+markers",line=dict(color="#3b82f6",width=2),fill="tozeroy",fillcolor="rgba(59,130,246,0.1)"),row=1,col=1)
    fig.add_trace(go.Scatter(x=kpi["week_start"],y=kpi["total_rooms_blocked"],mode="lines+markers",line=dict(color="#8b5cf6",width=2),fill="tozeroy",fillcolor="rgba(139,92,246,0.1)"),row=1,col=2)
    fig.add_trace(go.Bar(x=kpi["week_start"],y=kpi["expected_uplift_inr"],marker_color="#10b981"),row=2,col=1)
    fig.add_trace(go.Bar(x=kpi["week_start"],y=kpi["margin_recovered_inr"],marker_color="#f59e0b"),row=2,col=2)
    fig.update_layout(height=500,showlegend=False,template="plotly_white"); st.plotly_chart(fig,use_container_width=True)
    sdiv()

    st.markdown("## Rebooking Efficiency Rate")
    ibox("Rebook rate = rebookings / evaluations x 100. 20-40% is healthy (selective). Very high = loose thresholds. Very low = too strict.","a")
    if "rebook_evaluations" in kpi.columns:
        kpi["rebooking_rate_pct"]=np.where(kpi["rebook_evaluations"]>0,kpi["rebook_count"]/kpi["rebook_evaluations"]*100,0)
        fig=px.area(kpi,x="week_start",y="rebooking_rate_pct",labels={"rebooking_rate_pct":"Rate (%)","week_start":"Week"})
        fig.update_layout(height=280,template="plotly_white"); st.plotly_chart(fig,use_container_width=True)
    sdiv()

    # ── AI-Powered Recommendations ──
    st.markdown("## AI-Powered Recommendations")
    ibox("Data-driven recommendations generated by analyzing your KPI trends, blocking patterns, and rebooking performance. "
         "Click the button to get actionable insights from AI.","p")

    bl = D["demand_block_actions"]
    ev = D["rebooking_evaluations"]

    # Build data-driven recommendations (rule-based, always visible)
    recs = []
    if not kpi.empty and len(kpi) >= 2:
        latest = kpi.iloc[-1]
        prev = kpi.iloc[-2]

        # Blocking trend
        bl_change = latest.get("properties_blocked",0) - prev.get("properties_blocked",0)
        if bl_change > 0:
            recs.append(("📈","Blocking Activity Increasing",
                         f"Properties blocked grew by <b>{int(bl_change)}</b> vs previous week. "
                         "The system is detecting more demand spikes — verify if this aligns with upcoming events or seasonality.","g"))
        elif bl_change < 0:
            recs.append(("📉","Blocking Activity Declining",
                         f"Properties blocked dropped by <b>{abs(int(bl_change))}</b> vs previous week. "
                         "Consider lowering <code>MIN_BLOCKING_SCORE</code> if you want more aggressive blocking.","a"))

        # Margin recovery trend
        mr_curr = latest.get("margin_recovered_inr",0)
        mr_prev = prev.get("margin_recovered_inr",0)
        if mr_prev > 0:
            mr_pct = (mr_curr - mr_prev) / mr_prev * 100
            if mr_pct > 10:
                recs.append(("💰","Margin Recovery Improving",
                             f"Margin recovered increased by <b>{mr_pct:.0f}%</b> week-over-week (INR {fmt_inr(mr_curr)} vs {fmt_inr(mr_prev)}). "
                             "Rebooking strategy is becoming more effective.","g"))
            elif mr_pct < -10:
                recs.append(("⚠️","Margin Recovery Declining",
                             f"Margin recovered dropped by <b>{abs(mr_pct):.0f}%</b>. "
                             "Suppliers may have tightened rates or cancellation terms. Review <code>MAX_RISK_SCORE</code> and <code>MIN_SAVINGS_INR</code>.","r"))

    # Rebooking rate analysis
    if not ev.empty:
        ev_dec = ev.copy()
        ev_dec["decision"] = ev_dec["decision"].str.strip().str.title()
        rb_count = (ev_dec["decision"] == "Rebook").sum()
        total_eval = len(ev_dec)
        rb_rate = rb_count / max(total_eval, 1) * 100
        if rb_rate < 15:
            recs.append(("🔴","Low Rebook Rate",
                         f"Only <b>{rb_rate:.1f}%</b> of evaluations result in rebooking. "
                         "Thresholds may be too strict. Consider relaxing <code>MAX_RISK_SCORE</code> (currently {MAX_RISK_SCORE}) "
                         "or <code>MIN_SAVINGS_INR</code> (currently {MIN_SAVINGS_INR}).","a"))
        elif rb_rate > 60:
            recs.append(("🟡","High Rebook Rate",
                         f"Rebook rate is <b>{rb_rate:.1f}%</b> — unusually high. "
                         "Verify that quality checks are not being bypassed. Consider tightening <code>MIN_EQUIVALENCE_SCORE</code>.","a"))
        else:
            recs.append(("🟢","Healthy Rebook Rate",
                         f"Rebook rate is <b>{rb_rate:.1f}%</b> — within the ideal 15-60% range. "
                         "The system is being selective and profitable.","g"))

    # City concentration analysis
    if not bl.empty:
        city_counts = bl["city"].value_counts()
        top_city = city_counts.index[0]
        top_pct = city_counts.iloc[0] / len(bl) * 100
        if top_pct > 50:
            recs.append(("🏙️","High City Concentration",
                         f"<b>{top_pct:.0f}%</b> of all blocks are in <b>{top_city}</b>. "
                         "Consider diversifying blocking across more cities to reduce geographic risk.","a"))
        if len(city_counts) >= 3:
            underserved = city_counts.tail(3).index.tolist()
            recs.append(("🔍","Underserved Cities",
                         f"Cities with fewest blocks: <b>{', '.join(underserved)}</b>. "
                         "Investigate if these cities have low demand or if the system is missing opportunities.","b"))

    # Display recommendations
    if recs:
        for emoji, title, desc, color in recs:
            st.markdown(f"""<div class="ibox i{color}" style="padding:16px 20px">
                <div style="font-size:16px;font-weight:700;margin-bottom:6px">{emoji} {title}</div>
                <div style="font-size:14px;line-height:1.7">{desc}</div>
            </div>""", unsafe_allow_html=True)
    else:
        ibox("Not enough data to generate recommendations. Run the pipeline for at least 2 weeks.","a")

    spc(4)

    # AI Deep Analysis button
    if st.button("Generate AI Deep Analysis", key="ai_report_recs", use_container_width=True):
        with st.spinner("Running AI Analysis…"):
            ctx_parts = []
            if not kpi.empty:
                latest = kpi.iloc[-1]
                ctx_parts.append(f"Latest KPI: {int(latest.get('properties_blocked',0))} properties blocked, "
                                 f"{int(latest.get('total_rooms_blocked',0))} rooms, "
                                 f"INR {fmt_inr(latest.get('expected_uplift_inr',0))} uplift, "
                                 f"INR {fmt_inr(latest.get('margin_recovered_inr',0))} margin recovered.")
                if len(kpi) >= 2:
                    prev = kpi.iloc[-2]
                    ctx_parts.append(f"Previous week: {int(prev.get('properties_blocked',0))} blocked, "
                                     f"INR {fmt_inr(prev.get('margin_recovered_inr',0))} margin.")
            if not bl.empty:
                ctx_parts.append(f"Blocking: {len(bl)} total blocks across {bl['city'].nunique()} cities. "
                                 f"Top city: {bl['city'].value_counts().index[0]}.")
            if not ev.empty:
                ev_d = ev.copy(); ev_d["decision"] = ev_d["decision"].str.strip().str.title()
                rb_n = (ev_d["decision"]=="Rebook").sum()
                ctx_parts.append(f"Rebooking: {len(ev)} evaluations, {rb_n} rebooked ({rb_n/max(len(ev),1)*100:.1f}% rate).")

            prompt = ("You are a hotel revenue management expert. Based on this data, provide 5 specific, actionable recommendations "
                      "to improve the demand blocking and smart rebooking strategy. "
                      "For each recommendation, explain WHY and give a concrete action step. "
                      f"Data: {' '.join(ctx_parts)}")
            answer = ask_gpt(prompt, max_tokens=600)
            ai_insight_box(answer)

    spc(6)
    sdiv()
    st.markdown("## Raw Data")
    st_df(fmt_df_inr(kpi))
    spc(6)
    sdiv()

    # ── Download Reports ──
    st.markdown("## Download Reports")
    ibox("Download <b>Demand Blocking</b> and <b>Smart Rebooking</b> data as CSV or a combined PDF summary report.","p")

    bl = D["demand_block_actions"]
    ev = D["rebooking_evaluations"]

    dc1, dc2, dc3 = st.columns(3)

    # CSV: Demand Blocking
    with dc1:
        if not bl.empty:
            csv_bl = bl.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Demand Blocking (CSV)", data=csv_bl,
                               file_name=f"demand_blocking_{ws}.csv", mime="text/csv",
                               use_container_width=True)
        else:
            st.info("No blocking data")

    # CSV: Smart Rebooking
    with dc2:
        if not ev.empty:
            csv_ev = ev.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Smart Rebooking (CSV)", data=csv_ev,
                               file_name=f"smart_rebooking_{ws}.csv", mime="text/csv",
                               use_container_width=True)
        else:
            st.info("No rebooking data")

    # PDF: Combined Report
    with dc3:
        def _build_pdf():
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # Title
            pdf.set_font("Helvetica", "B", 20)
            pdf.set_text_color(30, 27, 75)
            pdf.cell(0, 12, "Agentic AI - Weekly Report", ln=True, align="C")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(0, 8, f"Week Start: {ws}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
            pdf.line(10, pdf.get_y()+2, 200, pdf.get_y()+2)
            pdf.ln(8)

            def _section(title, r, g, b):
                pdf.set_font("Helvetica", "B", 14)
                pdf.set_text_color(r, g, b)
                pdf.cell(0, 10, title, ln=True)
                pdf.ln(2)

            def _kv(label, value):
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(30, 41, 59)
                pdf.cell(80, 7, label, border=1)
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, str(value), border=1, ln=True)

            def _table(df, cols, col_widths=None):
                if df.empty:
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.cell(0, 7, "No data available", ln=True)
                    return
                if col_widths is None:
                    col_widths = [190 // len(cols)] * len(cols)
                # Header
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_fill_color(99, 102, 241)
                pdf.set_text_color(255, 255, 255)
                for i, c in enumerate(cols):
                    pdf.cell(col_widths[i], 6, str(c)[:20], border=1, fill=True)
                pdf.ln()
                # Rows
                pdf.set_font("Helvetica", "", 7)
                pdf.set_text_color(30, 41, 59)
                for _, row in df.head(10).iterrows():
                    for i, c in enumerate(cols):
                        val = str(row.get(c, ""))[:25]
                        pdf.cell(col_widths[i], 5, val, border=1)
                    pdf.ln()
                pdf.ln(4)

            # -- Demand Blocking --
            _section("Demand Blocking Summary", 59, 130, 246)
            if not bl.empty:
                bl_wk = bl[bl["week_start"].astype(str).str[:10] == ws] if "week_start" in bl.columns else bl
                bl_data = bl_wk if not bl_wk.empty else bl
                _kv("Total Blocks", len(bl_data))
                _kv("Unique Properties", bl_data["property_id"].nunique())
                _kv("Total Rooms Blocked", int(bl_data["rooms_blocked_per_night"].sum()))
                _kv("Expected Uplift (INR)", fmt_inr(bl_data["expected_revenue_uplift_inr"].sum()))
                _kv("Cities Covered", bl_data["city"].nunique())
                pdf.ln(4)
                pdf.set_font("Helvetica", "B", 11)
                pdf.set_text_color(71, 85, 105)
                pdf.cell(0, 8, "Top 10 Blocks by Uplift", ln=True)
                top_bl = bl_data.nlargest(10, "expected_revenue_uplift_inr")
                _table(top_bl, ["property_id","city","rooms_blocked_per_night","expected_revenue_uplift_inr"],
                       [45, 35, 50, 60])
            else:
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 7, "No blocking data available", ln=True)

            # -- Smart Rebooking --
            pdf.ln(4)
            _section("Smart Rebooking Summary", 16, 185, 129)
            if not ev.empty:
                ev_wk = ev[ev["eval_week"].astype(str).str[:10] == ws] if "eval_week" in ev.columns else ev
                ev_data = ev_wk if not ev_wk.empty else ev
                ev_d = ev_data.copy()
                ev_d["decision"] = ev_d["decision"].str.strip().str.title()
                ev_d["net_profit_inr"] = ev_d["savings_inr"] - ev_d["estimated_penalty_inr"]
                rb = ev_d[ev_d["decision"] == "Rebook"]
                sk = ev_d[ev_d["decision"] == "Skip"]
                _kv("Total Evaluations", len(ev_data))
                _kv("Rebooked", len(rb))
                _kv("Skipped", len(sk))
                _kv("Total Savings (INR)", fmt_inr(rb["savings_inr"].sum()) if not rb.empty else "0")
                _kv("Net Margin Recovered (INR)", fmt_inr(rb["net_profit_inr"].sum()) if not rb.empty else "0")
                _kv("Rebook Rate", f"{len(rb)/max(len(ev_data),1)*100:.1f}%")
                pdf.ln(4)
                pdf.set_font("Helvetica", "B", 11)
                pdf.set_text_color(71, 85, 105)
                pdf.cell(0, 8, "Top 10 Rebookings by Savings", ln=True)
                if not rb.empty:
                    top_rb = rb.nlargest(10, "savings_inr")
                    _table(top_rb, ["booking_id","city","old_supplier","new_supplier","savings_inr","decision"],
                           [35, 25, 30, 30, 35, 25])
            else:
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 7, "No rebooking data available", ln=True)

            # -- KPI Summary --
            if not kpi.empty:
                pdf.add_page()
                _section("Weekly KPI Summary", 139, 92, 246)
                kpi_cols = ["week_start","properties_blocked","total_rooms_blocked","expected_uplift_inr","margin_recovered_inr"]
                avail_cols = [c for c in kpi_cols if c in kpi.columns]
                _table(kpi, avail_cols, [38]*len(avail_cols))

            # Footer
            pdf.set_y(-20)
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(148, 163, 184)
            pdf.cell(0, 10, "Generated by Agentic AI: Demand Blocking & Smart Rebooking Intelligence", align="C")

            return pdf.output()

        pdf_bytes = bytes(_build_pdf())
        st.download_button("📥 Combined Report (PDF)", data=pdf_bytes,
                           file_name=f"weekly_report_{ws}.pdf", mime="application/pdf",
                           use_container_width=True)

    spc(6)
    page_nav("Report Agent")

# ═══════════════════════════════════════════════════════
# PAGE 7: PROPERTY EXPLORER
# ═══════════════════════════════════════════════════════
def pg_property(D, ws):
    st.markdown("# Property Explorer")
    ibox("Deep-dive into any property: occupancy history, sell-out patterns, blocking actions, rebooking evaluations. "
         "Useful for understanding why the system blocked (or didn't block) a specific hotel.","b")
    pm=D["property_master"]
    if pm.empty: st.warning("No data."); return
    c1,c2=st.columns([1,3])
    with c1:
        cities=sorted(pm["city"].dropna().unique().tolist())
        sel_city=st.selectbox("Select City",cities,key="prop_city")
    city_props=pm[pm["city"]==sel_city]
    with c2:
        sel=st.selectbox("Select Property",city_props["property_id"].unique(),key="prop_sel")
    if not sel: return
    pi=pm[pm["property_id"]==sel].iloc[0]
    sdiv()

    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Name",pi.get("property_id",""))
    c2.metric("City",f"{pi.get('city','')} ({pi.get('city_tier','')})")
    c3.metric("Stars",int(pi.get("star_rating",3)))
    c4.metric("Base ADR",f"INR {fmt_inr(pi.get('base_adr_inr',0))}")
    c5.metric("Popularity",f"{pi.get('popularity_index',0):.2f}")
    spc(6)
    sdiv()

    pd_df=D["property_daily"]
    if not pd_df.empty:
        pd_df["date"]=pd.to_datetime(pd_df["date"])
        pp=pd_df[(pd_df["property_id"]==sel)&(pd_df["date"]>=pd.Timestamp("2026-01-01"))].sort_values("date")
        if not pp.empty:
            pp["occ"]=np.where(pp["base_inventory_rooms"]>0,pp["rooms_sold"]/pp["base_inventory_rooms"],0)
            st.markdown("## Occupancy & Bookings")
            ibox("📊 <b>Daily occupancy rate (%):</b>  How full the hotel is each day. "
                 "<span style='color:#ef4444;font-weight:700'>Red markers</span> = sold-out days (100% full). "
                 "<span style='color:#f59e0b;font-weight:700'>Orange dashed line</span> = 85% danger zone.<br>"
                 "📊 <b>Daily booking requests:</b>  A leading indicator. "
                 "When requests spike while occupancy is already high, a sell-out is likely.","g")

            so=pp[pp["sold_out_flag"]==1]
            td=len(pp); sd2=int(so.shape[0]); ao=pp["occ"].mean()

            # -- Occupancy Chart --
            fig_occ = go.Figure()
            fig_occ.add_trace(go.Scatter(
                x=pp["date"], y=pp["occ"]*100, mode="lines+markers",
                line=dict(color="#6366f1", width=2.5),
                marker=dict(size=4, color="#6366f1"),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
                name="Occupancy %", hovertemplate="Date: %{x|%d %b}<br>Occupancy: %{y:.1f}%<extra></extra>"))
            if not so.empty:
                fig_occ.add_trace(go.Scatter(
                    x=so["date"], y=so["occ"]*100, mode="markers",
                    marker=dict(color="#ef4444", size=10, symbol="x", line=dict(width=2, color="#ef4444")),
                    name=f"Sold Out ({sd2} days)", hovertemplate="Date: %{x|%d %b}<br>SOLD OUT<extra></extra>"))
            fig_occ.add_hline(y=85, line_dash="dash", line_color="#f59e0b", line_width=1.5,
                              annotation_text="85% Danger Zone", annotation_position="top left",
                              annotation_font=dict(color="#f59e0b", size=11, family="Inter"))
            fig_occ.update_layout(
                title=dict(text="Daily Occupancy Rate", font=dict(size=15, family="Inter", color="#1e293b")),
                yaxis=dict(title="Occupancy %", range=[0, 105], ticksuffix="%", gridcolor="rgba(0,0,0,0.05)"),
                xaxis=dict(title="", gridcolor="rgba(0,0,0,0.05)"),
                height=320, template="plotly_white", margin=dict(t=50, b=10, l=60, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
                hovermode="x unified")
            st.plotly_chart(fig_occ, use_container_width=True)

            # -- Booking Requests Chart --
            fig_req = go.Figure()
            avg_req = pp["booking_requests"].mean()
            spike_mask = pp["booking_requests"] > avg_req * 1.3
            normal = pp[~spike_mask]
            spikes = pp[spike_mask]
            fig_req.add_trace(go.Bar(
                x=normal["date"], y=normal["booking_requests"],
                marker_color="rgba(99,102,241,0.5)", name="Normal Requests",
                hovertemplate="Date: %{x|%d %b}<br>Requests: %{y}<extra></extra>"))
            fig_req.add_trace(go.Bar(
                x=spikes["date"], y=spikes["booking_requests"],
                marker_color="#f59e0b", name="Spike (>130% of Avg)",
                hovertemplate="Date: %{x|%d %b}<br>Requests: %{y} (SPIKE)<extra></extra>"))
            fig_req.add_hline(y=avg_req, line_dash="dot", line_color="#64748b", line_width=1,
                              annotation_text=f"Avg: {avg_req:.0f}", annotation_position="top left",
                              annotation_font=dict(color="#64748b", size=11, family="Inter"))
            fig_req.update_layout(
                title=dict(text="Daily Booking Requests", font=dict(size=15, family="Inter", color="#1e293b")),
                yaxis=dict(title="Requests", gridcolor="rgba(0,0,0,0.05)"),
                xaxis=dict(title="Date", gridcolor="rgba(0,0,0,0.05)"),
                height=300, template="plotly_white", margin=dict(t=50, b=10, l=60, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
                barmode="overlay", hovermode="x unified")
            st.plotly_chart(fig_req, use_container_width=True)

            # -- Summary Metrics --
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total Days", f"{td}")
            mc2.metric("Avg Occupancy", f"{ao:.1%}")
            mc3.metric("Sold-out Days", f"{sd2} ({sd2/max(td,1):.0%})")
    spc(6)
    sdiv()

    bl=D["demand_block_actions"]
    if not bl.empty:
        pb=bl[bl["property_id"]==sel]
        st.markdown("## Blocking History")
        if not pb.empty:
            ibox(f"This property has been blocked <b>{len(pb)} time(s)</b>.","g")
            st_df(fmt_df_inr(pb))
        else:
            ibox("This property has never been blocked.","a")

    ev=D["rebooking_evaluations"]
    if not ev.empty:
        pe=ev[ev["property_id"]==sel]
        st.markdown("## Rebooking History")
        if not pe.empty:
            rb=pe[pe["decision"]=="Rebook"]
            ibox(f"<b>{len(pe)}</b> evaluations, <b>{len(rb)}</b> rebooked, <b>INR {fmt_inr(rb['savings_inr'].sum())}</b> saved.","g")
            st_df(fmt_df_inr(pe))
        else:
            ibox("No rebooking evaluations for this property.","a")
    page_nav("Property Explorer")

# ═══════════════════════════════════════════════════════
# PAGE 8: ML MODELS & DATA
# ═══════════════════════════════════════════════════════
def pg_ml(D, ws):
    st.markdown("# ML Models & Data")
    ibox("Shows model training status, feature importance, supplier reliability data, and the full database schema. "
         "All 3 models use <b>XGBoost</b> (gradient boosted trees) trained on real database data.","b")
    spc(6)
    sdiv()

    st.markdown("## Model Status")
    md=os.path.join(PROJECT_ROOT,"models")
    for name,desc in [("demand_model.pkl","Demand Spike Predictor (XGBRegressor) -- predicts city_demand_multiplier"),
                      ("sellout_model.pkl","Sell-out Probability (XGBClassifier) -- predicts sold_out_flag"),
                      ("price_model.pkl","Price Movement (XGBRegressor) -- predicts future net_rate_inr")]:
        p=os.path.join(md,name)
        if os.path.exists(p):
            sz=os.path.getsize(p)/1024
            st.success(f"**{name}** -- {desc} -- Trained ({sz:.1f} KB)")
        else:
            st.warning(f"**{name}** -- {desc} -- Not yet trained")

    # Feature importance
    meta_path = os.path.join(md, "label_encoders.pkl")
    if os.path.exists(meta_path):
        sdiv()
        st.markdown("## Feature Lists Used by Each Model")
        ibox("These are the actual features the trained models use for prediction. "
             "Features are engineered by the SENSE Agent from raw database columns.","g")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        for mname, key in [("Demand Model","demand_features"),("Sellout Model","sellout_features"),("Price Model","price_features")]:
            feats = meta.get(key, [])
            if feats:
                st.markdown(f"**{mname}** -- {len(feats)} features")
                st.code(", ".join(feats), language=None)

    sdiv()
    st.markdown("## Supplier Reliability")
    ibox("The 8 suppliers and their reliability metrics. <b>booking_failure_rate</b> = how often bookings fail. "
         "<b>preferred_supplier_flag</b> = 1 means the supplier gets priority in blocking decisions.","b")
    sup=D["supplier_reliability"]
    if not sup.empty:
        st_df(sup)
        fig=px.bar(sup,x="supplier_name",y=["booking_failure_rate","supplier_cancellation_rate","dispute_rate"],
                   barmode="group",color_discrete_sequence=["#ef4444","#f59e0b","#8b5cf6"],title="Supplier Risk Comparison")
        fig.update_layout(height=350,template="plotly_white"); st.plotly_chart(fig,use_container_width=True)

    spc(6)
    page_nav("ML Models & Data")

# ═══════════════════════════════════════════════════════
# PAGE 9: ASK AI ASSISTANT
# ═══════════════════════════════════════════════════════
def pg_ai(D, ws):
    hero_banner(
        "Revenue AI Insights",
        "Interact with our autonomous AI engine to analyze revenue leakage, demand surges, and smart rebooking opportunities in real time.",
        "Real-Time AI Processing"
    )

    # Build context summary for the AI
    kpi = D["weekly_kpi_summary"]
    bl = D["demand_block_actions"]
    ev = D["rebooking_evaluations"]
    pm = D["property_master"]
    sup = D["supplier_reliability"]

    context_parts = []
    if not bl.empty:
        bl["week_start"] = pd.to_datetime(bl["week_start"])
        bl_wk = bl[bl["week_start"] == pd.to_datetime(ws)]
        if not bl_wk.empty:
            context_parts.append(f"Blocking (week {ws}): {len(bl_wk)} total blocks, "
                                 f"{bl_wk['property_id'].nunique()} unique properties, "
                                 f"{int(bl_wk['rooms_blocked_per_night'].sum())} rooms, "
                                 f"INR {fmt_inr(bl_wk['expected_revenue_uplift_inr'].sum())} uplift, "
                                 f"{bl_wk['city'].nunique()} cities.")
        else:
            context_parts.append(f"Blocking: {len(bl)} total blocks across all weeks ({bl['property_id'].nunique()} properties, {bl['city'].nunique()} cities). No blocks for week {ws}.")
    if not kpi.empty:
        kpi["week_start"] = pd.to_datetime(kpi["week_start"])
        latest = kpi.sort_values("week_start").iloc[-1]
        context_parts.append(f"KPI summary (week {latest['week_start'].date()}): "
                             f"{int(latest.get('rebook_count',0))} rebookings, "
                             f"INR {fmt_inr(latest.get('margin_recovered_inr',0))} margin recovered.")
    if not ev.empty:
        if "week_start" in ev.columns:
            ev["week_start"] = pd.to_datetime(ev["week_start"])
            ev_wk = ev[ev["week_start"] == pd.to_datetime(ws)]
        else:
            ev_wk = ev
        if not ev_wk.empty:
            ev_dec = ev_wk["decision"].str.strip().str.title()
            rb_count = (ev_dec == "Rebook").sum()
            sk_count = (ev_dec == "Skip").sum()
            context_parts.append(f"Rebooking (week {ws}): {len(ev_wk)} evaluations, {rb_count} rebooked, {sk_count} skipped.")
        else:
            context_parts.append(f"Rebooking: No evaluations for week {ws}. {len(ev)} total across all weeks.")
    if not pm.empty:
        context_parts.append(f"Portfolio: {len(pm)} properties across {pm['city'].nunique()} cities.")
    if not sup.empty:
        context_parts.append(f"Suppliers: {len(sup)} suppliers. Best: {sup.loc[sup['booking_failure_rate'].idxmin(),'supplier_name']}.")

    data_context = " ".join(context_parts)

    system_msg = (
        "You are an expert hotel revenue management AI assistant for the Demand Blocking & Smart Rebooking Intelligence Agentic AI system. "
        "The system uses 7 autonomous agents: Sense, Predict (XGBoost), Decide, Reserve, Monitor, Optimize, Report. "
        "It performs demand blocking (proactively reserving rooms before spikes) and smart rebooking "
        "(switching to cheaper suppliers when safe). "
        f"Current data context: {data_context} "
        "Answer questions clearly with data-driven insights. Use bullet points for recommendations. "
        "If asked about technical details, explain the ML models (XGBoost), scoring formulas, and thresholds."
    )

    # Preset questions
    st.markdown("## Smart Revenue Queries")
    preset_cols = st.columns(4)
    presets = [
        "What is the overall health of our blocking strategy?",
        "Which cities should we prioritize for blocking next week?",
        "How can we improve our rebooking rate?",
        "Explain how the composite blocking score works.",
    ]
    if "ai_question" not in st.session_state:
        st.session_state["ai_question"] = ""
    for i, (col, q) in enumerate(zip(preset_cols, presets)):
        with col:
            if st.button(q, key=f"preset_{i}", use_container_width=True):
                st.session_state["ai_question"] = q
                st.rerun()
    spc(6)
    sdiv()
    st.markdown("## Ask the Revenue AI")
    user_q = st.text_area("Type your question about the data, strategy, or system:", height=100,
                          value=st.session_state.get("ai_question", ""),
                          placeholder="e.g., Why were properties in Mumbai blocked more than Delhi?")

    if st.button("Ask AI to Generate Revenue Insights", key="ask_ai_main", use_container_width=True):
        if user_q.strip():
            st.session_state["ai_question"] = ""
            with st.spinner("Running AI Analysis\u2026"):
                answer = ask_gpt(user_q, system_msg=system_msg, max_tokens=500)
                ai_insight_box(answer)
        else:
            st.warning("Please type a question or click a preset above.")
    spc(6)
    sdiv()
    st.markdown("## Live Revenue Data Coverage")
    ibox(f"Processing <b>{sum(len(v) for v in D.values()):,}+</b> records across "
         f"<b>{pm['city'].nunique() if not pm.empty else 0}</b> cities and "
         f"<b>{len(pm) if not pm.empty else 0}</b> properties with real-time ML scoring.", "p")
    page_nav("Revenue AI Insights")


#  Week Training Check 
def has_week_data(D, ws):
    """Check if pipeline data exists for the selected week."""
    kpi = D.get("weekly_kpi_summary", pd.DataFrame())
    if not kpi.empty:
        kpi["week_start"] = pd.to_datetime(kpi["week_start"])
        if not kpi[kpi["week_start"] == pd.to_datetime(ws)].empty:
            return True
    bl = D.get("demand_block_actions", pd.DataFrame())
    if not bl.empty:
        bl["week_start"] = pd.to_datetime(bl["week_start"])
        if not bl[bl["week_start"] == pd.to_datetime(ws)].empty:
            return True
    return False

def show_training_prompt(ws):
    """Show training required prompt with a button to trigger the pipeline."""
    st.markdown("---")
    st.warning(f"No pipeline data found for week **{ws}**. Training is required to generate predictions and blocking decisions.")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button(f"Train Pipeline for {ws}", use_container_width=True, type="primary", key="train_btn"):
            st.session_state["training_in_progress"] = True
            st.session_state["training_week"] = ws
            st.rerun()

    if st.session_state.get("training_in_progress") and st.session_state.get("training_week") == ws:
        with st.spinner(f"Running pipeline for week {ws}... This may take 2-3 minutes."):
            try:
                result = subprocess.run(
                    [sys.executable, "main.py", "--week-start", ws, "--train"],
                    cwd=PROJECT_ROOT,
                    capture_output=True, text=True, timeout=600
                )
                st.session_state["training_in_progress"] = False
                if result.returncode == 0:
                    st.success(f"Pipeline completed for week {ws}! Refreshing data...")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"Pipeline failed. Error:\n```\n{result.stderr[-500:]}\n```")
            except subprocess.TimeoutExpired:
                st.session_state["training_in_progress"] = False
                st.error("Pipeline timed out after 10 minutes.")
            except Exception as e:
                st.session_state["training_in_progress"] = False
                st.error(f"Error running pipeline: {e}")
    return False

# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    pg, ws = sidebar()

    with st.spinner("Agentic AI — Loading data..."):
        D = load_all()

    # Pages that need week data
    data_pages = {"Executive Overview", "Sense & Predict Agent",
                  "Decide & Reserve Agent", "Monitor & Optimize Agent",
                  "Report Agent"}

    if pg in data_pages and not has_week_data(D, ws):
        hero_banner("Training Required", f"No data for week {ws}. Run the pipeline to generate predictions.", f"Week: {ws}")
        show_training_prompt(ws)
    else:
        m = {"Business Context":pg_business_context,
             "Executive Overview":pg_overview,"How It Works":pg_how,
             "Sense & Predict Agent":pg_demand,"Decide & Reserve Agent":pg_blocking,
             "Monitor & Optimize Agent":pg_rebooking,"Report Agent":pg_kpi,
             "Property Explorer":pg_property,"ML Models & Data":pg_ml,
             "Revenue AI Insights":pg_ai}
        m.get(pg, pg_overview)(D, ws)

if __name__ == "__main__":
    main()
