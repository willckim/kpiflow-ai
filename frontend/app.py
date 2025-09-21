# frontend/app.py (Streamlit)
import os
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import streamlit as st
from requests.adapters import HTTPAdapter, Retry

# ---------- Paths ----------
FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parent
SAMPLE_PATH = REPO_ROOT / "data" / "sample_sales.csv"
if not SAMPLE_PATH.exists():
    SAMPLE_PATH = FRONTEND_DIR / "data" / "sample_sales.csv"

# ---------- Config ----------
st.set_page_config(page_title="KPIFlow AI — KPI Dashboard", page_icon="📊", layout="wide")
st.title("📊 KPIFlow AI — KPI & Insights Dashboard")

# ---------- Backend URL (supports multiple styles: secrets + env) ----------
API_BASE = (
    st.secrets.get("api", {}).get("base_url")          # [api][base_url] in Streamlit secrets
    or st.secrets.get("BACKEND_URL")                   # BACKEND_URL in Streamlit secrets
    or os.getenv("API_BASE")                            # env var API_BASE
    or os.getenv("BACKEND_URL")                         # env var BACKEND_URL
    or "http://localhost:8000"                          # local fallback
).rstrip("/")

# ---------- HTTP session with retries ----------
def _session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

# ---------- Sidebar ----------
with st.sidebar:
    st.caption("Connection status")
    try:
        r = _session().get(f"{API_BASE}/health", timeout=5)
        ok = r.ok and (r.json().get("ok") is True or r.json().get("status") == "ok")
    except Exception as _e:
        ok = False
    (st.success if ok else st.error)(f"API: {'connected' if ok else 'unreachable'}")
    st.caption(f"Endpoint: {API_BASE}")

    st.markdown("### Upload Data")
    uploaded = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
    use_sample = st.checkbox(
        f"Use sample dataset ({SAMPLE_PATH.as_posix()})",
        value=True,
        disabled=uploaded is not None,
    )
    analyze = st.button("Analyze")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def read_local_df(uploaded, use_sample):
    """Load the same data locally for charts."""
    # 1) Load
    if uploaded:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    elif use_sample:
        df = pd.read_csv(SAMPLE_PATH)
    else:
        return None

    # 2) Normalize headers (keep original map)
    df.columns = [c.strip() for c in df.columns]
    lc = {c.lower(): c for c in df.columns}

    # 3) Accept common aliases
    date_aliases = ["date", "order_date", "created_at", "transaction_date"]
    revenue_aliases = ["revenue", "sales", "amount", "total", "price"]

    date_key = next((a for a in date_aliases if a in lc), None)
    revenue_key = next((a for a in revenue_aliases if a in lc), None)

    if not date_key or not revenue_key:
        raise ValueError(
            "CSV must include a date-like column (e.g., 'date' or 'order_date') "
            "and a revenue-like column (e.g., 'revenue' or 'sales')."
        )

    # 4) Rename to canonical columns used by the charts
    df.rename(
        columns={
            lc[date_key]: "date",
            lc[revenue_key]: "revenue",
            lc.get("cogs", lc.get("cost")): "cogs" if (lc.get("cogs") or lc.get("cost")) else None,
            lc.get("customer_id", lc.get("customer")): "customer_id" if (lc.get("customer_id") or lc.get("customer")) else None,
        },
        inplace=True,
    )

    # 5) Types & fallbacks
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if "cogs" not in df.columns:
        df["cogs"] = np.nan
    if "customer_id" not in df.columns:
        df["customer_id"] = "unknown"

    return df.sort_values("date")


def post_analyze(file_bytes: bytes, filename: str):
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    resp = _session().post(f"{API_BASE}/analyze", files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()  # {kpis: dict, insights: str}


def kpi_card(label, value, fmt="{:,.2f}"):
    c = st.container(border=True)
    with c:
        st.caption(label)
        if value is None:
            st.markdown("### —")
        else:
            try:
                st.markdown(f"### {fmt.format(value)}")
            except Exception:
                st.markdown(f"### {value}")
    return c

# ---------- UI flow ----------
if not analyze:
    st.info("Upload a file or enable the sample dataset, then click **Analyze**.")
    st.stop()

if uploaded is None and not use_sample:
    st.warning("Please upload a CSV/Excel file or enable the sample dataset.")
    st.stop()

# Prepare file bytes for API
try:
    if uploaded:
        uploaded.seek(0)
        file_bytes = uploaded.read()
        filename = uploaded.name
    else:
        with open(SAMPLE_PATH, "rb") as f:
            file_bytes = f.read()
        filename = "sample_sales.csv"
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

# Call backend API
with st.spinner("Analyzing…"):
    try:
        result = post_analyze(file_bytes, filename)
    except Exception as e:
        st.error(f"API error: {e}")
        st.stop()

# Parse results
kpis = {k.lower(): v for k, v in (result or {}).get("kpis", {}).items()}
insights = (result or {}).get("insights", "")

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Revenue", kpis.get("revenue"))
with c2:
    kpi_card("Gross Profit", kpis.get("gross_profit"))
with c3:
    kpi_card("Gross Margin %", kpis.get("gross_margin_pct"), "{:,.2f}%")
with c4:
    kpi_card("Unique Customers", kpis.get("unique_customers"), "{:,.0f}")

c5, c6, c7, c8 = st.columns(4)
with c5:
    kpi_card("MoM Growth %", kpis.get("mom_growth_pct"), "{:,.2f}%")
with c6:
    kpi_card("Last Month Revenue", kpis.get("last_month_revenue"))
with c7:
    kpi_card("Avg Order Value", kpis.get("avg_order_value"))
with c8:
    kpi_card("Orders", kpis.get("orders"), "{:,.0f}")

# ---------- Insights ----------
with st.expander("💡 AI Insights", expanded=True):
    if insights:
        st.write(insights)
    else:
        st.caption("No insights returned from API.")

# ---------- Visualizations ----------
st.subheader("Visualizations")
try:
    df = read_local_df(uploaded, use_sample)
except Exception as e:
    st.error(f"Could not load data for charts: {e}")
    df = None

if df is not None and not df.empty:
    weekly = df.resample("W-MON", on="date").agg(revenue=("revenue", "sum")).reset_index()
    st.markdown("**Revenue Trend (Weekly)**")
    st.line_chart(weekly, x="date", y="revenue", use_container_width=True)

    if "customer_id" in df.columns:
        tops = (
            df.groupby("customer_id", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
            .head(10)
        )
        st.markdown("**Top Customers by Revenue**")
        st.bar_chart(tops, x="customer_id", y="revenue", use_container_width=True)

    with st.expander("Data preview"):
        st.dataframe(df.head(50), use_container_width=True)
else:
    st.info("Charts will appear after the data is readable locally.")

# ---------- Footer ----------
st.caption("KPIFlow AI • Streamlit frontend + FastAPI backend")
