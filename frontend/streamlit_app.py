import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from shared.analysis import (
    load_data, clean_data, compute_kpis,
    build_charts, generate_insights, export_pptx
)

st.set_page_config(page_title="KPIFlow AI", layout="wide")
st.title("📊 KPIFlow AI — KPI & Insights Dashboard")

# --- Provider status indicator ---
az_ok = all(os.getenv(k) for k in ["AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT"])
oa_ok = bool(os.getenv("OPENAI_API_KEY"))

st.sidebar.markdown("### Connection status")
if az_ok:
    st.sidebar.success(f"Azure: **{os.getenv('AZURE_OPENAI_DEPLOYMENT')}**")
elif oa_ok:
    st.sidebar.info(f"OpenAI: **{os.getenv('OPENAI_MODEL','gpt-5-mini')}**")
else:
    st.sidebar.warning("Heuristic mode (no API keys)")

# --- Uploads ---
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("CSV or Excel", type=["csv", "xls", "xlsx"])
use_sample = st.sidebar.checkbox("Use sample dataset (data/sample_sales.csv)", value=not uploaded)

if use_sample:
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_sales.csv")
    with open(sample_path, "rb") as f:
        file_bytes = f.read()
    filename = "sample_sales.csv"
elif uploaded:
    file_bytes = uploaded.read()
    filename = uploaded.name
else:
    st.info("Upload a dataset or use the sample to get started.")
    st.stop()

# --- Processing ---
df = load_data(file_bytes, filename)
df = clean_data(df)
kpis = compute_kpis(df)
charts = build_charts(df)
insights = generate_insights(df, kpis)

# KPI Cards
st.subheader("KPI Summary")
cols = st.columns(min(4, max(1, len(kpis))))
for i, (k, v) in enumerate(kpis.items()):
    with cols[i % len(cols)]:
        st.metric(k, f"{v:,.2f}" if isinstance(v, float) else v)

# Charts
st.subheader("Visualizations")
for name, fig in charts.items():
    st.plotly_chart(fig, use_container_width=True)

# Insights
st.subheader("AI Insights")
st.write(insights)

# Data preview
with st.expander("Preview cleaned data"):
    st.dataframe(df.head(50))

# Export
st.subheader("Export")
c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Download cleaned CSV",
        data=df.to_csv(index=False),
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )
with c2:
    out_path = os.path.join(os.getcwd(), "kpiflow_report.pptx")
    if st.button("📤 Export PPTX report"):
        export_pptx("KPIFlow AI Report", kpis, insights, charts, out_path)
        with open(out_path, "rb") as f:
            st.download_button(
                "Download PPTX",
                f,
                file_name="kpiflow_report.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                key="pptx_dl",
            )
