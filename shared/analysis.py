import os
import io
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, List, Any

# --- .env support ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- OpenAI / Azure OpenAI availability flags ---
OPENAI_AVAILABLE = False
AZURE_OPENAI_AVAILABLE = False
try:
    from openai import OpenAI  # Direct OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    from openai import AzureOpenAI  # Azure OpenAI client (same package)
    AZURE_OPENAI_AVAILABLE = True
except Exception:
    AZURE_OPENAI_AVAILABLE = False


def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith('.csv'):
        return pd.read_csv(io.BytesIO(file_bytes))
    elif filename.lower().endswith(('.xls', '.xlsx')):
        return pd.read_excel(io.BytesIO(file_bytes))
    else:
        raise ValueError('Unsupported file type. Please upload CSV or Excel.')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for col in df.columns:
        if 'date' in col:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(0)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')
    return df


def _maybe_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ''


def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    revenue_col = _maybe_col(df, ['revenue', 'sales', 'amount', 'total', 'price'])
    cost_col = _maybe_col(df, ['cost', 'cogs'])
    date_col = _maybe_col(df, ['order_date', 'date', 'created_at'])
    customer_col = _maybe_col(df, ['customer_id', 'customer', 'user_id'])

    kpis: Dict[str, Any] = {}
    if revenue_col:
        kpis['Revenue'] = float(df[revenue_col].sum())
    if cost_col and revenue_col:
        gross_profit = df[revenue_col].sum() - df[cost_col].sum()
        kpis['Gross Profit'] = float(gross_profit)
        if df[revenue_col].sum() > 0:
            kpis['Gross Margin %'] = float(100 * gross_profit / df[revenue_col].sum())
    if customer_col:
        kpis['Unique Customers'] = int(df[customer_col].nunique())
    if date_col and revenue_col:
        s = df.dropna(subset=[date_col]).copy()
        if not s.empty:
            s['month'] = s[date_col].dt.to_period('M')
            monthly = s.groupby('month')[revenue_col].sum().sort_index()
            if len(monthly) >= 2:
                last, prev = monthly.iloc[-1], monthly.iloc[-2]
                if prev != 0:
                    kpis['MoM Growth %'] = float(100 * (last - prev) / prev)
                kpis['Last Month Revenue'] = float(last)
    return kpis


def build_charts(df: pd.DataFrame) -> Dict[str, Any]:
    charts: Dict[str, Any] = {}
    revenue_col = _maybe_col(df, ['revenue', 'sales', 'amount', 'total', 'price'])
    cost_col = _maybe_col(df, ['cost', 'cogs'])
    date_col = _maybe_col(df, ['order_date', 'date', 'created_at'])
    region_col = _maybe_col(df, ['region', 'state', 'country'])
    category_col = _maybe_col(df, ['category', 'product', 'item'])

    if date_col and revenue_col:
        d = df.dropna(subset=[date_col]).copy()
        if not d.empty:
            d['date'] = pd.to_datetime(d[date_col])
            ts = d.groupby(pd.Grouper(key='date', freq='W'))[revenue_col].sum().reset_index()
            charts['revenue_trend'] = px.line(ts, x='date', y=revenue_col, title='Revenue Trend (Weekly)')

    if region_col and revenue_col:
        region = df.groupby(region_col)[revenue_col].sum().reset_index().sort_values(revenue_col, ascending=False)
        charts['by_region'] = px.bar(region, x=region_col, y=revenue_col, title='Revenue by Region')

    if category_col and revenue_col:
        cat = df.groupby(category_col)[revenue_col].sum().reset_index().sort_values(revenue_col, ascending=False)
        charts['by_category'] = px.bar(cat, x=category_col, y=revenue_col, title='Revenue by Category')

    if revenue_col and cost_col:
        df2 = df.copy()
        df2['gross_profit'] = df2[revenue_col] - df2[cost_col]
        agg = df2[[revenue_col, 'gross_profit']].sum()
        charts['profit_gauge'] = px.bar(
            x=['Revenue', 'Gross Profit'],
            y=[agg[revenue_col], agg['gross_profit']],
            title='Revenue vs Gross Profit'
        )
    return charts


def _format_kpis_for_prompt(kpis: Dict[str, Any]) -> str:
    parts = [f"- {k}: {round(v, 2) if isinstance(v, float) else v}" for k, v in kpis.items()]
    return "\n".join(parts)


def _ai_client_and_model():
    """Return (client, model_name, is_azure). Prefers Azure if fully configured."""
    az_key = os.getenv("AZURE_OPENAI_KEY")
    az_ep = os.getenv("AZURE_OPENAI_ENDPOINT")
    az_ver = os.getenv("AZURE_OPENAI_API_VERSION")
    az_deploy = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if AZURE_OPENAI_AVAILABLE and az_key and az_ep and az_ver and az_deploy:
        try:
            client = AzureOpenAI(api_key=az_key, api_version=az_ver, azure_endpoint=az_ep)
            return client, az_deploy, True
        except Exception:
            pass

    openai_key = os.getenv("OPENAI_API_KEY")
    if OPENAI_AVAILABLE and openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
            return client, model, False
        except Exception:
            pass

    return None, None, False


def generate_insights(df: pd.DataFrame, kpis: Dict[str, Any]) -> str:
    client, model, is_azure = _ai_client_and_model()
    if client is None or model is None:
        hints = []
        if 'MoM Growth %' in kpis:
            g = kpis['MoM Growth %']
            if g > 0:
                hints.append(f"Revenue increased {g:.1f}% month-over-month, indicating positive momentum.")
            elif g < 0:
                hints.append(f"Revenue decreased {abs(g):.1f}% month-over-month; investigate potential causes.")
        if 'Gross Margin %' in kpis:
            gm = kpis['Gross Margin %']
            hints.append(f"Gross margin is {gm:.1f}%. Consider pricing or cost optimization to improve profitability.")
        if not hints:
            hints.append("Data processed successfully. Add Azure/OpenAI keys in .env to unlock richer insights.")
        return "\n".join(hints)

    prompt = f"""You are an analytics copilot. Given the KPIs below, write 3–6 executive-ready bullet insights.
KPIs:
{_format_kpis_for_prompt(kpis)}
Keep it concise, factual, and helpful. Avoid overclaiming causality."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You write precise business insights."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=250,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        hints = [f"AI insights unavailable ({type(e).__name__}). Showing heuristic insights instead."]
        if 'MoM Growth %' in kpis:
            g = kpis['MoM Growth %']
            if g > 0:
                hints.append(f"Revenue increased {g:.1f}% month-over-month, indicating positive momentum.")
            elif g < 0:
                hints.append(f"Revenue decreased {abs(g):.1f}% month-over-month; investigate potential causes.")
        if 'Gross Margin %' in kpis:
            gm = kpis['Gross Margin %']
            hints.append(f"Gross margin is {gm:.1f}%. Consider pricing or cost optimization to improve profitability.")
        return "\n".join(hints)


def export_pptx(title: str, kpis: Dict[str, Any], insights: str, charts: Dict[str, Any], out_path: str):
    from pptx import Presentation
    from pptx.util import Inches
    import plotly.io as pio
    import os

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = "KPIFlow AI — Auto-generated KPI & Insights"

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "KPI Summary"
    body = slide.shapes.placeholders[1].text_frame
    for k, v in kpis.items():
        body.add_paragraph().text = f"{k}: {round(v, 2) if isinstance(v, float) else v}"

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "AI Insights"
    body = slide.shapes.placeholders[1].text_frame
    for line in insights.split("\n"):
        if line.strip():
            body.add_paragraph().text = line.strip()

    tmp_dir = os.path.join(os.path.dirname(out_path), "_tmp_imgs")
    os.makedirs(tmp_dir, exist_ok=True)
    for name, fig in charts.items():
        img_path = os.path.join(tmp_dir, f"{name}.png")
        pio.write_image(fig, img_path, format="png", width=1280, height=720, scale=2)
        slide = prs.slides.add_slide(prs.slide_layouts[5])  # Title Only
        slide.shapes.title.text = name.replace("_", " ").title()
        slide.shapes.add_picture(img_path, Inches(0.5), Inches(1.5), width=Inches(9))

    prs.save(out_path)
