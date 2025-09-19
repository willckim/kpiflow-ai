# KPIFlow AI — Starter

A lightweight KPI dashboard + AI insights demo. Upload a CSV to auto-build KPI cards, charts, and executive insights.

## Quickstart (VS Code)

```bash
# 1) Create & activate a venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Optional: set your OpenAI key
cp .env.example .env   # then add your key

# 4) Run the Streamlit app
streamlit run frontend/streamlit_app.py

# (Optional) Run the FastAPI backend for REST access
uvicorn backend.app:app --reload --port 8000
```

Open http://localhost:8501 for the UI. A sample CSV is in `data/sample_sales.csv`.
