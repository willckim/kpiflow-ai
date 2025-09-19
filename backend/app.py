# backend/app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from shared.analysis import load_data, clean_data, compute_kpis, generate_insights

app = FastAPI(title="KPIFlow AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeResponse(BaseModel):
    kpis: dict
    insights: str

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    df = load_data(content, file.filename)
    df = clean_data(df)
    kpis = compute_kpis(df)
    insights = generate_insights(df, kpis)
    return AnalyzeResponse(kpis=kpis, insights=insights)

@app.get("/health")
def health():
    return {"ok": True}
