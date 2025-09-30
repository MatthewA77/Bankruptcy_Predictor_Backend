import os
from fastapi import FastAPI, HTTPException, Response, APIRouter
from pydantic import BaseModel
from .inference import build_features_for_ticker_df, get_company_details, predict_from_features, shap_bar_png, get_ticker_object, shap_contributions_json
from .news import get_company_news
from fastapi.responses import JSONResponse
import yfinance as yf
from typing import List, Dict, Optional, Tuple

app = FastAPI(title="Bankruptcy Predictor API")
router = APIRouter()

class PredictTickerInput(BaseModel):
    ticker: str

class PredictManualInput(BaseModel):
    features: dict

class DebugFeaturesResponse(BaseModel):
    columns: List[str]
    values: Dict[str, float]

class CompanyBrief(BaseModel):
    symbol: str
    shortName: Optional[str] = None
    longName: Optional[str] = None
    industry: Optional[str] = None
    sector: Optional[str] = None
    country: Optional[str] = None
    website: Optional[str] = None
    ceo: Optional[str] = None
    summary: Optional[str] = None
    longBusinessSummary: Optional[str] = None

class PredictResponse(BaseModel):
    ticker: str
    probability: float
    top_contributions: List[Tuple[str, float]]
    company: Optional[CompanyBrief] = None
    news: Optional[list] = None

class NewsItemModel(BaseModel):
    title: str
    link: str
    publisher: Optional[str] = None
    published: Optional[str] = None
    summary: Optional[str] = None

def _resolve_ticker_or_404(ticker: str):
    t = get_ticker_object(ticker)
    if t is None:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")
    return t


@router.get("/debug/features", response_model=DebugFeaturesResponse)
def debug_features(ticker: str):
    _resolve_ticker_or_404(ticker)

    df = build_features_for_ticker_df(ticker)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No financial statements for '{ticker}'")

    if not df.to_numpy().any():
        raise HTTPException(status_code=404, detail=f"No usable features for '{ticker}'")

    return DebugFeaturesResponse(
        columns=list(df.columns),
        values=df.iloc[0].to_dict()
    )

@router.get("/news/{ticker}", response_model=List[NewsItemModel])
def news_for_ticker(ticker: str):
    t = _resolve_ticker_or_404(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    query = info.get("shortName") or ticker.strip().upper()

    items = get_company_news(query)
    return items or []

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict/ticker")
def predict_ticker(inp: PredictTickerInput):
    try:
        ticker = inp.ticker.upper().strip()
        yfTicker = get_ticker_object(ticker)
        if not ticker or yfTicker is None:
            raise HTTPException(status_code=400, detail=f"Ticker {ticker} not found.")

        # 1) Predict
        df = build_features_for_ticker_df(ticker)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Could not build features from yfinance.")
        prob, contrib, _ = predict_from_features(df)

        # 2) Company details
        company = get_company_details(ticker)

        # 3) News (shortName + ticker)
        try:
            tk = get_ticker_object(ticker)
            info = tk.info or {}
            query = f"{info.get('shortName', ticker)} {ticker}"
        except Exception:
            query = ticker
        news = get_company_news(query, limit=5) or []

        return {
            "ticker": ticker,
            "probability": prob,
            "top_contributions": contrib,
            "company": company,
            "news": news,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/predict/manual")
def predict_manual(inp: PredictManualInput):
    import pandas as pd
    df = pd.DataFrame([inp.features])
    prob, contrib, _ = predict_from_features(df)
    return {"probability": prob, "top_contributions": contrib}  

@app.get("/shap/bar.png")
def shap_bar_png_endpoint(ticker: str):
    img = shap_bar_png(ticker=ticker)
    return Response(content=img, media_type="image/png")

@app.get("/debug/features")
def debug_features(ticker: str):
    df = build_features_for_ticker_df(ticker)
    return {"columns": list(df.columns), "values": df.iloc[0].to_dict()}

@app.get("/debug/model")
def debug_model():
    from .inference import _ensure_models_loaded, _scaler
    _ensure_models_loaded()
    names = getattr(_scaler, "feature_names_in_", None)
    return {
        "loaded": _scaler is not None,
        "feature_names_in_present": names is not None,
        "scaler_feature_names": list(names) if names is not None else []
    }

@app.get("/debug/paths")
def debug_paths():
    import os, traceback
    from pathlib import Path
    from .inference import _ensure_models_loaded, MODELS_DIR, _model, _scaler, _explainer

    data = {
        "cwd": os.getcwd(),
        "MODELS_DIR": str(MODELS_DIR),
        "models_dir_listing": [],
        "loaded": False,
        "model_loaded": False,
        "scaler_loaded": False,
        "explainer_loaded": False,
        "error": None,
    }
    try:
        p = Path(MODELS_DIR)
        if p.exists() and p.is_dir():
            data["models_dir_listing"] = sorted([f.name for f in p.iterdir()])
        _ensure_models_loaded()
        data["loaded"] = True
        data["model_loaded"] = _model is not None
        data["scaler_loaded"] = _scaler is not None
        data["explainer_loaded"] = _explainer is not None
    except Exception as e:
        data["error"] = f"{type(e).__name__}: {e}"
    return data


@app.get("/debug/topfeatures")
def debug_topfeatures():
    from .inference import _ensure_models_loaded, _top_features
    _ensure_models_loaded()
    return {"top_features": _top_features}


@app.get("/company/{ticker}")
def company_details(ticker: str):
    try:
        data = get_company_details(ticker.upper())
        if not data or data.get("name") == "N/A":
            raise HTTPException(status_code=404, detail="Company details not found.")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch company details: {e}")
    
@app.get("/news/{ticker}")
def company_news(ticker: str, days: int = 7):
    try:
        tk = get_ticker_object(ticker)
        info = tk.info or {}
        query = f"{info.get('shortName', ticker)} {ticker}"
        articles = get_company_news(query, limit=5)
        if not articles:
            raise HTTPException(status_code=404, detail="No news found.")
        return articles
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch news: {e}")
    
@app.get("/explain/{ticker}")
def explain_ticker(ticker: str, k: int = 10):
    try:
        data = shap_contributions_json(ticker=ticker.upper(), top_k=k)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
    "https://matta779-bankruptcy-predictor-backend.hf.space",
    "https://bankruptcy-predictor-frontend.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
