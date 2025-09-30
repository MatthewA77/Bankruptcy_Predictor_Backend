# --- imports ---
import os, io, re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap
import yfinance as yf

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Models path: default to projectroot/models, or override via MODELS_DIR
# -----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_MODELS_DIR = (_THIS_DIR.parent / "models")  # ../models relative to app/
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(_DEFAULT_MODELS_DIR))).resolve()

_model = None
_scaler = None
_explainer = None
_top_features = None  # <--- authoratative feature list (strings)


def _ensure_models_loaded():
    """Lazy load model artifacts; raise clear error if any missing."""
    global _model, _scaler, _explainer, _top_features
    if _model is not None:
        return

    model_p = MODELS_DIR / "random_forest_model.joblib"
    scaler_p = MODELS_DIR / "scaler.joblib"
    expl_p  = MODELS_DIR / "shap_explainer.joblib"
    topf_p  = MODELS_DIR / "top_10_features.joblib"

    missing = [str(p) for p in [model_p, scaler_p, expl_p, topf_p] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Model artifacts not found:\n  " + "\n  ".join(missing) +
            f"\nResolved MODELS_DIR = {MODELS_DIR}\n"
            "Set MODELS_DIR env var or place files in the 'models/' folder."
        )

    _model = joblib.load(model_p)
    _scaler = joblib.load(scaler_p)
    _explainer = joblib.load(expl_p)
    _top_features = joblib.load(topf_p)  # list[str] – the exact columns used in training


# -----------------------------------------------------------------------------
# Feature name handling
# -----------------------------------------------------------------------------
# Fallback (only used if neither _top_features nor scaler names are available)
FALLBACK_TRAINING_FEATURE_NAMES = [
    " Net Income to Stockholder's Equity",
    " Net Value Growth Rate",
    " Persistent EPS in the Last Four Seasons",
    " Borrowing dependency",
    " Per Share Net profit before tax (Yuan ¥)",
    " Total debt/Total net worth",
    " Net Value Per Share (A)",
    " Net Income to Total Assets",
    " Degree of Financial Leverage (DFL)",
    " Interest Expense Ratio",
]

def _normalize_name(s: str) -> str:
    """Lowercase, strip, drop parentheses text (e.g., '(Yuan ¥)'), and non-alnum."""
    s = s.lower().strip()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def _needed_feature_names() -> list[str]:
    """
    Prefer the explicit top 10 saved at training time.
    Fall back to scaler.feature_names_in_ if present; else the constant fallback list.
    """
    if _top_features is not None:
        return list(_top_features)
    if _scaler is not None and hasattr(_scaler, "feature_names_in_"):
        names = list(getattr(_scaler, "feature_names_in_", []))
        if names:
            return names
    return FALLBACK_TRAINING_FEATURE_NAMES

def _align_columns(df: pd.DataFrame, needed_names: list[str]) -> pd.DataFrame:
    """
    Align df columns to needed_names via fuzzy matching:
    - ignore leading spaces, punctuation, and any '(...)' suffix like '(Yuan ¥)'
    - fill missing with zeros
    Output column ORDER exactly matches needed_names (the training order).
    """
    cur_norm_to_actual = {_normalize_name(c): c for c in df.columns}
    cols = []
    for need in needed_names:
        n_need = _normalize_name(need)
        actual = cur_norm_to_actual.get(n_need)
        if actual is not None:
            cols.append(df[actual])
        else:
            cols.append(pd.Series([0.0] * len(df), index=df.index))
    X = pd.concat(cols, axis=1)
    X.columns = needed_names
    return X


# -----------------------------------------------------------------------------
# Your displayed feature dictionary (for tooltips, docs, etc.)
# -----------------------------------------------------------------------------
FEATURE_DEFINITIONS = {
    "Net Income to Stockholder's Equity": "Also known as Return on Equity (ROE). Measures a company's profitability by revealing how much profit a company generates with the money shareholders have invested.",
    "Net Value Growth Rate": "Shows the percentage increase or decrease in a company's net worth (Assets - Liabilities) from one period to the next. A higher rate indicates a growing company.",
    "Persistent EPS in the Last Four Seasons": "The average Earnings Per Share (EPS) over the past four quarters. It indicates the consistency and stability of a company's profitability.",
    "Borrowing dependency": "Calculated as Total Liabilities / Total Assets. This ratio shows the extent to which a company relies on debt to finance its assets. A high ratio can indicate high risk.",
    "Per Share Net profit before tax": "The company's profit before taxes, divided by the number of outstanding shares. It shows profitability on a per-share basis.",
    "Total debt/Total net worth": "A leverage ratio that compares a company's total debt to its total net worth. It measures how much debt is used to finance the company's assets relative to the value owned by shareholders.",
    "Net Value Per Share (A)": "The company's net worth (Assets - Liabilities) divided by the number of outstanding shares. It represents the intrinsic value of a single share.",
    "Net Income to Total Assets": "Also known as Return on Assets (ROA). This ratio indicates how profitable a company is in relation to its total assets. It measures how efficiently a company is using its assets to generate earnings.",
    "Degree of Financial Leverage (DFL)": "Measures the sensitivity of a company's earnings per share to fluctuations in its operating income, as a result of changes in its capital structure. A high DFL means a small change in operating income will lead to a large change in earnings.",
    "Interest Expense Ratio": "Calculated as Interest Expense / Total Revenue. This ratio shows the proportion of a company's revenue that is used to pay the interest on its debt."
}
FEATURE_LIST = list(FEATURE_DEFINITIONS.keys())


# -----------------------------------------------------------------------------
# Helpers for ratios and yfinance pulls
# -----------------------------------------------------------------------------
def _safe_div(num, den):
    try:
        if den and float(den) != 0:
            return float(num) / float(den)
    except Exception:
        pass
    return 0.0

def _get_value(df: pd.DataFrame, key: str, col) -> float:
    try:
        if key in df.index and col in df.columns:
            val = df.loc[key, col]
            return float(val) if pd.notnull(val) else 0.0
    except Exception:
        pass
    return 0.0

def _pick_latest_year_col(df: pd.DataFrame):
    if df is None or df.empty:
        return None, None
    cols = list(df.columns)
    if not cols:
        return None, None
    cur = cols[0]
    prev = cols[1] if len(cols) > 1 else None
    return cur, prev

def _fetch_raw_financials(ticker: str):
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    income_stmt = tk.financials if tk.financials is not None else pd.DataFrame()
    balance_sheet = tk.balance_sheet if tk.balance_sheet is not None else pd.DataFrame()
    quarterly_earnings = tk.quarterly_earnings if tk.quarterly_earnings is not None else pd.DataFrame()
    return info, income_stmt, balance_sheet, quarterly_earnings


# -----------------------------------------------------------------------------
# Mapping from statements -> the 10 engineered features (column names = FEATURE_LIST)
# -----------------------------------------------------------------------------
def _map_data_to_features_for_year(
    info: dict,
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    quarterly_earnings: pd.DataFrame,
    feature_list: list[str],
    year_col,
    prev_year_col=None,
) -> pd.DataFrame:
    # Lines with simple fallbacks
    total_assets = _get_value(balance_sheet, 'Total Assets', year_col)
    total_liabilities = _get_value(balance_sheet, 'Total Liab', year_col)
    if total_liabilities == 0:
        total_liabilities = _get_value(balance_sheet, 'Total Liabilities Net Minority Interest', year_col)

    stockholders_equity = _get_value(balance_sheet, 'Stockholders Equity', year_col)
    if stockholders_equity == 0:
        stockholders_equity = _get_value(balance_sheet, 'Total Stockholder Equity', year_col)

    total_debt = _get_value(balance_sheet, 'Total Debt', year_col)

    net_worth = total_assets - total_liabilities

    net_income = _get_value(income_stmt, 'Net Income', year_col)

    ebt = _get_value(income_stmt, 'Pretax Income', year_col)
    if ebt == 0:
        ebt = _get_value(income_stmt, 'EBT', year_col)

    ebit = _get_value(income_stmt, 'EBIT', year_col)
    interest_expense = _get_value(income_stmt, 'Interest Expense', year_col)
    total_revenue = _get_value(income_stmt, 'Total Revenue', year_col)

    shares_outstanding = float(info.get('sharesOutstanding', 0) or 0)

    # Ratios
    net_income_to_equity = _safe_div(net_income, stockholders_equity)

    net_value_growth = 0.0
    if prev_year_col is not None:
        prev_assets = _get_value(balance_sheet, 'Total Assets', prev_year_col)
        prev_liabilities = _get_value(balance_sheet, 'Total Liab', prev_year_col)
        if prev_liabilities == 0:
            prev_liabilities = _get_value(balance_sheet, 'Total Liabilities Net Minority Interest', prev_year_col)
        prev_net_worth = prev_assets - prev_liabilities
        if prev_net_worth and float(prev_net_worth) != 0:
            net_value_growth = (net_worth - prev_net_worth) / abs(prev_net_worth)

    if quarterly_earnings is not None and not quarterly_earnings.empty and 'EPS' in quarterly_earnings.columns:
        persistent_eps = float(pd.to_numeric(quarterly_earnings['EPS'], errors='coerce').dropna().mean() or 0)
    else:
        persistent_eps = float(info.get('trailingEps', 0) or 0)

    borrowing_dependency = _safe_div(total_liabilities, total_assets)
    profit_before_tax_per_share = _safe_div(ebt, shares_outstanding)
    debt_to_net_worth = _safe_div(total_debt, net_worth)
    net_value_per_share = _safe_div(net_worth, shares_outstanding)
    net_income_to_assets = _safe_div(net_income, total_assets)

    dfl = 0.0
    denom = (ebit - interest_expense)
    if denom and float(denom) != 0:
        dfl = _safe_div(ebit, denom)

    interest_expense_ratio = _safe_div(interest_expense, total_revenue)

    feature_mapping = {
        "Net Income to Stockholder's Equity": net_income_to_equity,
        "Net Value Growth Rate": net_value_growth,
        "Persistent EPS in the Last Four Seasons": persistent_eps,
        "Borrowing dependency": borrowing_dependency,
        # provide BOTH keys so either name works
        "Per Share Net profit before tax": profit_before_tax_per_share,
        "Per Share Net profit before tax (Yuan ¥)": profit_before_tax_per_share,
        "Total debt/Total net worth": debt_to_net_worth,
        "Net Value Per Share (A)": net_value_per_share,
        "Net Income to Total Assets": net_income_to_assets,
        "Degree of Financial Leverage (DFL)": dfl,
        "Interest Expense Ratio": interest_expense_ratio
    }

    def _resolve(k: str):
        v = feature_mapping.get(k)
        if v is not None: return v
        ks = k.strip()
        v = feature_mapping.get(ks)
        if v is not None: return v
        ks2 = ks.replace(" (Yuan ¥)", "")
        return feature_mapping.get(ks2, 0.0)

    ordered_values = [_resolve(feat) for feat in feature_list]
    return pd.DataFrame([ordered_values], columns=feature_list)


def _build_feature_frame_for_ticker(ticker: str) -> pd.DataFrame:
    _ensure_models_loaded()
    needed_names = _needed_feature_names()

    info, income_stmt, balance_sheet, q_earn = _fetch_raw_financials(ticker)
    year_col, prev_col = _pick_latest_year_col(balance_sheet)
    if year_col is None:
        year_col, prev_col = _pick_latest_year_col(income_stmt)
    if year_col is None:
        return pd.DataFrame([[0.0]*len(needed_names)], columns=needed_names)

    df = _map_data_to_features_for_year(
        info=info,
        income_stmt=income_stmt if income_stmt is not None else pd.DataFrame(),
        balance_sheet=balance_sheet if balance_sheet is not None else pd.DataFrame(),
        quarterly_earnings=q_earn if q_earn is not None else pd.DataFrame(),
        feature_list=needed_names,
        year_col=year_col,
        prev_year_col=prev_col
    )
    return df

# add helper
def _scaler_names():
    if _scaler is not None and hasattr(_scaler, "feature_names_in_"):
        return list(_scaler.feature_names_in_)
    return None

def _decide_final_order(df_cols: list[str]) -> list[str]:
    """
    Decide the final column order for inference, prioritizing *actual* scaler names if available,
    else the saved top_10_features, else the fallback. Also ensure shape matches model/scaler.
    """
    _ensure_models_loaded()
    top_names = _needed_feature_names()
    scaler_names = _scaler_names()

    if scaler_names:
        return scaler_names

    return top_names

def _pick_pos_class_index():
    """Return the index of the 'positive' class (1 if present, else last)."""
    cls = list(getattr(_model, "classes_", []))
    if 1 in cls:
        return cls.index(1)
    return len(cls) - 1 if cls else 0

def _extract_shap_row(shap_values, sample_index: int = 0) -> np.ndarray:
    """
    Normalize various SHAP outputs into a 1D array (n_features,) for a single sample.
    Handles:
      - list of arrays (per class): [ (n_samples, n_features), ... ]
      - 3D array: (n_samples, n_features, n_classes)
      - 2D array: (n_samples, n_features)
    Picks the positive class (index per model.classes_) when there is a class axis.
    """
    pos_idx = _pick_pos_class_index()

    # Case A: TreeExplainer(list per class)
    if isinstance(shap_values, list):
        # choose positive class's matrix
        arr = shap_values[pos_idx]
        # arr: (n_samples, n_features) or (n_samples, n_features, ?)
        row = arr[sample_index]
        # if still has extra trailing dims, select pos_idx if possible, else flatten
        if row.ndim == 2:
            if row.shape[-1] > pos_idx:
                row = row[:, pos_idx]
            else:
                row = row.mean(axis=-1)
        return np.asarray(row)

    # Case B: single ndarray
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        if arr.shape[-1] > pos_idx:
            row = arr[sample_index, :, pos_idx]
        else:
            row = arr[sample_index].mean(axis=-1)
        return np.asarray(row)

    if arr.ndim == 2:
        return np.asarray(arr[sample_index])

    row = np.squeeze(arr)
    if row.ndim != 1:
        row = row.reshape(-1)
    return row

def _extract_company_details(info: dict) -> dict:
    def get_top_executive(officers):
        if not officers:
            return "N/A"
        for o in officers:
            if "ceo" in o.get("title", "").lower() or "president" in o.get("title", "").lower():
                return o.get("name", "N/A")
        return officers[0].get("name", "N/A")

    return {
        "name": info.get("longName", "N/A"),
        "symbol": info.get("symbol", "N/A"),
        "country": info.get("country", "N/A"),
        "industry": info.get("industry", "N/A"),
        "currency": info.get("currency", "N/A"),
        "website": info.get("website", "N/A"),
        "ceo": get_top_executive(info.get("companyOfficers", [])),
        "summary": info.get("longBusinessSummary", "N/A"),
    }

# -----------------------------------------------------------------------------
# Prediction & SHAP
# -----------------------------------------------------------------------------
def predict_from_features(df: pd.DataFrame):
    _ensure_models_loaded()

    # Decide exact feature order
    final_order = _decide_final_order(list(df.columns))

    # Reindex & type safety
    X = df.reindex(columns=final_order, fill_value=0.0)
    X = X.astype(float)
    # Validate shapes vs. scaler/model
    try:
        n_scaler = getattr(_scaler, "n_features_in_", None)
        n_model  = getattr(_model,  "n_features_in_", None)
        if n_scaler is not None and X.shape[1] != n_scaler:
            raise ValueError(f"Scaler expects {n_scaler} features but got {X.shape[1]}. Final order={final_order}")
        if n_model is not None and X.shape[1] != n_model:
            # Some pipelines wrap models and n_features_in_ may be missing; this check helps when present
            raise ValueError(f"Model expects {n_model} features but got {X.shape[1]}. Final order={final_order}")
    except Exception as e:
        # Re-raise with context so the endpoint can show it
        raise

    # Scale & predict
    X_scaled = _scaler.transform(X)
    prob = float(_model.predict_proba(X_scaled)[0][1])

    # SHAP (robust)
    contrib = []
    try:
        shap_values = _explainer.shap_values(X_scaled)
        shap_row = _extract_shap_row(shap_values, sample_index=0)  # <- unified 1D vector
        contrib = sorted(
            [(X.columns[i].lstrip(), float(shap_row[i])) for i in range(len(X.columns))],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
    except Exception:
        contrib = []

    return prob, contrib, None

def _shap_bar_png_from_df(df: pd.DataFrame) -> bytes:
    _ensure_models_loaded()
    final_order = _decide_final_order(list(df.columns))
    X = df.reindex(columns=final_order, fill_value=0.0).astype(float)
    X_scaled = _scaler.transform(X)

    try:
        shap_values = _explainer.shap_values(X_scaled)
        row = _extract_shap_row(shap_values, sample_index=0)  # <- unified 1D vector
    except Exception:
        fig = plt.figure() 
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0) 
        return buf.read()

    vals = np.abs(row)
    idxs = np.argsort(vals)[::-1][:10]
    labels = [X.columns[i] for i in idxs]
    heights = vals[idxs]

    plt.figure()
    plt.barh(range(len(labels)), heights)
    plt.yticks(range(len(labels)), labels)
    plt.gca().invert_yaxis()
    plt.title("Top 10 SHAP (abs)")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf.read()

def get_ticker_object(ticker: str):
    """Return a yfinance Ticker object."""
    t = yf.Ticker(ticker)

    # yfinance will always return an object, so check if it has data
    try:
        info = t.info
        if not info or "regularMarketPrice" not in info:
            return None
    except Exception:
        return None

    return t

def get_company_details(ticker: str) -> dict:
    """Return normalized company metadata from yfinance.info."""
    tk = get_ticker_object(ticker)
    info = tk.info or {}

    def get_top_executive(officers):
        if not officers:
            return "N/A"
        for o in officers:
            if "ceo" in (o.get("title") or "").lower() or "president" in (o.get("title") or "").lower():
                return o.get("name") or "N/A"
        return officers[0].get("name") or "N/A"

    return {
        "name": info.get("longName", "N/A"),
        "symbol": info.get("symbol", ticker.upper()),
        "country": info.get("country", "N/A"),
        "industry": info.get("industry", "N/A"),
        "currency": info.get("currency", "N/A"),
        "website": info.get("website", "N/A"),
        "ceo": get_top_executive(info.get("companyOfficers", [])),
        "summary": info.get("longBusinessSummary", "N/A"),
        # keep shortName too (handy for news)
        "shortName": info.get("shortName", None),
        "marketCap": info.get("marketCap", None),
        "trailingPE": info.get("trailingPE", None),
    }

# -----------------------------------------------------------------------------
# Public helpers used by FastAPI endpoints
# -----------------------------------------------------------------------------
def build_features_for_ticker_df(ticker: str) -> pd.DataFrame:
    return _build_feature_frame_for_ticker(ticker)

def shap_bar_png(ticker_df: pd.DataFrame | None = None, *, ticker: str | None = None) -> bytes:
    if ticker_df is None and ticker is not None:
        ticker_df = _build_feature_frame_for_ticker(ticker)
    return _shap_bar_png_from_df(ticker_df)

# --- New: SHAP contributions as JSON (top-k) ---
def shap_contributions_json(ticker_df: pd.DataFrame | None = None, *, ticker: str | None = None, top_k: int = 10) -> dict:
    """
    Returns top-k SHAP contributions with direction for positive (bankruptcy) class.
    Structure:
    {
      "ticker": "TSLA",
      "contributions": [{"feature": "...", "value": 0.0123, "abs": 0.0123, "direction": "increase"|"decrease"}],
      "positive": [...only 'increase'...],
      "negative": [...only 'decrease'...]
    }
    """
    if ticker_df is None and ticker is not None:
        ticker_df = _build_feature_frame_for_ticker(ticker)

    # Reuse existing predictor to get shap values in training order
    prob, contrib, shap_row = predict_from_features(ticker_df)

    # contrib is already sorted by |value| desc [(name, val), ...]
    items = []
    for name, val in contrib[:top_k]:
        items.append({
            "feature": name.strip().replace(' (Yuan ¥)', ''),
            "value": float(val),
            "abs": abs(float(val)),
            "direction": "increase" if val > 0 else "decrease"
        })

    pos = [it for it in items if it["direction"] == "increase"]
    neg = [it for it in items if it["direction"] == "decrease"]

    return {
        "ticker": ticker,
        "contributions": items,
        "positive": pos,
        "negative": neg
    }
