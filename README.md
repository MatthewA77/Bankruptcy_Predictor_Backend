# COBA-ION Backend – Bankruptcy Predictor API

This is the **backend service** for the COBA-ION project (Company Bankruptcy Prediction).  
It is built using **FastAPI** and serves as the core engine to process financial data, run machine learning models, and return results to the frontend.

---

## Features

- Predicts the probability of **company bankruptcy** using a trained **Random Forest** model.
- Provides **SHAP-based explanations** for feature contributions (increase/decrease bankruptcy risk).
- Returns **company details** (industry, sector, market cap, etc.) fetched via Yahoo Finance.
- Fetches and serves **related news articles** for the queried company.
- RESTful API endpoints with **interactive documentation** (Swagger UI and ReDoc).
- Fast, lightweight, and easy to deploy via **Hugging Face Spaces** or other platforms.

---

## Tech Stack

- **Framework:** [FastAPI](https://fastapi.tiangolo.com/)
- **Modeling:** Scikit-learn (Random Forest)
- **Data Handling:** Pandas, NumPy
- **Finance Data:** yfinance
- **Explainability:** SHAP
- **Deployment:** Hugging Face Spaces (Docker)

---
