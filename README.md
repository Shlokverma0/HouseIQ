# HouseIQ

<div align="center">

**A property estimate API using sourced city reference rates, with a synthetic-data ML comparison and assumption-based forecasts.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063.svg)](https://docs.pydantic.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-API%20tests%20included-blue.svg)]()

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Details](#-model-details)
- [Setup & Installation](#-setup--installation)
- [Running the API](#-running-the-api)
- [API Endpoints](#-api-endpoints)
- [Example Request & Response](#-example-request--response)
- [Test Coverage](#-test-coverage)
- [Production Features](#-production-features)
- [Author](#-author)

---

## 🎯 Overview

**HouseIQ** is a REST API that:

- Estimates current values from dated, editable MagicBricks reference rates where available.
- Keeps the synthetic-data ML estimate as a secondary comparison only.
- Forecasts **future prices** (up to 50 years) using economic rates supplied in the request.
- Follows a **clean 4-Layer Architecture** for separation of concerns.
- Includes **logging, rate limiting, input validation, and API tests**.

Built with **FastAPI**, **scikit-learn**, and **Pydantic v2**, this project reflects how ML services are structured in real production environments.

---

## ✨ Key Features

- 🏘️ **Reference-Rate Estimate** — Uses low/typical/high MagicBricks source rates only where they have been recorded and dated.
- 🤖 **Demo ML Comparison** — Trained on 250,000 synthetic Kaggle records; not real market data and not the main estimate.
- 📈 **Future Forecasting** — Predicts prices up to 50 years ahead.
- 🌍 **Economic Assumptions** — Forecasts use inflation, GDP growth, interest, and migration rates supplied in the request.
- 🏗️ **4-Layer Architecture** — Routes → Controllers → Services → Repositories.
- ✅ **Strict Input Validation** — Pydantic v2 with training-range size validation and 42 supported cities.
- 🚦 **Rate Limiting** — 30 requests/min per IP (`slowapi`).
- 📝 **Centralized Logging** — Console + file-based logs.
- ⚡ **Performance Monitoring** — `X-Process-Time` header on every response.
- 🧪 **API Tests** — Root, health, prediction, and compact future forecast responses.
- 📄 **Auto-Generated Docs** — Interactive Swagger UI + ReDoc.

---

## 🏗 Architecture

This project implements the **Controller-Service-Repository pattern** (a variant of Clean Architecture) with 4 distinct layers:

| Layer | Folder | Responsibility |
|-------|--------|----------------|
| **1️⃣ Routes** | `app/routes/` | API endpoints, HTTP method definitions |
| **2️⃣ Controllers** | `app/controllers/` | Request handling, response orchestration |
| **3️⃣ Services** | `app/services/` | Business logic, feature engineering |
| **4️⃣ Repositories** | `app/repositories/` | Model loading, data access |
| **Support** | `app/schemas/` | Pydantic validation schemas |
| **Support** | `app/utils/` | Logger & rate limiter |

### 🔄 Request Flow

```
Client Request
     ↓
[ Routes ] → [ Controllers ] → [ Services ] → [ Repositories ] → Trained Model
     ↑                                                                ↓
     └────────────────────── JSON Response ───────────────────────────┘
```

**Benefits:**
- ✅ **Separation of Concerns** — Each layer has a single responsibility.
- ✅ **Testability** — Layers can be tested independently.
- ✅ **Maintainability** — Model swaps require changes in only 1-2 layers.
- ✅ **Scalability** — Teams can work on different layers in parallel.

---

## 🧰 Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.10+ |
| **Web Framework** | FastAPI |
| **Data Validation** | Pydantic v2 |
| **ML Library** | scikit-learn |
| **Data Processing** | pandas, NumPy |
| **Model Serialization** | joblib |
| **Dataset Download** | kagglehub |
| **Server** | Uvicorn |
| **Rate Limiting** | slowapi |
| **Testing** | pytest, httpx |

---

## 📁 Project Structure

```
HouseIQ/
│
├── app/
│   ├── main.py                         # FastAPI entrypoint
│   ├── __init__.py
│   │
│   ├── routes/                         # 1️⃣ ROUTES LAYER
│   │   ├── __init__.py
│   │   └── predict.py
│   │
│   ├── controllers/                    # 2️⃣ CONTROLLER LAYER
│   │   ├── __init__.py
│   │   └── prediction_controller.py
│   │
│   ├── services/                       # 3️⃣ SERVICE LAYER
│   │   ├── __init__.py
│   │   └── prediction_service.py
│   │
│   ├── repositories/                   # 4️⃣ REPOSITORY LAYER
│   │   ├── __init__.py
│   │   └── model_repository.py
│   │
│   ├── schemas/                        # PYDANTIC VALIDATION
│   │   ├── __init__.py
│   │   └── house.py
│   │
│   └── utils/                          # UTILITIES
│       ├── __init__.py
│       ├── limiter.py                  # Rate limiter
│       └── logger.py                   # Centralized logger
│
├── data/
│   └── house_data_clean.csv            # Cleaned dataset
│
├── models/                             # ML ARTIFACTS
│   ├── house_columns.pkl
│   ├── house_imputer.pkl
│   └── house_model.pkl
│
├── scripts/
│   └── train.py                        # Training script
│
├── tests/
│   └── test_api.py                     # API tests
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

- **Source:** [Indian House Price Prediction Dataset](https://www.kaggle.com/datasets/srisyra02/house-price-prediction-dataset), by Srimathy Sivanessan (srisyra02).
- **Size:** 250,000 rows × 23 columns.
- **Model target:** INR per square foot, derived as total price in lakhs × 100,000 ÷ size in square feet.

The Kaggle source explicitly describes these as **synthetic residential property records** for educational and ML use, not actual market prices. Kaggle lists the license as CC0 Public Domain. In this source the reported Price_per_SqFt values are in lakh rupees per sq-ft; training derives the model target from total price and area to normalize it to INR/sq-ft. Rows with zero/invalid price rate, total price, or area are removed. Exact duplicate rows and IDs are checked. IQR-flagged price tails are reported and retained with absolute-error training so high-rate cases are not removed by an arbitrary threshold.

### Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `BHK` | int | Bedrooms, Hall, Kitchen |
| `Size_in_SqFt` | float | Property size in sq ft |
| `Year_Built` | int | Year of construction |
| `Parking_Space` | int | 1 = Yes, 0 = No |
| City | one-hot | 42 source cities |
| `rate_per_sqft` | finite optional input | Editable city rate used by the main estimate when configured |

The dataset is **automatically downloaded** via `kagglehub` when you run `scripts/train.py`.

---

## 📍 Main Reference-Rate Estimate

The main estimate is **reference rate × property size × year/parking adjustment**; it is not ML-learned. Rates in `data/city_reference_rates.json` are MagicBricks listing/reference prices, not registered sale prices. The configured source periods are Jan–Mar 2026, except Kolkata (Oct–Dec 2025) and Ahmedabad (Apr–Jun 2026). Entries with a sourced band store low, typical, and high INR/sq ft values; entries whose source published only an average use `average_only` and deliberately omit low/high. Each configured entry records its source, URL, reporting period, access date, and notes. Of the 42 dropdown cities, 34 are `not_set`; users can enter a rate manually for those cities, without a locality range or source-based rate warning.

The web form pre-fills the editable **Rate per sq ft** from the selected city's configured typical rate. Changing city replaces the value; **Reset to reference rate** restores it. If `city_reference_rates.json` changes, the prediction service detects the file's modification time and reloads the rates automatically; restart Uvicorn and refresh the browser if an already-open session still shows old content. A user-entered rate drives the main estimate (rate × size), followed by a transparent year adjustment of 0.25% per year from 2020 (capped at ±10%) and a 2% parking adjustment. Where source low/high values exist, the locality range is scaled around the edited rate and a non-blocking warning appears below half the configured low or above 1.5× the high. For average-only sources the UI says “Range unavailable: source gave only an average.” Unconfigured cities allow a manual estimate, with no source range or comparison warning. The live-rate link opens a Google search; the app does not fetch or scrape it. BHK is used for training-supported size validation.

## 🤖 Demo ML Model (Comparison Only)

- **Algorithm:** HistGradientBoostingRegressor with absolute-error loss.
- **Target:** INR per square foot; the model remains available and is shown as **Demo ML model estimate (synthetic data, comparison only)**.
- **Features:** BHK, size, year built, parking, and one-hot city. User-entered market rate is never sent to the model.
- **Splits:** 70% train, 15% interval calibration, 15% held-out test.
- **Forecast:** Assumptions-based, not city-specific; growth is capped at 2%–12% and gradually approaches capped inflation.

The ML estimate is secondary because the Kaggle source explicitly contains synthetic records; price has little meaningful relationship to size, city, or BHK in that dataset. Its held-out MAPE is about 118%, describing fit to synthetic data only. The model demonstrates the training/inference pipeline and provides a comparison; it is not comparable to real prices and is not the main estimate. A missing source rate stays unset rather than falling back to the synthetic model.

Future forecasts are assumption-based and not city-specific. Inflation is the main long-run growth driver: the initial scenario also considers GDP growth, migration, and interest, while long-horizon growth gradually approaches capped inflation.

### 📈 Evaluation Metrics

Training prints held-out R², MAE, MAPE, 90% interval calibration/coverage, and grouped permutation importance. These metrics describe fit to the synthetic Kaggle data only; they do not establish accuracy for real property prices.

### 📦 Saved Artifacts

| File | Purpose |
|------|---------|
| house_model.pkl | Trained PSF model |
| house_columns.pkl | Exact feature column order |
| house_metadata.json | Training limits, source counts, metrics, and interval calibration |

------|---------|
| `house_model.pkl` | Trained LinearRegression model |
| `house_imputer.pkl` | Fitted SimpleImputer |
| `house_columns.pkl` | Exact column order for inference |

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Shlokverma0/house-price-api.git HouseIQ
cd HouseIQ
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

**Activate:**

- **Windows:** `.\venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Kaggle Credentials

To download the dataset, you need Kaggle API credentials:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. Place `kaggle.json` in:
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`
   - **macOS/Linux:** `~/.kaggle/kaggle.json`

### 5️⃣ Train the Model

```bash
python scripts/train.py
```

This will:
- Download the dataset from Kaggle
- Clean and preprocess the data
- Train the Linear Regression model
- Save `.pkl` artifacts to `models/` and cleaned data to `data/`

Run this after changing training preprocessing so the model files reflect the current code. Kaggle credentials and network access are required to download the source dataset.

---

## 🚀 Running the API

Start the FastAPI server:

```bash
python -m uvicorn app.main:app --reload
```

**Available URLs:**

| URL | Purpose |
|-----|---------|
| `http://127.0.0.1:8000/app` | HouseIQ property estimate and forecast dashboard |
| `http://127.0.0.1:8000/` | API root |
| `http://127.0.0.1:8000/docs` | Swagger UI (interactive) |
| `http://127.0.0.1:8000/redoc` | ReDoc documentation |
| `http://127.0.0.1:8000/health` | Health check |

---

## 🔌 API Endpoints

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| `GET` | `/` | Root — welcome message | — |
| `GET` | `/health` | Health check + model status | — |
| `POST` | `/predict` | Predict current house price | 30/min |
| `POST` | `/predict/future` | Forecast prices for N years | 10/min |

The dashboard is served by the FastAPI app at `/app`; it uses the same-origin prediction endpoints and does not need a separate frontend server.

### Forecast assumptions

The future endpoint starts from the current reference-rate estimate and compounds a user-assumptions scenario. The initial rate combines inflation, GDP growth, migration, and interest, is capped at 2%–12%, and gradually moves toward inflation over the horizon. It is not city-specific or trained on future economic data. If a city reference rate is unset, the API returns no reference forecast and explains why; it still returns the demo ML comparison.

### Request Fields (for `/predict/future`)

| Field | Type | Constraint | Description |
|-------|------|------------|-------------|
| `BHK` | int | 1–5 | Used for training-supported size validation; it does not change the main reference estimate |
| `Size_in_SqFt` | integer | 500–5000, also checked against the selected BHK's training range | Property size |
| `rate_per_sqft` | finite number | 1,000–150,000 | Editable INR/sq ft; a manual rate can be used when a city is not configured |
| `Year_Built` | integer | 1900–current year (dynamic) | Registry/construction year; years before 1990 receive a rough-extrapolation warning |
| `Parking_Space` | int | 0 or 1 | Parking availability |
| `location` | enum | 42 cities | Property city; reference estimate is available only where a sourced rate is set |
| `years` | int | 1–50 | Forecast horizon |
| `inflation_rate` | finite number | 0–0.15 | Annual inflation as a decimal fraction |
| `interest_rate` | finite number | 0.01–0.20 | Home loan rate as a decimal fraction |
| `gdp_growth_rate` | finite number | -0.05–0.15 | GDP growth as a decimal fraction |
| `migration_rate` | finite number | 0–0.10 | Migration as a decimal fraction |
| `gdp_growth_rate` | float | 0–0.2 | GDP growth |
| `migration_rate` | float | 0–0.2 | Migration rate |

---

## 📥 Example Request & Response

### Request

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/future' \
  -H 'Content-Type: application/json' \
  -d '{
    "BHK": 3,
    "Size_in_SqFt": 1500,
    "rate_per_sqft": 37930,
    "Year_Built": 2015,
    "Parking_Space": 1,
    "location": "Mumbai",
    "years": 10,
    "inflation_rate": 0.06,
    "interest_rate": 0.085,
    "gdp_growth_rate": 0.07,
    "migration_rate": 0.02
  }'
```

### Response

The API returns the main reference-rate estimate and its **Locality spread range** from configured low/typical/high source rates, plus the separate demo ML comparison estimate. Forecast period ranges propagate the source locality spread through the chosen assumptions; they do not represent a probability or confidence interval. Browser validation shows inline errors and disables submission while invalid; the API independently rejects out-of-range, non-finite, empty, and fractional size/year values.

Forecast output is grouped into 10-year periods to keep long forecasts readable. Each period shows the price before that period and its ending price. Amounts below ₹1 crore display in lakhs; amounts of ₹1 crore or more display in crores. Numeric price fields remain in lakhs for consistent calculations.

### 📐 Forecasting Formula

The short-run rate uses the submitted inflation, GDP growth, migration, and interest assumptions. It is capped at 2%–12%. Each year's rate gradually moves toward the inflation assumption (also capped at 2%–12%). It does not use city-specific growth because the dataset has no time-series market history.

---

## 🧪 Test Coverage

The API test suite covers validation, reference-rate × area arithmetic, source metadata, separate demo ML results, unset-city behavior for current/future API requests, long-run growth, and API health.

- Root endpoint and HouseIQ title.
- Health endpoint and model status.
- Successful current-price prediction with a concise response.
- A 50-year forecast summarized into five 10-year ranges.

### Run Tests

```bash
python -m pytest tests/test_api.py -v
```

---

## 🚀 Production Features

| Feature | Implementation |
|---------|---------------|
| **Rate Limiting** | `slowapi` — 30 req/min per IP |
| **Logging** | Console + `logs/app.log` file |
| **Input Validation** | Pydantic v2 with enum cities |
| **Performance Monitoring** | `X-Process-Time` header |
| **CORS** | Enabled for all origins |
| **Auto Docs** | Swagger UI + ReDoc |
| **Error Handling** | Structured `422` validation errors |

---

## 👨‍💻 Author

**Shlok Verma**  
Full-Stack Developer & AI/ML Engineer

- 🐙 GitHub: [@Shlokverma0](https://github.com/Shlokverma0)
- 💼 LinkedIn: [shlok-verma](https://www.linkedin.com/in/shlok-verma-113713363)
- 📧 Email: vshlok24@gmail.com

---

## 🙏 Acknowledgements

- [Kaggle](https://www.kaggle.com/) for the dataset
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent framework
- [scikit-learn](https://scikit-learn.org/) for ML tools
- [slowapi](https://github.com/laurentS/slowapi) for rate limiting

---

<div align="center">

**⭐ If you found this project useful, please consider giving it a star! ⭐**

</div>
