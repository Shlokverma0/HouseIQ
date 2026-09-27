# 🏠 House Price Prediction API

<div align="center">

**A production-grade ML REST API that predicts property prices and forecasts up to 50 years ahead using real economic factors.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063.svg)](https://docs.pydantic.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-9%2F9%20Passing-brightgreen.svg)]()

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

**House Price Prediction API** is a production-ready REST API that:

- Predicts **current property prices** in Indian cities based on property features.
- Forecasts **future prices** (up to 50 years) using real economic indicators.
- Follows a **clean 4-Layer Architecture** for separation of concerns.
- Includes **logging, rate limiting, input validation, and full test coverage**.

Built with **FastAPI**, **scikit-learn**, and **Pydantic v2**, this project reflects how ML services are structured in real production environments.

---

## ✨ Key Features

- 🤖 **ML-Powered Predictions** — Trained on 250,000+ real Indian property records.
- 📈 **Future Forecasting** — Predicts prices up to 50 years ahead.
- 🌍 **Economic Factor Modeling** — Uses Inflation, GDP Growth, Interest Rate, and Migration Rate.
- 🏗️ **4-Layer Architecture** — Routes → Controllers → Services → Repositories.
- ✅ **Strict Input Validation** — Pydantic v2 with enum-based city validation (41 cities).
- 🚦 **Rate Limiting** — 30 requests/min per IP (`slowapi`).
- 📝 **Centralized Logging** — Console + file-based logs.
- ⚡ **Performance Monitoring** — `X-Process-Time` header on every response.
- 🧪 **9/9 Test Coverage** — Positive and negative cases covered.
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
house-price-ml-api/
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
│   └── test_api.py                     # 9 automated tests
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

- **Source:** [Indian House Price Prediction Dataset](https://www.kaggle.com/datasets/srisyra02/house-price-prediction-dataset) (Kaggle)
- **Size:** 250,000 rows × 23 columns
- **Target Variable:** `Price_in_Lakhs`

### Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `BHK` | int | Bedrooms, Hall, Kitchen |
| `Size_in_SqFt` | float | Property size in sq ft |
| `Price_per_SqFt` | float | Rate per sq ft |
| `Year_Built` | int | Year of construction |
| `Parking_Space` | int | 1 = Yes, 0 = No |
| `City` | one-hot | 41 major Indian cities |

The dataset is **automatically downloaded** via `kagglehub` when you run `scripts/train.py`.

---

## 🤖 Model Details

- **Algorithm:** Multiple Linear Regression
- **Preprocessing:** `SimpleImputer(strategy="mean")` for missing values
- **Features:** 47 (5 base + 42 one-hot city columns)
- **Train/Test Split:** 80/20

### 📈 Evaluation Metrics

| Metric | Value |
|--------|-------|
| **MAE** | 77.28 |
| **RMSE** | 97.11 |
| **R² Score** | 0.510 |

### 📦 Saved Artifacts

| File | Purpose |
|------|---------|
| `house_model.pkl` | Trained LinearRegression model |
| `house_imputer.pkl` | Fitted SimpleImputer |
| `house_columns.pkl` | Exact column order for inference |

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Shlokverma0/house-price-api.git
cd house-price-api
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

---

## 🚀 Running the API

Start the FastAPI server:

```bash
python -m uvicorn app.main:app --reload
```

**Available URLs:**

| URL | Purpose |
|-----|---------|
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

### Request Fields (for `/predict/future`)

| Field | Type | Constraint | Description |
|-------|------|------------|-------------|
| `BHK` | int | 1–10 | Bedrooms, Hall, Kitchen |
| `Size_in_SqFt` | float | 100–20000 | Property size |
| `Price_per_SqFt` | float | 500–50000 | Rate per sq ft |
| `Year_Built` | int | 1900–2026 | Year of construction |
| `Parking_Space` | int | 0 or 1 | Parking availability |
| `location` | enum | 41 cities | Property city |
| `years` | int | 1–50 | Forecast horizon |
| `inflation_rate` | float | 0–0.5 | Annual inflation |
| `interest_rate` | float | 0–0.5 | Home loan rate |
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
    "Price_per_SqFt": 8000,
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

### Response (`200 OK`)

```json
{
  "input_summary": {
    "location": "Mumbai",
    "BHK": 3,
    "Size_in_SqFt": 1500,
    "years": 10
  },
  "current_price_in_lakhs": 239.32,
  "effective_growth_rate": 0.1075,
  "forecast": [
    { "year": 1, "price_in_lakhs": 265.05 },
    { "year": 2, "price_in_lakhs": 293.54 },
    { "year": 10, "price_in_lakhs": 664.38 }
  ],
  "currency": "INR",
  "disclaimer": "This is an ML-based estimate, not financial advice."
}
```

### 📐 Forecasting Formula

```
Effective Growth Rate = Inflation + GDP Growth + Migration Rate − (Interest Rate × 0.5)

Future Price = Current Price × (1 + Effective Growth Rate) ^ Years
```

---

## 🧪 Test Coverage

| Category | Tests | Status |
|----------|-------|--------|
| Positive Tests | 3 | ✅ Pass |
| Negative Tests | 6 | ✅ Pass |
| **Total** | **9** | **✅ 9/9 Pass** |

### Sample Test Cases

| # | Test | Input | Expected |
|---|------|-------|----------|
| 1 | Valid Mumbai prediction | BHK=3, Mumbai | `200 OK` |
| 2 | Synonym location | "Bombay" | `200 OK` (→ Mumbai) |
| 3 | 50-year forecast | years=50 | `200 OK` |
| 4 | Invalid city | "Atlantis" | `422` |
| 5 | Negative BHK | BHK=-5 | `422` |
| 6 | Years > 50 | years=100 | `422` |
| 7 | Negative inflation | -0.05 | `422` |
| 8 | Parking > 1 | Parking=2 | `422` |
| 9 | Years = 0 | years=0 | `422` |

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