"""
Train the HouseIQ price-per-square-foot model from the Kaggle source.
The Kaggle dataset is synthetic and is not a source of real market valuations.
"""
import json
import os
from pathlib import Path

import joblib
import kagglehub
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_REF = "srisyra02/house-price-prediction-dataset"
CSV_NAME = "indian_house_price_prediction_data.csv"
TARGET = "Target_Price_Per_SqFt_INR"
RANDOM_STATE = 17


def load_dataset() -> pd.DataFrame:
    dataset_path = Path(kagglehub.dataset_download(DATASET_REF))
    csv_path = dataset_path / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected Kaggle file not found: {csv_path}")
    return pd.read_csv(csv_path)


def report_and_clean(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ID", "City", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
        "Price_per_SqFt", "Year_Built", "Parking_Space",
    }
    missing_columns = required - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    print(f"Dataset: {DATASET_REF} / {CSV_NAME}")
    print(f"Raw rows: {len(raw):,}; columns: {len(raw.columns)}")
    print(f"Missing cells: {int(raw.isna().sum().sum()):,}")
    duplicate_rows = int(raw.duplicated().sum())
    duplicate_ids = int(raw["ID"].duplicated().sum())
    print(f"Exact duplicate rows: {duplicate_rows:,}")
    print(f"Duplicate IDs: {duplicate_ids:,}")

    df = raw.drop_duplicates().copy()
    zero_rate = pd.to_numeric(df["Price_per_SqFt"], errors="coerce").le(0)
    invalid = (
        zero_rate
        | pd.to_numeric(df["Price_in_Lakhs"], errors="coerce").le(0)
        | pd.to_numeric(df["Size_in_SqFt"], errors="coerce").le(0)
    )
    print(f"Rows removed (zero/invalid price-per-sq-ft, price or size): {int(invalid.sum()):,}")
    df = df.loc[~invalid].copy()

    df["BHK"] = pd.to_numeric(df["BHK"], errors="coerce")
    df["Size_in_SqFt"] = pd.to_numeric(df["Size_in_SqFt"], errors="coerce")
    df["Year_Built"] = pd.to_numeric(df["Year_Built"], errors="coerce")
    df["Price_in_Lakhs"] = pd.to_numeric(df["Price_in_Lakhs"], errors="coerce")
    df["Price_per_SqFt"] = pd.to_numeric(df["Price_per_SqFt"], errors="coerce")
    df = df.dropna(subset=["BHK", "Size_in_SqFt", "Year_Built", "Price_in_Lakhs", "Price_per_SqFt"])
    df = df[df["Price_per_SqFt"] > 0].copy()

    # The source stores Price_per_SqFt in lakh rupees/sq-ft. Derive the
    # rupee target from total price and area so units and total-price math agree.
    df[TARGET] = df["Price_in_Lakhs"] * 100_000 / df["Size_in_SqFt"]
    print(
        "Clean target PSF (INR/sq-ft): "
        f"min={df[TARGET].min():,.0f}, p01={df[TARGET].quantile(.01):,.0f}, "
        f"median={df[TARGET].median():,.0f}, p99={df[TARGET].quantile(.99):,.0f}, "
        f"max={df[TARGET].max():,.0f}"
    )
    print(
        "Clean feature ranges: "
        f"BHK={int(df.BHK.min())}..{int(df.BHK.max())}, "
        f"size={int(df.Size_in_SqFt.min())}..{int(df.Size_in_SqFt.max())} sq-ft, "
        f"year={int(df.Year_Built.min())}..{int(df.Year_Built.max())}"
    )
    print("City row counts after cleaning:")
    print(df["City"].value_counts().sort_index().to_string())
    print("BHK-wise size ranges after cleaning:")
    print(
        df.groupby("BHK")["Size_in_SqFt"]
        .agg(rows="size", min_size="min", max_size="max")
        .to_string()
    )
    return df


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    parking = df["Parking_Space"].astype(str).str.strip().str.casefold()
    parking_values = parking.map({"yes": 1, "no": 0})
    if parking_values.isna().any():
        bad = sorted(parking[parking_values.isna()].unique())
        raise ValueError(f"Unexpected Parking_Space values: {bad}")

    features = pd.DataFrame({
        "BHK": df["BHK"].astype(float),
        "Size_in_SqFt": df["Size_in_SqFt"].astype(float),
        "Year_Built": df["Year_Built"].astype(float),
        "Parking_Space": parking_values.astype(float),
        "City": df["City"].astype(str).str.strip(),
    })
    encoded = pd.get_dummies(features, columns=["City"], prefix="loc", dtype=float)
    city_columns = [column for column in encoded.columns if column.startswith("loc_")]
    groups = {
        "BHK": ["BHK"],
        "Size_in_SqFt": ["Size_in_SqFt"],
        "Year_Built": ["Year_Built"],
        "Parking_Space": ["Parking_Space"],
        "City": city_columns,
    }
    return encoded, list(encoded.columns), groups


def print_feature_importance(model, X_test, y_test, groups):
    base_score = r2_score(y_test, model.predict(X_test))
    rng = np.random.default_rng(RANDOM_STATE)
    results = []
    for name, columns in groups.items():
        permuted = X_test.copy()
        order = rng.permutation(len(permuted))
        permuted.loc[:, columns] = X_test.iloc[order][columns].to_numpy()
        score_drop = base_score - r2_score(y_test, model.predict(permuted))
        results.append((name, score_drop))
    print("Grouped permutation importance (held-out R2 drop; higher = more useful):")
    for name, score_drop in sorted(results, key=lambda item: item[1], reverse=True):
        print(f"  {name}: {score_drop:.4f}")


def main():
    raw = load_dataset()
    clean = report_and_clean(raw)

    X, columns, groups = make_features(clean)
    y = clean[TARGET].astype(float)

    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_hold, y_hold, test_size=0.50, random_state=RANDOM_STATE
    )
    print(
        f"Split rows: train={len(X_train):,}, calibration={len(X_cal):,}, "
        f"held-out test={len(X_test):,}"
    )
    q1, q3 = y_train.quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_low, outlier_high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    train_outliers = int(((y_train < outlier_low) | (y_train > outlier_high)).sum())
    test_outliers = int(((y_test < outlier_low) | (y_test > outlier_high)).sum())
    print(
        "Train-only IQR outlier check (INR/sq-ft): "
        f"bounds={outlier_low:,.0f}..{outlier_high:,.0f}; "
        f"train flagged={train_outliers:,}; held-out flagged={test_outliers:,}"
    )
    print("IQR-flagged tails are retained; absolute-error loss limits their influence.")

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=120,
        max_leaf_nodes=31,
        l2_regularization=2.0,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    calibration_residuals = np.abs(y_cal.to_numpy() - model.predict(X_cal))
    q90 = float(np.quantile(calibration_residuals, 0.90, method="higher"))

    predictions = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "mae_rupees_per_sqft": float(mean_absolute_error(y_test, predictions)),
        "mape_percent": float(mean_absolute_percentage_error(y_test, predictions) * 100),
    }
    covered = (
        (y_test.to_numpy() >= np.maximum(0, predictions - q90))
        & (y_test.to_numpy() <= predictions + q90)
    )
    print("Held-out test metrics (rupees per sq-ft target):")
    print(f"  R2: {metrics['r2']:.4f}")
    print(f"  MAE: INR {metrics['mae_rupees_per_sqft']:,.0f}/sq-ft")
    print(f"  MAPE: {metrics['mape_percent']:.2f}%")
    print(
        f"  Calibrated 90% interval: +/- INR {q90:,.0f}/sq-ft; "
        f"held-out coverage={covered.mean():.1%}"
    )
    print_feature_importance(model, X_test, y_test, groups)

    bhk_limits = {}
    for bhk, group in clean.groupby("BHK"):
        bhk_limits[str(int(bhk))] = {
            "min_size_sqft": int(group["Size_in_SqFt"].min()),
            "max_size_sqft": int(group["Size_in_SqFt"].max()),
            "rows": int(len(group)),
        }

    city_counts = {str(k): int(v) for k, v in clean["City"].value_counts().sort_index().items()}
    metadata = {
        "dataset_ref": DATASET_REF,
        "dataset_title": "Indian House Price Prediction Dataset",
        "dataset_description": "250,000 synthetic residential property records; educational/ML use only, not actual market prices.",
        "dataset_license": "CC0: Public Domain",
        "target_unit": "INR per square foot",
        "features": columns,
        "bhk_limits": bhk_limits,
        "bhk_min": int(clean["BHK"].min()),
        "bhk_max": int(clean["BHK"].max()),
        "year_min": int(clean["Year_Built"].min()),
        "year_max": int(clean["Year_Built"].max()),
        "size_min_sqft": int(clean["Size_in_SqFt"].min()),
        "size_max_sqft": int(clean["Size_in_SqFt"].max()),
        "market_rate_min_rupees_per_sqft": float(clean[TARGET].min()),
        "market_rate_max_rupees_per_sqft": float(clean[TARGET].max()),
        "city_counts": city_counts,
        "clean_rows": int(len(clean)),
        "raw_rows": int(len(raw)),
        "exact_duplicates_removed": int(raw.duplicated().sum()),
        "zero_or_invalid_rows_removed": int(
            (pd.to_numeric(raw["Price_per_SqFt"], errors="coerce").le(0)
             | pd.to_numeric(raw["Price_in_Lakhs"], errors="coerce").le(0)
             | pd.to_numeric(raw["Size_in_SqFt"], errors="coerce").le(0)).sum()
        ),
        "iqr_outlier_count_retained_in_train": train_outliers,
        "iqr_outlier_count_in_held_out_test": test_outliers,
        "iqr_bounds_from_train_rupees_per_sqft": [float(outlier_low), float(outlier_high)],
        "metrics_held_out_test": metrics,
        "calibration": {
            "coverage": 0.90,
            "absolute_error_q_rupees_per_sqft": q90,
            "calibration_rows": int(len(X_cal)),
            "held_out_coverage": float(covered.mean()),
        },
    }

    model_dir = BASE_DIR / "models"
    data_dir = BASE_DIR / "data"
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "house_model.pkl")
    joblib.dump(columns, model_dir / "house_columns.pkl")
    (model_dir / "house_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    preview_columns = [
        "City", "BHK", "Size_in_SqFt", "Year_Built", "Parking_Space",
        "Price_in_Lakhs", "Price_per_SqFt", TARGET,
    ]
    preview = clean[preview_columns].sample(
        n=min(2000, len(clean)), random_state=RANDOM_STATE
    )
    preview.to_csv(data_dir / "house_data_clean.csv", index=False)
    print("Saved: models/house_model.pkl, house_columns.pkl, house_metadata.json")
    print(f"Saved a {len(preview):,}-row cleaned preview to data/house_data_clean.csv")


if __name__ == "__main__":
    main()
