"""
train.py
--------
Trains a house-price model from the dataset's original target values.
"""
import os
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "Price_in_Lakhs"


def load_dataset() -> pd.DataFrame:
    import kagglehub
    dataset_path = kagglehub.dataset_download("srisyra02/house-price-prediction-dataset")
    csv_path = f"{dataset_path}/indian_house_price_prediction_data.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def main():
    df = load_dataset()

    if 'City' not in df.columns:
        raise KeyError("Dataset me 'City' column nahi mila.")

    df['location'] = df['City']
    df['Parking_Space'] = df['Parking_Space'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    top_cities = df['location'].value_counts().nlargest(50).index
    df['location'] = df['location'].apply(lambda x: x if x in top_cities else 'Other')

    df = pd.get_dummies(df, columns=['location'], prefix='loc', drop_first=False)
    location_cols = [col for col in df.columns if col.startswith('loc_')]

    base_features = ['BHK', 'Size_in_SqFt', 'Price_per_SqFt', 'Year_Built', 'Parking_Space']
    FEATURES = base_features + location_cols

    df = df.dropna(subset=[TARGET])

    # Keep observed target prices unchanged; city effects are learned from features.
    df = df[FEATURES + [TARGET]].copy()

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test), columns=FEATURES, index=X_test.index
    )

    model = LinearRegression()
    model.fit(X_train_imputed, y_train)

    preds = model.predict(X_test_imputed)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    print(f"\nMAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R2:   {r2:.3f}")

    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    # Sirf 2000 rows save karo (50MB -> 200KB)
    clean_df = pd.concat(
        [pd.concat([X_train_imputed, X_test_imputed]), pd.concat([y_train, y_test])],
        axis=1,
    )
    clean_df.head(2000).to_csv(
        os.path.join(BASE_DIR, "data", "house_data_clean.csv"), index=False
    )

    joblib.dump(model, os.path.join(BASE_DIR, "models", "house_model.pkl"))
    joblib.dump(FEATURES, os.path.join(BASE_DIR, "models", "house_columns.pkl"))
    joblib.dump(imputer, os.path.join(BASE_DIR, "models", "house_imputer.pkl"))

    print("\nSaved: models/*.pkl, data/house_data_clean.csv")


if __name__ == "__main__":
    main()
