"""Load the trained HouseIQ model and metadata."""
import json
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "house_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "house_columns.pkl"
METADATA_PATH = BASE_DIR / "models" / "house_metadata.json"


class ModelRepository:
    def __init__(self):
        self.model = None
        self.columns = []
        self.metadata = {}
        self.loaded = False

    def load(self):
        missing = [path for path in (MODEL_PATH, COLUMNS_PATH, METADATA_PATH) if not path.exists()]
        if missing:
            names = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"HouseIQ model artifacts are missing: {names}. Run 'python scripts/train.py' first."
            )
        self.model = joblib.load(MODEL_PATH)
        self.columns = joblib.load(COLUMNS_PATH)
        self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.loaded = True
        print(f"[ModelRepository] Loaded PSF model with {len(self.columns)} features.")


model_repository = ModelRepository()
