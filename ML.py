import os
import json
import numpy as np
import pandas as pd
from typing import List, Optional
from pathlib import Path


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


train_url   = "https://github.com/akshayraj0674/MachineLearning/blob/main/ml-4127-e-project-2/train_data.xlsx"
test_url    = "https://github.com/akshayraj0674/MachineLearning/blob/main/ml-4127-e-project-2/test_data.xlsx"
sample_url  = "https://github.com/akshayraj0674/MachineLearning/blob/main/ml-4127-e-project-2/sample_submission_probs.csv"


TARGET_COL = "action"
N_FOLDS = 5
RANDOM_STATE = 42

USE_CLASS_WEIGHT = True
CALIBRATE = False
CALIBRATION_METHOD = "isotonic"
SOLVER = "lbfgs"
MAX_ITER = 3000
VERBOSE = 1


def read_csv_url(url: str) -> pd.DataFrame:

    try:
        return pd.read_csv(url)
    except Exception as e:
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise RuntimeError(f"Failed to read {url}. If private, set GITHUB_TOKEN. Original error: {e}")
        try:
            import requests
        except ImportError:
            raise ImportError("Install requests for private repo access: pip install requests")
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        from io import BytesIO
        return pd.read_csv(BytesIO(r.content))


def split_features(df: pd.DataFrame, target: str, id_col: str):
    feature_cols = [c for c in df.columns if c not in [target, id_col]]
    cat_cols = [c for c in feature_cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return feature_cols, num_cols, cat_cols


def build_logreg_pipeline(num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    numeric_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=True))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols)
    ])

    class_weight = "balanced" if USE_CLASS_WEIGHT else None

    logreg = LogisticRegression(
        multi_class="multinomial",
        solver=SOLVER,
        class_weight=class_weight,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        n_jobs=-1 if SOLVER in ("lbfgs", "saga") else None
    )

    return Pipeline([
        ("prep", preprocessor),
        ("model", logreg)
    ])