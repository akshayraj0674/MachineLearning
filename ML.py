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


def main():
    print("Loading datasets...")
    train = read_csv_url(train_url)
    test = read_csv_url(test_url)
    sample = read_csv_url(sample_url)

    required_sample_cols = ["id", "0", "1", "2"]
    if list(sample.columns)[:4] != required_sample_cols:
        print(f"[Warning] sample_submission first 4 columns expected {required_sample_cols}, got: {list(sample.columns)[:4]}")

    if TARGET_COL not in train.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in train. Available: {list(train.columns)}")

    if "id" not in train.columns or "id" not in test.columns:
        raise ValueError("Both train.csv and test.csv must contain an 'id' column for this script.")

    id_col = "id"

    feature_cols, num_cols, cat_cols = split_features(train, TARGET_COL, id_col)
    print(f"Detected {len(feature_cols)} features (numeric={len(num_cols)}, categorical={len(cat_cols)})")

    X = train[feature_cols].copy()
    y_raw = train[TARGET_COL]

    # Ensure target is integers 0,1,2
    # If already int, keep; if not, attempt conversion
    if not np.issubdtype(y_raw.dtype, np.integer):
        try:
            y = y_raw.astype(int)
        except Exception:
            # Map unique labels to integers sorted by value
            unique_labels = sorted(y_raw.unique())
            mapping = {lab: i for i, lab in enumerate(unique_labels)}
            y = y_raw.map(mapping).astype(int)
            print(f"[Info] Converted non-integer labels to integers via mapping: {mapping}")
    else:
        y = y_raw.astype(int)

    missing_classes = set([0, 1, 2]) - set(np.unique(y))
    if missing_classes:
        raise ValueError(f"Training data is missing classes: {missing_classes}. Cannot produce probabilities for absent classes.")

    X_test = test[feature_cols].copy()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Align final class order explicitly: 0,1,2
    class_order = [0, 1, 2]
    oof_probs = np.zeros((len(train), len(class_order)), dtype=float)
    test_fold_probs = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        if VERBOSE:
            print(f"\n--- Fold {fold}/{N_FOLDS} ---")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        pipeline = build_logreg_pipeline(num_cols, cat_cols)
        pipeline.fit(X_tr, y_tr)

        model = pipeline
        if CALIBRATE:
            calibrator = CalibratedClassifierCV(pipeline, method=CALIBRATION_METHOD, cv=3)
            calibrator.fit(X_tr, y_tr)
            model = calibrator

        val_pred_full = model.predict_proba(X_val)
        test_pred_full = model.predict_proba(X_test)

        # Ensure columns correspond to model.classes_
        model_classes = list(model.classes_)
        # Build index mapping from class_order to model class indices
        idx_map = [model_classes.index(c) for c in class_order]
        val_pred = val_pred_full[:, idx_map]
        test_pred = test_pred_full[:, idx_map]

        oof_probs[val_idx] = val_pred
        test_fold_probs.append(test_pred)

        fold_logloss = log_loss(y_val, val_pred)
        if VERBOSE:
            print(f"Fold {fold} log_loss: {fold_logloss:.6f}")

    overall_logloss = log_loss(y, oof_probs)
    print(f"\nOOF log_loss: {overall_logloss:.6f}")

    test_probs = np.mean(test_fold_probs, axis=0)

    # Build submission with exact required columns id,0,1,2
    submission = pd.DataFrame({
        "id": test[id_col].values,
        "0": test_probs[:, class_order.index(0)],
        "1": test_probs[:, class_order.index(1)],
        "2": test_probs[:, class_order.index(2)]
    })

    # Final sanity: probabilities sum
    prob_sums = submission[["0", "1", "2"]].sum(axis=1)
    print("\nProbability sum stats (should be ~1.0):")
    print(prob_sums.describe())

    out_path = Path("submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path.resolve()}")

    diagnostics = {
        "oof_log_loss": float(overall_logloss),
        "folds": N_FOLDS,
        "calibrated": CALIBRATE,
        "calibration_method": CALIBRATION_METHOD if CALIBRATE else None,
        "solver": SOLVER,
        "max_iter": MAX_ITER,
        "use_class_weight": USE_CLASS_WEIGHT,
        "n_features": len(feature_cols),
        "n_numeric": len(num_cols),
        "n_categorical": len(cat_cols),
        "class_order": class_order
    }
    with open("training_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    print("Wrote training_diagnostics.json")


if __name__ == "__main__":
    main()