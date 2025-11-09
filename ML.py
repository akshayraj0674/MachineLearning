import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


train_url   = "https://raw.githubusercontent.com/akshayraj0674/MachineLearning/refs/heads/main/ml-4127-e-project-2/train_data.csv"
test_url    = "https://raw.githubusercontent.com/akshayraj0674/MachineLearning/refs/heads/main/ml-4127-e-project-2/test_data.csv"
sample_url  = "https://raw.githubusercontent.com/akshayraj0674/MachineLearning/refs/heads/main/ml-4127-e-project-2/sample_submission_probs.csv"


TARGET_COL = "Action"          # Target column name
ID_COL = "id"                  # ID column (must be present in train & test)
CLASS_ORDER = [0, 1, 2]        # Expected classes
N_FOLDS = 5
RANDOM_STATE = 42
VERBOSE = 1


USE_CV = True                  # True: use LogisticRegressionCV to tune C & l1_ratio; False: fixed settings
FIXED_C = 1.0                  # Used only if USE_CV=False
FIXED_L1_RATIO = 0.5           # Used only if USE_CV=False (0<L1<=1)
CLASS_WEIGHT_BALANCED = True


C_VALUES = np.logspace(-2, 2, 8)           # 0.01 ... 100
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]      # Mix of L1/L2
INNER_CV_FOLDS = 5                         # Inner CV for LogisticRegressionCV
MAX_ITER = 8000                            # Increase if convergence warnings
TOL = 1e-4


def read_csv_url(url: str) -> pd.DataFrame:
    """
    Read CSV directly. If fails and GITHUB_TOKEN is set, attempt authenticated request (for private repos).
    """
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


def split_features(df: pd.DataFrame, target: str, id_col: str) -> Tuple[List[str], List[str], List[str]]:
    features = [c for c in df.columns if c not in [target, id_col]]
    cat_cols = [c for c in features if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in features if c not in cat_cols]
    return features, num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols)
    ])
    return preprocessor


def build_fixed_logreg() -> LogisticRegression:
    class_weight = "balanced" if CLASS_WEIGHT_BALANCED else None
    return LogisticRegression(
        penalty="elasticnet",
        l1_ratio=FIXED_L1_RATIO,
        C=FIXED_C,
        multi_class="multinomial",
        solver="saga",
        class_weight=class_weight,
        max_iter=MAX_ITER,
        tol=TOL,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )


def build_cv_logreg() -> LogisticRegressionCV:
    class_weight = "balanced" if CLASS_WEIGHT_BALANCED else None
    return LogisticRegressionCV(
        Cs=C_VALUES,
        cv=INNER_CV_FOLDS,
        penalty="elasticnet",
        solver="saga",
        l1_ratios=L1_RATIOS,
        scoring="neg_log_loss",
        multi_class="multinomial",
        max_iter=MAX_ITER,
        tol=TOL,
        refit=True,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )


def main():
    print("Loading data...")
    train = read_csv_url(train_url)
    test = read_csv_url(test_url)
    sample = read_csv_url(sample_url)

    # Basic checks
    for name, df in [("train", train), ("test", test)]:
        if ID_COL not in df.columns:
            raise ValueError(f"{name}.csv missing required ID column '{ID_COL}'. Columns: {list(df.columns)}")
    if TARGET_COL not in train.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not in train. Columns: {list(train.columns)}")

    # Warn if sample header mismatch (but continue)
    expected_prefix = ["id", "0", "1", "2"]
    if list(sample.columns)[:4] != expected_prefix:
        print(f"[Warning] sample_submission first 4 columns should be {expected_prefix}, got {list(sample.columns)[:4]}")

    features, num_cols, cat_cols = split_features(train, TARGET_COL, ID_COL)
    print(f"Features total={len(features)} numeric={len(num_cols)} categorical={len(cat_cols)}")

    # Extract X, y
    X = train[features].copy()
    y_raw = train[TARGET_COL]
    # Force integer classes
    if not np.issubdtype(y_raw.dtype, np.integer):
        try:
            y = y_raw.astype(int)
        except Exception:
            unique_labels = sorted(y_raw.unique())
            mapping = {lab: i for i, lab in enumerate(unique_labels)}
            y = y_raw.map(mapping).astype(int)
            print(f"[Info] Converted labels to integers with mapping: {mapping}")
    else:
        y = y_raw.astype(int)

    # Check that all expected classes exist
    missing = set(CLASS_ORDER) - set(np.unique(y))
    if missing:
        raise ValueError(f"Training data missing classes: {missing}. Cannot output full probability distribution.")

    X_test = test[features].copy()

    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Outer CV for unbiased OOF estimates
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros((len(train), len(CLASS_ORDER)), dtype=float)
    test_fold_probs = []
    fold_details = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        if VERBOSE:
            print(f"\n--- Fold {fold}/{N_FOLDS} ---")

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        if USE_CV:
            logreg = build_cv_logreg()
        else:
            logreg = build_fixed_logreg()

        pipeline = Pipeline([
            ("prep", preprocessor),
            ("clf", logreg)
        ])

        pipeline.fit(X_tr, y_tr)

        # Extract model & its classes
        model = pipeline.named_steps["clf"]
        model_classes = list(model.classes_)
        # Map probabilities to CLASS_ORDER
        idx_map = [model_classes.index(c) for c in CLASS_ORDER]

        val_pred_full = pipeline.predict_proba(X_val)
        val_pred = val_pred_full[:, idx_map]

        test_pred_full = pipeline.predict_proba(X_test)
        test_pred = test_pred_full[:, idx_map]

        oof_probs[val_idx] = val_pred
        test_fold_probs.append(test_pred)

        fold_logloss = log_loss(y_val, val_pred)
        if VERBOSE:
            if USE_CV:
                # Show chosen C & l1_ratio
                chosen_C = model.C_[0] if hasattr(model, "C_") else None
                chosen_l1 = model.l1_ratio_[0] if hasattr(model, "l1_ratio_") else None
                print(f"Fold {fold} log_loss: {fold_logloss:.6f} | best_C={chosen_C} best_l1_ratio={chosen_l1}")
            else:
                print(f"Fold {fold} log_loss: {fold_logloss:.6f}")

        detail = {
            "fold": fold,
            "log_loss": float(fold_logloss)
        }
        if USE_CV:
            detail["best_C"] = float(model.C_[0])
            detail["best_l1_ratio"] = float(model.l1_ratio_[0])
        fold_details.append(detail)

    overall_logloss = log_loss(y, oof_probs)
    print(f"\nOOF log_loss: {overall_logloss:.6f}")

    # Average test probabilities
    test_probs = np.mean(test_fold_probs, axis=0)

    # Build submission with exact columns id,0,1,2
    submission = pd.DataFrame({
        "id": test[ID_COL].values,
        "0": test_probs[:, CLASS_ORDER.index(0)],
        "1": test_probs[:, CLASS_ORDER.index(1)],
        "2": test_probs[:, CLASS_ORDER.index(2)]
    })

    # Sanity check probability sums
    sums = submission[["0", "1", "2"]].sum(axis=1)
    if VERBOSE:
        print("\nProbability sum stats (should be ~1):")
        print(sums.describe())

    submission_path = Path("submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path.resolve()}")

    diagnostics = {
        "model_family": "Multinomial Logistic Regression (Elastic Net)",
        "use_cv": USE_CV,
        "oof_log_loss": float(overall_logloss),
        "folds": N_FOLDS,
        "class_weight_balanced": CLASS_WEIGHT_BALANCED,
        "fixed_C": float(FIXED_C) if not USE_CV else None,
        "fixed_l1_ratio": float(FIXED_L1_RATIO) if not USE_CV else None,
        "cv_C_grid": list(map(float, C_VALUES)) if USE_CV else None,
        "cv_l1_ratios": L1_RATIOS if USE_CV else None,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "n_features": len(features),
        "n_numeric": len(num_cols),
        "n_categorical": len(cat_cols),
        "class_order": CLASS_ORDER,
        "fold_details": fold_details
    }

    with open("training_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    print("Wrote training_diagnostics.json")

    print("\nDone.")


if __name__ == "__main__":
    main()