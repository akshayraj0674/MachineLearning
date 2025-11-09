import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from catboost import CatBoostClassifier


train_url   = "https://raw.githubusercontent.com/akshayraj0674/MachineLearning/refs/heads/main/ml-4127-e-project-2/train_data.csv"
test_url    = "https://raw.githubusercontent.com/akshayraj0674/MachineLearning/refs/heads/main/ml-4127-e-project-2/test_data.csv"
sample_url  = "https://raw.githubusercontent.com/akshayraj0674/MachineLearning/refs/heads/main/ml-4127-e-project-2/sample_submission_probs.csv"


TARGET_COL = "Action"
N_FOLDS = 5
RANDOM_STATE = 42
VERBOSE = 1


def read_csv_url(url: str) -> pd.DataFrame:
    """
    Try direct pandas read. If it fails and GITHUB_TOKEN is set, retry with authenticated request.
    """
    try:
        return pd.read_csv(url)
    except Exception as e:
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise RuntimeError(f"Failed to read {url}. If it's private, set GITHUB_TOKEN. Original error: {e}")
        try:
            import requests
        except ImportError:
            raise ImportError("Install requests for private repo access: pip install requests")
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        from io import BytesIO
        return pd.read_csv(BytesIO(r.content))


def get_feature_sets(df: pd.DataFrame, target_col: str, id_col: str):
    feature_cols = [c for c in df.columns if c not in [target_col, id_col]]
    cat_cols = [c for c in feature_cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return feature_cols, num_cols, cat_cols


def main():
    print("Loading datasets...")
    train = read_csv_url(train_url)
    test = read_csv_url(test_url)
    sample = read_csv_url(sample_url)

    # Validate required columns
    for name, df in [("train", train), ("test", test)]:
        if "id" not in df.columns:
            raise ValueError(f"'{name}.csv' must contain an 'id' column.")
    if TARGET_COL not in train.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in train. Columns: {list(train.columns)}")

    # Optional: warn if sample headers aren't exactly id,0,1,2
    expected_headers = ["id", "0", "1", "2"]
    if list(sample.columns)[:4] != expected_headers:
        print(f"[Warning] sample_submission first 4 columns expected {expected_headers}, got {list(sample.columns)[:4]}")

    id_col = "id"
    feature_cols, num_cols, cat_cols = get_feature_sets(train, TARGET_COL, id_col)
    print(f"Detected {len(feature_cols)} features (numeric={len(num_cols)}, categorical={len(cat_cols)})")

    X = train[feature_cols].copy()
    y = train[TARGET_COL].astype(int).copy()
    X_test = test[feature_cols].copy()

    # Ensure classes 0,1,2 all present
    present = set(np.unique(y))
    missing = set([0, 1, 2]) - present
    if missing:
        raise ValueError(f"Training data is missing classes: {missing}. Add data or relabel before training.")

    # CatBoost categorical feature indices
    cat_indices = [X.columns.get_loc(c) for c in cat_cols]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    class_order = [0, 1, 2]
    oof_probs = np.zeros((len(train), len(class_order)), dtype=float)
    test_fold_probs: List[np.ndarray] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        if VERBOSE:
            print(f"\n--- Fold {fold}/{N_FOLDS} ---")

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=2000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=5.0,
            random_seed=RANDOM_STATE + fold,  # slightly varied per fold
            early_stopping_rounds=100,
            bootstrap_type="Bayesian",
            bagging_temperature=1.0,
            border_count=254,
            grow_policy="SymmetricTree",
            allow_writing_files=False,
            verbose=False
        )

        model.fit(
            X_tr, y_tr,
            cat_features=cat_indices if cat_indices else None,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False
        )

        val_pred_full = model.predict_proba(X_val)  # shape (n_val, n_classes)
        test_pred_full = model.predict_proba(X_test)

        # Align probabilities to class_order [0,1,2]
        # CatBoost uses sorted unique labels; y is ints 0,1,2, so order should match.
        # Still, we compute mapping defensively if available.
        try:
            model_classes = list(model.classes_)  # scikit-learn compat
            idx_map = [model_classes.index(c) for c in class_order]
            val_pred = val_pred_full[:, idx_map]
            test_pred = test_pred_full[:, idx_map]
        except Exception:
            val_pred = val_pred_full
            test_pred = test_pred_full

        oof_probs[val_idx] = val_pred
        test_fold_probs.append(test_pred)

        fold_logloss = log_loss(y_val, val_pred)
        if VERBOSE:
            print(f"Fold {fold} log_loss: {fold_logloss:.6f}")

    overall_logloss = log_loss(y, oof_probs)
    print(f"\nOOF log_loss: {overall_logloss:.6f}")

    # Average test probabilities over folds
    test_probs = np.mean(test_fold_probs, axis=0)

    # Build submission with exact headers: id,0,1,2
    submission = pd.DataFrame({
        "id": test[id_col].values,
        "0": test_probs[:, class_order.index(0)],
        "1": test_probs[:, class_order.index(1)],
        "2": test_probs[:, class_order.index(2)]
    })

    # Sanity check: row-wise sums ~ 1
    sums = submission[["0", "1", "2"]].sum(axis=1)
    print("\nProbability sum stats (should be ~1.0):")
    print(sums.describe())

    out_path = Path("submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path.resolve()}")

    diagnostics = {
        "model_name": "CatBoostClassifier (Ordered Gradient Boosting with Oblivious Trees)",
        "oof_log_loss": float(overall_logloss),
        "folds": N_FOLDS,
        "random_state": RANDOM_STATE,
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