from __future__ import annotations

from pathlib import Path
import json
import platform
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)

from imblearn.ensemble import EasyEnsembleClassifier
from xgboost import XGBClassifier

_BALANCE_CASCADE_IMPORT_ERROR = None
try:
    from imbens.ensemble import BalanceCascadeClassifier
except Exception as exc_public:
    try:
        from imbens.ensemble._under_sampling.balance_cascade import BalanceCascadeClassifier
    except Exception as exc_internal:
        BalanceCascadeClassifier = None
        _BALANCE_CASCADE_IMPORT_ERROR = exc_internal

import config


ORDER = [
    "Zero Times",
    "1-2 Times",
    "3-5 Times",
    "6-9 Times",
    "10 or More Times",
]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def package_versions():
    import sklearn
    import imblearn
    import xgboost
    out = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": imblearn.__version__,
        "xgboost": xgboost.__version__,
    }
    try:
        import imbens
        out["imbalanced_ensemble"] = getattr(imbens, "__version__", "unknown")
    except Exception as exc:
        out["imbalanced_ensemble"] = f"IMPORT ERROR: {exc!r}"
    try:
        import shap
        out["shap"] = shap.__version__
    except Exception:
        out["shap"] = None
    return out


def save_json(obj, path):
    path = Path(path)
    ensure_dir(path.parent)

    def _default(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def _to_level(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    lookup = {v.lower(): i for i, v in enumerate(ORDER)}
    return lookup.get(s.lower(), np.nan)


def load_primary_dataset(data_path=None):
    data_path = Path(data_path or config.DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Primary data not found: {data_path}\n"
            "Edit config.DATA_PATH or set DSH_DATA_PATH."
        )

    df = pd.read_csv(data_path)
    df = df.drop(columns=["label_year", "label_month"], errors="ignore")
    required = {"FL708", "FL709"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"processed_Feb21 must still contain FL708 and FL709. Missing: {sorted(missing)}"
        )

    level_12 = df["FL708"].map(_to_level)
    level_30 = df["FL709"].map(_to_level)

    valid_pair = level_12.notna() & level_30.notna()
    violation = valid_pair & (level_30 > level_12)

    level_12_reconciled = level_12.copy()
    level_12_reconciled.loc[violation] = level_30.loc[violation]

    y = pd.Series(
        np.where(
            level_12_reconciled.isna(),
            np.nan,
            (level_12_reconciled > 0).astype(float),
        ),
        index=df.index,
        name=config.TARGET_NAME,
    )

    keep = y.notna()
    df_target = df.loc[keep].copy()
    y = y.loc[keep].astype(int)

    X = df_target.drop(
        columns=[
            "FL708", "FL709",
            "label_year", "label_month",
            "level_12", "level_30",
            "D8_nan", "D3_nan",
        ],
        errors="ignore",
    ).copy()

    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    all_missing_cols = X.columns[X.isna().all()].tolist()
    if all_missing_cols:
        warnings.warn(
            "Columns are entirely missing after numeric coercion and will be dropped: "
            + ", ".join(all_missing_cols)
        )
        X = X.drop(columns=all_missing_cols)

    excluded_present = [
        c for c in getattr(config, "EXCLUDE_MODEL_FEATURES", [])
        if c in X.columns
    ]
    if excluded_present:
        X = X.drop(columns=excluded_present)

    X.index.name = "sample_index"
    y.index = X.index

    meta = {
        "data_path": str(data_path),
        "n_rows_file": int(len(df)),
        "n_primary_sample": int(len(X)),
        "n_features": int(X.shape[1]),
        "class_0": int((y == 0).sum()),
        "class_1": int((y == 1).sum()),
        "prevalence": float(y.mean()),
        "n_temporal_violations_repaired": int(violation.sum()),
        "n_missing_primary_outcome": int((~keep).sum()),
        "object_columns_coerced": obj_cols,
        "all_missing_columns_dropped": all_missing_cols,
        "primary_predictor_exclusions": excluded_present,
    }
    return df_target, X, y, meta


def make_fold_manifest(y, seeds=None, n_splits=None):
    seeds = list(config.CV_SEEDS if seeds is None else seeds)
    n_splits = int(config.N_SPLITS if n_splits is None else n_splits)

    rows = []
    for seed in seeds:
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(seed),
        )
        for fold, (_, test_pos) in enumerate(cv.split(np.zeros(len(y)), y.to_numpy())):
            ids = y.index.to_numpy()[test_pos]
            rows.extend(
                {"sample_index": int(i), "cv_seed": int(seed), "fold": int(fold)}
                for i in ids
            )
    out = pd.DataFrame(rows)
    expected = len(y) * len(seeds)
    if len(out) != expected:
        raise RuntimeError(f"Fold manifest size {len(out)} != expected {expected}")
    return out


def fold_indices_from_manifest(manifest, y, cv_seed, fold):
    test_ids = manifest.loc[
        (manifest["cv_seed"] == int(cv_seed)) &
        (manifest["fold"] == int(fold)),
        "sample_index",
    ].to_numpy()
    test_ids = pd.Index(test_ids)
    train_ids = y.index.difference(test_ids, sort=False)
    return train_ids, test_ids


def _xgb_base(*, scale_pos_weight=None, n_jobs=-1):
    kwargs = dict(
        n_estimators=config.N_TREES,
        max_depth=config.MAX_DEPTH,
        random_state=config.MODEL_SEED,
        eval_metric="aucpr",
        objective="binary:logistic",
        tree_method=config.XGB_TREE_METHOD,
        n_jobs=n_jobs,
    )
    if scale_pos_weight is not None:
        kwargs["scale_pos_weight"] = float(scale_pos_weight)
    return XGBClassifier(**kwargs)


def build_model(model_name, y_train):
    name = str(model_name)

    if name == "LogisticRegression":
        estimator = LogisticRegression(
            max_iter=config.LR_MAX_ITER,
            random_state=config.MODEL_SEED,
            class_weight="balanced",
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])

    if name == "RandomForest":
        estimator = RandomForestClassifier(
            n_estimators=config.N_TREES,
            max_depth=config.MAX_DEPTH,
            random_state=config.MODEL_SEED,
            class_weight="balanced",
            n_jobs=-1,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])

    if name == "XGBoost":
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        spw = neg / max(pos, 1)
        estimator = _xgb_base(scale_pos_weight=spw, n_jobs=-1)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])

    if name == "EasyEnsembleXGB":
        base_xgb = _xgb_base(scale_pos_weight=None, n_jobs=1)
        estimator = EasyEnsembleClassifier(
            estimator=base_xgb,
            n_estimators=config.EASY_ENSEMBLE_N_ESTIMATORS,
            random_state=config.MODEL_SEED,
            n_jobs=-1,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])

    if name == "BalanceCascadeXGB":
        if BalanceCascadeClassifier is None:
            detail = repr(_BALANCE_CASCADE_IMPORT_ERROR)
            raise ImportError(
                "BalanceCascadeXGB requires the PyPI package "
                "'imbalanced-ensemble==0.2.3' (import name: 'imbens'). "
                "Run `python -m pip install -r requirements.txt` and restart the "
                f"runtime if needed. Original import error: {detail}"
            ) from _BALANCE_CASCADE_IMPORT_ERROR
        base_xgb = _xgb_base(scale_pos_weight=None, n_jobs=1)
        estimator = BalanceCascadeClassifier(
            estimator=base_xgb,
            n_estimators=config.BALANCE_CASCADE_N_ESTIMATORS,
            replacement=True,
            random_state=config.MODEL_SEED,
            n_jobs=-1,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])

    raise ValueError(f"Unknown model: {model_name}")


def topk_metrics(y_true, y_prob, capacity):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    k = max(1, min(int(np.ceil(n * float(capacity))), n))

    order = np.argsort(-y_prob, kind="mergesort")
    selected = order[:k]

    y_hat = np.zeros(n, dtype=int)
    y_hat[selected] = 1
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    prevalence = y_true.mean()
    lift = precision / prevalence if prevalence > 0 else np.nan
    fp_per_tp = fp / tp if tp > 0 else np.inf

    return {
        "capacity": float(capacity),
        "n": int(n),
        "n_flagged": int(k),
        "flagged_fraction": float(k / n),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall_case_capture": float(recall),
        "lift": float(lift),
        "fp_per_tp": float(fp_per_tp),
    }


def compute_metrics(y_true, y_prob, threshold=0.5, include_screening=True):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)

    out = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if include_screening:
        m20 = topk_metrics(y_true, y_prob, 0.20)
        out["recall_at_20pct"] = m20["recall_case_capture"]
        out["precision_at_20pct"] = m20["precision"]
        out["lift_at_20pct"] = m20["lift"]
    return out


BOOTSTRAP_METRICS = [
    "roc_auc", "pr_auc", "brier",
    "balanced_accuracy", "sensitivity", "specificity",
    "precision", "f1",
    "recall_at_20pct", "precision_at_20pct", "lift_at_20pct",
]


def bootstrap_metric_ci(
    y_true,
    y_prob,
    *,
    n_boot=None,
    random_state=None,
    metrics=None,
):
    n_boot = int(config.BOOTSTRAP_REPS if n_boot is None else n_boot)
    random_state = int(config.BOOTSTRAP_SEED if random_state is None else random_state)
    metrics = list(BOOTSTRAP_METRICS if metrics is None else metrics)

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(random_state)

    screening_metrics = {"recall_at_20pct", "precision_at_20pct", "lift_at_20pct"}
    needs_screening = any(m in screening_metrics for m in metrics)

    values = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = y_prob[idx]
        if np.unique(yb).size < 2:
            continue
        mb = compute_metrics(yb, pb, include_screening=needs_screening)
        for m in metrics:
            values[m].append(mb[m])

    rows = []
    point = compute_metrics(y_true, y_prob, include_screening=needs_screening)
    for m in metrics:
        arr = np.asarray(values[m], dtype=float)
        rows.append({
            "metric": m,
            "estimate": float(point[m]),
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
            "n_boot_valid": int(len(arr)),
        })
    return pd.DataFrame(rows)


def aggregate_repeated_oof(preds_long):
    required = {"sample_index", "model_name", "cv_seed", "y_true", "y_prob"}
    missing = required - set(preds_long.columns)
    if missing:
        raise ValueError(f"Missing OOF columns: {sorted(missing)}")

    check = (
        preds_long.groupby(["model_name", "cv_seed", "sample_index"])
        .size()
    )
    if check.max() != 1:
        raise ValueError("OOF rows are not unique by model × seed × sample.")

    out = (
        preds_long
        .groupby(["model_name", "sample_index"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_prob=("y_prob", "mean"),
            n_repeats=("cv_seed", "nunique"),
            prob_sd_across_repeats=("y_prob", "std"),
        )
    )
    out["y_pred_0p5"] = (out["y_prob"] >= 0.5).astype(int)
    return out


def present_columns(X, cols):
    return [c for c in cols if c in X.columns]


def combined_domain_columns(X, domain_names):
    cols = []
    for name in domain_names:
        if name not in config.DOMAIN_GROUPS:
            raise KeyError(f"Unknown domain: {name}")
        cols.extend(config.DOMAIN_GROUPS[name])
    return sorted(set(present_columns(X, cols)))


def feature_to_domain_map(X, *, include_unassigned=True):
    mapping = {}
    overlaps = {}
    for domain, cols in config.DOMAIN_GROUPS.items():
        for col in present_columns(X, cols):
            if col in mapping and mapping[col] != domain:
                overlaps.setdefault(col, [mapping[col]]).append(domain)
            else:
                mapping[col] = domain

    if overlaps:
        msg = "; ".join(f"{c}: {v}" for c, v in overlaps.items())
        raise ValueError(
            "DOMAIN_GROUPS overlap. Resolve before domain-level SHAP aggregation: " + msg
        )

    if include_unassigned:
        for col in X.columns:
            mapping.setdefault(col, "Other_unassigned")
    return mapping
