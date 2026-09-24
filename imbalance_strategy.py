from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import EasyEnsembleClassifier

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
from dsh_utils import (
    load_primary_dataset,
    make_fold_manifest,
    fold_indices_from_manifest,
    bootstrap_metric_ci,
    aggregate_repeated_oof,
    ensure_dir,
)


def _xgb(*, weighted, y_train=None, n_jobs=-1):
    kwargs = dict(
        n_estimators=config.N_TREES,
        max_depth=config.MAX_DEPTH,
        random_state=config.MODEL_SEED,
        eval_metric="aucpr",
        objective="binary:logistic",
        tree_method=config.XGB_TREE_METHOD,
        n_jobs=n_jobs,
    )
    if weighted:
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        kwargs["scale_pos_weight"] = neg / max(pos, 1)
    return XGBClassifier(**kwargs)


def build_imbalance_model(strategy, y_train):
    if strategy == "PlainXGBoost":
        return SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", _xgb(weighted=False, y_train=y_train)),
        ])

    if strategy == "WeightedXGBoost":
        return SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", _xgb(weighted=True, y_train=y_train)),
        ])

    if strategy == "ROSXGBoost":
        return ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("ros", RandomOverSampler(
                sampling_strategy=1.0,
                random_state=config.MODEL_SEED,
            )),
            ("model", _xgb(weighted=False, y_train=y_train)),
        ])

    if strategy == "EasyEnsembleXGB":
        base = _xgb(weighted=False, y_train=y_train, n_jobs=1)
        ee = EasyEnsembleClassifier(
            estimator=base,
            n_estimators=config.EASY_ENSEMBLE_N_ESTIMATORS,
            random_state=config.MODEL_SEED,
            n_jobs=-1,
        )
        return SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ee),
        ])

    if strategy == "BalanceCascadeXGB":
        if BalanceCascadeClassifier is None:
            raise ImportError(
                "BalanceCascadeXGB requires 'imbalanced-ensemble==0.2.3' "
                "(import name: 'imbens'). Run `python -m pip install -r requirements.txt` "
                "and restart the runtime if needed. Original import error: "
                f"{_BALANCE_CASCADE_IMPORT_ERROR!r}"
            ) from _BALANCE_CASCADE_IMPORT_ERROR
        base = _xgb(weighted=False, y_train=y_train, n_jobs=1)
        bc = BalanceCascadeClassifier(
            estimator=base,
            n_estimators=config.BALANCE_CASCADE_N_ESTIMATORS,
            replacement=True,
            random_state=config.MODEL_SEED,
            n_jobs=-1,
        )
        return SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", bc),
        ])

    raise ValueError(strategy)


def run():
    outdir = ensure_dir(config.OUTPUT_ROOT / "imbalance_strategy")
    _, X, y, _ = load_primary_dataset()

    main_manifest = config.OUTPUT_ROOT / "main_models" / "fold_manifest.csv"
    manifest = pd.read_csv(main_manifest) if main_manifest.exists() else make_fold_manifest(y)

    rows = []
    for strategy in config.IMBALANCE_STRATEGIES:
        print(f"\n=== {strategy} ===")
        for cv_seed in config.CV_SEEDS:
            for fold in range(config.N_SPLITS):
                train_ids, test_ids = fold_indices_from_manifest(manifest, y, cv_seed, fold)
                model = build_imbalance_model(strategy, y.loc[train_ids])
                model.fit(X.loc[train_ids], y.loc[train_ids])
                prob = model.predict_proba(X.loc[test_ids])[:, 1]

                rows.extend({
                    "sample_index": int(idx),
                    "model_name": strategy,
                    "cv_seed": int(cv_seed),
                    "fold": int(fold),
                    "y_true": int(yt),
                    "y_prob": float(pp),
                    "y_pred_0p5": int(pp >= 0.5),
                } for idx, yt, pp in zip(test_ids, y.loc[test_ids], prob))

    long = pd.DataFrame(rows)
    long.to_csv(outdir / "imbalance_oof_predictions_long.csv", index=False)

    avg = aggregate_repeated_oof(long)
    avg.to_csv(outdir / "imbalance_oof_predictions_averaged.csv", index=False)

    ci_rows = []
    for i, (strategy, g) in enumerate(avg.groupby("model_name")):
        ci = bootstrap_metric_ci(
            g["y_true"].to_numpy(),
            g["y_prob"].to_numpy(),
            random_state=config.BOOTSTRAP_SEED + 5000 + i,
            metrics=[
                "pr_auc", "roc_auc", "balanced_accuracy",
                "sensitivity", "precision",
                "recall_at_20pct", "precision_at_20pct", "lift_at_20pct",
            ],
        )
        ci.insert(0, "model_name", strategy)
        ci_rows.append(ci)

    summary = pd.concat(ci_rows, ignore_index=True)
    summary.to_csv(outdir / "imbalance_strategy_summary_95ci.csv", index=False)

    print(summary.to_string(index=False))
    return summary, avg


if __name__ == "__main__":
    run()
