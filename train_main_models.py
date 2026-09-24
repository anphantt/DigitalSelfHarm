from __future__ import annotations

import numpy as np
import pandas as pd

import config
from dsh_utils import (
    load_primary_dataset,
    make_fold_manifest,
    fold_indices_from_manifest,
    build_model,
    compute_metrics,
    aggregate_repeated_oof,
    ensure_dir,
    save_json,
    package_versions,
)


MANUSCRIPT_METRICS = [
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "sensitivity",
    "precision",
]

METRIC_LABELS = {
    "balanced_accuracy": "Balanced Accuracy",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "sensitivity": "Sensitivity",
    "precision": "Precision",
}

CONFUSION_COLS = ["tn", "fp", "fn", "tp"]


def _round_count(x: float) -> int:
    return int(np.floor(float(x) + 0.5))


def _summarize_repetitions(per_rep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "roc_auc", "pr_auc", "brier", "balanced_accuracy",
        "sensitivity", "specificity", "precision", "f1",
    ]
    for model_name, g in per_rep.groupby("model_name", sort=False):
        for metric in metric_cols:
            if metric not in g.columns:
                continue
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append({
                "model_name": model_name,
                "metric": metric,
                "mean_across_repetitions": float(vals.mean()),
                "sd_across_repetitions": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "min_across_repetitions": float(vals.min()),
                "max_across_repetitions": float(vals.max()),
                "n_repetitions": int(len(vals)),
            })
    return pd.DataFrame(rows)


def _confusion_summaries(per_rep: pd.DataFrame):
    by_rep = per_rep[["model_name", "cv_seed", *CONFUSION_COLS]].copy()

    mean_rows = []
    rounded_rows = []
    for model_name, g in by_rep.groupby("model_name", sort=False):
        means = g[CONFUSION_COLS].mean()
        sds = g[CONFUSION_COLS].std(ddof=1)
        mean_rows.append({
            "model_name": model_name,
            **{f"{c}_mean": float(means[c]) for c in CONFUSION_COLS},
            **{f"{c}_sd": float(sds[c]) for c in CONFUSION_COLS},
        })
        rounded = {c: _round_count(means[c]) for c in CONFUSION_COLS}
        rounded_rows.append({
            "model_name": model_name,
            **rounded,
            "confusion_matrix": (
                f"[[{rounded['tn']}, {rounded['fp']}], "
                f"[{rounded['fn']}, {rounded['tp']}]]"
            ),
        })
    return by_rep, pd.DataFrame(mean_rows), pd.DataFrame(rounded_rows)


def _make_manuscript_table(
    repetition_summary: pd.DataFrame,
    confusion_rounded: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for model_name in config.MAIN_MODELS:
        s = repetition_summary[repetition_summary["model_name"] == model_name]
        cm = confusion_rounded[confusion_rounded["model_name"] == model_name]
        if s.empty or cm.empty:
            continue
        by_metric = s.set_index("metric")
        row = {
            "model_name": model_name,
            "Model": config.MODEL_LABELS.get(model_name, model_name),
        }
        for metric in MANUSCRIPT_METRICS:
            row[metric] = float(by_metric.loc[metric, "mean_across_repetitions"])
            row[METRIC_LABELS[metric]] = f"{row[metric]:.4f}"
        row["Confusion Matrix"] = cm.iloc[0]["confusion_matrix"]
        rows.append(row)
    return pd.DataFrame(rows)


def run():
    outdir = ensure_dir(config.OUTPUT_ROOT / "main_models")
    _, X, y, meta = load_primary_dataset()

    manifest = make_fold_manifest(y)
    manifest.to_csv(outdir / "fold_manifest.csv", index=False)

    pred_rows = []
    fold_rows = []

    for model_name in config.MAIN_MODELS:
        print(f"\n{'='*80}\nMODEL: {model_name}\n{'='*80}")
        for cv_seed in config.CV_SEEDS:
            for fold in range(config.N_SPLITS):
                train_ids, test_ids = fold_indices_from_manifest(
                    manifest, y, cv_seed, fold
                )
                X_train, X_test = X.loc[train_ids], X.loc[test_ids]
                y_train, y_test = y.loc[train_ids], y.loc[test_ids]

                model = build_model(model_name, y_train)
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_test)[:, 1]

                fm = compute_metrics(y_test.to_numpy(), prob)
                fm.update({
                    "model_name": model_name,
                    "cv_seed": int(cv_seed),
                    "fold": int(fold),
                    "n_train": int(len(train_ids)),
                    "n_test": int(len(test_ids)),
                })
                fold_rows.append(fm)

                pred_rows.extend({
                    "sample_index": int(idx),
                    "model_name": model_name,
                    "cv_seed": int(cv_seed),
                    "fold": int(fold),
                    "y_true": int(yt),
                    "y_prob": float(pp),
                    "y_pred_0p5": int(pp >= 0.5),
                } for idx, yt, pp in zip(test_ids, y_test, prob))

                print(
                    f"{model_name} seed={cv_seed} fold={fold}: "
                    f"PR-AUC={fm['pr_auc']:.4f}, ROC-AUC={fm['roc_auc']:.4f}"
                )

    preds = pd.DataFrame(pred_rows)
    folds = pd.DataFrame(fold_rows)
    preds.to_csv(outdir / "oof_predictions_long.csv", index=False)
    folds.to_csv(outdir / "fold_metrics_diagnostic.csv", index=False)

    per_rep_rows = []
    for (model_name, cv_seed), g in preds.groupby(["model_name", "cv_seed"], sort=False):
        g = g.sort_values("sample_index")
        if len(g) != len(y):
            raise RuntimeError(
                f"Expected {len(y)} pooled OOF rows for {model_name}, seed={cv_seed}; "
                f"got {len(g)}"
            )
        m = compute_metrics(g["y_true"], g["y_prob"])
        m.update({"model_name": model_name, "cv_seed": int(cv_seed)})
        per_rep_rows.append(m)

    per_rep = pd.DataFrame(per_rep_rows)
    per_rep.to_csv(outdir / "per_repetition_oof_metrics.csv", index=False)
    per_rep.to_csv(outdir / "per_seed_oof_metrics.csv", index=False)

    repetition_summary = _summarize_repetitions(per_rep)
    repetition_summary.to_csv(
        outdir / "performance_summary_across_repetitions.csv", index=False
    )

    cm_by_rep, cm_mean, cm_rounded = _confusion_summaries(per_rep)
    cm_by_rep.to_csv(outdir / "confusion_matrix_by_repetition.csv", index=False)
    cm_mean.to_csv(outdir / "confusion_matrix_mean_sd_across_repetitions.csv", index=False)
    cm_rounded.to_csv(outdir / "confusion_matrix_manuscript_rounded.csv", index=False)

    manuscript = _make_manuscript_table(repetition_summary, cm_rounded)
    manuscript.to_csv(outdir / "manuscript_performance_table.csv", index=False)

    averaged = aggregate_repeated_oof(preds)
    averaged.to_csv(outdir / "oof_predictions_averaged.csv", index=False)

    save_json({
        "sample": meta,
        "cv_seeds": config.CV_SEEDS,
        "n_splits": config.N_SPLITS,
        "model_seed": config.MODEL_SEED,
        "models": config.MAIN_MODELS,
        "manuscript_performance_reporting": (
            "For each model, metrics are computed separately on the complete pooled "
            "10-fold OOF prediction vector from each CV split seed, then averaged "
            "across the three repetitions. No bootstrap CI is attached to the main "
            "performance table."
        ),
        "manuscript_confusion_matrix_reporting": (
            "Confusion matrices are computed separately for each repetition at "
            "threshold 0.5; cells are averaged across repetitions and rounded to "
            "nearest integer for manuscript display."
        ),
        "averaged_oof_usage": (
            "Each student's three OOF probabilities are averaged only for downstream "
            "risk-score deciles and fixed-capacity prioritization, not for the main "
            "performance table."
        ),
        "reported_metrics": MANUSCRIPT_METRICS,
        "threshold_for_label_metrics": 0.5,
        "package_versions": package_versions(),
    }, outdir / "main_model_manifest.json")

    print("\nMANUSCRIPT PERFORMANCE TABLE — MEAN ACROSS 3 CV REPETITIONS")
    print(manuscript.to_string(index=False))
    print("\nAveraged OOF scores were saved for risk stratification only.")
    return manuscript, averaged, preds, per_rep


if __name__ == "__main__":
    run()
