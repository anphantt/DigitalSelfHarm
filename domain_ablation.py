from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from dsh_utils import (
    load_primary_dataset,
    make_fold_manifest,
    fold_indices_from_manifest,
    build_model,
    compute_metrics,
    topk_metrics,
    feature_to_domain_map,
    ensure_dir,
)


import seaborn as sns
sns.set_theme(style="white", palette="Set2")

def _canonical_domain_columns(X, include_unassigned=False):
    mapping = feature_to_domain_map(X, include_unassigned=True)
    out = {}
    for col in X.columns:
        domain = mapping.get(col, "Unassigned")
        if domain is None:
            domain = "Unassigned"
        domain = str(domain)
        if (not include_unassigned) and domain.lower() == "unassigned":
            continue
        out.setdefault(domain, []).append(col)
    return out


def _norm_name(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def _resolve_combo_domains(combo_domains, domain_cols):
    lookup = {_norm_name(k): k for k in domain_cols}
    resolved = []
    missing = []
    for d in combo_domains:
        key = _norm_name(d)
        if key in lookup:
            resolved.append(lookup[key])
        else:
            missing.append(str(d))
    return resolved, missing


def _feature_sets(X):
    sets = {"full": list(X.columns)}
    domain_cols = _canonical_domain_columns(X, include_unassigned=False)

    for domain, cols in domain_cols.items():
        present = set(cols)
        if not present:
            continue
        sets[f"minus::{domain}"] = [c for c in X.columns if c not in present]
        sets[f"only::{domain}"] = list(cols)

    combos = getattr(config, "ABLATION_COMBINATIONS", {})
    for combo, requested_domains in combos.items():
        resolved, missing = _resolve_combo_domains(requested_domains, domain_cols)
        if missing:
            print(
                f"WARNING: skipping combo {combo!r}; canonical domains not found: {missing}. "
                f"Available domains: {sorted(domain_cols)}"
            )
            continue
        cols = []
        for d in resolved:
            cols.extend(domain_cols[d])
        selected = [c for c in X.columns if c in set(cols)]
        if selected:
            sets[f"only::{combo}"] = selected
            remove = set(selected)
            sets[f"minus::{combo}"] = [c for c in X.columns if c not in remove]

    return {k: v for k, v in sets.items() if len(v) > 0}, domain_cols


def _metric_value(y, p, metric):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if metric == "pr_auc":
        return float(average_precision_score(y, p))
    if metric == "roc_auc":
        return float(roc_auc_score(y, p))
    if metric == "brier":
        return float(brier_score_loss(y, p))
    if metric == "recall_at_20pct":
        return topk_metrics(y, p, 0.20)["recall_case_capture"]
    raise KeyError(metric)


def _paired_bootstrap_delta(y, p_candidate, p_full, metric, n_boot, seed):
    y = np.asarray(y, int)
    pc = np.asarray(p_candidate, float)
    pf = np.asarray(p_full, float)
    n = len(y)
    rng = np.random.default_rng(seed)

    point = _metric_value(y, pc, metric) - _metric_value(y, pf, metric)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue
        vals.append(
            _metric_value(yb, pc[idx], metric)
            - _metric_value(yb, pf[idx], metric)
        )
    vals = np.asarray(vals)
    if len(vals) == 0:
        return point, np.nan, np.nan
    return point, float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def _clean_label(name):
    return str(name).replace("minus::", "").replace("only::", "").replace("_", " ")


def _is_single_domain_feature_set(fs_name, domain_names):
    clean = str(fs_name).replace("minus::", "").replace("only::", "")
    return clean in set(domain_names)


def plot_domain_only(df, save_path):    
    models = ["LogisticRegression","XGBoost"]

    plot_df = df[df["feature_set"].str.startswith("only::") & df["model_name"].isin(models)].copy()
    plot_df["set_name"] = (plot_df["feature_set"].str.replace("only::", "", regex=False))
    plot_df = plot_df[plot_df["set_name"].isin(config.SELECTED_SETS)].copy()
    plot_df["display_name"] = (plot_df["set_name"].map(config.DISPLAY_LABELS))

    full = (
        df[(df["feature_set"] == "full") & df["model_name"].isin(models)]
        .set_index("model_name")["pr_auc"]
        .to_dict()
    )

    order = [config.DISPLAY_LABELS[x] for x in config.SELECTED_SETS]

    pivot = (
        plot_df
        .pivot(
            index="display_name",
            columns="model_name",
            values="pr_auc"
        )
        .reindex(order)
    )

    if pivot.isna().any().any():
        print("Warning: missing values in sufficiency plot:\n", pivot[pivot.isna().any(axis=1)])

    y = np.arange(len(pivot))
    height = 0.34

    fig, ax = plt.subplots(figsize=(10,7))

    ax.barh(
        y - height / 2,
        pivot["LogisticRegression"],
        height,
        label="Logistic Regression",
        edgecolor="0.2",
        linewidth=0.8,
    )

    ax.barh(
        y + height / 2,
        pivot["XGBoost"],
        height,
        label="XGBoost",
        edgecolor="0.2",
        linewidth=0.8,
    )

    ax.axvline(
        full["LogisticRegression"],
        linestyle="--",
        color="red",
        linewidth=1.2,
        label=f"LR full model ({full['LogisticRegression']:.3f})"
    )

    ax.axvline(
        full["XGBoost"],
        linestyle=":",
        color="black",
        linewidth=1.5,
        label=f"XGBoost full model ({full['XGBoost']:.3f})"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("PR-AUC")
    ax.set_title("Predictive performance using selected domains")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="center right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.show()


def plot_domain_remove(df, save_path):
    models = ["LogisticRegression","XGBoost"]

    plot_df = df[
        (df["metric"] == "pr_auc")
        & df["feature_set"].str.startswith("minus::")
        & df["model_name"].isin(models)
    ].copy()

    plot_df["set_name"] = (plot_df["feature_set"].str.replace("minus::", "", regex=False))
    plot_df = plot_df[plot_df["set_name"].isin(config.SELECTED_SETS)].copy()
    plot_df["display_name"] = (plot_df["set_name"].map(config.DISPLAY_LABELS))
    order = [config.DISPLAY_LABELS[x] for x in config.SELECTED_SETS]

    y = np.arange(len(order))

    height = 0.34
    fig, ax = plt.subplots(figsize=(10,7))

    for j, model in enumerate(models):
        sub = (
            plot_df[plot_df["model_name"] == model]
            .set_index("display_name")
            .reindex(order)
        )

        if sub[["delta_candidate_minus_full", "ci_low", "ci_high"]].isna().any().any():
            print(
                f"Warning: missing values for {model}:\n",
                sub[sub[["delta_candidate_minus_full", "ci_low", "ci_high"]].isna().any(axis=1)]
            )

        values = (sub["delta_candidate_minus_full"].to_numpy())

        low = (sub["ci_low"].to_numpy())
        high = (sub["ci_high"].to_numpy())
        xerr = np.vstack([values - low, high - values])
        offset = (-height / 2 if j == 0 else height / 2)
        
        ax.errorbar(
            values,
            y + offset,
            xerr=xerr,
            fmt="o",
            capsize=2,
            label=config.MODEL_LABELS[model]
        )

    ax.axvline(0, linestyle="--", linewidth=1, color="black")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(r"Change in PR-AUC after removal ($\Delta$ = removed $-$ full)")
    ax.set_title("Change in predictive performance after removing selected domains")

    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.show()


def run():
    outdir = ensure_dir(config.OUTPUT_ROOT / "domain_ablation")
    _, X, y, meta = load_primary_dataset()

    main_manifest_path = config.OUTPUT_ROOT / "main_models" / "fold_manifest.csv"
    if main_manifest_path.exists():
        manifest = pd.read_csv(main_manifest_path)
    else:
        manifest = make_fold_manifest(y)

    feature_sets, domain_cols = _feature_sets(X)
    domain_names = list(domain_cols)
    pred_rows = []
    for model_name in config.ABLATION_MODELS:
        print(f"\n{'='*80}\nABLATION MODEL: {model_name}\n{'='*80}")
        for fs_name, cols in feature_sets.items():
            Xs = X[cols]
            print(f"\n{fs_name}: p={Xs.shape[1]}")
            for cv_seed in config.CV_SEEDS:
                for fold in range(config.N_SPLITS):
                    train_ids, test_ids = fold_indices_from_manifest(
                        manifest, y, cv_seed, fold
                    )
                    model = build_model(model_name, y.loc[train_ids])
                    model.fit(Xs.loc[train_ids], y.loc[train_ids])
                    prob = model.predict_proba(Xs.loc[test_ids])[:, 1]

                    pred_rows.extend({
                        "sample_index": int(idx),
                        "model_name": model_name,
                        "feature_set": fs_name,
                        "cv_seed": int(cv_seed),
                        "fold": int(fold),
                        "y_true": int(yt),
                        "y_prob": float(pp),
                    } for idx, yt, pp in zip(test_ids, y.loc[test_ids], prob))

    preds = pd.DataFrame(pred_rows)
    preds.to_csv(outdir / "ablation_oof_predictions_long.csv", index=False)

    avg = (
        preds.groupby(["model_name", "feature_set", "sample_index"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_prob=("y_prob", "mean"),
            n_repeats=("cv_seed", "nunique"),
        )
    )
    avg.to_csv(outdir / "ablation_oof_predictions_averaged.csv", index=False)

    summary_rows = []
    delta_rows = []
    metrics_for_delta = ["pr_auc", "roc_auc", "brier", "recall_at_20pct"]

    for model_name, gm in avg.groupby("model_name"):
        full = gm[gm["feature_set"] == "full"].sort_values("sample_index")
        full_idx = full["sample_index"].to_numpy()
        full_y = full["y_true"].to_numpy()
        full_p = full["y_prob"].to_numpy()

        for fs_name, g in gm.groupby("feature_set"):
            g = g.sort_values("sample_index")
            if not np.array_equal(g["sample_index"].to_numpy(), full_idx):
                raise RuntimeError(f"Sample mismatch for {model_name} {fs_name}")

            m = compute_metrics(g["y_true"], g["y_prob"])
            summary_rows.append({
                "model_name": model_name,
                "feature_set": fs_name,
                "set_type": (
                    "full"
                    if fs_name == "full"
                    else "single_domain"
                    if _is_single_domain_feature_set(fs_name, domain_names)
                    else "targeted_combo"
                ),
                "n_features": len(feature_sets[fs_name]),
                **m,
            })

            if fs_name == "full":
                continue

            for j, metric in enumerate(metrics_for_delta):
                d, lo, hi = _paired_bootstrap_delta(
                    full_y,
                    g["y_prob"].to_numpy(),
                    full_p,
                    metric,
                    config.BOOTSTRAP_REPS,
                    config.BOOTSTRAP_SEED + j + 1000,
                )
                delta_rows.append({
                    "model_name": model_name,
                    "feature_set": fs_name,
                    "set_type": (
                        "single_domain"
                        if _is_single_domain_feature_set(fs_name, domain_names)
                        else "targeted_combo"
                    ),
                    "metric": metric,
                    "delta_candidate_minus_full": d,
                    "ci_low": lo,
                    "ci_high": hi,
                })

    summary = pd.DataFrame(summary_rows)
    deltas = pd.DataFrame(delta_rows)
    summary.to_csv(outdir / "ablation_summary.csv", index=False)
    deltas.to_csv(outdir / "ablation_delta_vs_full_95ci.csv", index=False)
    plot_domain_only(summary, save_path=os.path.join(outdir, "domain_ablation_only.png"))
    plot_domain_remove(deltas, save_path=os.path.join(outdir, "domain_ablation_remove.png"))

    return summary, deltas


if __name__ == "__main__":
    run()
