from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

import config
from dsh_utils import (
    load_primary_dataset,
    make_fold_manifest,
    fold_indices_from_manifest,
    build_model,
    feature_to_domain_map,
    ensure_dir,
    save_json,
)


SUPPORTED = {"XGBoost", "RandomForest"}


def _extract_tree_shap(fitted_pipeline, X_test):
    imputer = fitted_pipeline.named_steps["imputer"]
    tree_model = fitted_pipeline.named_steps["model"]

    X_imp = imputer.transform(X_test)
    explainer = shap.TreeExplainer(tree_model)

    try:
        sv = explainer(X_imp, check_additivity=False)
        vals = sv.values
        base = sv.base_values
    except TypeError:
        sv = explainer(X_imp)
        vals = sv.values
        base = sv.base_values
    except Exception:
        vals = explainer.shap_values(X_imp)
        base = getattr(explainer, "expected_value", np.nan)

    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    base = np.asarray(base)
    if base.ndim == 2:
        base = base[:, 1]
    elif base.ndim == 1 and len(base) == 2 and vals.shape[0] != 2:
        base = np.repeat(base[1], vals.shape[0])
    elif base.ndim == 0:
        base = np.repeat(float(base), vals.shape[0])

    return vals, X_imp, base


def _plot_domain_bar(domain, value_col, xlabel, title, outfile, annotate_n=False):
    dg = domain.sort_values(value_col)
    fig, ax = plt.subplots(figsize=(8, max(5, .35 * len(dg))))
    ax.barh(dg["domain"], dg[value_col])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if annotate_n:
        xmax = float(dg[value_col].max()) if len(dg) else 1.0
        pad = 0.01 * xmax if xmax > 0 else 0.01
        for i, (_, r) in enumerate(dg.iterrows()):
            ax.text(float(r[value_col]) + pad, i, f"n={int(r['n_features'])}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def run():
    if config.SHAP_MODEL not in SUPPORTED:
        raise ValueError(
            f"SHAP_MODEL={config.SHAP_MODEL!r} is not supported in this module. "
            f"Use one of {sorted(SUPPORTED)}. "
            "EasyEnsemble-XGB requires averaging SHAP across constituent learners "
            "and is intentionally kept separate from this clean TreeSHAP analysis."
        )

    outdir = ensure_dir(config.OUTPUT_ROOT / "shap")
    _, X, y, meta = load_primary_dataset()

    main_manifest_path = config.OUTPUT_ROOT / "main_models" / "fold_manifest.csv"
    if main_manifest_path.exists():
        manifest = pd.read_csv(main_manifest_path)
    else:
        manifest = make_fold_manifest(y)

    all_vals = []
    all_data = []
    row_meta = []

    for cv_seed in config.CV_SEEDS:
        for fold in range(config.N_SPLITS):
            train_ids, test_ids = fold_indices_from_manifest(
                manifest, y, cv_seed, fold
            )
            model = build_model(config.SHAP_MODEL, y.loc[train_ids])
            model.fit(X.loc[train_ids], y.loc[train_ids])

            vals, X_imp, _ = _extract_tree_shap(model, X.loc[test_ids])
            all_vals.append(vals)
            all_data.append(X_imp)
            row_meta.extend({
                "sample_index": int(idx),
                "cv_seed": int(cv_seed),
                "fold": int(fold),
            } for idx in test_ids)

            print(
                f"SHAP {config.SHAP_MODEL}: seed={cv_seed}, fold={fold}, "
                f"rows={len(test_ids)}"
            )

    values = np.vstack(all_vals)
    data = np.vstack(all_data)
    row_meta = pd.DataFrame(row_meta)

    if values.shape != data.shape or values.shape[1] != X.shape[1]:
        raise RuntimeError(
            f"SHAP/data shape mismatch: SHAP={values.shape}, data={data.shape}, p={X.shape[1]}"
        )

    np.savez_compressed(
        outdir / "oof_shap_values.npz",
        shap_values=values.astype(np.float32),
        feature_values=data.astype(np.float32),
        feature_names=np.asarray(X.columns, dtype=object),
        sample_index=row_meta["sample_index"].to_numpy(),
        cv_seed=row_meta["cv_seed"].to_numpy(),
        fold=row_meta["fold"].to_numpy(),
    )

    mean_abs = np.mean(np.abs(values), axis=0)
    mean_signed = np.mean(values, axis=0)
    feat = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
    }).sort_values("mean_abs_shap", ascending=False)
    feat["rank"] = np.arange(1, len(feat) + 1)

    mapping = feature_to_domain_map(X, include_unassigned=True)
    feat["domain"] = feat["feature"].map(mapping).fillna("Unassigned")
    feat.to_csv(outdir / "shap_feature_importance_all.csv", index=False)
    feat.head(config.SHAP_TOP_K).to_csv(
        outdir / f"shap_top{config.SHAP_TOP_K}_features.csv", index=False
    )
    feat[["feature", "domain"]].to_csv(
        outdir / "canonical_domain_membership.csv", index=False
    )

    domain = (
        feat.groupby("domain", as_index=False)
        .agg(
            n_features=("feature", "count"),
            sum_mean_abs_shap=("mean_abs_shap", "sum"),
            mean_per_feature_abs_shap=("mean_abs_shap", "mean"),
            median_per_feature_abs_shap=("mean_abs_shap", "median"),
            sum_mean_signed_shap=("mean_signed_shap", "sum"),
        )
    )
    total = domain["sum_mean_abs_shap"].sum()
    domain["share_of_total_abs_shap"] = (
        domain["sum_mean_abs_shap"] / total if total > 0 else np.nan
    )
    domain["rank_total_abs_shap"] = (
        domain["sum_mean_abs_shap"].rank(method="min", ascending=False).astype(int)
    )
    domain["rank_mean_per_feature"] = (
        domain["mean_per_feature_abs_shap"].rank(method="min", ascending=False).astype(int)
    )
    domain = domain.sort_values("sum_mean_abs_shap", ascending=False)
    domain.to_csv(outdir / "shap_domain_importance.csv", index=False)

    size_corr = domain["n_features"].corr(domain["sum_mean_abs_shap"], method="spearman")
    rank_corr = domain["rank_total_abs_shap"].corr(
        domain["rank_mean_per_feature"], method="spearman"
    )
    pd.DataFrame([{
        "spearman_n_features_vs_total_abs_shap": size_corr,
        "spearman_total_rank_vs_mean_per_feature_rank": rank_corr,
    }]).to_csv(outdir / "shap_domain_size_diagnostic.csv", index=False)

    top = feat.head(config.SHAP_TOP_K).sort_values("mean_abs_shap")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value| across repeated OOF predictions")
    plt.title(f"Top {config.SHAP_TOP_K} features — {config.SHAP_MODEL}")
    plt.tight_layout()
    plt.savefig(outdir / f"shap_top{config.SHAP_TOP_K}_bar.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / f"shap_top{config.SHAP_TOP_K}_bar.svg", bbox_inches="tight")
    plt.close()

    _plot_domain_bar(
        domain,
        value_col="sum_mean_abs_shap",
        xlabel="Total mean |SHAP| across encoded features",
        title=f"Domain-level SHAP attribution — {config.SHAP_MODEL}",
        outfile=outdir / "shap_domain_total_importance",
        annotate_n=True,
    )

    _plot_domain_bar(
        domain,
        value_col="mean_per_feature_abs_shap",
        xlabel="Mean |SHAP| per encoded feature",
        title=f"Domain-level SHAP attribution normalized by feature count — {config.SHAP_MODEL}",
        outfile=outdir / "shap_domain_importance_per_feature",
        annotate_n=True,
    )
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    nplot = min(config.SHAP_PLOT_MAX_ROWS, len(values))
    idx = np.sort(rng.choice(len(values), size=nplot, replace=False))
    shap.summary_plot(
        values[idx],
        data[idx],
        feature_names=X.columns.tolist(),
        max_display=config.SHAP_TOP_K,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(outdir / f"shap_top{config.SHAP_TOP_K}_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / f"shap_top{config.SHAP_TOP_K}_beeswarm.svg", bbox_inches="tight")
    plt.close()

    print("\nTOP FEATURES")
    print(feat.head(config.SHAP_TOP_K).to_string(index=False))
    print("\nDOMAIN SHAP ATTRIBUTION")
    print(domain.to_string(index=False))
    print(f"\nSpearman(domain n_features, total |SHAP|) = {size_corr:.3f}")
    print(f"Spearman(total-SHAP rank, per-feature-SHAP rank) = {rank_corr:.3f}")
    return feat, domain


if __name__ == "__main__":
    run()
