from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from dsh_utils import ensure_dir, topk_metrics


def _load_averaged_oof() -> pd.DataFrame:
    path = config.OUTPUT_ROOT / "main_models" / "oof_predictions_averaged.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run train_main_models.py first.")
    df = pd.read_csv(path)
    required = {"model_name", "sample_index", "y_true", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Averaged OOF file is missing columns: {sorted(missing)}")
    return df


def assign_score_deciles(df: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    if n_deciles < 2:
        raise ValueError("n_deciles must be >= 2")

    out = df.sort_values(["y_prob", "sample_index"], ascending=[True, True]).copy()
    if out.empty:
        raise ValueError("Cannot assign deciles to an empty dataframe")

    rank = out["y_prob"].rank(method="first")
    out["score_decile"] = (
        pd.qcut(rank, q=n_deciles, labels=False, duplicates="drop").astype(int) + 1
    )
    return out


def summarize_score_deciles(df: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    ranked = assign_score_deciles(df, n_deciles=n_deciles)
    total_cases = int(ranked["y_true"].sum())
    overall_prevalence = float(ranked["y_true"].mean())

    rows = []
    for decile, g in ranked.groupby("score_decile", sort=True):
        n_students = int(len(g))
        n_dsh = int(g["y_true"].sum())
        rows.append({
            "score_decile": int(decile),
            "n_students": n_students,
            "n_dsh": n_dsh,
            "observed_dsh_prevalence": n_dsh / n_students,
            "mean_score": float(g["y_prob"].mean()),
            "share_of_all_dsh_cases": n_dsh / total_cases if total_cases else np.nan,
            "overall_prevalence": overall_prevalence,
        })
    return pd.DataFrame(rows)


def plot_score_decile_prevalence(summary: pd.DataFrame, out_base: Path) -> None:
    s = summary.sort_values("score_decile").copy()
    x = np.arange(1, len(s) + 1)
    y = 100.0 * s["observed_dsh_prevalence"].to_numpy(float)
    overall = 100.0 * float(s["overall_prevalence"].iloc[0])

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(x, y, edgecolor="0.25", linewidth=0.8)
    ax.axhline(overall, linestyle="--", linewidth=1.2, label=f"Overall prevalence ({overall:.1f}%)")
    ax.set_xlabel("Predicted-risk score decile (lowest to highest)")
    ax.set_ylabel("Observed DSH prevalence (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=500, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _bootstrap_capacity(y, p, capacity, n_boot, seed):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    wanted = ("recall_case_capture", "precision")
    values = {k: [] for k in wanted}

    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        m = topk_metrics(y[idx], p[idx], capacity)
        for k in wanted:
            values[k].append(m[k])

    out = {}
    for k, vals in values.items():
        a = np.asarray(vals, dtype=float)
        out[f"{k}_ci_low"] = float(np.quantile(a, 0.025))
        out[f"{k}_ci_high"] = float(np.quantile(a, 0.975))
    return out


def summarize_fixed_capacities(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mi, model_name in enumerate(config.PRIORITIZATION_MODELS):
        g = preds.loc[preds["model_name"] == model_name].sort_values("sample_index")
        if g.empty:
            raise ValueError(f"Missing averaged OOF predictions for prioritization model {model_name}")
        y = g["y_true"].to_numpy(dtype=int)
        p = g["y_prob"].to_numpy(dtype=float)

        for ci, capacity in enumerate(config.PRIORITIZATION_CAPACITIES):
            m = topk_metrics(y, p, capacity)
            boot = _bootstrap_capacity(
                y,
                p,
                capacity,
                config.BOOTSTRAP_REPS,
                config.BOOTSTRAP_SEED + 100 * mi + ci,
            )
            rows.append({
                "model_name": model_name,
                "capacity": float(capacity),
                "n_selected": int(m["n_flagged"]),
                "selected_fraction": float(m["flagged_fraction"]),
                "case_capture": float(m["recall_case_capture"]),
                "case_capture_ci_low": boot["recall_case_capture_ci_low"],
                "case_capture_ci_high": boot["recall_case_capture_ci_high"],
                "selected_dsh_prevalence": float(m["precision"]),
                "selected_dsh_prevalence_ci_low": boot["precision_ci_low"],
                "selected_dsh_prevalence_ci_high": boot["precision_ci_high"],
                "lift": float(m["lift"]),
                "non_dsh_per_dsh": float(m["fp_per_tp"]),
                "tp": int(m["tp"]),
                "fp": int(m["fp"]),
                "fn": int(m["fn"]),
                "tn": int(m["tn"]),
            })
    return pd.DataFrame(rows)


def _pct_ci(est, lo, hi, digits=1):
    return f"{100*est:.{digits}f}% ({100*lo:.{digits}f}–{100*hi:.{digits}f})"


def manuscript_capacity_table(fixed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in fixed.iterrows():
        rows.append({
            "Model": config.MODEL_LABELS.get(r["model_name"], r["model_name"]),
            "Capacity": f"{100*r['capacity']:.0f}%",
            "DSH cases captured": _pct_ci(
                r["case_capture"], r["case_capture_ci_low"], r["case_capture_ci_high"]
            ),
            "DSH prevalence": _pct_ci(
                r["selected_dsh_prevalence"],
                r["selected_dsh_prevalence_ci_low"],
                r["selected_dsh_prevalence_ci_high"],
            ),
            "Lift": f"{r['lift']:.2f}",
            "Non-DSH per DSH case": f"{r['non_dsh_per_dsh']:.2f}",
        })
    return pd.DataFrame(rows)


def run():
    outdir = ensure_dir(config.OUTPUT_ROOT / "risk_stratification")
    preds = _load_averaged_oof()

    primary = preds.loc[
        preds["model_name"] == config.RISK_STRATIFICATION_MODEL
    ].sort_values("sample_index")
    if primary.empty:
        raise ValueError(
            f"RISK_STRATIFICATION_MODEL={config.RISK_STRATIFICATION_MODEL!r} "
            "not found in averaged OOF predictions"
        )

    deciles = summarize_score_deciles(primary, config.RISK_SCORE_DECILES)
    deciles.to_csv(outdir / "xgboost_score_decile_prevalence.csv", index=False)
    plot_score_decile_prevalence(
        deciles, outdir / "xgboost_score_decile_prevalence"
    )

    fixed = summarize_fixed_capacities(preds)
    fixed.to_csv(outdir / "fixed_capacity_4models.csv", index=False)
    manuscript_capacity_table(fixed).to_csv(
        outdir / "manuscript_fixed_capacity_table.csv", index=False
    )

    print("\nXGBoost score-decile prevalence")
    print(deciles.to_string(index=False))
    print("\nFixed-capacity prioritization")
    print(fixed.to_string(index=False))
    return deciles, fixed


if __name__ == "__main__":
    run()
