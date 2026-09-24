from pathlib import Path
import numpy as np
import pandas as pd

import config
from dsh_utils import load_primary_dataset, ensure_dir, save_json


def _fmt_n_pct(n, denom):
    pct = 100.0 * n / denom if denom else np.nan
    return f"{int(n)} ({pct:.1f}%)"


def _continuous_row(name, s, y, include_overall=False):
    rows = []
    row = {"Characteristic": name, "Level": ""}
    for lab, col in [(0, "No DSH"), (1, "DSH")]:
        vals = pd.to_numeric(s[y == lab], errors="coerce").dropna()
        row[col] = f"{vals.mean():.1f} ({vals.std(ddof=1):.1f})" if len(vals) else "NA"
    if include_overall:
        vals = pd.to_numeric(s, errors="coerce").dropna()
        row["Overall"] = f"{vals.mean():.1f} ({vals.std(ddof=1):.1f})" if len(vals) else "NA"
    row["DSH prevalence within subgroup"] = ""
    row["Valid N"] = int(s.notna().sum())
    rows.append(row)
    return rows


def _categorical_rows(name, s, y, include_overall=False):
    s = s.copy()
    s = s.where(s.notna(), "Missing/Unknown").astype(str)

    rows = []
    levels = list(pd.unique(s))
    # Natural sort numeric-like levels, otherwise alphabetical.
    try:
        levels = sorted(levels, key=lambda z: float(z) if z != "Missing/Unknown" else 1e18)
    except Exception:
        levels = sorted(levels)

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())

    for level in levels:
        mask = s == level
        c0 = int(((y == 0) & mask).sum())
        c1 = int(((y == 1) & mask).sum())
        total = int(mask.sum())
        row = {
            "Characteristic": name,
            "Level": str(level),
            "No DSH": _fmt_n_pct(c0, n0),
            "DSH": _fmt_n_pct(c1, n1),
            "DSH prevalence within subgroup": (
                f"{100*c1/total:.1f}%" if total else "NA"
            ),
            "Valid N": total,
        }
        if include_overall:
            row["Overall"] = _fmt_n_pct(total, len(s))
        rows.append(row)
    return rows


def _onehot_to_category(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return None, []

    block = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    arr = block.to_numpy()
    labels = []

    stripped = [c[len(prefix):] for c in cols]
    for row in arr:
        idx = np.flatnonzero(row > 0.5)
        if len(idx) == 1:
            labels.append(stripped[idx[0]])
        elif len(idx) == 0:
            labels.append("Missing/Unknown")
        else:
            labels.append("Multiple/Other")
    return pd.Series(labels, index=df.index), cols


def run():
    outdir = ensure_dir(config.OUTPUT_ROOT / "demographics")
    df_target, X, y, meta = load_primary_dataset()

    rows = []
    skipped = []

    for display, col in config.DEMOGRAPHIC_CONTINUOUS.items():
        if col in X.columns:
            rows.extend(_continuous_row(display, X[col], y, config.INCLUDE_OVERALL_COLUMN))
        else:
            skipped.append((display, col))

    for display, col in config.DEMOGRAPHIC_CATEGORICAL.items():
        if col in X.columns:
            rows.extend(_categorical_rows(display, X[col], y, config.INCLUDE_OVERALL_COLUMN))
        else:
            skipped.append((display, col))

    for display, prefix in config.DEMOGRAPHIC_ONEHOT.items():
        s, used = _onehot_to_category(X, prefix)
        if s is not None:
            rows.extend(_categorical_rows(display, s, y, config.INCLUDE_OVERALL_COLUMN))
        else:
            skipped.append((display, prefix + "*"))

    for display, col in config.DEMOGRAPHIC_BINARY.items():
        if col not in X.columns:
            skipped.append((display, col))
            continue
        s = pd.to_numeric(X[col], errors="coerce")
        # Endorsed/Yes row only, common for overlapping race/ethnicity indicators.
        mask = s > 0
        c0 = int(((y == 0) & mask).sum())
        c1 = int(((y == 1) & mask).sum())
        total = int(mask.sum())
        row = {
            "Characteristic": display,
            "Level": "Yes",
            "No DSH": _fmt_n_pct(c0, int((y == 0).sum())),
            "DSH": _fmt_n_pct(c1, int((y == 1).sum())),
            "DSH prevalence within subgroup": (
                f"{100*c1/total:.1f}%" if total else "NA"
            ),
            "Valid N": total,
        }
        if config.INCLUDE_OVERALL_COLUMN:
            row["Overall"] = _fmt_n_pct(total, len(X))
        rows.append(row)

    table = pd.DataFrame(rows)
    ordered = ["Characteristic", "Level"]
    if config.INCLUDE_OVERALL_COLUMN:
        ordered += ["Overall"]
    ordered += ["No DSH", "DSH", "DSH prevalence within subgroup", "Valid N"]
    table = table[[c for c in ordered if c in table.columns]]

    table.to_csv(outdir / "demographic_table.csv", index=False)
    table.to_latex(
        outdir / "demographic_table.tex",
        index=False,
        escape=True,
        longtable=False,
    )

    # Demographic missingness, separate from the main table.
    demo_cols = set(config.DEMOGRAPHIC_CONTINUOUS.values()) | set(config.DEMOGRAPHIC_CATEGORICAL.values())
    for prefix in config.DEMOGRAPHIC_ONEHOT.values():
        demo_cols.update([c for c in X.columns if c.startswith(prefix)])
    demo_cols.update(config.DEMOGRAPHIC_BINARY.values())
    demo_cols = [c for c in demo_cols if c in X.columns]

    miss = pd.DataFrame({
        "feature": demo_cols,
        "missing_n": [int(X[c].isna().sum()) for c in demo_cols],
        "missing_pct": [float(100 * X[c].isna().mean()) for c in demo_cols],
    }).sort_values("missing_pct", ascending=False)
    miss.to_csv(outdir / "demographic_missingness.csv", index=False)

    save_json(
        {"sample": meta, "skipped_demographic_specs": skipped},
        outdir / "demographic_manifest.json",
    )

    print(table.to_string(index=False))
    if skipped:
        print("\nSkipped demographic specs (not found):", skipped)
    return table


if __name__ == "__main__":
    run()
