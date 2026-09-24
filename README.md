# Digital Self-Harm (DSH) final analysis pipeline

This repository contains the cleaned Python pipeline corresponding to the submitted manuscript analysis.

## Evaluation design

The modeling sample is evaluated with **three repetitions of stratified 10-fold cross-validation**:

```python
CV_SEEDS = [42, 100, 2222]
N_SPLITS = 10
MODEL_SEED = 42
```

For each repetition, every student receives exactly one out-of-fold (OOF) prediction. The ten held-out folds from that repetition are pooled to obtain one complete OOF prediction vector for the full analytic sample.

### Main performance reporting

The manuscript performance table is **not** computed from probabilities averaged across the three repetitions.

Instead:

1. compute ROC-AUC, PR-AUC, balanced accuracy, sensitivity, precision, and the confusion matrix separately for each CV repetition;
2. take the arithmetic mean of each performance metric across the three repetitions;
3. compute the confusion matrix separately in each repetition, then take the element-wise mean across the three matrices and round the cell means to integer counts for manuscript display.

The main performance table does **not** attach student-level bootstrap 95% confidence intervals to these mean repeated-CV metrics.

The averaged OOF score is still saved for each student:

```text
mean(OOF score from seed 42,
     OOF score from seed 100,
     OOF score from seed 2222)
```

but this averaged cross-fitted score is used only for downstream **risk-score deciles and fixed-capacity prioritization**, not for the main performance table.

## Primary models

The five models in the performance comparison are:

1. Logistic Regression — `class_weight="balanced"`
2. Random Forest — `class_weight="balanced"`
3. XGBoost — fold-specific `scale_pos_weight = n_negative / n_positive`
4. EasyEnsemble-XGBoost
5. BalanceCascade-XGBoost

Tree models use 300 trees and maximum depth 6. EasyEnsemble and BalanceCascade use 100 ensemble members/stages. Model randomization is held fixed at `MODEL_SEED=42`; only the CV split seed changes across repetitions.

## Interpretation analyses

### SHAP

`shap_importance.py` uses the **class-weighted XGBoost** model. SHAP values are calculated only for held-out students in each CV fold and then aggregated across the repeated OOF predictions.

### Domain ablation

`domain_ablation.py` uses the same class-weighted models as the main analysis:

- Logistic Regression
- XGBoost

Both domain-only and leave-domain-out analyses reuse the repeated stratified 10-fold design.

### Risk stratification

`risk_stratification_analysis.py` consumes the **averaged OOF scores**.

- The score-decile figure uses class-weighted XGBoost.
- Students are ranked and divided into 10 approximately equal-sized score groups.
- The plotted quantity is observed DSH prevalence within each score decile.
- The fixed-capacity table evaluates the four models used in the submitted capacity analysis: Logistic Regression, Random Forest, XGBoost, and EasyEnsemble-XGBoost at 5%, 10%, and 20% selection capacity.
- Student-level bootstrap intervals are retained for the fixed-capacity case-capture/prevalence quantities because these intervals belong to that risk-stratification analysis, not to the main performance table.

**Calibration is not part of the final pipeline.**

## BalanceCascade dependency

The PyPI package is named `imbalanced-ensemble`, while the Python import namespace is `imbens`:

```python
from imbens.ensemble import BalanceCascadeClassifier
```

Install all dependencies before importing the analysis modules:

```bash
python -m pip install -r requirements.txt
```

The pinned dependency is:

```text
imbalanced-ensemble==0.2.3
```

If a Colab runtime previously loaded incompatible versions, restart the runtime after installation. The pipeline preserves and displays the original import exception if BalanceCascade cannot be loaded.

## Main output files

### `train_main_models.py`

- `main_models/oof_predictions_long.csv` — one OOF prediction per student × model × repetition
- `main_models/per_repetition_oof_metrics.csv` — metrics computed separately for each repetition
- `main_models/per_seed_oof_metrics.csv` — compatibility alias of the previous file
- `main_models/performance_summary_across_repetitions.csv` — mean/SD/min/max across the three repetitions
- `main_models/confusion_matrix_by_repetition.csv`
- `main_models/confusion_matrix_mean_sd_across_repetitions.csv`
- `main_models/confusion_matrix_manuscript_rounded.csv`
- `main_models/manuscript_performance_table.csv` — values used for the manuscript-style performance table
- `main_models/oof_predictions_averaged.csv` — averaged cross-fitted scores for risk stratification only

### `shap_importance.py`

- `shap/oof_shap_values.npz`
- `shap/shap_feature_importance_all.csv`
- `shap/shap_domain_importance.csv`
- manuscript SHAP figures

### `domain_ablation.py`

- `domain_ablation/ablation_oof_predictions_long.csv`
- `domain_ablation/ablation_oof_predictions_averaged.csv`
- `domain_ablation/ablation_summary.csv`
- `domain_ablation/ablation_delta_vs_full_95ci.csv`
- domain-only and leave-domain-out figures

### `risk_stratification_analysis.py`

- `risk_stratification/xgboost_score_decile_prevalence.csv`
- `risk_stratification/xgboost_score_decile_prevalence.png`
- `risk_stratification/fixed_capacity_4models.csv`
- `risk_stratification/manuscript_fixed_capacity_table.csv`

## Running in Colab

Open `run_all.ipynb`, update `PIPELINE_DIR` if necessary, install `requirements.txt`, verify the dependency check, and then run the final modules.

From a shell:

```bash
python run_all.py
```

`imbalance_strategy.py` is retained only as an optional sensitivity script and is not executed by `run_all.py`.

## Data privacy

Do not commit student-level data, OOF predictions, SHAP arrays, or other row-level outputs to a public repository. The `.gitignore` excludes the usual local data/output locations, but repository contents should still be reviewed before publishing.
