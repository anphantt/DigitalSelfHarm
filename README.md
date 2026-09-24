# Digital Self-Harm (DSH) Analysis Pipeline

Python code for the machine-learning analyses used in the submitted DSH manuscript.

## Setup

```bash
git clone <repository-url>
cd <repository-name>
python -m pip install -r requirements.txt
```

Update `DATA_PATH` and `OUTPUT_ROOT` in `config.py`, then run:

```bash
python run_all.py
```

For Google Colab, use `run_all.ipynb`.

## Analysis

The pipeline uses three repetitions of stratified 10-fold cross-validation:

```python
CV_SEEDS = [42, 100, 2222]
```

Main models:

- Logistic Regression (`class_weight="balanced"`)
- Random Forest (`class_weight="balanced"`)
- XGBoost (fold-specific `scale_pos_weight`)
- EasyEnsemble-XGBoost
- BalanceCascade-XGBoost

Performance metrics are computed separately for each repetition and then averaged across the three repetitions. Averaged OOF scores are used only for risk-score deciles and fixed-capacity prioritization.

Additional analyses:

- `shap_importance.py`: SHAP analysis using class-weighted XGBoost
- `domain_ablation.py`: domain-only and leave-domain-out analyses using Logistic Regression and XGBoost
- `risk_stratification_analysis.py`: XGBoost score deciles and 5%, 10%, 20% prioritization analyses

Calibration is not included in the final pipeline.

## Outputs

Results are written under `OUTPUT_ROOT`, including repeated-CV performance, OOF scores, SHAP results, domain-ablation results, and risk-stratification tables/figures.

## Data

The processed study dataset is not included. Do not commit student-level data or row-level prediction/SHAP outputs to a public repository.