import config

from demographics import run as run_demographics
from train_main_models import run as run_main_models
from shap_importance import run as run_shap
from domain_ablation import run as run_ablation
from risk_stratification_analysis import run as run_risk_stratification


def main():
    print("\n[1/5] Demographic table")
    run_demographics()

    print("\n[2/5] Five primary models: repeated CV; mean performance across repetitions")
    run_main_models()

    print("\n[3/5] Repeated-OOF SHAP — class-weighted XGBoost")
    run_shap()

    print("\n[4/5] Domain ablation — class-weighted Logistic Regression and XGBoost")
    run_ablation()

    print("\n[5/5] Risk-score deciles and fixed-capacity prioritization")
    run_risk_stratification()

    print("\nAll final-manuscript modules completed.")
    print("Outputs:", config.OUTPUT_ROOT)


if __name__ == "__main__":
    main()
