"""Small environment check for the DSH pipeline."""
from __future__ import annotations


def main():
    import numpy
    import pandas
    import sklearn
    import imblearn
    import xgboost
    import shap

    print("numpy:", numpy.__version__)
    print("pandas:", pandas.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("imbalanced-learn:", imblearn.__version__)
    print("xgboost:", xgboost.__version__)
    print("shap:", shap.__version__)

    try:
        import imbens
        from imbens.ensemble import BalanceCascadeClassifier
        print("imbalanced-ensemble / imbens:", getattr(imbens, "__version__", "unknown"))
        print("BalanceCascadeClassifier:", BalanceCascadeClassifier)
    except Exception as exc:
        print("\nBalanceCascade import FAILED.")
        print("Install with: python -m pip install -r requirements.txt")
        print("Original exception:", repr(exc))
        raise


if __name__ == "__main__":
    main()
