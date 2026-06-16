"""
Entry point for the Parkinson's Disease Detection project.

The full pipeline is split into individual scripts under src/, meant to be
run in order the first time the project is set up:

    1. python src/data_analysis.py      - exploratory data analysis
    2. python src/train.py              - compare baseline models (5-fold CV)
    3. python src/feature_selection.py  - rank features, pick the best subset
    4. python src/final_model.py        - tune hyperparameters and save the
                                           final model, threshold and features
    5. python src/predict.py            - run example predictions in the console
    6. streamlit run app/ui.py          - launch the web interface

Running this script directly checks whether a trained model is already
available and, if so, runs the example predictions from src/predict.py.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"


def main():
    if not MODEL_PATH.exists():
        print("No trained model found yet.")
        print("Run the training pipeline first:")
        print("  1. python src/feature_selection.py")
        print("  2. python src/final_model.py")
        return

    from predict import main as run_prediction_examples

    run_prediction_examples()


if __name__ == "__main__":
    main()
