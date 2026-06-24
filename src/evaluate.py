"""
Evaluation utilities for the Parkinson's disease model.

This module is the single place where test-set metrics, the classification
report, the confusion matrix and the ROC curve are computed, so that logic
is not duplicated between the training script (final_model.py) and a
standalone evaluation run.

Running this file directly evaluates whatever model is currently saved in
models/ against the test set, without retraining anything (useful to
re-check a saved model, e.g. after pulling new artifacts). Results are
written to results/evaluation/, separate from the training-time report in
results/final_model/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from preprocessing import load_data, prepare_features_and_target, split_data
from predict import load_prediction_artifacts


DIVIDER = "=" * 80
SUBDIVIDER = "-" * 80


def write_title(file, title):
    file.write(f"{DIVIDER}\n{title}\n{DIVIDER}\n")


def write_section(file, title):
    file.write(f"\n{SUBDIVIDER}\n{title}\n{SUBDIVIDER}\n")


def write_feature_list(file, features):
    for index, feature in enumerate(features, start=1):
        file.write(f"  {index:>2}. {feature}\n")


BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = RESULTS_DIR / "evaluation_metrics.csv"
REPORT_PATH = RESULTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = RESULTS_DIR / "roc_curve.png"
PR_CURVE_PATH = RESULTS_DIR / "precision_recall_curve.png"


def evaluate_model(model, selected_features, decision_threshold, X_test, y_test):
    """
    Evaluates a fitted model on the test set using a fixed decision
    threshold. Returns the metrics dict, the textual classification report,
    and the predictions/probabilities (needed for the confusion matrix and
    ROC curve plots).
    """
    X_test_selected = X_test[selected_features]

    y_prob = model.predict_proba(X_test_selected)[:, 1]
    y_pred = (y_prob >= decision_threshold).astype(int)

    metrics = {
        "Threshold": decision_threshold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Average Precision": average_precision_score(y_test, y_prob),
    }

    report = classification_report(y_test, y_pred)

    return metrics, report, y_pred, y_prob


def save_confusion_matrix(y_test, y_pred, output_path, model_name="Random Forest"):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_roc_curve(y_test, y_prob, output_path, model_name="Random Forest"):
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f"ROC Curve - {model_name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_pr_curve(y_test, y_prob, output_path, model_name="Random Forest"):
    """
    Precision-Recall curve, informativnija od ROC krive kod neravnomernih
    klasa (147 Parkinson vs 48 zdravih u ovom dataset-u).
    """
    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    df = load_data()

    X, y, groups = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_data(
        X, y, groups
    )

    model, selected_features, decision_threshold = load_prediction_artifacts()

    metrics, report, y_pred, y_prob = evaluate_model(
        model, selected_features, decision_threshold, X_test, y_test
    )

    print("\nEvaluating saved model on the test set:")
    print("Selected features:", selected_features)
    print(f"Decision threshold: {decision_threshold:.4f}")

    print("\nClassification report:")
    print(report)

    metrics_df = pd.DataFrame([metrics])
    print("\nMetrics:")
    print(metrics_df)

    metrics_df.to_csv(METRICS_PATH, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        write_title(file, "EVALUATION OF THE SAVED MODEL ON THE TEST SET")

        write_section(file, f"SELECTED FEATURES ({len(selected_features)})")
        write_feature_list(file, selected_features)

        write_section(file, "DECISION THRESHOLD")
        file.write(f"  {decision_threshold:.4f}\n")

        write_section(file, "CLASSIFICATION REPORT")
        file.write(f"\n{report}")

    save_confusion_matrix(y_test, y_pred, CONFUSION_MATRIX_PATH)
    save_roc_curve(y_test, y_prob, ROC_CURVE_PATH)
    save_pr_curve(y_test, y_prob, PR_CURVE_PATH)

    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
