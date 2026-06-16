from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from preprocessing import load_data, prepare_features_and_target, split_data


BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
FEATURES_PATH = MODELS_DIR / "selected_features.joblib"

SELECTED_FEATURES = [
    "PPE",
    "spread1",
    "MDVP:Fo(Hz)",
    "NHR",
    "Jitter:DDP",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "spread2",
    "Shimmer:APQ5",
]


def build_model():
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42))
    ])

    return model


def main():
    df = load_data()

    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_selected = X_train[SELECTED_FEATURES]
    X_test_selected = X_test[SELECTED_FEATURES]

    model = build_model()

    model.fit(X_train_selected, y_train)

    y_pred = model.predict(X_test_selected)
    y_prob = model.predict_proba(X_test_selected)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    metrics_df = pd.DataFrame([metrics])

    print("\nFinal model: Random Forest")
    print("\nSelected features:")
    for feature in SELECTED_FEATURES:
        print(f"- {feature}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nFinal metrics:")
    print(metrics_df)

    metrics_df.to_csv(RESULTS_DIR / "final_model_metrics.csv", index=False)

    with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write("Final model: Random Forest\n")
        file.write("Selected features:\n")
        for feature in SELECTED_FEATURES:
            file.write(f"- {feature}\n")
        file.write("\nClassification report:\n")
        file.write(classification_report(y_test, y_pred))

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test_selected,
        y_test
    )
    plt.title("Confusion Matrix - Random Forest")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png")
    plt.close()

    RocCurveDisplay.from_estimator(
        model,
        X_test_selected,
        y_test
    )
    plt.title("ROC Curve - Random Forest")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve.png")
    plt.close()

    joblib.dump(model, MODEL_PATH)
    joblib.dump(SELECTED_FEATURES, FEATURES_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Selected features saved to: {FEATURES_PATH}")


if __name__ == "__main__":
    main()