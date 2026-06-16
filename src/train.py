from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocessing import load_data, prepare_features_and_target, split_data
from metrics import SCORING, scores_to_row


BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results" / "model_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = RESULTS_DIR / "model_comparison.csv"


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    return models


def evaluate_models(X_train, y_train, models):
    results = []

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring=SCORING
        )

        results.append({
            "Model": model_name,
            **scores_to_row(scores),
        })

    return pd.DataFrame(results)


def main():
    df = load_data()

    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = get_models()
    results_df = evaluate_models(X_train, y_train, models)

    results_df = results_df.sort_values(
        by=["Recall", "F1-score", "ROC-AUC"],
        ascending=False
    )

    print("\nPoređenje modela pomoću 5-fold Cross Validation:")
    print(results_df)

    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nRezultati su sačuvani u: {RESULTS_PATH}")


if __name__ == "__main__":
    main()