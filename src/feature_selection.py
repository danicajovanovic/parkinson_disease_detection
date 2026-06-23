from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedGroupKFold

from preprocessing import load_data, prepare_features_and_target, split_data
from metrics import SCORING, scores_to_row


BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results" / "feature_selection"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_IMPORTANCE_CSV = RESULTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_PNG = RESULTS_DIR / "feature_importance.png"
FEATURE_SUBSET_CSV = RESULTS_DIR / "feature_subset_comparison.csv"
FEATURE_SUBSET_PNG = RESULTS_DIR / "feature_subset_performance.png"


def build_random_forest():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42))
    ])


def calculate_feature_importance(X_train, y_train, X_columns):
    model = build_random_forest()
    model.fit(X_train, y_train)

    rf = model.named_steps["model"]

    importance = pd.DataFrame({
        "Feature": X_columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return importance


def evaluate_feature_subsets(X_train, y_train, groups_train, ranked_features):
    results = []
    cv = StratifiedGroupKFold(n_splits=5)

    for number_of_features in range(1, len(ranked_features) + 1):
        selected_features = ranked_features[:number_of_features]

        model = build_random_forest()

        scores = cross_validate(
            model,
            X_train[selected_features],
            y_train,
            cv=cv,
            groups=groups_train,
            scoring=SCORING
        )

        results.append({
            "Number of features": number_of_features,
            "Selected features": ", ".join(selected_features),
            **scores_to_row(scores),
        })

    return pd.DataFrame(results)


def plot_feature_importance(importance):
    plt.figure(figsize=(10, 6))
    plt.barh(importance["Feature"], importance["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance - Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PNG)
    plt.close()


def plot_feature_subset_results(results):
    plt.figure(figsize=(8, 5))

    plt.plot(
        results["Number of features"],
        results["F1-score"],
        marker="o",
        label="F1-score"
    )

    plt.plot(
        results["Number of features"],
        results["Recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        results["Number of features"],
        results["ROC-AUC"],
        marker="o",
        label="ROC-AUC"
    )

    plt.title("Model performance by number of selected features")
    plt.xlabel("Number of selected features")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FEATURE_SUBSET_PNG)
    plt.close()


def main():
    df = load_data()

    X, y, groups = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_data(
        X, y, groups
    )

    importance = calculate_feature_importance(X_train, y_train, X.columns)
    importance.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    print("\nFeature importance:")
    print(importance)

    plot_feature_importance(importance)

    ranked_features = importance["Feature"].tolist()

    subset_results = evaluate_feature_subsets(
        X_train,
        y_train,
        groups_train,
        ranked_features
    )

    subset_results.to_csv(FEATURE_SUBSET_CSV, index=False)

    print("\nFeature subset comparison:")
    print(subset_results)

    plot_feature_subset_results(subset_results)

    best_subset = subset_results.sort_values(
        by=["F1-score", "Recall", "ROC-AUC"],
        ascending=False
    ).iloc[0]

    print("\nBest feature subset based on F1-score, Recall and ROC-AUC:")
    print(best_subset)


if __name__ == "__main__":
    main()