from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedGroupKFold

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocessing import load_data, prepare_features_and_target, split_data
from metrics import SCORING, scores_to_row


BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results" / "model_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = RESULTS_DIR / "model_comparison.csv"
COMPARISON_BAR_PATH = RESULTS_DIR / "model_comparison_bar.png"
CV_BOXPLOT_PATH = RESULTS_DIR / "cv_scores_boxplot.png"


def get_models():
    models = {
        # Baseline - uvek predviđa većinsku klasu (Parkinson). Svaki ozbiljan
        # model mora da bude bolji od ovoga, inače ne donosi nikakvu vrednost.
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting (XGBoost)": XGBClassifier(
            random_state=42, eval_metric="logloss"
        ),
    }

    return models


def evaluate_models(X_train, y_train, groups_train, models):
    results = []
    raw_fold_scores = {}

    # StratifiedGroupKFold - snimci iste osobe ostaju u istom fold-u (groups),
    # dok se raspodela klasa po fold-ovima i dalje stratifikuje koliko je to
    # moguce uz grupno ogranicenje.
    cv = StratifiedGroupKFold(n_splits=5)

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            groups=groups_train,
            scoring=SCORING
        )

        results.append({
            "Model": model_name,
            **scores_to_row(scores),
        })
        raw_fold_scores[model_name] = scores

    return pd.DataFrame(results), raw_fold_scores


def plot_model_comparison_bar(results_df, output_path):
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    models = results_df["Model"].tolist()

    x = np.arange(len(models))
    width = 0.15

    plt.figure(figsize=(13, 6))
    for index, metric in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2) * width
        plt.bar(x + offset, results_df[metric], width=width, label=metric)

    plt.xticks(x, models, rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score (5-fold CV mean)")
    plt.title("Poređenje baznih modela")
    plt.legend(loc="lower right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_cv_score_boxplots(raw_fold_scores, output_path):
    """
    Raspon F1-score-a kroz 5 fold-ova za svaki model, ne samo prosek - pokazuje
    koliko je svaki model stabilan na ovako malom dataset-u (195 uzoraka).
    """
    model_names = list(raw_fold_scores.keys())
    fold_f1_scores = [raw_fold_scores[name]["test_f1"] for name in model_names]

    plt.figure(figsize=(11, 7))
    plt.boxplot(fold_f1_scores, tick_labels=model_names)
    plt.scatter(
        np.repeat(np.arange(1, len(model_names) + 1), 5),
        np.concatenate(fold_f1_scores),
        alpha=0.5,
        color="black",
        s=15,
        zorder=3,
    )
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("F1-score (po fold-u)")
    plt.title("Raspodela F1-score-a kroz 5 CV fold-ova")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    df = load_data()

    X, y, groups = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_data(
        X, y, groups
    )

    models = get_models()
    results_df, raw_fold_scores = evaluate_models(
        X_train, y_train, groups_train, models
    )

    results_df = results_df.sort_values(
        by=["Recall", "F1-score", "ROC-AUC"],
        ascending=False
    )

    print("\nPoređenje modela pomoću 5-fold Group Cross Validation:")
    print(results_df)

    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nRezultati su sačuvani u: {RESULTS_PATH}")

    plot_model_comparison_bar(results_df, COMPARISON_BAR_PATH)
    plot_cv_score_boxplots(raw_fold_scores, CV_BOXPLOT_PATH)
    print(f"Grafici su sačuvani u: {COMPARISON_BAR_PATH} i {CV_BOXPLOT_PATH}")


if __name__ == "__main__":
    main()