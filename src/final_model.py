from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_predict,
    StratifiedGroupKFold,
)
from sklearn.metrics import roc_curve

from preprocessing import load_data, prepare_features_and_target, split_data
from evaluate import (
    evaluate_model,
    save_confusion_matrix,
    save_roc_curve,
    save_pr_curve,
    write_title,
    write_section,
    write_feature_list,
)


BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results" / "final_model"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "best_model.joblib"
FEATURES_PATH = MODELS_DIR / "selected_features.joblib"
THRESHOLD_PATH = MODELS_DIR / "decision_threshold.joblib"

METRICS_PATH = RESULTS_DIR / "final_model_metrics.csv"
HYPERPARAMETERS_PATH = RESULTS_DIR / "best_hyperparameters.csv"
REPORT_PATH = RESULTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = RESULTS_DIR / "roc_curve.png"
PR_CURVE_PATH = RESULTS_DIR / "precision_recall_curve.png"
HYPERPARAMETER_HEATMAP_PATH = RESULTS_DIR / "hyperparameter_heatmap.png"

SELECTED_FEATURES = [
    "PPE",
    "MDVP:Flo(Hz)",
    "spread1",
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "NHR",
    "RPDE",
    "Jitter:DDP",
    "MDVP:Jitter(Abs)",
    "spread2",
]


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        # class_weight="balanced" - dataset je neravnomeran (147 Parkinson vs
        # 48 zdravih), pa bez ovoga model favorizuje većinsku klasu.
        ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ])


def tune_hyperparameters(X_train, y_train, groups_train):
    pipeline = build_pipeline()

    param_grid = {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__max_depth": [None, 3, 5, 10],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    # StratifiedGroupKFold - snimci iste osobe ostaju u istom fold-u, da
    # pretraga hiperparametara ne favorizuje model koji "prepoznaje" osobu.
    cv = StratifiedGroupKFold(n_splits=5)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        # f1 balansira precision i recall - "recall" bi nagrađivao model koji
        # uvek predviđa Parkinson, što šteti precision-u zdrave klase.
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train, groups=groups_train)

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
        grid_search.cv_results_,
    )


def plot_hyperparameter_heatmap(cv_results, best_params, output_path):
    """
    Heatmap F1-score (mean_test_score) za n_estimators x max_depth, sa
    min_samples_split i min_samples_leaf fiksiranim na najbolje pronađene
    vrednosti - pokazuje da li je izabrana kombinacija stvarni optimum ili
    je na ivici ispitanog grida.
    """
    results = pd.DataFrame(cv_results)

    fixed_split = best_params["model__min_samples_split"]
    fixed_leaf = best_params["model__min_samples_leaf"]

    subset = results[
        (results["param_model__min_samples_split"] == fixed_split)
        & (results["param_model__min_samples_leaf"] == fixed_leaf)
    ].copy()

    subset["param_model__max_depth"] = subset["param_model__max_depth"].apply(
        lambda value: "None" if value is None else value
    )

    pivot = subset.pivot(
        index="param_model__max_depth",
        columns="param_model__n_estimators",
        values="mean_test_score"
    )

    plt.figure(figsize=(8, 5))
    plt.imshow(pivot.values, cmap="viridis", aspect="auto")
    plt.colorbar(label="Mean CV F1-score")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("n_estimators")
    plt.ylabel("max_depth")
    plt.title(
        f"GridSearchCV F1-score "
        f"(min_samples_split={fixed_split}, min_samples_leaf={fixed_leaf})"
    )

    for row in range(len(pivot.index)):
        for col in range(len(pivot.columns)):
            value = pivot.values[row, col]
            plt.text(col, row, f"{value:.3f}", ha="center", va="center", color="white")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def find_optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    youden_index = tpr - fpr
    best_index = np.argmax(youden_index)

    return thresholds[best_index]


def main():
    df = load_data()

    X, y, groups = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_data(
        X, y, groups
    )

    X_train_selected = X_train[SELECTED_FEATURES]

    model, best_params, best_cv_f1, cv_results = tune_hyperparameters(
        X_train_selected,
        y_train,
        groups_train
    )

    # The decision threshold must not be chosen using the test set, otherwise
    # the test labels leak into the model selection process and the final
    # metrics become optimistically biased. Instead, the threshold is chosen
    # using out-of-fold predicted probabilities on the training set only.
    # Grouped by subject, same as the hyperparameter search above.
    cv_probabilities = cross_val_predict(
        model,
        X_train_selected,
        y_train,
        groups=groups_train,
        cv=StratifiedGroupKFold(n_splits=5),
        method="predict_proba"
    )[:, 1]

    decision_threshold = find_optimal_threshold(y_train, cv_probabilities)

    eval_metrics, report, y_pred, y_prob = evaluate_model(
        model, SELECTED_FEATURES, decision_threshold, X_test, y_test
    )

    metrics = {"Best CV F1-score": best_cv_f1, **eval_metrics}
    metrics_df = pd.DataFrame([metrics])

    hyperparameters_df = pd.DataFrame([{
        "Best CV F1-score": best_cv_f1,
        **best_params
    }])

    print("\nFinal model: Random Forest")

    print("\nSelected features:")
    for feature in SELECTED_FEATURES:
        print(f"- {feature}")

    print("\nBest hyperparameters:")
    print(best_params)
    print(f"Best CV F1-score: {best_cv_f1:.4f}")

    print(f"\nOptimal decision threshold: {decision_threshold:.4f}")

    print("\nClassification report:")
    print(report)

    print("\nFinal metrics:")
    print(metrics_df)

    print("\nWrong classifications:")
    for real, prob, pred in zip(y_test, y_prob, y_pred):
        if real != pred:
            print(
                f"True class={real}, "
                f"Probability={prob:.2f}, "
                f"Prediction={pred}"
            )

    metrics_df.to_csv(METRICS_PATH, index=False)
    hyperparameters_df.to_csv(HYPERPARAMETERS_PATH, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        write_title(file, "FINAL MODEL TRAINING REPORT")
        file.write(
            "  Model: Random Forest\n"
            "  Hyperparameter tuning: GridSearchCV\n"
            "  Threshold selection: ROC curve + Youden's index "
            "(cross-validated on the training set, not the test set)\n"
        )

        write_section(file, f"SELECTED FEATURES ({len(SELECTED_FEATURES)})")
        write_feature_list(file, SELECTED_FEATURES)

        write_section(file, "BEST HYPERPARAMETERS")
        labels = [*best_params.keys(), "Best CV F1-score", "Optimal decision threshold"]
        name_width = max(len(label) for label in labels)
        for key, value in best_params.items():
            file.write(f"  {key:<{name_width}} : {value}\n")
        file.write(f"\n  {'Best CV F1-score':<{name_width}} : {best_cv_f1:.4f}\n")
        file.write(
            f"  {'Optimal decision threshold':<{name_width}} : "
            f"{decision_threshold:.4f}\n"
        )

        write_section(file, "CLASSIFICATION REPORT")
        file.write(f"\n{report}")

    save_confusion_matrix(y_test, y_pred, CONFUSION_MATRIX_PATH)
    save_roc_curve(y_test, y_prob, ROC_CURVE_PATH)
    save_pr_curve(y_test, y_prob, PR_CURVE_PATH)
    plot_hyperparameter_heatmap(cv_results, best_params, HYPERPARAMETER_HEATMAP_PATH)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(SELECTED_FEATURES, FEATURES_PATH)
    joblib.dump(decision_threshold, THRESHOLD_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Selected features saved to: {FEATURES_PATH}")
    print(f"Decision threshold saved to: {THRESHOLD_PATH}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()