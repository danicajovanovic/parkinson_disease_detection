"""
Provera koliko je jedan konkretan 80/20 train/test split (po osobi) "sreća".

Dataset ima samo 32 osobe, pa jedna konkretna podela na trening i test
subjekte može slučajno biti lakša ili teža za model. Ovaj skript ponavlja
ceo postupak (split -> trening finalnog modela -> evaluacija na test skupu)
za više različitih random_state vrednosti i prikazuje raspon metrika, kao
dopunu jednoj tačkastoj proceni iz results/final_model/.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from preprocessing import load_data, prepare_features_and_target, split_data
from final_model import SELECTED_FEATURES


BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results" / "final_model"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = RESULTS_DIR / "split_robustness.csv"
RESULTS_PNG = RESULTS_DIR / "split_robustness.png"

SEEDS = [0, 1, 7, 42, 100, 123, 2024]


def build_model():
    # Isti tip modela kao final_model.py (RandomForest, balansirane klase),
    # ali sa fiksnim hiperparametrima - puni GridSearchCV po svakom seed-u bi
    # bio nepotrebno skup samo za proveru robusnosti split-a.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )),
    ])


def evaluate_split(X, y, groups, seed):
    X_train, X_test, y_train, y_test, groups_train, _ = split_data(
        X, y, groups, random_state=seed
    )

    model = build_model()
    model.fit(X_train[SELECTED_FEATURES], y_train)

    y_prob = model.predict_proba(X_test[SELECTED_FEATURES])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "Seed": seed,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-macro": f1_score(y_test, y_pred, average="macro"),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }


def plot_split_robustness(results_df, output_path):
    plt.figure(figsize=(8, 5))
    metrics = ["Accuracy", "F1-macro", "ROC-AUC"]
    plt.boxplot(
        [results_df[metric] for metric in metrics],
        tick_labels=metrics,
    )
    for index, metric in enumerate(metrics, start=1):
        plt.scatter(
            [index] * len(results_df),
            results_df[metric],
            alpha=0.6,
            color="black",
            zorder=3,
        )
    plt.ylabel("Vrednost metrike")
    plt.title(f"Robusnost rezultata na izbor train/test split-a ({len(results_df)} seed-ova)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    df = load_data()
    X, y, groups = prepare_features_and_target(df)

    results = [evaluate_split(X, y, groups, seed) for seed in SEEDS]
    results_df = pd.DataFrame(results)

    print("Metrike po seed-u (random_state korišćen za train/test split):")
    print(results_df)

    print("\nProsek i standardna devijacija:")
    print(results_df[["Accuracy", "F1-macro", "ROC-AUC"]].agg(["mean", "std"]))

    results_df.to_csv(RESULTS_CSV, index=False)
    plot_split_robustness(results_df, RESULTS_PNG)

    print(f"\nRezultati sačuvani u: {RESULTS_CSV}")
    print(f"Grafik sačuvan u: {RESULTS_PNG}")


if __name__ == "__main__":
    main()
