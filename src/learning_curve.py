"""
Learning curve za finalni model - train vs cross-validation F1-score u
zavisnosti od veličine trening skupa.

Svrha: vizuelno pokazati da li model underfituje (i train i CV skor niski),
overfituje (train skor visok, CV skor nizak i razmak između njih veliki) ili
je dobro balansiran (oba skora se približavaju sa više podataka).
"""

from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve, StratifiedGroupKFold

from preprocessing import load_data, prepare_features_and_target


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "selected_features.joblib"

RESULTS_DIR = BASE_DIR / "results" / "final_model"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "learning_curve.png"

TRAIN_SIZES = np.linspace(0.2, 1.0, 8)


def main():
    df = load_data()
    X, y, groups = prepare_features_and_target(df)

    model = joblib.load(MODEL_PATH)
    selected_features = joblib.load(FEATURES_PATH)

    # StratifiedGroupKFold - isti razlog kao u train.py/final_model.py: snimci
    # iste osobe ne smeju da završe u različitim fold-ovima.
    cv = StratifiedGroupKFold(n_splits=5)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model,
        X[selected_features],
        y,
        groups=groups,
        cv=cv,
        train_sizes=TRAIN_SIZES,
        scoring="f1",
        n_jobs=-1,
    )

    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)

    print("Veličina trening podskupa | Train F1 | CV F1")
    for size, tr, te in zip(train_sizes_abs, train_mean, test_mean):
        print(f"{size:>24} | {tr:.4f}   | {te:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes_abs, train_mean, marker="o", label="Train F1-score")
    plt.fill_between(
        train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15
    )
    plt.plot(train_sizes_abs, test_mean, marker="o", label="CV F1-score")
    plt.fill_between(
        train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.15
    )
    plt.xlabel("Broj snimaka u trening podskupu")
    plt.ylabel("F1-score")
    plt.title("Learning curve - finalni model (Random Forest)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    plt.close()

    gap = train_mean[-1] - test_mean[-1]
    print(f"\nRazmak (train - CV) F1 na punom trening skupu: {gap:.4f}")
    print(f"Grafik sačuvan u: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
