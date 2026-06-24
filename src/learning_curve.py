"""
Learning curve za finalni model - train vs cross-validation F1-score u
zavisnosti od veličine trening skupa.

Svrha: pokazati da li model underfituje (i train i CV skor niski), overfituje
(train skor visok, CV skor nizak i razmak između njih veliki) ili je dobro
balansiran (oba skora se približavaju sa više podataka).

Rezultat je tabela (CSV), ne grafik: sa samo 32 osobe, svaka tačka je CV
procena na grupama od ~5-6 osoba po fold-u, pa je linija na grafiku
šumovita (par konkretnih osoba pomeri CV F1 gore-dole nekoliko procentnih
poena) - tabela sa std. devijacijom po veličini je iskreniji prikaz te
nesigurnosti nego glatka linija koja sugeriše čistiji trend nego što ga
ovaj dataset stvarno može da pokaže.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import learning_curve, StratifiedGroupKFold

from preprocessing import load_data, prepare_features_and_target


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "selected_features.joblib"

RESULTS_DIR = BASE_DIR / "results" / "final_model"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "learning_curve.csv"

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
    gap = train_mean - test_mean

    table = pd.DataFrame({
        "Veličina trening skupa": train_sizes_abs,
        "Train F1 (prosek)": train_mean,
        "Train F1 (std)": train_std,
        "CV F1 (prosek)": test_mean,
        "CV F1 (std)": test_std,
        "Razmak (Train - CV)": gap,
    })

    print(table.to_string(index=False))

    table.to_csv(OUTPUT_PATH, index=False)

    print(f"\nRazmak (train - CV) F1 na punom trening skupu: {gap[-1]:.4f}")
    print(f"Tabela sačuvana u: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
