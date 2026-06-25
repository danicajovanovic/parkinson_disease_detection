import random
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "selected_features.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "decision_threshold.joblib"


def load_prediction_artifacts():
    model = joblib.load(MODEL_PATH)
    selected_features = joblib.load(FEATURES_PATH)
    decision_threshold = joblib.load(THRESHOLD_PATH)

    return model, selected_features, decision_threshold


def predict_from_features(model, selected_features, decision_threshold, input_df):
    """
    Predikcija na osnovu DataFrame-a koji već sadrži (ili sadrži bar) kolone
    iz selected_features. Ovo je jedino mesto gde se računa probability i
    finalna odluka, da se ta logika ne bi ponavljala na više mesta.
    """
    input_df = input_df[selected_features]

    probability = model.predict_proba(input_df)[0][1]
    prediction = 1 if probability >= decision_threshold else 0

    return prediction, probability


def predict_parkinson(input_data):
    """
    Pomoćna funkcija: učitava artefakte i predviđa na osnovu dict-a sa
    sirovim vrednostima atributa (npr. iz UI forme).
    """
    model, selected_features, decision_threshold = load_prediction_artifacts()

    input_df = pd.DataFrame([input_data])
    prediction, probability = predict_from_features(
        model, selected_features, decision_threshold, input_df
    )

    return prediction, probability, decision_threshold


def _print_example(label, model, selected_features, decision_threshold, sample):
    prediction, probability = predict_from_features(
        model, selected_features, decision_threshold, sample
    )

    print(f"\n===== {label} =====")
    print(f"Decision threshold: {decision_threshold:.2f}")
    print(f"Parkinson probability: {probability:.2%}")

    if prediction == 1:
        print("Parkinson disease risk detected")
    else:
        print("Healthy voice pattern")


def main():
    from preprocessing import load_data, prepare_features_and_target, split_data

    df = load_data()

    X, y, groups = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_data(
        X, y, groups
    )

    model, selected_features, decision_threshold = load_prediction_artifacts()

    for label, status_value in [("HEALTHY EXAMPLE", 0), ("PARKINSON EXAMPLE", 1)]:
        candidates = [i for i in range(len(y_test)) if y_test.iloc[i] == status_value]
        random.shuffle(candidates)

        # Bira se primer koji model tačno predviđa, da demo ne bude zbunjujuć
        # u retkim slučajevima kada model pogreši (nije 100% tačan - vidi
        # results/evaluation/classification_report.txt).
        # Redosled je promešan iznad, pa se pri svakom pokretanju prikazuje
        # drugi primer.
        chosen = candidates[0]
        for i in candidates:
            sample = X_test.iloc[i:i + 1]
            prediction, _ = predict_from_features(
                model, selected_features, decision_threshold, sample
            )
            if prediction == status_value:
                chosen = i
                break

        sample = X_test.iloc[chosen:chosen + 1]
        _print_example(label, model, selected_features, decision_threshold, sample)


if __name__ == "__main__":
    main()
