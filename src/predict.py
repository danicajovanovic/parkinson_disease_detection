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


def predict_parkinson(input_data):
    model, selected_features, decision_threshold = load_prediction_artifacts()

    input_df = pd.DataFrame([input_data])
    input_df = input_df[selected_features]

    probability = model.predict_proba(input_df)[0][1]
    prediction = 1 if probability >= decision_threshold else 0

    return prediction, probability, decision_threshold


def main():
    from preprocessing import load_data, prepare_features_and_target, split_data

    df = load_data()

    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model, selected_features, decision_threshold = load_prediction_artifacts()

    X_test = X_test[selected_features]

    print("\n===== HEALTHY EXAMPLE =====")

    for i in range(len(y_test)):
        if y_test.iloc[i] == 0:
            sample = X_test.iloc[i:i + 1]

            probability = model.predict_proba(sample)[0][1]
            prediction = 1 if probability >= decision_threshold else 0

            print(f"Decision threshold: {decision_threshold:.2f}")
            print(f"Parkinson probability: {probability:.2%}")

            if prediction == 1:
                print("Parkinson disease risk detected")
            else:
                print("Healthy voice pattern")

            break

    print("\n===== PARKINSON EXAMPLE =====")

    for i in range(len(y_test)):
        if y_test.iloc[i] == 1:
            sample = X_test.iloc[i:i + 1]

            probability = model.predict_proba(sample)[0][1]
            prediction = 1 if probability >= decision_threshold else 0

            print(f"Decision threshold: {decision_threshold:.2f}")
            print(f"Parkinson probability: {probability:.2%}")

            if prediction == 1:
                print("Parkinson disease risk detected")
            else:
                print("Healthy voice pattern")

            break


if __name__ == "__main__":
    main()