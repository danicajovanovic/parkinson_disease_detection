from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "selected_features.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "decision_threshold.joblib"


def load_prediction_artifacts():
    model = joblib.load(MODEL_PATH)
    selected_features = joblib.load(FEATURES_PATH)
    decision_threshold = joblib.load(THRESHOLD_PATH)

    return model, selected_features, decision_threshold


def main():
    st.set_page_config(
        page_title="Parkinson Disease Detection",
        page_icon="🧠",
        layout="centered"
    )

    st.title("Parkinson Disease Detection")
    st.write(
        "Machine learning system for Parkinson's disease risk prediction "
        "based on voice biomarkers."
    )

    st.warning(
        "This application is for educational purposes only and does not represent "
        "a medical diagnosis."
    )

    model, selected_features, decision_threshold = load_prediction_artifacts()

    st.subheader("Input voice biomarkers")

    input_data = {}

    input_data["PPE"] = st.number_input(
        "PPE",
        value=0.284654,
        format="%.6f"
    )

    input_data["spread1"] = st.number_input(
        "spread1",
        value=-4.813031,
        format="%.6f"
    )

    input_data["MDVP:Fo(Hz)"] = st.number_input(
        "MDVP:Fo(Hz)",
        value=119.992,
        format="%.3f"
    )

    input_data["NHR"] = st.number_input(
        "NHR",
        value=0.02211,
        format="%.6f"
    )

    input_data["Jitter:DDP"] = st.number_input(
        "Jitter:DDP",
        value=0.01109,
        format="%.6f"
    )

    input_data["MDVP:Fhi(Hz)"] = st.number_input(
        "MDVP:Fhi(Hz)",
        value=157.302,
        format="%.3f"
    )

    input_data["MDVP:Flo(Hz)"] = st.number_input(
        "MDVP:Flo(Hz)",
        value=74.997,
        format="%.3f"
    )

    input_data["spread2"] = st.number_input(
        "spread2",
        value=0.266482,
        format="%.6f"
    )

    input_data["Shimmer:APQ5"] = st.number_input(
        "Shimmer:APQ5",
        value=0.03130,
        format="%.6f"
    )

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_df = input_df[selected_features]

        probability = model.predict_proba(input_df)[0][1]
        prediction = 1 if probability >= decision_threshold else 0

        st.subheader("Prediction result")

        st.write(f"Decision threshold: **{decision_threshold:.2f}**")
        st.write(f"Parkinson probability: **{probability:.2%}**")

        if prediction == 1:
            st.error("Parkinson disease risk detected")
        else:
            st.success("Healthy voice pattern")


if __name__ == "__main__":
    main()