import sys
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

# The prediction logic lives in src/predict.py and is reused here instead of
# being duplicated, since src/ is not set up as an installable package.
sys.path.insert(0, str(SRC_DIR))
from predict import (  # noqa: E402
    load_prediction_artifacts as _load_prediction_artifacts,
    predict_from_features,
)

METRICS_PATH = BASE_DIR / "results" / "final_model" / "final_model_metrics.csv"

# Display metadata for each biomarker input (label, tooltip, default value and
# a sensible min/max/step/format derived from the training dataset).
FEATURE_INFO = {
    "PPE": {
        "label": "PPE (Pitch Period Entropy)",
        "help": "Measures irregularity in the fundamental frequency of the voice. "
                "Higher values indicate less stable pitch control.",
        "value": 0.284654, "min": 0.0, "max": 0.6, "step": 0.0001, "format": "%.6f",
    },
    "spread1": {
        "label": "spread1",
        "help": "Nonlinear measure of fundamental frequency variation.",
        "value": -4.813031, "min": -8.0, "max": -2.0, "step": 0.0001, "format": "%.6f",
    },
    "MDVP:Fo(Hz)": {
        "label": "MDVP:Fo(Hz) — Average vocal frequency",
        "help": "Average vocal fundamental frequency, measured in Hz.",
        "value": 119.992, "min": 80.0, "max": 270.0, "step": 0.001, "format": "%.3f",
    },
    "NHR": {
        "label": "NHR (Noise-to-Harmonics Ratio)",
        "help": "Ratio of noise to harmonic components in the voice signal.",
        "value": 0.02211, "min": 0.0, "max": 0.35, "step": 0.0001, "format": "%.6f",
    },
    "Jitter:DDP": {
        "label": "Jitter:DDP",
        "help": "Average absolute difference of differences between consecutive "
                "pitch periods, a measure of frequency instability.",
        "value": 0.01109, "min": 0.0, "max": 0.07, "step": 0.0001, "format": "%.6f",
    },
    "MDVP:Fhi(Hz)": {
        "label": "MDVP:Fhi(Hz) — Maximum vocal frequency",
        "help": "Maximum vocal fundamental frequency, measured in Hz.",
        "value": 157.302, "min": 100.0, "max": 600.0, "step": 0.001, "format": "%.3f",
    },
    "MDVP:Flo(Hz)": {
        "label": "MDVP:Flo(Hz) — Minimum vocal frequency",
        "help": "Minimum vocal fundamental frequency, measured in Hz.",
        "value": 74.997, "min": 60.0, "max": 245.0, "step": 0.001, "format": "%.3f",
    },
    "spread2": {
        "label": "spread2",
        "help": "Nonlinear measure of fundamental frequency variation.",
        "value": 0.266482, "min": 0.0, "max": 0.5, "step": 0.0001, "format": "%.6f",
    },
    "Shimmer:APQ5": {
        "label": "Shimmer:APQ5",
        "help": "Five-point amplitude perturbation quotient, a measure of "
                "amplitude variation in the voice signal.",
        "value": 0.03130, "min": 0.0, "max": 0.09, "step": 0.0001, "format": "%.6f",
    },
}

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }
    h1 {
        font-weight: 700;
    }
    .app-subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-top: -0.5rem;
        margin-bottom: 1.25rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(120, 120, 120, 0.06);
        border: 1px solid rgba(120, 120, 120, 0.15);
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    .footer-note {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
</style>
"""


@st.cache_resource
def load_prediction_artifacts():
    return _load_prediction_artifacts()


@st.cache_data
def load_model_metrics():
    if not METRICS_PATH.exists():
        return None

    return pd.read_csv(METRICS_PATH).iloc[0]


def render_sidebar(metrics):
    with st.sidebar:
        st.header("About this tool")
        st.write(
            "This application uses a **Random Forest** classifier trained on "
            "voice recordings to estimate the likelihood of Parkinson's disease "
            "based on biomedical voice measurements."
        )

        if metrics is not None:
            st.subheader("Model performance")
            st.caption("Evaluated on a held-out test set")

            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
            col2.metric("ROC-AUC", f"{metrics['ROC-AUC']:.1%}")

            col3, col4 = st.columns(2)
            col3.metric("Precision", f"{metrics['Precision']:.1%}")
            col4.metric("Recall", f"{metrics['Recall']:.1%}")

        st.divider()
        st.caption(
            "⚠️ Educational project only. Not a certified medical device and "
            "must not be used for real diagnostic decisions."
        )


def render_input_form():
    st.subheader("1. Input voice biomarkers")
    st.caption(
        "Enter the acoustic measurements extracted from a sustained vowel "
        "phonation recording. Hover over the ⓘ icon next to each field for "
        "details."
    )

    input_data = {}

    with st.form("prediction_form"):
        feature_keys = list(FEATURE_INFO.keys())
        columns = st.columns(3)

        for index, key in enumerate(feature_keys):
            info = FEATURE_INFO[key]
            with columns[index % 3]:
                input_data[key] = st.number_input(
                    info["label"],
                    help=info["help"],
                    value=info["value"],
                    min_value=info["min"],
                    max_value=info["max"],
                    step=info["step"],
                    format=info["format"],
                )

        submitted = st.form_submit_button(
            "Predict", use_container_width=True, type="primary"
        )

    return input_data, submitted


def render_result(model, selected_features, decision_threshold, input_data):
    input_df = pd.DataFrame([input_data])
    prediction, probability = predict_from_features(
        model, selected_features, decision_threshold, input_df
    )

    st.subheader("2. Prediction result")

    col1, col2 = st.columns(2)
    col1.metric("Parkinson probability", f"{probability:.1%}")
    col2.metric("Decision threshold", f"{decision_threshold:.0%}")

    st.progress(min(max(probability, 0.0), 1.0))

    if prediction == 1:
        st.error("🔴 **Parkinson disease risk detected** based on the provided measurements.")
    else:
        st.success("🟢 **Healthy voice pattern** based on the provided measurements.")

    st.caption(
        "This result reflects statistical patterns learned from training data "
        "and is **not** a medical diagnosis. Please consult a healthcare "
        "professional for any medical concerns."
    )


def main():
    st.set_page_config(
        page_title="Parkinson Disease Detection",
        page_icon="🧠",
        layout="centered",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("🧠 Parkinson Disease Detection")
    st.markdown(
        '<p class="app-subtitle">Machine learning system for Parkinson\'s '
        "disease risk prediction based on voice biomarkers.</p>",
        unsafe_allow_html=True,
    )

    st.warning(
        "This application is for **educational purposes only** and does not "
        "represent a medical diagnosis."
    )

    model, selected_features, decision_threshold = load_prediction_artifacts()
    metrics = load_model_metrics()

    render_sidebar(metrics)

    input_data, submitted = render_input_form()

    if submitted:
        render_result(model, selected_features, decision_threshold, input_data)

    st.markdown(
        '<p class="footer-note">Built with Streamlit · Random Forest model '
        "trained on the UCI Parkinson's voice dataset</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
