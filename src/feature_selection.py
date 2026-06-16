from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from preprocessing import (
    load_data,
    prepare_features_and_target,
    split_data
)


BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"


def main():

    df = load_data()

    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    rf = model.named_steps["model"]

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(importance)

    importance.to_csv(
        RESULTS_DIR / "feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )

    plt.gca().invert_yaxis()

    plt.title(
        "Feature Importance - Random Forest"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "feature_importance.png"
    )

    plt.show()


if __name__ == "__main__":
    main()