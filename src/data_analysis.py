import re
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "parkinsons.data"

RESULTS_DIR = BASE_DIR / "results" / "graphics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CORRELATION_HEATMAP_PATH = RESULTS_DIR / "correlation_heatmap.png"
CLASS_DISTRIBUTION_PATH = RESULTS_DIR / "class_distribution.png"
PCA_SCATTER_PATH = RESULTS_DIR / "pca_scatter.png"

TOP_N_FEATURES = 9


def sanitize_feature_name(feature):
    """
    "MDVP:Fo(Hz)" -> "MDVP_FoHz", da ime fajla bude bezbedno za sve OS.
    """
    return re.sub(r"[^0-9a-zA-Z]+", "_", feature).strip("_")


def plot_correlation_heatmap(numeric_df, output_path):
    correlation = numeric_df.corr(numeric_only=True)

    plt.figure(figsize=(14, 11))
    sns.heatmap(correlation, cmap="coolwarm", center=0, annot=False)
    plt.title("Korelacija između atributa")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return correlation


def plot_class_distribution(df, output_path):
    plt.figure(figsize=(5, 5))
    df["status"].value_counts().sort_index().plot(
        kind="bar", color=["seagreen", "indianred"]
    )
    plt.title("Raspodela klasa (0 = zdrav, 1 = Parkinson)")
    plt.xlabel("status")
    plt.ylabel("Broj uzoraka")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def top_correlated_features(correlation, target="status", n=TOP_N_FEATURES):
    return (
        correlation[target]
        .drop(target)
        .abs()
        .sort_values(ascending=False)
        .head(n)
        .index
        .tolist()
    )


def plot_feature_boxplot(df, feature, output_path):
    plt.figure(figsize=(5, 4))
    sns.boxplot(x="status", y=feature, data=df, hue="status", legend=False)
    plt.title(f"{feature} po klasama")
    plt.xlabel("status (0 = zdrav, 1 = Parkinson)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pca_scatter(numeric_df, output_path):
    """
    2D PCA projekcija svih ulaznih atributa (bez "status"), obojena po
    klasi - vizuelni uvid u to koliko su klase razdvojive u prostoru
    atributa, i da li postoje očigledni outlieri daleko od oba klastera.
    """
    features = numeric_df.drop(columns=["status"])
    status = numeric_df["status"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ])
    components = pipeline.named_steps["pca"].fit_transform(
        pipeline.named_steps["scaler"].fit_transform(features)
    )
    explained_variance = pipeline.named_steps["pca"].explained_variance_ratio_

    plt.figure(figsize=(7, 6))
    for status_value, label, color in [(0, "Zdrav", "seagreen"), (1, "Parkinson", "indianred")]:
        mask = status == status_value
        plt.scatter(
            components[mask, 0], components[mask, 1],
            label=label, color=color, alpha=0.7, edgecolor="black", linewidth=0.3,
        )

    plt.xlabel(f"PC1 ({explained_variance[0]:.1%} varijanse)")
    plt.ylabel(f"PC2 ({explained_variance[1]:.1%} varijanse)")
    plt.title("PCA projekcija (2D) svih atributa, po klasi")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return explained_variance


def plot_feature_histogram(df, feature, output_path):
    plt.figure(figsize=(5, 4))
    sns.histplot(data=df, x=feature, hue="status", kde=True, element="step")
    plt.title(f"Raspodela atributa {feature}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    df = pd.read_csv(DATA_PATH)

    print("Prvih 5 redova dataseta:")
    print(df.head())

    print("\nDimenzija dataseta:")
    print(df.shape)

    print("\nInformacije o kolonama:")
    print(df.info())

    print("\nNedostajuće vrednosti:")
    print(df.isnull().sum())

    print("\nDuplikati:")
    print(df.duplicated().sum())

    print("\nRaspodela klasa:")
    print(df["status"].value_counts())

    print("\nOsnovna statistika:")
    print(df.describe())

    # "name" nije atribut korisan za korelacionu/distribucionu analizu - nosi
    # samo identifikator snimka/osobe, ne akustičku informaciju.
    numeric_df = df.drop(columns=["name"])

    correlation = plot_correlation_heatmap(numeric_df, CORRELATION_HEATMAP_PATH)
    plot_class_distribution(df, CLASS_DISTRIBUTION_PATH)

    explained_variance = plot_pca_scatter(numeric_df, PCA_SCATTER_PATH)
    print(
        f"\nPCA - objašnjena varijansa: PC1={explained_variance[0]:.1%}, "
        f"PC2={explained_variance[1]:.1%}, "
        f"ukupno={explained_variance.sum():.1%}"
    )

    top_features = top_correlated_features(correlation)

    print(f"\nAtributi najjače korelisani sa 'status' (top {TOP_N_FEATURES}):")
    print(correlation["status"][top_features])

    for feature in top_features:
        safe_name = sanitize_feature_name(feature)
        plot_feature_boxplot(
            numeric_df, feature, RESULTS_DIR / f"boxplot_{safe_name}.png"
        )
        plot_feature_histogram(
            numeric_df, feature, RESULTS_DIR / f"histogram_{safe_name}.png"
        )

    print(f"\nGrafici su sačuvani u: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
