from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "parkinsons.data"


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


if __name__ == "__main__":
    main()