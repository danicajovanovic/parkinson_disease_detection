from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "parkinsons.data"


def load_data():
    """
    Ucitava Parkinson dataset.
    """
    df = pd.read_csv(DATA_PATH)
    return df


def prepare_features_and_target(df):
    """
    Uklanja kolonu name i odvaja ulazne atribute X i ciljnu promenljivu y.
    """
    X = df.drop(columns=["name", "status"])
    y = df["status"]

    return X, y


def split_data(X, y):
    """
    Deli podatke na trening i test skup.

    Ne pravi se poseban validation skup jer dataset ima samo 195 uzoraka.
    Validacija ce se kasnije raditi pomocu 5-fold Cross Validation
    na trening skupu.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def main():
    df = load_data()

    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Ukupan broj uzoraka:", len(df))
    print("Broj ulaznih atributa:", X.shape[1])

    print("\nX:", X.shape)
    print("y:", y.shape)

    print("\nTrening skup:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nTest skup:")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("\nRaspodela klasa u trening skupu:")
    print(y_train.value_counts())

    print("\nRaspodela klasa u test skupu:")
    print(y_test.value_counts())

    print("\nNapomena:")
    print("Poseban validation skup nije izdvojen.")
    print("Za validaciju modela koristi se 5-fold Cross Validation na trening skupu.")


if __name__ == "__main__":
    main()