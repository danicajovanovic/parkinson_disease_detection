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

    Takođe izvlači subjekta (osobu) iz kolone name (npr.
    "phon_R01_S01_1" -> "S01"). Dataset ima 195 snimaka od samo 32 osobe -
    više snimaka po osobi - pa se subjekat koristi za grupisanje pri deljenju
    podataka i cross-validaciji, da snimci iste osobe ne završe istovremeno
    u trening i test skupu (curenje podataka na nivou osobe).
    """
    groups = df["name"].str.extract(r"(S\d+)")[0]
    groups.name = "subject"

    X = df.drop(columns=["name", "status"])
    y = df["status"]

    return X, y, groups


def split_data(X, y, groups):
    """
    Deli podatke na trening i test skup, na nivou osobe (subjekta), ne na
    nivou pojedinačnog snimka.

    Pošto svaka osoba ima više snimaka sa istim statusom, običan
    train_test_split na nivou snimka bi gotovo sigurno stavio snimke iste
    osobe i u trening i u test skup, što bi model učilo da prepoznaje
    konkretnu osobu umesto generalizovanog biomarkera bolesti. Split se zato
    pravi na nivou jedinstvenih subjekata (stratifikovano po njihovom
    statusu), a zatim se mapira nazad na snimke.

    Ne pravi se poseban validation skup jer dataset ima samo 32 osobe.
    Validacija ce se kasnije raditi pomocu 5-fold Group Cross Validation
    na trening skupu.
    """
    subject_status = (
        pd.DataFrame({"subject": groups, "status": y})
        .drop_duplicates("subject")
        .set_index("subject")["status"]
    )

    train_subjects, test_subjects = train_test_split(
        subject_status.index.to_numpy(),
        test_size=0.2,
        random_state=42,
        stratify=subject_status.to_numpy()
    )

    train_mask = groups.isin(train_subjects)
    test_mask = groups.isin(test_subjects)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    groups_train, groups_test = groups[train_mask], groups[test_mask]

    return X_train, X_test, y_train, y_test, groups_train, groups_test


def main():
    df = load_data()

    X, y, groups = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_data(
        X, y, groups
    )

    print("Ukupan broj uzoraka:", len(df))
    print("Broj jedinstvenih osoba:", groups.nunique())
    print("Broj ulaznih atributa:", X.shape[1])

    print("\nX:", X.shape)
    print("y:", y.shape)

    print("\nTrening skup:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("Broj osoba u trening skupu:", groups_train.nunique())

    print("\nTest skup:")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    print("Broj osoba u test skupu:", groups_test.nunique())

    print("\nPreklapanje osoba izmedju trening i test skupa (mora biti 0):")
    print(len(set(groups_train) & set(groups_test)))

    print("\nRaspodela klasa u trening skupu:")
    print(y_train.value_counts())

    print("\nRaspodela klasa u test skupu:")
    print(y_test.value_counts())

    print("\nNapomena:")
    print("Poseban validation skup nije izdvojen.")
    print(
        "Za validaciju modela koristi se 5-fold Group Cross Validation "
        "na trening skupu (grupisano po osobi, da snimci iste osobe ne "
        "zavrse u razlicitim fold-ovima)."
    )


if __name__ == "__main__":
    main()