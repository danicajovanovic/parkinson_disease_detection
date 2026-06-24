"""
Detekcija anomalija u skupu podataka, prateći pristupe iz predavanja
"Metode zasnovane na rastojanju i sličnosti" (KNN, LOF, Isolation Forest).

Ovo je deo eksplorativne analize / preprocesiranja (faza "Proveriti da li
postoje nedostajuće vrednosti ili anomalije") - svrha je da se potvrdi da li
postoje snimci koji značajno odskaču od ostatka skupa, a ne da se nužno
nešto ukloni. Sve tri metode su nenadgledane (ne koriste kolonu "status").

Koraci (za svaku metodu):
1. standardizovati atribute (rastojanje ima smisla samo na uporedivoj skali)
2. izračunati anomaly score za svaki uzorak
3. proglasiti top-N uzoraka sa najvećim score-om za potencijalne anomalije
4. upoutiti se da li se metode slažu (presek skupova) - ako se tri nezavisne
   metode slažu oko istog uzorka, to je jači signal nego kad se ne slažu
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from preprocessing import load_data, prepare_features_and_target


BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results" / "anomaly_detection"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = RESULTS_DIR / "anomaly_scores.csv"
OVERLAP_TXT = RESULTS_DIR / "anomaly_overlap.txt"
PLOT_PATH = RESULTS_DIR / "anomaly_scores.png"

K_NEIGHBORS = 5
TOP_N = 10


def knn_anomaly_score(X_scaled, k=K_NEIGHBORS):
    """
    Anomaly score = rastojanje do K-tog najbližeg suseda. Veće rastojanje
    znači da je tačka izolovanija od ostatka skupa.
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    return distances[:, -1]


def lof_anomaly_score(X_scaled, k=K_NEIGHBORS):
    """
    Local Outlier Factor - posmatra lokalnu gustinu, ne samo rastojanje.
    sklearn vraća negative_outlier_factor_ (manje = sumnjivije), pa se
    okreće znak da veći score = sumnjiviji uzorak, dosledno ostalim merama.
    """
    lof = LocalOutlierFactor(n_neighbors=k)
    lof.fit_predict(X_scaled)
    return -lof.negative_outlier_factor_


def isolation_forest_anomaly_score(X_scaled, random_state=42):
    """
    Isolation Forest - tačka je sumnjivija ako se izoluje u manje podela.
    score_samples vraća veće (manje negativne) vrednosti za normalne tačke,
    pa se okreće znak da veći score = sumnjiviji uzorak.
    """
    model = IsolationForest(random_state=random_state)
    model.fit(X_scaled)
    return -model.score_samples(X_scaled)


def top_n_indices(scores, n=TOP_N):
    return set(pd.Series(scores).sort_values(ascending=False).head(n).index)


def plot_anomaly_scores(names, scores_df, flagged_any, output_path):
    plt.figure(figsize=(11, 6))
    x = range(len(names))
    plt.scatter(
        x, scores_df["KNN (normalizovano)"],
        label="KNN", alpha=0.7, s=25,
    )
    plt.scatter(
        x, scores_df["LOF (normalizovano)"],
        label="LOF", alpha=0.7, s=25,
    )
    plt.scatter(
        x, scores_df["Isolation Forest (normalizovano)"],
        label="Isolation Forest", alpha=0.7, s=25,
    )
    for index in flagged_any:
        plt.axvline(index, color="grey", alpha=0.15, linewidth=1)

    plt.xlabel("Indeks uzorka (snimka)")
    plt.ylabel("Anomaly score (min-max normalizovan po metodi)")
    plt.title(
        f"Anomaly score po metodi (sive linije = uzorci u top-{TOP_N} "
        "bar jedne metode)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    df = load_data()
    X, y, groups = prepare_features_and_target(df)

    X_scaled = StandardScaler().fit_transform(X)

    knn_scores = knn_anomaly_score(X_scaled)
    lof_scores = lof_anomaly_score(X_scaled)
    iso_scores = isolation_forest_anomaly_score(X_scaled)

    scores_df = pd.DataFrame({
        "name": df["name"],
        "subject": groups,
        "status": y,
        "KNN distance": knn_scores,
        "LOF score": lof_scores,
        "Isolation Forest score": iso_scores,
    })

    # Min-max normalizacija samo za vizuelno poređenje na istom grafiku -
    # tri metode imaju potpuno različite skale score-a.
    for raw_col, norm_col in [
        ("KNN distance", "KNN (normalizovano)"),
        ("LOF score", "LOF (normalizovano)"),
        ("Isolation Forest score", "Isolation Forest (normalizovano)"),
    ]:
        values = scores_df[raw_col]
        scores_df[norm_col] = (values - values.min()) / (values.max() - values.min())

    knn_top = top_n_indices(knn_scores)
    lof_top = top_n_indices(lof_scores)
    iso_top = top_n_indices(iso_scores)

    agreement = knn_top & lof_top & iso_top
    flagged_any = knn_top | lof_top | iso_top

    print(f"Top-{TOP_N} po KNN distanci:\n{scores_df.loc[sorted(knn_top), ['name', 'subject', 'status']]}\n")
    print(f"Top-{TOP_N} po LOF:\n{scores_df.loc[sorted(lof_top), ['name', 'subject', 'status']]}\n")
    print(f"Top-{TOP_N} po Isolation Forest:\n{scores_df.loc[sorted(iso_top), ['name', 'subject', 'status']]}\n")

    print(f"Uzorci koje SVE TRI metode oglašavaju anomalijom ({len(agreement)}):")
    if agreement:
        print(scores_df.loc[sorted(agreement), ["name", "subject", "status"]])
    else:
        print("(nema preseka - metode se ne slažu oko istih uzoraka)")

    scores_df.to_csv(SUMMARY_CSV, index=False)
    plot_anomaly_scores(df["name"], scores_df, flagged_any, PLOT_PATH)

    with open(OVERLAP_TXT, "w", encoding="utf-8") as file:
        file.write("Detekcija anomalija - KNN, LOF, Isolation Forest\n\n")
        file.write(f"Top-{TOP_N} po metodi se preklapaju u {len(agreement)} uzoraka.\n\n")
        file.write("Uzorci u preseku sve tri metode:\n")
        if agreement:
            file.write(
                scores_df.loc[sorted(agreement), ["name", "subject", "status"]].to_string()
            )
        else:
            file.write("(nema preseka)\n")
        file.write(
            "\n\nZaključak: uzorci nisu uklonjeni iz skupa - dataset je vec mali "
            "(195 snimaka, 32 osobe), a izolovanost u prostoru atributa kod "
            "biomedicinskih signala često odražava prirodnu varijaciju glasa "
            "(npr. teže izražen Parkinson), ne grešku u merenju. Uklanjanje bi "
            "rizikovalo da se izgubi baš signal koji model treba da nauči."
        )

    print(f"\nRezultati sačuvani u: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
