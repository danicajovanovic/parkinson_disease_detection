# Detekcija Parkinsonove bolesti

Projekat iz oblasti mašinskog učenja koji na osnovu biomedicinskih
glasovnih mera procenjuje rizik od Parkinsonove bolesti, korišćenjem
[UCI Oxford Parkinson's Disease Detection dataset-a](https://archive.ics.uci.edu/dataset/174/parkinsons).

> ⚠️ **Napomena:** Ovo je edukativni projekat. Model nije sertifikovano
> medicinsko sredstvo i ne sme se koristiti za stvarno postavljanje
> dijagnoze.

## Sadržaj

- [O dataset-u](#o-dataset-u)
- [Struktura projekta](#struktura-projekta)
- [Pipeline](#pipeline)
- [Metodologija](#metodologija)
- [Trenutni rezultati](#trenutni-rezultati)
- [Web interfejs](#web-interfejs)
- [Instalacija i pokretanje](#instalacija-i-pokretanje)

## O dataset-u

- 195 glasovnih snimaka od 31 osobe (23 sa Parkinsonovom bolešću, 8 zdravih).
- 22 biomedicinske glasovne karakteristike (jitter, shimmer, odnosi šuma,
  nelinearne mere varijacije fundamentalne frekvencije, ...) i binarna
  ciljna promenljiva `status` (1 = Parkinson, 0 = zdrav).
- Klase su neravnomerno raspodeljene: **147 Parkinson vs 48 zdravih**
  uzoraka (~75% / 25%), što je uzeto u obzir prilikom treniranja modela
  (videti [Metodologija](#metodologija)).
- Pun opis atributa se nalazi u [`data/parkinsons.names`](data/parkinsons.names).

## Struktura projekta

```
app/            Streamlit web interfejs
data/           Sirovi dataset
docs/           Specifikacija i dokumentacija projekta
models/         Sačuvan model, izabrane karakteristike i decision threshold
results/        Metrike, grafici i izveštaji koje generiše pipeline
src/            Skripte pipeline-a (analiza, treniranje, evaluacija, predikcija)
main.py         Ulazna tačka projekta
```

## Pipeline

Svaki korak pipeline-a je posebna skripta u `src/` i upisuje rezultate u
`results/`:

| Korak | Skripta | Izlaz |
|---|---|---|
| 1. Eksplorativna analiza podataka | [`src/data_analysis.py`](src/data_analysis.py) | konzolni ispis |
| 2. Poređenje baznih modela (5-fold CV) | [`src/train.py`](src/train.py) | `results/model_comparison/` |
| 3. Feature importance i izbor podskupa karakteristika | [`src/feature_selection.py`](src/feature_selection.py) | `results/feature_selection/` |
| 4. Finalni model: tuning hiperparametara, izbor threshold-a, evaluacija | [`src/final_model.py`](src/final_model.py) | `results/final_model/`, `models/` |
| 5. Samostalna evaluacija sačuvanog modela | [`src/evaluate.py`](src/evaluate.py) | `results/evaluation/` |
| 6. Primer predikcija | [`src/predict.py`](src/predict.py) | konzolni ispis |

Pokretanje iz korena projekta, po redosledu:

```bash
python src/data_analysis.py
python src/train.py
python src/feature_selection.py
python src/final_model.py
python src/evaluate.py
python src/predict.py
```

`main.py` je pomoćna ulazna tačka: ako već postoji istreniran model u
`models/`, pokreće primer predikcija; ako ne postoji, ispisuje koje
skripte treba prvo pokrenuti.

## Metodologija

- **Model:** Random Forest, podešen pomoću `GridSearchCV` (5-fold CV) na
  9 najznačajnijih karakteristika izabranih u koraku feature selection-a:
  `PPE`, `spread1`, `MDVP:Fo(Hz)`, `NHR`, `Jitter:DDP`, `MDVP:Fhi(Hz)`,
  `MDVP:Flo(Hz)`, `spread2`, `Shimmer:APQ5`.
- **Neravnomerna raspodela klasa:** pošto dataset ima 3x više Parkinson
  nego zdravih uzoraka, model koristi `class_weight="balanced"`, a
  pretraga hiperparametara optimizuje **F1-score** (a ne čist recall),
  čime se izbegava da model favorizuje većinsku klasu.
- **Decision threshold:** umesto podrazumevanog cutoff-a od 0.5,
  threshold se bira preko ROC krive i Youden indeksa, koristeći
  **out-of-fold predikcije na trening skupu** (`cross_val_predict`).
  Test skup se nikada ne koristi za biranje threshold-a — koristi se
  isključivo za finalnu, nezavisnu evaluaciju. Ovo je svesna odluka:
  biranje threshold-a na test skupu bi "iscurelo" test labele u proces
  selekcije modela i veštački naduvalo izveštene metrike.
- **Train/test podela:** 80/20, stratifikovano po klasama. Nije izdvojen
  poseban validation skup (dataset ima samo 195 uzoraka) — i izbor
  modela i izbor threshold-a se zasnivaju na 5-fold cross-validation na
  trening skupu.

## Trenutni rezultati

Rezultati finalnog modela na test skupu (videti
[`results/final_model/classification_report.txt`](results/final_model/classification_report.txt)
i [`results/final_model/final_model_metrics.csv`](results/final_model/final_model_metrics.csv)):

| Metrika | Vrednost |
|---|---|
| Accuracy | 92.3% |
| Precision | 93.3% |
| Recall | 96.6% |
| F1-score | 94.9% |
| ROC-AUC | 97.9% |
| Decision threshold | 0.447 |

Poređenje baznih modela (5-fold CV, pre feature selection-a) se nalazi u
[`results/model_comparison/model_comparison.csv`](results/model_comparison/model_comparison.csv),
a uticaj broja izabranih karakteristika na performanse u
[`results/feature_selection/feature_subset_comparison.csv`](results/feature_selection/feature_subset_comparison.csv).

## Web interfejs

Streamlit aplikacija omogućava ručni unos glasovnih biomarkera i
trenutnu predikciju:

```bash
streamlit run app/ui.py
```

Aplikacija koristi istu logiku predikcije iz `src/predict.py` i
artefakte koje čuva `src/final_model.py`
(`models/best_model.joblib`, `models/selected_features.joblib`,
`models/decision_threshold.joblib`).

## Instalacija i pokretanje

Projekat koristi [uv](https://docs.astral.sh/uv/) za upravljanje
zavisnostima.

```bash
uv sync
```

Zahteva Python >= 3.14 (videti [`pyproject.toml`](pyproject.toml)).
