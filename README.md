# Detekcija Parkinsonove bolesti

![Python](https://img.shields.io/badge/Python-3.14-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

Projekat iz mašinskog učenja za binarnu klasifikaciju Parkinsonove bolesti na osnovu akustičkih karakteristika glasa korišćenjem UCI Parkinson's Disease dataseta.

> ⚠️ Edukativan projekat — model nije medicinsko sredstvo i njegov izlaz ne treba tumačiti kao dijagnozu.

---

## Struktura projekta

```text
├── src/
│   ├── data_analysis.py        # eksplorativna analiza podataka
│   ├── anomaly_detection.py    # detekcija anomalija
│   ├── train.py                # poređenje baznih modela
│   ├── feature_selection.py    # izbor karakteristika
│   ├── final_model.py          # treniranje finalnog modela
│   ├── evaluate.py             # evaluacija modela
│   ├── learning_curve.py       # analiza train/CV performansi
│   ├── split_robustness.py     # stabilnost različitih split-ova
│   └── predict.py              # primer predikcije

├── app/ui.py                   # Streamlit aplikacija
├── data/                       # dataset
├── models/                     # sačuvani model i artefakti
├── results/                    # metrike, grafici i izveštaji
└── docs/documentation.md       # kompletna dokumentacija
```

---

## Dataset

Korišćen je **UCI Parkinson's Disease dataset** koji sadrži:

* 195 glasovnih snimaka
* 32 osobe
* 24 osobe sa Parkinsonovom bolešću
* 8 zdravih osoba
* 22 numeričke karakteristike glasa
* ciljnu promenljivu `status`

Svaka osoba poseduje više snimaka (u proseku oko 6), zbog čega je podela podataka vršena na nivou osobe.

Dataset je nebalansiran:

* 147 Parkinson snimaka
* 48 zdravih snimaka

Pun opis atributa nalazi se u:

```text
data/parkinsons.names
```

---

## Pipeline

| Korak | Skripta                | Opis                           |
| ----- | ---------------------- | ------------------------------ |
| 1     | `data_analysis.py`     | Eksplorativna analiza podataka |
| 2     | `anomaly_detection.py` | Detekcija anomalija            |
| 3     | `train.py`             | Poređenje modela               |
| 4     | `feature_selection.py` | Izbor najvažnijih atributa     |
| 5     | `final_model.py`       | Tuning i treniranje            |
| 6     | `evaluate.py`          | Evaluacija                     |
| 7     | `learning_curve.py`    | Analiza generalizacije         |
| 8     | `split_robustness.py`  | Stabilnost rezultata           |
| 9     | `predict.py`           | Predikcija                     |

---

## Metodologija

Evaluirani modeli:

* Logistic Regression
* KNN
* SVM
* Decision Tree
* Random Forest
* XGBoost
* Dummy baseline classifier

Kao finalni model izabran je:

```text
Random Forest
```

Hiperparametri su optimizovani pomoću:

```text
GridSearchCV
```

Originalnih 22 atributa redukovano je na 10 najznačajnijih karakteristika.

---

## Sprečavanje curenja podataka

Pošto dataset sadrži više snimaka iste osobe, train/test podela i cross-validacija vršene su na nivou osobe korišćenjem:

* StratifiedGroupKFold
* grupisane train/test podele

Na ovaj način nijedan snimak iste osobe ne može da se pojavi istovremeno u trening i test skupu.

---

## Neravnomerne klase

Zbog odnosa 147:48 korišćeno je:

```python
class_weight="balanced"
```

Optimizacija modela vršena je korišćenjem F1-score metrike.

---

## Izbor decision threshold-a

Threshold nije fiksiran na 0.5.

Optimalna vrednost određena je pomoću:

* ROC krive
* Youden indeksa
* out-of-fold predikcija

Threshold je određen isključivo na trening skupu kako bi se sprečilo curenje informacija iz test skupa.

---

## Rezultati

| Metrika            | Vrednost |
| ------------------ | -------- |
| Accuracy           | 93.0%    |
| Precision          | 91.2%    |
| Recall             | 100.0%   |
| F1-score           | 95.4%    |
| ROC-AUC            | 97.0%    |
| Decision threshold | 0.769    |

Pošto test skup sadrži samo 7 osoba, dodatne analize stabilnosti modela sprovedene su korišćenjem:

* 5-fold group cross-validation
* analize različitih train/test split-ova

---

## Dodatne analize

Implementirane su i dodatne analize:

* korelaciona analiza
* PCA vizualizacija
* analiza značaja atributa
* learning curves
* Precision-Recall analiza
* analiza stabilnosti split-ova
* detekcija anomalija

Svi rezultati se nalaze u direktorijumu:

```text
results/
```

---

## Web aplikacija

Pokretanje aplikacije:

```bash
streamlit run app/ui.py
```

Aplikacija koristi isti model, izabrane karakteristike i threshold kao i ostatak sistema.

---

## Pokretanje projekta

Instalacija zavisnosti:

```bash
uv sync
```

Pokretanje kompletnog pipeline-a:

```bash
python src/data_analysis.py
python src/anomaly_detection.py
python src/train.py
python src/feature_selection.py
python src/final_model.py
python src/evaluate.py
```

---

## Tehnologije

* Python
* scikit-learn
* pandas
* numpy
* matplotlib
* seaborn
* Streamlit
* XGBoost

---

## Autor

**Danica Jovanović**

Fakultet tehničkih nauka, Novi Sad
