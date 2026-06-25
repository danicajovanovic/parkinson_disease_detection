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
│   ├── train.py                # poređenje modela
│   ├── feature_selection.py    # izbor karakteristika
│   ├── final_model.py          # treniranje finalnog modela (ne dotiče test skup)
│   ├── evaluate.py             # evaluacija sačuvanog modela na test skupu
│   ├── learning_curve.py       # analiza train/CV performansi
│   ├── split_robustness.py     # stabilnost različitih split-ova
│   └── predict.py              # primer predikcije

├── app/ui.py                   # Streamlit aplikacija
├── data/                       # dataset
├── models/                     # sačuvani model i artefakti
├── results/
│   ├── final_model/            # artefakti treniranja (hiperparametri, training_report.txt, ...)
│   └── evaluation/             # rezultati evaluacije na test skupu (metrike, izveštaj, grafici)
└── dokumentacija/documentation.pdf  # kompletna dokumentacija
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

| Korak | Skripta | Opis |
|------|----------|------|
| 1 | `data_analysis.py` | Eksplorativna analiza podataka |
| 2 | `anomaly_detection.py` | Detekcija anomalija |
| 3 | `train.py` | Poređenje modela |
| 4 | `feature_selection.py` | Izbor najvažnijih atributa |
| 5 | `final_model.py` | Tuning i treniranje finalnog modela |
| 6 | `evaluate.py` | Evaluacija sačuvanog modela na test skupu |
| 7 | `learning_curve.py` | Analiza generalizacije |
| 8 | `split_robustness.py` | Stabilnost rezultata |
| 9 | `predict.py` | Predikcija |

Treniranje (`final_model.py`) i evaluacija (`evaluate.py`) su striktno razdvojeni. `final_model.py` trenira model, bira atribute i određuje decision threshold isključivo na trening skupu i ni na koji način ne koristi test skup. Evaluacija se vrši tek nakon toga pokretanjem `evaluate.py`, koji učitava već sačuvane modele i artefakte.

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

Logistic Regression korišćena je kao osnovni model binarne klasifikacije i služila je kao referentni model za poređenje sa ostalim algoritmima. Ovaj model računa verovatnoću pripadnosti klasi pomoću logističke funkcije i predstavlja jedan od najčešće korišćenih modela za probleme binarne klasifikacije.

Svi modeli evaluirani su korišćenjem grupisane 5-fold cross-validacije i odgovarajućih metrika evaluacije.

Nakon poređenja svih modela kao finalni model izabran je:

```text
Random Forest
```

Random Forest predstavlja ansambl metodu zasnovanu na većem broju stabala odlučivanja. Svako stablo daje svoju predikciju, dok se konačna odluka dobija većinskim glasanjem svih stabala.

Ovaj model ostvario je najbolje ukupne performanse i pokazao se pogodnijim za modelovanje složenih i nelinearnih odnosa između glasovnih biomarkera.

Logistic Regression korišćena je tokom faze poređenja modela, dok se Random Forest koristi kao konačni model sistema i u Streamlit aplikaciji.

Hiperparametri finalnog modela optimizovani su pomoću:

```text
GridSearchCV
```

Originalnih 22 atributa redukovano je na 10 najznačajnijih karakteristika.

---

## Sprečavanje curenja podataka

Pošto dataset sadrži više snimaka iste osobe, train/test podela i cross-validacija vršene su na nivou osobe korišćenjem:

* StratifiedGroupKFold
* grupisane train/test podele

Na ovaj način nijedan snimak iste osobe ne može se pojaviti istovremeno u trening i test skupu.

Validacija se vrši pomoću 5-fold Group Cross Validation na trening skupu. Poseban validation skup nije izdvojen zbog malog broja ispitanika.

---

## Neravnomerne klase

Zbog odnosa 147:48 korišćeno je:

```python
class_weight="balanced"
```

Optimizacija modela vršena je korišćenjem F1-score metrike.

---

## Izbor decision threshold-a

Threshold nije fiksiran na vrednost 0.5.

Optimalna vrednost određena je pomoću:

* ROC krive
* Youden indeksa
* out-of-fold predikcija

Threshold je određen isključivo na trening skupu kako bi se sprečilo curenje informacija iz test skupa.

---

## Rezultati

| Metrika | Vrednost |
|---------|----------|
| Accuracy | 93.0% |
| Precision | 91.2% |
| Recall | 100.0% |
| F1-score | 95.4% |
| ROC-AUC | 97.0% |
| Decision threshold | 0.769 |

Pošto test skup sadrži samo 7 osoba, dodatne analize stabilnosti modela sprovedene su korišćenjem:

* 5-fold group cross-validation
* analize različitih train/test split-ova

Rezultati treniranja (hiperparametri, `training_report.txt`, heatmap, learning curve i stabilnost split-ova) nalaze se u `results/final_model/`, dok se rezultati evaluacije (`classification_report.txt`, `evaluation_metrics.csv`, `confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`) nalaze u `results/evaluation/`.

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

Svi rezultati nalaze se u direktorijumu:

```text
results/
```

---

## Web aplikacija

Pokretanje aplikacije:

```bash
streamlit run app/ui.py
```

Aplikacija koristi isti model, izabrane karakteristike i decision threshold kao i ostatak sistema.

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
