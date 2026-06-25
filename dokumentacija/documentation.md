# Detekcija Parkinsonove bolesti iz glasovnih biomarkera

Predmetni projekat — SAUSAU, 2026.

## 1. Opis problema

Cilj projekta je da se na osnovu akustičkih karakteristika glasa proceni
da li osoba ima Parkinsonovu bolest. Koristi se javno dostupan
[UCI Parkinson's dataset](https://archive.ics.uci.edu/dataset/174/parkinsons):
195 snimaka glasa od 32 osobe (24 sa Parkinsonom, 8 zdravih; svaka osoba
ima u proseku ~6 snimaka), opisanih sa 22 numeričke karakteristike
(jitter, shimmer, odnos šuma i tona, nelinearne mere frekvencije...) i
binarnom ciljnom promenljivom `status` (1 = Parkinson, 0 = zdrava osoba).
Pun opis kolona je u [data/parkinsons.names](../data/parkinsons.names).

Problem je formulisan kao **binarna klasifikacija**: na osnovu vektora
akustičkih atributa jednog snimka predvideti `status` osobe od koje snimak
potiče.

Napomena: projekat je edukativan. Model nije medicinsko sredstvo i njegov
izlaz se ne tumači kao dijagnoza.

## 2. Početno preprocesiranje podataka

Implementirano u [src/data_analysis.py](../src/data_analysis.py) i
[src/preprocessing.py](../src/preprocessing.py).

- **Nedostajuće vrednosti i duplikati**: provereno `df.isnull().sum()` i
  `df.duplicated().sum()` — dataset nema nedostajuće vrednosti ni
  duplikate redova.
- **Enkodiranje**: svi ulazni atributi su već numerički (kontinualne
  akustičke mere), tako da enkodiranje kategoričkih promenljivih nije
  potrebno.
- **Uklanjanje irelevantnih atributa**: kolona `name` (identifikator
  snimka/osobe, npr. `phon_R01_S01_1`) se uklanja iz ulaznih atributa jer ne
  nosi akustičku informaciju. Umesto toga se iz nje izvlači identifikator
  osobe (`S01`), koji se koristi kao `groups` za podelu podataka i
  cross-validaciju (videti tačku 3).
- **Skaliranje**: svi modeli se obučavaju kroz `Pipeline` sa
  `StandardScaler`-om (z-score standardizacija), neophodnom za modele
  zasnovane na rastojanju/margini (KNN, SVM, logistička regresija).

## 3. Eksplorativna analiza skupa

Implementirano u [src/data_analysis.py](../src/data_analysis.py),
rezultati u [results/graphics/](../results/graphics/).

- **Raspodela klasa** (`class_distribution.png`): 147 snimaka klase
  Parkinson, 48 snimaka klase zdrav — dataset je **neravnomeran**
  (~3:1), što direktno utiče na izbor metrike i treniranje modela
  (tačke 5 i 6).
- **Korelacije** (`correlation_heatmap.png`): atributi izvedeni iz istog
  fizičkog fenomena su međusobno snažno korelisani (npr. `MDVP:Jitter(%)`,
  `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP` mere variraju oko istog svojstva
  glasa — jitter; isto za `Shimmer:APQ3/APQ5`, `MDVP:APQ`, `Shimmer:DDA`).
  Ova multikolinearnost je razlog zbog kojeg je korisno izvršiti odabir
  atributa (tačka 7) umesto korišćenja svih 22 atributa.
- **Atributi najjače korelisani sa ciljnom promenljivom**: `spread1`,
  `PPE`, `spread2`, `MDVP:Fo(Hz)`, `MDVP:Flo(Hz)`, `MDVP:Shimmer`,
  `MDVP:APQ`, `HNR`, `Shimmer:APQ5` (apsolutna Pearson korelacija sa
  `status`). Za njih su generisani `boxplots_top_features.png` (raspodela
  po klasi, sva 9 atributa na jednoj slici, grid 3×3) i
  `histograms_top_features.png` (raspodela vrednosti, isti format).
- **Anomalije/ekstremne vrednosti**: boxplot-ovi pokazuju prirodno
  rasipanje karakteristično za biomedicinske signale (par outlier-a po
  atributu), bez vrednosti koje ukazuju na grešku u merenju/unosu (npr.
  negativne frekvencije ili nemoguće odnose).

  Pored vizuelne provere, [src/anomaly_detection.py](../src/anomaly_detection.py)
  primenjuje tri nenadgledane metode detekcije anomalija nad standardizovanim
  atributima: **KNN** (rastojanje do K-tog najbližeg suseda), **LOF**
  (Local Outlier Factor — lokalna gustina) i **Isolation Forest** (broj
  podela potrebnih za izolaciju tačke). Svaka metoda nezavisno proglašava
  top-10 najsumnjivijih snimaka; presek sve tri metode (jači signal nego
  slaganje samo jedne) su 4 snimka, sva od osoba `S24` i `S35`
  ([results/anomaly_detection/anomaly_overlap.txt](../results/anomaly_detection/anomaly_overlap.txt)).

  Uzorci **nisu uklonjeni**: model (Random Forest) je dovoljno robustan na
  ovakav nivo šuma, skup je već mali (195 redova, 32 osobe) — uklanjanje bi
  rizikovalo gubitak baš onog signala koji model treba da nauči (kod
  Parkinsonovih pacijenata izraženost simptoma prirodno varira po snimku),
  a ne grešku u merenju.
- **PCA vizualizacija** (`pca_scatter.png`): 2D projekcija svih 22 atributa
  (PC1 = 58.9%, PC2 = 11.3% varijanse, ukupno ≈ 70%) pokazuje delimično
  preklapanje klasa — potvrđuje da problem nije trivijalno linearno
  razdvojiv u 2D, što opravdava korišćenje nelinearnih modela
  (Random Forest, Gradient Boosting) u poređenju modela.

## 4. Odabir i treniranje modela

Implementirano u [src/train.py](../src/train.py), rezultati u
[results/model_comparison/](../results/model_comparison/).

Isprobano je šest algoritama pogodnih za binarnu klasifikaciju na malom
tabelarnom skupu: logistička regresija, KNN, SVM, stablo odlučivanja,
Random Forest i Gradient Boosting (XGBoost), uz **baseline** model
(`DummyClassifier`, uvek predviđa većinsku klasu) kao donju referentnu
granicu. Svaki je evaluiran pomoću **5-fold Group Cross Validation**
(`StratifiedGroupKFold`) na trening skupu.

**Zašto grupna (group) CV, a ne obična stratifikovana CV?** Dataset ima
195 snimaka od samo 32 osobe, sa po nekoliko snimaka po osobi i istim
statusom za sve snimke jedne osobe. Obična podela na nivou snimka bi
gotovo sigurno stavila snimke iste osobe i u trening i u validacioni/test
fold, čime bi model delom učio da prepoznaje konkretnu osobu po glasu
umesto generalizovanog biomarkera bolesti (**curenje podataka na nivou
subjekta**). Zato se i podela na trening/test skup (80/20) i sva
cross-validacija rade grupisano po osobi.

Rezultati poređenja (5-fold group CV, prosek):

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (majority class) | 0.764 | 0.764 | 1.000 | 0.864 | 0.500 |
| SVM | 0.856 | 0.859 | 0.992 | 0.914 | 0.704 |
| Random Forest | 0.803 | 0.837 | 0.951 | 0.881 | 0.769 |
| Gradient Boosting (XGBoost) | 0.763 | 0.826 | 0.909 | 0.855 | 0.791 |
| Logistic Regression | 0.796 | 0.871 | 0.901 | 0.872 | 0.755 |
| KNN | 0.744 | 0.823 | 0.885 | 0.840 | 0.796 |
| Decision Tree | 0.731 | 0.835 | 0.844 | 0.827 | 0.647 |

Baseline ima F1 = 0.864 unatoč tome što ne razlikuje klase — posledica
neravnomernih klasa (accuracy paradox, pomenut na predavanju 3): kad jedna
klasa dominira, i F1 i accuracy mogu izgledati dobro samo zato što se
uvek predviđa većinska klasa. ROC-AUC = 0.5 razotkriva to (model bez
prediktivne moći nužno ima ROC-AUC = 0.5), pa je svaki ozbiljan model u
tabeli ocenjivan i po ROC-AUC, ne samo po F1/accuracy. Po ovom kriterijumu,
svi modeli osim Decision Tree-a jasno nadmašuju baseline.

SVM ima najviši F1, ali najslabiji ROC-AUC od preostalih modela (loše
rangira verovatnoće, što je važno jer se threshold posebno bira —
tačka 6). Gradient Boosting (XGBoost) ima najviši ROC-AUC, ali niži F1 od
SVM/RF/LogReg na default hiperparametrima (nije podešavan u ovoj fazi).
**Random Forest** je
izabran kao finalni model: drugo najbolje F1, najbolji ROC-AUC od
top modela, prirodno daje feature importance (potrebno za tačku 7),
otporan je na multikolinearnost uočenu u EDA i ne zahteva intenzivno
podešavanje da bi bio stabilan na malom skupu. `cv_scores_boxplot.png`
pokazuje i raspon F1-score-a po fold-u, ne samo prosek — bitno za skup od
samo 195 uzoraka, gde je varijansa po fold-u značajna.

## 5. Odabir najznačajnijih atributa

Implementirano u [src/feature_selection.py](../src/feature_selection.py),
rezultati u [results/feature_selection/](../results/feature_selection/).

Značajnost atributa je računata pomoću `feature_importances_` iz
Random Forest-a (`feature_importance.png`). Atributi su zatim rangirani i
testirani u podskupovima rastuće veličine (1, 2, ..., 22 atributa), svaki
put sa istim 5-fold group CV protokolom kao u tački 4
(`feature_subset_performance.png`, `feature_subset_comparison.csv`).

Najbolji rezultat (po F1, zatim Recall, zatim ROC-AUC) postiže podskup od
**10 atributa**: `PPE`, `MDVP:Flo(Hz)`, `spread1`, `MDVP:Fo(Hz)`,
`MDVP:Fhi(Hz)`, `NHR`, `RPDE`, `Jitter:DDP`, `MDVP:Jitter(Abs)`,
`spread2` — F1 = 0.896, Recall = 0.959, ROC-AUC = 0.804, naspram
F1 = 0.881 sa svih 22 atributa u poređenju modela iz tačke 4. Korišćenje
samo najbitnijih atributa dakle **ne pogoršava** model — naprotiv, malo ga
poboljšava i smanjuje rizik od overfitting-a na ovom malom skupu, pa je
ovih 10 atributa korišćeno u finalnom modelu.

## 6. Podešavanje hiperparametara i analiza rezultata predikcije

Treniranje finalnog modela ([src/final_model.py](../src/final_model.py),
rezultati u [results/final_model/](../results/final_model/)) i evaluacija na
test skupu ([src/evaluate.py](../src/evaluate.py), rezultati u
[results/evaluation/](../results/evaluation/)) su odvojeni u dva samostalna
koraka: `final_model.py` trenira model, bira atribute i prag odlučivanja
isključivo na trening skupu i ne dotiče test skup ni na jedan način, a tek
nakon toga `evaluate.py` učitava već sačuvane artefakte iz `models/` i
evaluira ih na test skupu, bez ponovnog treniranja.

**Hiperparametri** Random Forest-a su podešeni pomoću `GridSearchCV` (5-fold
group CV, scoring = F1) nad gridom za `n_estimators`, `max_depth`,
`min_samples_split`, `min_samples_leaf`
(`hyperparameter_heatmap.png` prikazuje F1 za sve kombinacije
`n_estimators` × `max_depth`). Najbolja kombinacija:
`max_depth=None, min_samples_leaf=2, min_samples_split=5,
n_estimators=100` (Best CV F1-score = 0.871).

**Dve dodatne odluke, motivisane EDA i prirodom problema:**

- `class_weight="balanced"` — zbog neravnomernih klasa (147 vs 48), bez
  ovoga model favorizuje većinsku (Parkinson) klasu.
- **Decision threshold** se ne uzima podrazumevano kao 0.5, već se bira
  preko ROC krive i Youden indeksa, isključivo na osnovu out-of-fold
  predikcija na trening skupu (`cross_val_predict`, grupisano po osobi) —
  nikad na test skupu, da test labele ne "iscure" u izbor modela. Izabrani
  threshold: **0.769**.

**Rezultati na test skupu** (7 osoba, 43 snimka, nikada korišćeno tokom
selekcije modela/hiperparametara/threshold-a), dobijeni pokretanjem
`src/evaluate.py` kao posebnog koraka nakon treniranja:

| Metrika | Vrednost |
|---|---|
| Accuracy | 93.0% |
| Precision | 91.2% |
| Recall | 100% |
| F1-score | 95.4% |
| ROC-AUC | 97.0% |

Matrica konfuzije, ROC kriva i Precision-Recall kriva su u
[results/evaluation/](../results/evaluation/): `confusion_matrix.png`,
`roc_curve.png`, `precision_recall_curve.png`.

**Diskusija i ograničenja.** Recall od 100% na test skupu znači da model
nije propustio nijedan slučaj Parkinsona u testu, ali test skup ima samo 7
osoba — jedna pogrešno klasifikovana osoba bi promenila metrike za
desetak procentnih poena, pa su ove vrednosti procena sa visokom
varijansom. Da bi se to kvantifikovalo, [src/split_robustness.py](../src/split_robustness.py)
ponavlja čitav postupak (split → trening → evaluacija) za 7 različitih
`random_state` vrednosti pri podeli na osobe
(`results/final_model/split_robustness.csv/png`): Accuracy se kreće
0.67–0.88 (prosek ≈ 0.78, std ≈ 0.09), a F1-macro 0.40–0.83 — potvrđujući
da je tačkasta procena na jednom test skupu samo deo slike, i da je
pouzdaniji uvid u kvalitet modela 5-fold group CV rezultat iz tačke 4/5
(Best CV F1-score = 0.871).

**Bias-variance / over-underfitting.** [src/learning_curve.py](../src/learning_curve.py)
upoređuje train i CV F1-score finalnog modela u zavisnosti od veličine
trening skupa, sačuvano kao **tabela**, ne grafik
(`results/final_model/learning_curve.csv`) - sa samo 32 osobe, svaka tačka
je CV procena na grupama od ~5-6 osoba po fold-u (CV F1 std. devijacija po
veličini je 0.05-0.07, istog reda veličine kao razlike između susednih
tačaka), pa bi glatka linija na grafiku sugerisala čistiji trend nego što
ovaj dataset stvarno može pokazati.

Train F1 brzo dostiže ~0.99 (već sa 31 uzoraka), dok se CV F1 kreće
0.84–0.88 kroz čitav opseg veličina, sa blagim padom na najveće dve
veličine (0.846, 0.844 na 139/158 uzoraka) — razmak (~0.10-0.15 kroz ceo
opseg, ~0.15 na punom trening skupu) je znak **blagog overfitting-a**,
očekivan kod Random Forest-a (bagging) na ovoliko malom skupu. Bitno:
razmak je prisutan **kroz ceo opseg veličina**, ne samo na najvećoj — to
je znak da je strukturna karakteristika modela na ovom dataset-u, ne
nedostatak podataka koji bi nestao samo postepenim dodavanjem uzoraka iz
**istog**, već postojećeg skupa od 32 osobe. Probano je i sa jačom
regularizacijom (manji `max_depth`, veći `min_samples_leaf`): razmak se
ne smanjuje značajno (0.125 → 0.108), dok CV F1 pada (0.871 → 0.818) - kod
bagging metoda kao Random Forest, razmak delom dolazi iz same prirode
bagginga (svako stablo skoro savršeno klasifikuje sopstveni bootstrap
uzorak), pa ga je teško ukloniti bez žrtvovanja generalizacije.

## 7. Deployment modela

- **Eksport modela**: finalni `Pipeline` (StandardScaler + RandomForest),
  lista izabranih atributa i decision threshold se čuvaju pomoću `joblib`
  u `models/best_model.joblib`, `models/selected_features.joblib`,
  `models/decision_threshold.joblib` (videti `src/final_model.py`).
- **Korišćenje modela**: [src/predict.py](../src/predict.py) centralizuje
  učitavanje artefakata i logiku predikcije (`predict_from_features`), da
  se ne ponavlja na više mesta.
- **UI**: [app/ui.py](../app/ui.py) — Streamlit forma za unos glasovnih
  biomarkera, koja prikazuje predikciju i procenjenu verovatnoću,
  koristeći isti `predict.py` modul i iste sačuvane artefakte:

  ```bash
  streamlit run app/ui.py
  ```

## 8. Zaključak

Finalno rešenje — Random Forest sa 10 izabranih atributa, balansiranim
klasama i pažljivo (cross-validacijom na trening skupu) biranim decision
threshold-om — daje F1-score od ~0.87–0.95 (CV / test), uz recall od 100%
na test skupu, što je poželjno za ovaj tip problema (propušten pozitivan
slučaj je gore od lažnog alarma). Tri metodološke odluke su ovde imale
najveći uticaj na pouzdanost rezultata: grupna podela po subjektu (sprečava
curenje na nivou osobe), `class_weight="balanced"` (sprečava da model samo
predviđa većinsku klasu) i biranje threshold-a isključivo na trening
podacima (sprečava curenje test labela u model). Glavno ograničenje
projekta je veličina dataset-a (32 osobe) — zbog toga je svaka tačkasta
metrika praćena merom njene varijanse (CV raspon po fold-u, robusnost na
izbor split-a).
