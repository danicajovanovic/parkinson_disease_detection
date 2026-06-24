# Detekcija Parkinsonove bolesti

Mašinsko učenje za procenu rizika od Parkinsonove bolesti na osnovu glasovnih
snimaka, rađeno nad [UCI Parkinson's dataset-om](https://archive.ics.uci.edu/dataset/174/parkinsons).
Projekat je edukativan — model nije medicinsko sredstvo i njegov izlaz ne
treba tumačiti kao dijagnozu.

## Podaci

Dataset sadrži 195 snimaka glasa od 32 osobe (vidljivo iz `name` kolone,
npr. `phon_R01_S01_1` → osoba `S01`, sa ~6 snimaka po osobi), od kojih 24
imaju Parkinsonovu bolest, a 8 je zdravo. Svaki red je jedan snimak opisan
sa 22 akustičke karakteristike (jitter, shimmer, odnos šuma i tona,
nelinearne mere frekvencije...) i ciljnom promenljivom `status`
(1 = Parkinson, 0 = zdrava osoba). Pun opis kolona je u
[data/parkinsons.names](data/parkinsons.names).

Bitno je napomenuti da klase nisu ravnomerno raspoređene — 147 uzoraka je
Parkinson, a samo 48 zdravo — i to je direktno uticalo na to kako je model
treniran (videti odeljak [Metodologija](#metodologija)).

## Struktura projekta

```
app/      Streamlit interfejs
data/     Sirovi podaci
docs/     Specifikacija projekta
models/   Sačuvan model i prateći artefakti (izabrane karakteristike, threshold)
results/  Sve što pipeline generiše - metrike, grafici, izveštaji
src/      Skripte: analiza, treniranje, evaluacija, predikcija
```

## Pipeline

Projekat je podeljen u nekoliko skripti koje se pokreću jedna za drugom, iz
korena projekta:

```bash
python src/data_analysis.py      # eksplorativna analiza (statistike, korelacije, PCA, grafici u results/graphics)
python src/anomaly_detection.py  # KNN/LOF/Isolation Forest provera anomalija (results/anomaly_detection)
python src/train.py              # poređenje baznih modela, uključujući baseline (5-fold CV)
python src/feature_selection.py  # rangiranje i izbor karakteristika
python src/final_model.py        # tuning hiperparametara i treniranje finalnog modela
python src/evaluate.py           # evaluacija sačuvanog modela na test skupu
python src/learning_curve.py     # train vs CV F1-score u zavisnosti od veličine trening skupa
python src/split_robustness.py   # koliko rezultat zavisi od izbora train/test split-a
python src/predict.py            # par primera predikcije u konzoli
```

Svaka skripta upisuje svoje rezultate u odgovarajući poddirektorijum unutar
`results/`. `main.py` je samo prečica: ako model već postoji u `models/`,
pokreće primer predikcije; ako ne postoji, kaže ti koje skripte treba prvo
pokrenuti.

## Metodologija

Finalni model je Random Forest, podešen pomoću `GridSearchCV` (5-fold CV)
nad 10 najznačajnijih karakteristika: `PPE`, `MDVP:Flo(Hz)`, `spread1`,
`MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `NHR`, `RPDE`, `Jitter:DDP`,
`MDVP:Jitter(Abs)`, `spread2`.

Tri detalja su ovde namerno drugačija od "naivnog" pristupa.

Dataset sadrži 195 snimaka, ali samo od **32 osobe** — svaka osoba ima
nekoliko (oko 6) snimaka, sa istim statusom za sve svoje snimke. Da bi se
izbeglo curenje podataka na nivou osobe (model bi inače delom učio da
prepozna konkretnu osobu po glasu, umesto generalizovanog biomarkera
bolesti), train/test podela i sva cross-validacija (poređenje modela, izbor
karakteristika, tuning hiperparametara) se rade na nivou **osobe**, ne
snimka — koristeći stratifikovanu podelu po subjektima i
`StratifiedGroupKFold`. Posle ove ispravke su izveštene metrike primetno
niže nego sa naivnim (curećim) split-om po snimcima, ali realnije
predstavljaju kako bi se model ponašao na potpuno novoj osobi.

Pošto u dataset-u ima skoro tri puta više Parkinson nego zdravih uzoraka,
model koristi `class_weight="balanced"`, a pretraga hiperparametara
optimizuje F1-score umesto čistog recall-a. Bez ovoga model ima tendenciju
da skoro sve proglasi Parkinsonom — recall onda izgleda odlično, ali na
štetu zdrave klase, koja ispadne sistematski lošije prepoznata.

Decision threshold se ne uzima podrazumevano kao 0.5, već se bira preko ROC
krive i Youden indeksa — ali isključivo na osnovu out-of-fold predikcija na
trening skupu (`cross_val_predict`, grupisano po osobi), nikad na test
skupu. Razlog je jednostavan: kad bi se threshold birao gledajući test
labele, te labele bi efektivno "iscurele" u proces selekcije modela, a
izveštene metrike bi ispale veštački bolje nego što model zaista jeste.

Train/test podela je 80/20 **po osobi**, stratifikovana po klasama (25
osoba u treningu, 7 u testu). Poseban validation skup nije izdvojen jer
dataset ima samo 32 osobe — i izbor modela i izbor threshold-a se oslanjaju
na 5-fold group cross-validation nad trening skupom.

## Rezultati

Na test skupu, finalni model postiže:

| Metrika | Vrednost |
|---|---|
| Accuracy | 93.0% |
| Precision | 91.2% |
| Recall | 100% |
| F1-score | 95.4% |
| ROC-AUC | 97.0% |
| Decision threshold | 0.769 |

Test skup ima samo 7 osoba (43 snimka), pa su ove vrednosti procena sa
relativno visokom varijansom — pri ovako malom broju subjekata, jedna
pogrešno klasifikovana osoba menja metrike za nekoliko procentnih poena.
Pouzdaniji uvid u stabilnost modela daje 5-fold group cross-validation na
trening skupu (vidi `Best CV F1-score` u
[results/final_model/classification_report.txt](results/final_model/classification_report.txt)
i raspon po fold-ovima u
[results/model_comparison/cv_scores_boxplot.png](results/model_comparison/cv_scores_boxplot.png)).

Detaljniji izveštaj je u
[results/final_model/classification_report.txt](results/final_model/classification_report.txt),
poređenje baznih modela u
[results/model_comparison/model_comparison.csv](results/model_comparison/model_comparison.csv),
a kako broj izabranih karakteristika utiče na performanse u
[results/feature_selection/feature_subset_comparison.csv](results/feature_selection/feature_subset_comparison.csv).

Pošto test skup ima samo 7 osoba, [src/split_robustness.py](src/split_robustness.py)
dodatno proverava koliko su ove metrike osetljive na konkretan izbor
train/test split-a, ponavljajući split+trening+evaluaciju za 7 različitih
random seed-ova (videti [results/final_model/split_robustness.csv](results/final_model/split_robustness.csv)).

[src/learning_curve.py](src/learning_curve.py) prikazuje train vs CV
F1-score u zavisnosti od veličine trening skupa
([results/final_model/learning_curve.png](results/final_model/learning_curve.png)):
na punom trening skupu train F1 ≈ 0.99, CV F1 ≈ 0.84 — razmak od ~0.15
ukazuje na blagi overfitting, što je realno za Random Forest na skupu od
ovolike veličine, ali nije alarmantno (CV skor ostaje stabilan, ne pada sa
više podataka).

Baseline (model koji uvek predviđa većinsku klasu) ima F1 = 0.86 i
ROC-AUC = 0.50 — visok F1 baseline-a je posledica neravnomernih klasa
(accuracy paradox), zato je ROC-AUC ovde informativniji pokazatelj da
baseline ne razlikuje klase, dok svaki ozbiljan model treba da ga pobedi
po ROC-AUC. Svi bazni modeli (uključujući novododati Gradient Boosting /
XGBoost) su u
[results/model_comparison/model_comparison.csv](results/model_comparison/model_comparison.csv).

[src/anomaly_detection.py](src/anomaly_detection.py) primenjuje tri
nenadgledane metode (KNN distanca, Local Outlier Factor, Isolation Forest)
da provere postoje li snimci koji značajno odskaču od ostatka skupa. Sve
tri metode se slažu oko 4 snimka (osobe `S24` i `S35`,
[results/anomaly_detection/anomaly_overlap.txt](results/anomaly_detection/anomaly_overlap.txt)) -
nisu uklonjeni, jer kod ovako malog dataset-a izolovanost u prostoru
atributa pre liči na prirodnu varijaciju izraženosti bolesti nego na
grešku u merenju.

Potpuna dokumentacija projekta (opis problema, metodologija, rezultati i
diskusija po fazama specifikacije) je u
[docs/documentation.md](docs/documentation.md).

## Web aplikacija

```bash
streamlit run app/ui.py
```

Forma za unos glasovnih biomarkera, sa rezultatom predikcije i procenjenom
verovatnoćom. Koristi isti `predict.py` modul i iste sačuvane artefakte kao
i ostatak pipeline-a, samo umotane u Streamlit interfejs.

## Pokretanje

Zavisnosti se upravljaju preko [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Potreban je Python 3.14+ (videti `pyproject.toml`).
