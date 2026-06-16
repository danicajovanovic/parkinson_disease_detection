# Detekcija Parkinsonove bolesti

Mašinsko učenje za procenu rizika od Parkinsonove bolesti na osnovu glasovnih
snimaka, rađeno nad [UCI Parkinson's dataset-om](https://archive.ics.uci.edu/dataset/174/parkinsons).
Projekat je edukativan — model nije medicinsko sredstvo i njegov izlaz ne
treba tumačiti kao dijagnozu.

## Podaci

Dataset sadrži 195 snimaka glasa od 31 osobe, od kojih 23 imaju Parkinsonovu
bolest. Svaki red je jedan snimak opisan sa 22 akustičke karakteristike
(jitter, shimmer, odnos šuma i tona, nelinearne mere frekvencije...) i
ciljnom promenljivom `status` (1 = Parkinson, 0 = zdrava osoba). Pun opis
kolona je u [data/parkinsons.names](data/parkinsons.names).

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
python src/data_analysis.py      # eksplorativna analiza
python src/train.py              # poređenje nekoliko baznih modela (5-fold CV)
python src/feature_selection.py  # rangiranje i izbor karakteristika
python src/final_model.py        # tuning hiperparametara i treniranje finalnog modela
python src/evaluate.py           # evaluacija sačuvanog modela na test skupu
python src/predict.py            # par primera predikcije u konzoli
```

Svaka skripta upisuje svoje rezultate u odgovarajući poddirektorijum unutar
`results/`. `main.py` je samo prečica: ako model već postoji u `models/`,
pokreće primer predikcije; ako ne postoji, kaže ti koje skripte treba prvo
pokrenuti.

## Metodologija

Finalni model je Random Forest, podešen pomoću `GridSearchCV` (5-fold CV)
nad 9 najznačajnijih karakteristika: `PPE`, `spread1`, `MDVP:Fo(Hz)`, `NHR`,
`Jitter:DDP`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`, `spread2`, `Shimmer:APQ5`.

Dva detalja su ovde namerno drugačija od "naivnog" pristupa.

Pošto u dataset-u ima skoro tri puta više Parkinson nego zdravih uzoraka,
model koristi `class_weight="balanced"`, a pretraga hiperparametara
optimizuje F1-score umesto čistog recall-a. Bez ovoga model ima tendenciju
da skoro sve proglasi Parkinsonom — recall onda izgleda odlično, ali na
štetu zdrave klase, koja ispadne sistematski lošije prepoznata.

Decision threshold se ne uzima podrazumevano kao 0.5, već se bira preko ROC
krive i Youden indeksa — ali isključivo na osnovu out-of-fold predikcija na
trening skupu (`cross_val_predict`), nikad na test skupu. Razlog je
jednostavan: kad bi se threshold birao gledajući test labele, te labele bi
efektivno "iscurele" u proces selekcije modela, a izveštene metrike bi
ispale veštački bolje nego što model zaista jeste.

Train/test podela je 80/20, stratifikovana po klasama. Poseban validation
skup nije izdvojen jer dataset ima samo 195 uzoraka — i izbor modela i izbor
threshold-a se oslanjaju na 5-fold cross-validation nad trening skupom.

## Rezultati

Na test skupu, finalni model postiže:

| Metrika | Vrednost |
|---|---|
| Accuracy | 92.3% |
| Precision | 93.3% |
| Recall | 96.6% |
| F1-score | 94.9% |
| ROC-AUC | 97.9% |
| Decision threshold | 0.447 |

Detaljniji izveštaj je u
[results/final_model/classification_report.txt](results/final_model/classification_report.txt),
poređenje baznih modela u
[results/model_comparison/model_comparison.csv](results/model_comparison/model_comparison.csv),
a kako broj izabranih karakteristika utiče na performanse u
[results/feature_selection/feature_subset_comparison.csv](results/feature_selection/feature_subset_comparison.csv).

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
