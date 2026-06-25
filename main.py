"""
Ulazna tačka projekta za detekciju Parkinsonove bolesti.

Ceo pipeline je podeljen u pojedinačne skripte u src/, koje treba pokrenuti
ovim redosledom prilikom prvog podešavanja projekta:

    1. python src/data_analysis.py      - eksplorativna analiza podataka
    2. python src/train.py              - poređenje baznih modela (5-fold CV)
    3. python src/feature_selection.py  - rangiranje atributa, izbor najboljeg podskupa
    4. python src/final_model.py        - podešavanje hiperparametara i čuvanje
                                           finalnog modela, praga i atributa
                                           (trening, ne dotiče test skup)
    5. python src/evaluate.py           - evaluacija sačuvanog modela na test
                                           skupu (poseban korak od treniranja)
    6. python src/predict.py            - primer predikcija u konzoli
    7. streamlit run app/ui.py          - pokretanje veb interfejsa

Direktno pokretanje ove skripte provera da li je treniran model već
dostupan i, ako jeste, pokreće primer predikcija iz src/predict.py.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"


def main():
    if not MODEL_PATH.exists():
        print("No trained model found yet.")
        print("Run the training pipeline first:")
        print("  1. python src/feature_selection.py")
        print("  2. python src/final_model.py")
        return

    from predict import main as run_prediction_examples

    run_prediction_examples()


if __name__ == "__main__":
    main()
