"""
Zajedničke pomoćne funkcije za evaluaciju koje koriste skripte za poređenje
modela, izbor atributa i finalni model, da definicija metrika ostane na
jednom mestu.
"""

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def scores_to_row(scores):
    """
    Pretvara izlaz sklearn cross_validate() funkcije u flat dict sa
    prosečnim vrednostima metrika.
    """
    return {
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1-score": scores["test_f1"].mean(),
        "ROC-AUC": scores["test_roc_auc"].mean(),
    }
