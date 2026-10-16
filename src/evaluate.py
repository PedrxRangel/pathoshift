"""Métricas de avaliação para classificação binária (0 = normal, 1 = tumor)."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

CLASS_NAMES = ["normal", "tumor"]
MAIN_METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "auc",
                "precision_normal", "recall_normal", "f1_normal",
                "precision_tumor", "recall_tumor", "f1_tumor"]


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Métricas por classe e macro, acurácia balanceada, AUC e matriz de confusão. Aula 08 – Classificação: avaliação.

    y_prob é a probabilidade de tumor; o patch é classificado como tumor se y_prob >= threshold.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    # Acurácia balanceada e AUC só fazem sentido se as duas classes aparecem em y_true
    duas_classes = len(np.unique(y_true)) == 2
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if duas_classes else float("nan"),
        "f1_macro": f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if duas_classes else float("nan"),
    }
    for i, name in enumerate(CLASS_NAMES):
        m[f"precision_{name}"] = prec[i]
        m[f"recall_{name}"] = rec[i]
        m[f"f1_{name}"] = f1[i]
    m["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return m


def print_metrics(m, title=""):
    """Imprime as métricas principais e a matriz de confusão. Apoio."""
    if title:
        print(f"== {title} ==")
    for k in MAIN_METRICS:
        print(f"  {k:18s} {m[k]:.4f}")
    print("  matriz de confusão (linhas = verdadeiro, colunas = predito):")
    print("  " + str(m["confusion_matrix"]).replace("\n", "\n  "))


def fold_table(metrics_per_fold):
    """Tabela com as métricas de cada fold e as linhas 'média' e 'desvio'. Apoio: validação cruzada."""
    df = pd.DataFrame([{k: m[k] for k in MAIN_METRICS} for m in metrics_per_fold])
    df.index.name = "fold"
    df.loc["média"] = df.mean()
    df.loc["desvio"] = df.iloc[:-1].std()
    return df


def mean_std_text(table):
    """Resumo 'média ± desvio' de cada métrica, a partir da tabela de fold_table. Apoio."""
    return pd.Series({k: f"{table.loc['média', k]:.3f} ± {table.loc['desvio', k]:.3f}" for k in MAIN_METRICS})
