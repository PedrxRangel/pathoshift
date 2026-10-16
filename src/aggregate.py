"""Agregação das predições dos patches em uma predição por lâmina (patient_id, node_id).

Não vem de uma aula específica: são estatísticas simples (média, máximo, top-k)
sobre as probabilidades que o classificador deu a cada patch da lâmina.
Funções prontas e testadas (tests/test_aggregate.py); os experimentos ficam para depois.
"""

import numpy as np
import pandas as pd


def mean_score(probs):
    """Média das probabilidades dos patches. Apoio: agregação por lâmina (não é de aula)."""
    return float(np.mean(probs))


def max_score(probs):
    """Maior probabilidade entre os patches (basta 1 patch suspeito). Apoio: agregação por lâmina (não é de aula)."""
    return float(np.max(probs))


def topk_score(probs, k=10):
    """Média das k maiores probabilidades (meio-termo entre média e máximo). Apoio: agregação por lâmina (não é de aula)."""
    probs = np.sort(np.asarray(probs))[::-1]
    return float(probs[:k].mean())


def aggregate_slides(pred_df, method="mean", k=10):
    """Uma linha por lâmina com o score agregado e o rótulo da lâmina (1 se tiver algum patch de tumor). Apoio: agregação por lâmina.

    pred_df precisa das colunas: centre_id, patient_id, node_id, label, prob.
    """
    funcs = {"mean": mean_score, "max": max_score, "topk": lambda p: topk_score(p, k)}
    if method not in funcs:
        raise ValueError(f"Método de agregação desconhecido: {method}")
    rows = []
    for (centre, patient, node), g in pred_df.groupby(["centre_id", "patient_id", "node_id"]):
        rows.append({
            "centre_id": centre, "patient_id": patient, "node_id": node,
            "n_patches": len(g),
            "label": int(g["label"].max()),
            "score": funcs[method](g["prob"].to_numpy()),
        })
    return pd.DataFrame(rows)
