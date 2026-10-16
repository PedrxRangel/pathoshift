"""Testes da agregação por lâmina."""

import pandas as pd
import pytest

from src.aggregate import aggregate_slides, max_score, mean_score, topk_score


def test_scores_simples():
    p = [0.1, 0.2, 0.9, 0.4]
    assert mean_score(p) == pytest.approx(0.4)
    assert max_score(p) == 0.9
    assert topk_score(p, k=2) == pytest.approx(0.65)
    assert topk_score(p, k=10) == pytest.approx(0.4)  # k maior que o nº de patches = média


def test_aggregate_slides():
    df = pd.DataFrame({
        "centre_id": [0, 0, 0, 1, 1],
        "patient_id": [1, 1, 1, 7, 7],
        "node_id": [0, 0, 1, 0, 0],
        "label": [0, 1, 0, 0, 0],
        "prob": [0.2, 0.8, 0.1, 0.3, 0.5],
    })
    out = aggregate_slides(df, "max").set_index(["patient_id", "node_id"])
    assert len(out) == 3
    assert out.loc[(1, 0), "label"] == 1 and out.loc[(1, 0), "score"] == 0.8
    assert out.loc[(1, 0), "n_patches"] == 2
    assert out.loc[(7, 0), "label"] == 0
    assert aggregate_slides(df, "mean").set_index(["patient_id", "node_id"]).loc[(7, 0), "score"] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        aggregate_slides(df, "mediana")
