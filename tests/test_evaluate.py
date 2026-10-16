"""Testes das métricas por centro/paciente e do formato das predições."""

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluate import build_metrics, group_metrics, patient_summary
from src.predict import to_records


def toy():
    """2 centros; o centro 1 só tem normais. Paciente 1 tem tumor, pacientes 2 e 3 não."""
    return pd.DataFrame({
        "id": range(8),
        "centre_id": [0, 0, 0, 0, 1, 1, 1, 1],
        "patient_id": [1, 1, 1, 2, 3, 3, 3, 3],
        "node_id": 0,
        "label": [1, 1, 0, 0, 0, 0, 0, 0],
        "prob": [0.9, 0.2, 0.1, 0.7, 0.1, 0.2, 0.6, 0.3],
    })


def test_centro_sem_tumor_so_tem_especificidade():
    t = group_metrics(toy(), "centre_id").set_index("centre_id")
    assert t.loc[1, "especificidade"] == pytest.approx(0.75)    # 3 de 4 normais certos
    assert np.isnan(t.loc[1, "recall_tumor"]) and np.isnan(t.loc[1, "auc"])
    assert t.loc[0, "recall_tumor"] == pytest.approx(0.5)       # 1 de 2 tumores
    assert t.loc[0, "especificidade"] == pytest.approx(0.5)     # 1 de 2 normais


def test_resumo_por_paciente_pesa_igual():
    s = patient_summary(toy())
    assert s["pacientes"] == 3 and s["pacientes_com_tumor"] == 1
    assert s["recall_medio_pacientes_tumor"] == pytest.approx(0.5)
    # especificidade de cada paciente: p1 = 1/1, p2 = 0/1, p3 = 3/4  -> média 0,5833
    assert s["especificidade_media_pacientes"] == pytest.approx((1 + 0 + 0.75) / 3)


def test_predicoes_e_metrics_json():
    df = toy().assign(fold=[0, 0, 0, 0, 1, 1, 1, 1], origem="cv")
    rec = to_records(df, "cnn")
    assert list(rec.columns) == ["id", "modelo", "origem", "fold", "prob_tumor", "pred"]
    assert rec["pred"].tolist() == [1, 0, 0, 1, 0, 0, 1, 0]
    m = build_metrics(rec, toy())
    json.dumps(m)  # tem que ser serializável (NaN vira None)
    cv = m["modelos"]["cnn"]["cv"]
    assert set(cv) >= {"por_patch", "por_paciente", "por_centro", "detalhe_pacientes"}
    assert len(cv["por_patch"]["f1_macro"]["folds"]) == 2
