"""Testes dos splits: pacientes em um único fold, centro de teste isolado e tumor em todo fold.

Os testes com dados sintéticos sempre rodam; o teste do manifest real só roda
se data/manifest.jsonl já existir (depois do `python pathoshift.py prepare`).
"""

import numpy as np
import pandas as pd
import pytest

from src.manifest import load_manifest
from src.splits import make_splits
from src.utils import ROOT, load_config, resolve_path


def fake_data(seed=0):
    """Dados sintéticos: 6 centros, 10 pacientes por centro, 2 lâminas por paciente."""
    rng = np.random.default_rng(seed)
    rows = []
    for centre in range(6):
        for p in range(10):
            has_tumor = p % 3 == 0  # 4 pacientes com tumor por centro
            for node in (0, 1):
                for _ in range(rng.integers(100, 200)):
                    rows.append((centre * 100 + p, node, centre, int(has_tumor and rng.random() < 0.5)))
    return pd.DataFrame(rows, columns=["patient_id", "node_id", "centre_id", "label"])


def check_splits(df, test_centre, n_folds, min_tumor):
    """As regras que os splits precisam respeitar."""
    dev = df[df["split"] == "dev"]

    # 1) Cada paciente aparece em um único fold (e nunca no dev e no teste ao mesmo tempo)
    folds_por_paciente = df.groupby("patient_id")["fold"].nunique()
    assert (folds_por_paciente == 1).all(), "Paciente em mais de um fold"

    # 2) O centro de teste está todo (e só ele) em test_ext, com fold -1
    ext = df[df["split"] == "test_ext"]
    assert set(ext["centre_id"]) == {test_centre}
    assert (ext["fold"] == -1).all()
    assert test_centre not in set(dev["centre_id"]), "Centro de teste aparece no dev"

    # 3) Todos os folds existem e têm tumor suficiente
    assert sorted(dev["fold"].unique()) == list(range(n_folds))
    tumor_por_fold = dev.groupby("fold")["label"].sum()
    assert (tumor_por_fold >= min_tumor).all(), f"Fold com pouco tumor: {tumor_por_fold.to_dict()}"
    pac_tumor_por_fold = dev[dev["label"] == 1].groupby("fold")["patient_id"].nunique()
    assert len(pac_tumor_por_fold) == n_folds and (pac_tumor_por_fold >= 1).all()


@pytest.mark.parametrize("test_centre", [0, 2, 4])
def test_splits_dados_sinteticos(test_centre):
    df = fake_data()
    df["split"], df["fold"] = make_splits(df, test_centre, n_folds=4, seed=42, min_tumor=50)
    check_splits(df, test_centre, n_folds=4, min_tumor=50)


def test_splits_reprodutiveis():
    df = fake_data()
    s1, f1 = make_splits(df, 2, seed=42)
    s2, f2 = make_splits(df, 2, seed=42)
    assert s1.equals(s2) and f1.equals(f2)


def test_centro_inexistente():
    with pytest.raises(ValueError):
        make_splits(fake_data(), test_centre=99)


def test_manifest_real():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    path = resolve_path(cfg["paths"]["manifest"])
    if not path.exists():
        pytest.skip("data/manifest.jsonl não existe; rode `python pathoshift.py prepare`")
    sp = cfg["splits"]
    check_splits(load_manifest(path), sp["test_centre"], sp["n_folds"], sp["min_tumor_patches_per_fold"])
