"""Divisão dos dados: teste externo + validação cruzada por paciente.

- test_ext: um centro inteiro (padrão: centro 2) fica de fora e só é usado UMA vez, no final.
- dev (demais centros): StratifiedGroupKFold com grupo = patient_id. Todos os patches
  de um paciente ficam no mesmo fold, então o modelo nunca vê na validação um paciente
  com o qual treinou.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def assign_folds(dev, n_folds=4, seed=42, min_tumor=100, max_tries=50):
    """Devolve o fold (0 a n_folds-1) de cada patch, estratificado pelo rótulo e agrupado por paciente. Apoio: validação cruzada (metodologia).

    Como poucos pacientes têm tumor, a divisão é refeita com outra semente
    (seed, seed+1, ...) até todo fold ter pelo menos `min_tumor` patches de tumor.
    """
    y = dev["label"].to_numpy()
    for attempt in range(max_tries):
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + attempt)
        fold = np.full(len(dev), -1)
        for k, (_, val_idx) in enumerate(sgkf.split(dev, y, groups=dev["patient_id"])):
            fold[val_idx] = k
        tumor_por_fold = np.bincount(fold, weights=y, minlength=n_folds)
        if tumor_por_fold.min() >= min_tumor:
            return pd.Series(fold, index=dev.index, name="fold")
    raise RuntimeError("Não foi possível ter tumor em todos os folds; reduza n_folds ou min_tumor.")


def early_stopping_split(train_df, n_splits=5, seed=42, min_tumor=100, max_tries=50):
    """Marca ~1/n_splits dos PACIENTES do treino (estratificado pelo rótulo) como conjunto de parada do early stopping. Apoio: metodologia.

    O conjunto de parada sai só do treino do fold; assim o fold de validação nunca
    influencia o treino e serve apenas para medir. Escolhe o primeiro grupo de pacientes
    com pelo menos `min_tumor` patches de tumor (e deixa pelo menos isso no treino).
    Devolve uma Series booleana (True = conjunto de parada) com o índice de train_df.
    """
    y = train_df["label"].to_numpy()
    for attempt in range(max_tries):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + attempt)
        for _, es_idx in sgkf.split(train_df, y, groups=train_df["patient_id"]):
            is_es = np.zeros(len(train_df), dtype=bool)
            is_es[es_idx] = True
            if y[is_es].sum() >= min_tumor and y[~is_es].sum() >= min_tumor:
                return pd.Series(is_es, index=train_df.index, name="parada")
    raise RuntimeError("Não foi possível montar um conjunto de parada com tumor; reduza es_min_tumor.")


def make_splits(df, test_centre, n_folds=4, seed=42, min_tumor=100):
    """Devolve (split, fold): split é 'dev' ou 'test_ext'; fold é 0..n_folds-1 no dev e -1 no teste externo. Apoio: metodologia."""
    if test_centre not in set(df["centre_id"]):
        raise ValueError(f"Centro de teste {test_centre} não existe nos dados.")
    is_ext = df["centre_id"] == test_centre
    split = pd.Series(np.where(is_ext, "test_ext", "dev"), index=df.index, name="split")
    fold = pd.Series(-1, index=df.index, name="fold")
    fold[~is_ext] = assign_folds(df[~is_ext], n_folds, seed, min_tumor)
    return split, fold
