"""Geração e leitura do manifest.jsonl (uma linha JSON por patch).

Campos: id, patient_id, node_id, centre_id, x, y, label, split, fold
  split: 'dev' (desenvolvimento, com validação cruzada) ou 'test_ext' (centro de teste)
  fold:  0 a n_folds-1 no dev; -1 no test_ext
"""

import pandas as pd

MANIFEST_COLUMNS = ["id", "patient_id", "node_id", "centre_id", "x", "y", "label", "split", "fold"]


def build_manifest(meta):
    """Renomeia as colunas do HF para o padrão do projeto (split e fold são preenchidos depois). Apoio."""
    df = meta.rename(columns={"patch_x": "x", "patch_y": "y", "tumor": "label"})
    df["split"] = ""
    df["fold"] = -1
    return df[MANIFEST_COLUMNS]


def save_manifest(df, path):
    """Salva o manifest em JSON Lines. Apoio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True)


def load_manifest(path):
    """Lê o manifest. Apoio."""
    return pd.read_json(path, lines=True)


def summarize(df, by):
    """Resumo por grupo: patches, tumor, pacientes e lâminas (total e com tumor). Apoio: estatística descritiva."""
    rows = []
    for key, g in df.groupby(by):
        slides = g.groupby(["patient_id", "node_id"])["label"].max()  # 1 se a lâmina tem tumor
        patients = g.groupby("patient_id")["label"].max()
        rows.append({
            by: key,
            "patches": len(g),
            "tumor": int(g["label"].sum()),
            "tumor_%": round(100 * g["label"].mean(), 1),
            "pacientes": len(patients),
            "pac_c_tumor": int(patients.sum()),
            "laminas": len(slides),
            "lam_c_tumor": int(slides.sum()),
        })
    return pd.DataFrame(rows).set_index(by)
