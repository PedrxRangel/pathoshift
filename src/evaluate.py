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


# ---------------------------------------------------------------- por centro e por paciente

def group_metrics(df, by, threshold=0.5):
    """Métricas de cada grupo (centro ou paciente); grupo sem tumor só tem especificidade. Aula 08 – Classificação: avaliação.

    df precisa de: label, prob e a coluna `by`.
    """
    rows = []
    for key, g in df.groupby(by):
        y = g["label"].to_numpy()
        pred = g["prob"].to_numpy() >= threshold
        row = {by: key, "patches": len(g), "tumor": int(y.sum()),
               "especificidade": float(np.mean(~pred[y == 0])) if (y == 0).any() else np.nan,
               "recall_tumor": float(np.mean(pred[y == 1])) if (y == 1).any() else np.nan,
               "acuracia_balanceada": np.nan, "f1_macro": np.nan, "auc": np.nan}
        if len(np.unique(y)) == 2:
            m = compute_metrics(y, g["prob"], threshold)
            row.update(acuracia_balanceada=m["balanced_accuracy"], f1_macro=m["f1_macro"], auc=m["auc"])
        rows.append(row)
    return pd.DataFrame(rows)


def patient_summary(df, threshold=0.5):
    """Resumo em que cada PACIENTE pesa igual (e não cada patch). Aula 08 – Classificação: avaliação.

    - recall_medio_pacientes_tumor: média, entre os pacientes com tumor, do recall dos seus patches de tumor
    - especificidade_media_pacientes: média, entre os pacientes, da especificidade dos seus patches normais
    - acuracia_balanceada_pacientes: média das duas
    """
    t = group_metrics(df, "patient_id", threshold)
    rec = t["recall_tumor"].mean()      # NaN dos pacientes sem tumor é ignorado
    spec = t["especificidade"].mean()
    return {"pacientes": len(t), "pacientes_com_tumor": int((t["tumor"] > 0).sum()),
            "recall_medio_pacientes_tumor": rec, "especificidade_media_pacientes": spec,
            "acuracia_balanceada_pacientes": (rec + spec) / 2 if (t["tumor"] > 0).any() else np.nan}


PATIENT_METRICS = ["recall_medio_pacientes_tumor", "especificidade_media_pacientes", "acuracia_balanceada_pacientes"]


def _clean(x):
    """Converte números do NumPy para JSON (NaN vira null). Apoio."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), 4)
    return x


def _mean_std(values, keys):
    """{métrica: {media, desvio, folds}} a partir de uma lista de dicionários (um por fold). Apoio."""
    out = {}
    for k in keys:
        v = np.array([d[k] for d in values], dtype=float)
        ok = v[~np.isnan(v)]
        out[k] = {"media": ok.mean() if len(ok) else np.nan,
                  "desvio": ok.std(ddof=1) if len(ok) > 1 else np.nan, "folds": v.tolist()}
    return out


def model_report(preds, threshold=0.5):
    """Relatório completo de um modelo: CV (por patch, por paciente, por centro) e teste externo. Apoio.

    preds: predictions.jsonl de UM modelo já juntado com o manifest (label, patient_id, centre_id).
    """
    rep = {}
    cv = preds[preds["origem"] == "cv"]
    if len(cv):
        folds = [g for _, g in cv.groupby("fold")]
        all_m = compute_metrics(cv["label"], cv["prob"], threshold)
        rep["cv"] = {
            "por_patch": _mean_std([compute_metrics(g["label"], g["prob"], threshold) for g in folds], MAIN_METRICS),
            "por_paciente": _mean_std([patient_summary(g, threshold) for g in folds], PATIENT_METRICS),
            "folds_juntos": {k: all_m[k] for k in MAIN_METRICS},
            "matriz_confusao_folds_juntos": all_m["confusion_matrix"],
            "por_centro": group_metrics(cv, "centre_id", threshold).to_dict("records"),
            "detalhe_pacientes": group_metrics(cv, "patient_id", threshold).to_dict("records"),
        }
    te = preds[preds["origem"] == "teste_externo"]
    if len(te):
        m = compute_metrics(te["label"], te["prob"], threshold)
        rep["teste_externo"] = {
            "por_patch": {k: m[k] for k in MAIN_METRICS},
            "matriz_confusao": m["confusion_matrix"],
            "por_paciente": patient_summary(te, threshold),
            "detalhe_pacientes": group_metrics(te, "patient_id", threshold).to_dict("records"),
        }
    return _clean(rep)


def build_metrics(predictions, manifest, threshold=0.5):
    """metrics.json: junta as predições com os rótulos do manifest e gera o relatório de cada modelo. Apoio."""
    df = predictions.merge(manifest[["id", "label", "patient_id", "node_id", "centre_id"]], on="id")
    df = df.rename(columns={"prob_tumor": "prob"})
    return {"limiar": threshold,
            "modelos": {name: model_report(g, threshold) for name, g in df.groupby("modelo")}}
