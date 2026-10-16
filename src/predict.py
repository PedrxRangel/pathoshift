"""Montagem do outputs/predictions.jsonl (uma linha JSON por patch e por modelo).

Formato de cada linha:
    {"id": 123, "modelo": "cnn", "origem": "cv", "fold": 2, "prob_tumor": 0.8731, "pred": 1}
  id          índice do patch no dataset do Hugging Face (o mesmo do manifest)
  modelo      "logreg" ou "cnn"
  origem      "cv" (predição de validação cruzada) ou "teste_externo" (centro 2)
  fold        fold de validação (0 a 3) na CV; -1 no teste externo
  prob_tumor  probabilidade de tumor dada pelo modelo
  pred        1 se prob_tumor >= limiar (0,5), senão 0

O arquivo NÃO tem os rótulos: o `evaluate` junta com o manifest pelo id.
Se o enunciado pedir outro formato, só esta função (`to_records`) precisa mudar.
"""

import pandas as pd

from src import utils
from src.train import load_data, load_final, predict_model

MODELS = ["logreg", "cnn"]


def cv_predictions(cfg, model_name):
    """Predições de validação cruzada salvas pelo `train` (outputs/<modelo>/cv_oof.csv). Apoio."""
    path = utils.resolve_path(cfg["paths"]["outputs_dir"]) / model_name / "cv_oof.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe; rode `python pathoshift.py train --model {model_name}`")
    return pd.read_csv(path).assign(origem="cv")


def test_predictions_path(cfg, model_name):
    """Arquivo com as predições do teste externo de um modelo. Apoio."""
    return utils.resolve_path(cfg["paths"]["outputs_dir"]) / model_name / "teste_externo.csv"


def predict_test(cfg, model_name):
    """Aplica o modelo FINAL ao centro de teste externo e salva as probabilidades. Aula 08 – Classificação: predição."""
    utils.set_seed(cfg["seed"])
    man, hf_ds = load_data(cfg, model_name)
    test = man[man["split"] == "test_ext"]
    prob = predict_model(cfg, model_name, load_final(cfg, model_name), test, hf_ds)
    out = test[["id"]].assign(fold=-1, prob=prob)
    out.to_csv(test_predictions_path(cfg, model_name), index=False)
    return out


def to_records(df, model_name, threshold=0.5):
    """Converte (id, fold, prob, origem) para as linhas do predictions.jsonl. Apoio."""
    return pd.DataFrame({
        "id": df["id"].astype(int), "modelo": model_name, "origem": df["origem"],
        "fold": df["fold"].astype(int), "prob_tumor": df["prob"].astype(float),  # sem arredondar: evita empates na AUC
        "pred": (df["prob"] >= threshold).astype(int),
    })


def build_predictions(cfg, threshold=0.5):
    """Junta CV + teste externo (se já existir) dos dois modelos num único DataFrame. Apoio."""
    parts = []
    for name in MODELS:
        parts.append(to_records(cv_predictions(cfg, name), name, threshold))
        tpath = test_predictions_path(cfg, name)
        if tpath.exists():
            parts.append(to_records(pd.read_csv(tpath).assign(origem="teste_externo"), name, threshold))
    return pd.concat(parts, ignore_index=True)


def save_predictions(df, path):
    """Grava em JSON Lines. Apoio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True)


def load_predictions(path):
    """Lê o predictions.jsonl. Apoio."""
    return pd.read_json(path, lines=True)
