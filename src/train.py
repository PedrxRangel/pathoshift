"""Treino, validação cruzada e modelo final dos dois modelos: regressão logística (features) e SimpleCNN.

Protocolo:
- Validação cruzada por paciente (4 folds) no conjunto dev. O fold de validação só MEDE:
  nenhuma escolha (época, hiperparâmetro) olha para ele.
- CNN: dentro do treino de cada fold, ~1/5 dos pacientes (estratificado) formam o conjunto
  de PARADA, usado só para o early stopping (splits.early_stopping_split).
- Modelo final: treinado no dev inteiro (a CNN separa de novo pacientes do dev para a parada).
  O teste externo (centro 2) é feito depois, pelo comando `predict --teste-externo`.

Saídas em outputs/<modelo>/:
  cv_oof.csv        probabilidade de tumor de cada patch do dev quando ele estava na validação
  cv_folds.csv      métricas por patch de cada fold + média e desvio
  cv_history.csv    (CNN) loss de treino, F1 macro do conjunto de parada e tempo de cada época
  cv_info.json      pacientes de parada, melhor época e tempo de cada fold
  fold<k>.pt        (CNN) melhor checkpoint do fold k
  final.pt / final.joblib   modelo final (treinado no dev inteiro)
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src import utils
from src.data import PatchDataset, load_hf_dataset
from src.evaluate import compute_metrics, fold_table, mean_std_text, print_metrics
from src.features import feature_columns, load_features
from src.manifest import load_manifest
from src.models import SimpleCNN, build_logreg
from src.splits import early_stopping_split


# ---------------------------------------------------------------- CNN

def class_weights(labels):
    """Peso de cada classe = N / (2 * N_classe): a classe rara (tumor) pesa mais na loss. Aula 08 – Classificação: desbalanceamento."""
    counts = np.bincount(labels, minlength=2)
    return torch.tensor(len(labels) / (2 * counts), dtype=torch.float32)


def make_loader(hf_ds, df, cfg, train):
    """DataLoader do PyTorch (embaralha só no treino; semente fixa). Apoio."""
    c = cfg["cnn"]
    ds = PatchDataset(hf_ds, df, c["image_size"], train=train)
    return torch.utils.data.DataLoader(
        ds, batch_size=c["batch_size"], shuffle=train, num_workers=c["num_workers"],
        generator=torch.Generator().manual_seed(cfg["seed"]),
        persistent_workers=c["num_workers"] > 0,
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Uma época: para cada lote calcula a loss, faz o backpropagation e atualiza os pesos. Aula 08 – Classificação: treino da CNN."""
    model.train()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc="treino", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def predict_proba(model, loader, device):
    """Probabilidade de tumor (softmax da classe 1) para cada patch do loader. Aula 08 – Classificação: predição."""
    model.eval()
    probs = []
    for x, _ in tqdm(loader, desc="predição", leave=False):
        probs.append(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def fit_cnn(hf_ds, fit_df, stop_df, cfg, ckpt_path, max_epochs):
    """Treina a SimpleCNN (CrossEntropy ponderada + Adam) com early stopping pela F1 macro do conjunto de PARADA. Aula 08 – Classificação: CNN.

    fit_df: patches usados para ajustar os pesos; stop_df: pacientes separados só para decidir a parada.
    Devolve (modelo com os pesos da melhor época, histórico por época).
    """
    device = utils.get_device()
    c = cfg["cnn"]
    model = SimpleCNN(num_classes=2, image_size=c["image_size"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(fit_df["label"].to_numpy()).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=c["lr"])
    fit_loader = make_loader(hf_ds, fit_df, cfg, train=True)
    stop_loader = make_loader(hf_ds, stop_df, cfg, train=False)

    history, best_f1, best_epoch = [], -1.0, 0
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        loss = train_one_epoch(model, fit_loader, optimizer, criterion, device)
        m = compute_metrics(stop_df["label"], predict_proba(model, stop_loader, device))
        row = {"epoch": epoch, "train_loss": loss, "parada_f1_macro": m["f1_macro"],
               "parada_auc": m["auc"], "tempo_s": time.time() - start}
        history.append(row)
        print("  " + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

        # Early stopping: guarda o melhor modelo e para após `patience` épocas sem melhora
        if m["f1_macro"] > best_f1:
            best_f1, best_epoch = m["f1_macro"], epoch
            torch.save(model.state_dict(), ckpt_path)
        elif epoch - best_epoch >= c["patience"]:
            print(f"  early stopping na época {epoch} (melhor: {best_epoch})")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    hist = pd.DataFrame(history)
    hist["best_epoch"] = best_epoch
    return model, hist


def load_cnn(cfg, ckpt_path):
    """Recria a SimpleCNN e carrega os pesos de um checkpoint. Apoio."""
    model = SimpleCNN(num_classes=2, image_size=cfg["cnn"]["image_size"]).to(utils.get_device())
    model.load_state_dict(torch.load(ckpt_path, map_location=utils.get_device()))
    return model


# ---------------------------------------------------------------- treino genérico (os dois modelos)

def load_data(cfg, model_name):
    """Manifest (CNN) ou manifest + features (regressão logística), e o dataset HF se for a CNN. Apoio."""
    man = load_manifest(utils.resolve_path(cfg["paths"]["manifest"]))
    if model_name == "logreg":
        man = load_features(utils.resolve_path(cfg["paths"]["features"]), man)
    hf_ds = load_hf_dataset(cfg) if model_name == "cnn" else None
    return man, hf_ds


def train_model(cfg, model_name, train_df, hf_ds, ckpt_path, max_epochs=None):
    """Treina o modelo em train_df e devolve (modelo, info). Na CNN, separa os pacientes de parada. Apoio."""
    if model_name == "logreg":
        cols = feature_columns(train_df)
        return build_logreg(cfg).fit(train_df[cols], train_df["label"]), {}

    sp = cfg["splits"]
    is_stop = early_stopping_split(train_df, sp["es_n_splits"], cfg["seed"], sp["es_min_tumor"])
    fit_df, stop_df = train_df[~is_stop], train_df[is_stop]
    print(f"  ajuste: {len(fit_df)} patches | parada: {len(stop_df)} patches, "
          f"pacientes {sorted(stop_df['patient_id'].unique().tolist())} "
          f"(com tumor: {sorted(stop_df.loc[stop_df['label'] == 1, 'patient_id'].unique().tolist())})")
    model, hist = fit_cnn(hf_ds, fit_df, stop_df, cfg, ckpt_path, max_epochs or cfg["cnn"]["max_epochs"])
    info = {
        "pacientes_parada": sorted(stop_df["patient_id"].unique().tolist()),
        "pacientes_parada_com_tumor": sorted(stop_df.loc[stop_df["label"] == 1, "patient_id"].unique().tolist()),
        "patches_ajuste": len(fit_df), "patches_parada": len(stop_df),
        "melhor_epoca": int(hist["best_epoch"].iloc[0]), "epocas_treinadas": len(hist),
        "history": hist,
    }
    return model, info


def predict_model(cfg, model_name, model, df, hf_ds):
    """Probabilidade de tumor de cada patch de df. Aula 08 – Classificação: predição."""
    if model_name == "logreg":
        return model.predict_proba(df[feature_columns(df)])[:, 1]
    return predict_proba(model, make_loader(hf_ds, df, cfg, train=False), utils.get_device())


def run_cv(cfg, model_name, epochs=None):
    """Validação cruzada por paciente no dev; o fold de validação só é usado para medir. Apoio: metodologia."""
    utils.set_seed(cfg["seed"])
    out_dir = utils.resolve_path(cfg["paths"]["outputs_dir"]) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    man, hf_ds = load_data(cfg, model_name)
    dev = man[man["split"] == "dev"]

    metrics, oof, histories, infos = [], [], [], {}
    for k in sorted(dev["fold"].unique()):
        tr, va = dev[dev["fold"] != k], dev[dev["fold"] == k]
        print(f"\n--- {model_name} | fold {k}: treino {len(tr)} patches, validação {len(va)} ---")
        start = time.time()
        model, info = train_model(cfg, model_name, tr, hf_ds, out_dir / f"fold{k}.pt", epochs)
        prob = predict_model(cfg, model_name, model, va, hf_ds)  # só agora o fold k é usado
        m = compute_metrics(va["label"], prob)
        print_metrics(m, f"fold {k} ({time.time() - start:.1f} s)")
        metrics.append(m)
        oof.append(va[["id", "fold"]].assign(prob=prob))
        if "history" in info:
            histories.append(info.pop("history").assign(fold=k))
        infos[int(k)] = {**info, "segundos": round(time.time() - start, 1)}

    table = fold_table(metrics)
    table.to_csv(out_dir / "cv_folds.csv")
    pd.concat(oof).to_csv(out_dir / "cv_oof.csv", index=False)
    if histories:
        pd.concat(histories).to_csv(out_dir / "cv_history.csv", index=False)
    (out_dir / "cv_info.json").write_text(json.dumps(infos, indent=2), encoding="utf-8")
    print(f"\n=== {model_name}: média ± desvio em {len(metrics)} folds (por patch) ===")
    print(mean_std_text(table).to_string())
    return table


def final_path(cfg, model_name):
    """Arquivo do modelo final: outputs/logreg/final.joblib ou outputs/cnn/final.pt. Apoio."""
    out_dir = utils.resolve_path(cfg["paths"]["outputs_dir"]) / model_name
    return out_dir / ("final.joblib" if model_name == "logreg" else "final.pt")


def run_final(cfg, model_name, epochs=None):
    """Treina o modelo final no dev inteiro e salva; NÃO olha o teste externo. Apoio: metodologia."""
    utils.set_seed(cfg["seed"])
    path = final_path(cfg, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    man, hf_ds = load_data(cfg, model_name)
    dev = man[man["split"] == "dev"]
    model, info = train_model(cfg, model_name, dev, hf_ds, path, epochs)
    if model_name == "logreg":
        joblib.dump(model, path)
    if "history" in info:
        info.pop("history").to_csv(path.parent / "final_history.csv", index=False)
    (path.parent / "final_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Modelo final salvo em {path}")
    return info


def load_final(cfg, model_name):
    """Carrega o modelo final salvo por run_final. Apoio."""
    path = final_path(cfg, model_name)
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe; rode `python pathoshift.py train --model {model_name} --final`")
    return joblib.load(path) if model_name == "logreg" else load_cnn(cfg, path)
