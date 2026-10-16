"""Treino e validação cruzada dos dois modelos: regressão logística (features) e SimpleCNN.

Saídas em outputs/<modelo>/:
  cv_folds.csv       métricas de cada fold + média e desvio
  cv_oof.csv         predição de cada patch do dev quando ele estava na validação
  cv_history.csv     (CNN) loss, F1 e tempo de cada época de cada fold
  fold<k>.pt         (CNN) melhor checkpoint do fold k
  final_*            treino no dev inteiro + teste único no centro externo
"""

import time

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


def fit_cnn(hf_ds, train_df, val_df, cfg, ckpt_path, max_epochs):
    """Treina a SimpleCNN com CrossEntropy ponderada e Adam; early stopping pela F1 macro da validação. Aula 08 – Classificação: CNN.

    Se val_df for None (treino final), treina exatamente max_epochs épocas.
    Devolve (modelo com os melhores pesos, histórico por época).
    """
    device = utils.get_device()
    c = cfg["cnn"]
    model = SimpleCNN(num_classes=2, image_size=c["image_size"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_df["label"].to_numpy()).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=c["lr"])
    train_loader = make_loader(hf_ds, train_df, cfg, train=True)
    val_loader = make_loader(hf_ds, val_df, cfg, train=False) if val_df is not None else None

    history, best_f1, best_epoch = [], -1.0, 0
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        row = {"epoch": epoch, "train_loss": loss}
        if val_loader is not None:
            m = compute_metrics(val_df["label"], predict_proba(model, val_loader, device))
            row.update(val_f1_macro=m["f1_macro"], val_auc=m["auc"])
        row["tempo_s"] = time.time() - start
        history.append(row)
        print("  " + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

        # Early stopping: guarda o melhor modelo e para após `patience` épocas sem melhora
        f1 = row.get("val_f1_macro", epoch)  # sem validação, a última época é a "melhor"
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            torch.save(model.state_dict(), ckpt_path)
        elif epoch - best_epoch >= c["patience"]:
            print(f"  early stopping na época {epoch} (melhor: {best_epoch})")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    hist = pd.DataFrame(history)
    hist["best_epoch"] = best_epoch
    return model, hist


# ---------------------------------------------------------------- validação cruzada e teste final

def _load_dev_and_test(cfg, model_name):
    """Carrega o manifest (CNN) ou manifest + features (regressão logística). Apoio."""
    man = load_manifest(utils.resolve_path(cfg["paths"]["manifest"]))
    if model_name == "logreg":
        man = load_features(utils.resolve_path(cfg["paths"]["features"]), man)
    return man[man["split"] == "dev"], man[man["split"] == "test_ext"]


def _fit_predict(cfg, model_name, train_df, eval_df, hf_ds, out_dir, tag, epochs):
    """Treina em train_df e devolve (probabilidades em eval_df, histórico ou None). Apoio."""
    if model_name == "logreg":
        cols = feature_columns(train_df)
        clf = build_logreg(cfg).fit(train_df[cols], train_df["label"])
        return clf.predict_proba(eval_df[cols])[:, 1], None
    val_df = eval_df if tag.startswith("fold") else None  # no treino final não há validação
    model, hist = fit_cnn(hf_ds, train_df, val_df, cfg, out_dir / f"{tag}.pt", epochs)
    loader = make_loader(hf_ds, eval_df, cfg, train=False)
    return predict_proba(model, loader, utils.get_device()), hist


def run_cv(cfg, model_name, epochs=None):
    """Validação cruzada por paciente no conjunto dev; imprime e salva média ± desvio entre os folds. Apoio: metodologia."""
    utils.set_seed(cfg["seed"])
    out_dir = utils.resolve_path(cfg["paths"]["outputs_dir"]) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dev, _ = _load_dev_and_test(cfg, model_name)
    hf_ds = load_hf_dataset(cfg) if model_name == "cnn" else None
    epochs = epochs or cfg["cnn"]["max_epochs"]

    metrics, oof, histories = [], [], []
    for k in sorted(dev["fold"].unique()):
        tr, va = dev[dev["fold"] != k], dev[dev["fold"] == k]
        print(f"\n--- {model_name} | fold {k}: treino {len(tr)} patches, validação {len(va)} ---")
        start = time.time()
        prob, hist = _fit_predict(cfg, model_name, tr, va, hf_ds, out_dir, f"fold{k}", epochs)
        m = compute_metrics(va["label"], prob)
        print_metrics(m, f"fold {k} ({time.time() - start:.1f} s)")
        metrics.append(m)
        oof.append(va[["id", "centre_id", "patient_id", "node_id", "label", "fold"]].assign(prob=prob))
        if hist is not None:
            histories.append(hist.assign(fold=k))

    table = fold_table(metrics)
    table.to_csv(out_dir / "cv_folds.csv")
    oof = pd.concat(oof)
    oof.to_csv(out_dir / "cv_oof.csv", index=False)
    if histories:
        pd.concat(histories).to_csv(out_dir / "cv_history.csv", index=False)
    print(f"\n=== {model_name}: média ± desvio em {len(metrics)} folds ===")
    print(mean_std_text(table).to_string())
    return table, oof


def run_final(cfg, model_name):
    """Treina no dev inteiro e avalia UMA vez no centro de teste externo. Apoio: metodologia.

    Na CNN não há validação para o early stopping, então usamos o nº de épocas
    médio que o early stopping escolheu na validação cruzada (cv_history.csv).
    """
    utils.set_seed(cfg["seed"])
    out_dir = utils.resolve_path(cfg["paths"]["outputs_dir"]) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dev, test = _load_dev_and_test(cfg, model_name)
    hf_ds = load_hf_dataset(cfg) if model_name == "cnn" else None

    epochs = None
    if model_name == "cnn":
        hist = pd.read_csv(out_dir / "cv_history.csv")  # precisa rodar a CV antes
        epochs = int(round(hist.groupby("fold")["best_epoch"].first().mean()))
        print(f"Treino final da CNN com {epochs} épocas (média da CV)")

    prob, _ = _fit_predict(cfg, model_name, dev, test, hf_ds, out_dir, "final", epochs)
    m = compute_metrics(test["label"], prob)
    print_metrics(m, f"{model_name} | TESTE EXTERNO (centro {cfg['splits']['test_centre']})")
    test[["id", "centre_id", "patient_id", "node_id", "label"]].assign(prob=prob).to_csv(
        out_dir / "final_test_preds.csv", index=False)
    pd.Series({k: v for k, v in m.items() if k != "confusion_matrix"}).to_csv(out_dir / "final_test_metrics.csv")
    return m
