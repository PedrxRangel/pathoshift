"""Figuras para a EDA, os baselines e o relatório (matplotlib)."""

import matplotlib.pyplot as plt
import numpy as np

from src.color import HSV_RANGES, rgb_to_hsv
from src.data import read_image


def show_grid(images, titles=None, ncols=6, size=2.0, cmap=None):
    """Mostra uma lista de imagens em grade. Apoio: visualização."""
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, img in enumerate(images):
        ax = axes.ravel()[i]
        # máscara binária: escala fixa 0-1 (senão uma máscara toda 1 apareceria preta)
        scale = {"vmin": 0, "vmax": 1} if img.ndim == 2 and img.max() <= 1 else {}
        ax.imshow(img, cmap=cmap, **scale)
        if titles is not None:
            ax.set_title(titles[i], fontsize=8)
    fig.tight_layout()
    return fig


def examples_by_centre(hf_ds, manifest, n=6, seed=0):
    """Grade com n patches aleatórios de cada centro (uma linha por centro). Apoio: visualização da mudança de domínio."""
    rng = np.random.default_rng(seed)
    images, titles = [], []
    for c in sorted(manifest["centre_id"].unique()):
        sub = manifest[manifest["centre_id"] == c]
        for _, row in sub.sample(n, random_state=int(rng.integers(1e6))).iterrows():
            images.append(read_image(hf_ds, row["id"]))
            titles.append(f"centro {c} | y={row['label']}")
    return show_grid(images, titles, ncols=n)


def mean_hsv_hist_by_centre(hf_ds, manifest, n=100, bins=32, seed=0):
    """Histograma médio de H, S e V de cada centro (amostra de n patches). Aula 01 – Fundamentos: histograma e HSV."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    for c in sorted(manifest["centre_id"].unique()):
        sub = manifest[manifest["centre_id"] == c].sample(n, random_state=seed, replace=True)
        hists = np.zeros((3, bins))
        for idx in sub["id"]:
            hsv = rgb_to_hsv(read_image(hf_ds, idx))
            for ch, (lo, hi) in enumerate(HSV_RANGES):
                hists[ch] += np.histogram(hsv[..., ch], bins=bins, range=(lo, hi))[0]
        for ch, ax in enumerate(axes):
            lo, hi = HSV_RANGES[ch]
            ax.plot(np.linspace(lo, hi, bins), hists[ch] / hists[ch].sum(), label=f"centro {c}")
    for ax, name in zip(axes, "HSV"):
        ax.set_title(f"canal {name}")
        ax.set_xlabel("valor")
    axes[0].set_ylabel("fração de pixels")
    axes[-1].legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, title="", class_names=("normal", "tumor")):
    """Matriz de confusão com as contagens escritas em cada célula. Aula 08 – Classificação: avaliação."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    ax.set_xticks([0, 1], class_names)
    ax.set_yticks([0, 1], class_names)
    ax.set_xlabel("predito")
    ax.set_ylabel("verdadeiro")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def error_grid(hf_ds, preds, kind="fp", n=12, threshold=0.5):
    """Grade dos erros mais confiantes: 'fp' (normal predito como tumor) ou 'fn' (tumor predito como normal). Aula 08 – Classificação: análise de erros."""
    if kind == "fp":
        err = preds[(preds["label"] == 0) & (preds["prob"] >= threshold)].sort_values("prob", ascending=False)
    else:
        err = preds[(preds["label"] == 1) & (preds["prob"] < threshold)].sort_values("prob")
    err = err.head(n)
    images = [read_image(hf_ds, i) for i in err["id"]]
    titles = [f"c{c} p={p:.2f}" for c, p in zip(err["centre_id"], err["prob"])]
    return show_grid(images, titles, ncols=6) if images else None


def slide_heatmap(preds, patient_id, node_id, manifest):
    """Mapa de uma lâmina: cada patch é um ponto em (x, y) colorido pela probabilidade de tumor. Apoio: visualização por lâmina."""
    sub = preds.merge(manifest[["id", "x", "y"]], on="id")
    sub = sub[(sub["patient_id"] == patient_id) & (sub["node_id"] == node_id)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, col, title in [(axes[0], "label", "rótulo verdadeiro"), (axes[1], "prob", "prob. de tumor")]:
        sc = ax.scatter(sub["x"], sub["y"], c=sub[col], cmap="coolwarm", vmin=0, vmax=1, s=12)
        ax.set_title(title)
        ax.invert_yaxis()  # y cresce para baixo, como numa imagem
    fig.colorbar(sc, ax=axes)
    fig.suptitle(f"lâmina paciente {patient_id}, nó {node_id}")
    return fig
