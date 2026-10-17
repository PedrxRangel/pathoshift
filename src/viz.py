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


# ---------------------------------------------------------------- figuras da Entrega 1 (outputs/figuras/)

# Cores por modelo (paleta categórica validada: slot 1 azul, slot 2 laranja); texto fica em cinza escuro
MODEL_COLORS = {"logreg": "#2a78d6", "cnn": "#eb6834"}
MODEL_NAMES = {"logreg": "Regressão logística", "cnn": "SimpleCNN"}


def _draw_cm(ax, cm, title):
    """Desenha uma matriz de confusão 2x2 num eixo. Apoio."""
    cm = np.asarray(cm)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#222222")
    ax.set_xticks([0, 1], ["normal", "tumor"])
    ax.set_yticks([0, 1], ["normal", "tumor"])
    ax.set_xlabel("predito")
    ax.set_ylabel("verdadeiro")
    ax.set_title(title)


def confusion_figure(report, model_name):
    """Matrizes de confusão de um modelo: CV (4 folds juntos) e teste externo, se houver. Aula 08 – Classificação: avaliação."""
    panels = [("CV (4 folds juntos)", report["cv"]["matriz_confusao_folds_juntos"])]
    if "teste_externo" in report:
        panels.append(("teste externo (centro 2)", report["teste_externo"]["matriz_confusao"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 3.6), squeeze=False)
    for ax, (title, cm) in zip(axes[0], panels):
        _draw_cm(ax, cm, title)
    fig.suptitle(MODEL_NAMES[model_name])
    fig.tight_layout()
    return fig


def errors_figure(hf_ds, preds, model_name, n=6, threshold=0.5):
    """Linha 1: falsos positivos mais confiantes; linha 2: falsos negativos mais confiantes (predições da CV). Aula 08 – Classificação: análise de erros."""
    fp = preds[(preds["label"] == 0) & (preds["prob"] >= threshold)].sort_values("prob", ascending=False).head(n)
    fn = preds[(preds["label"] == 1) & (preds["prob"] < threshold)].sort_values("prob").head(n)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.8), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for r, (rows, kind) in enumerate([(fp, "FP"), (fn, "FN")]):
        for ax, (_, row) in zip(axes[r], rows.iterrows()):
            ax.imshow(read_image(hf_ds, row["id"]))
            ax.set_title(f"{kind} c{row['centre_id']} pac{row['patient_id']} p={row['prob']:.2f}", fontsize=8)
    fig.suptitle(f"{MODEL_NAMES[model_name]}: falsos positivos (cima) e falsos negativos (baixo) mais confiantes")
    fig.tight_layout()
    return fig


def cnn_curves_figure(hist):
    """Loss de treino e F1 macro do conjunto de parada por época, em cada fold. Aula 08 – Classificação: treino da CNN."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for k, h in hist.groupby("fold"):
        axes[0].plot(h["epoch"], h["train_loss"], marker="o", ms=4, lw=2, label=f"fold {k}")
        axes[1].plot(h["epoch"], h["parada_f1_macro"], marker="o", ms=4, lw=2, label=f"fold {k}")
        best = h["best_epoch"].iloc[0]
        axes[1].plot(best, h.loc[h["epoch"] == best, "parada_f1_macro"], marker="*", ms=12, color="#222222")
    axes[0].set_title("loss de treino")
    axes[1].set_title("F1 macro no conjunto de parada (★ = época escolhida)")
    for ax in axes:
        ax.set_xlabel("época")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    return fig


def comparison_figure(metrics):
    """Comparação dos dois modelos: CV (média ± desvio, barra cheia) e teste externo (barra hachurada). Aula 08 – Classificação: avaliação."""
    keys = [("balanced_accuracy", "acurácia\nbalanceada"), ("f1_macro", "F1 macro"), ("auc", "AUC"),
            ("recall_tumor", "recall\ntumor"), ("recall_normal", "especificidade")]
    models = [m for m in ("logreg", "cnn") if m in metrics["modelos"]]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    width = 0.2
    x = np.arange(len(keys))
    for i, name in enumerate(models):
        rep = metrics["modelos"][name]
        cv = [rep["cv"]["por_patch"][k]["media"] for k, _ in keys]
        sd = [rep["cv"]["por_patch"][k]["desvio"] for k, _ in keys]
        pos = x + (i - len(models) / 2) * 2 * width
        ax.bar(pos, cv, width, yerr=sd, capsize=3, color=MODEL_COLORS[name], edgecolor="white",
               linewidth=2, label=f"{MODEL_NAMES[name]} — CV")
        if "teste_externo" in rep:
            te = [rep["teste_externo"]["por_patch"][k] for k, _ in keys]
            ax.bar(pos + width, te, width, color="white", edgecolor=MODEL_COLORS[name], hatch="//",
                   linewidth=1.5, label=f"{MODEL_NAMES[name]} — teste centro 2")
    ax.set_xticks(x - width / 2 if len(models) == 2 else x, [k[1] for k in keys])
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="#999999", lw=1, ls="--")
    ax.set_ylabel("valor")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title("Comparação dos modelos (por patch; barras de erro = desvio entre os 4 folds)")
    fig.tight_layout()
    return fig


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
