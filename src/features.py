"""Vetor de features clássicas por patch (para o baseline com regressão logística).

Todas as features são calculadas no patch original (256x256) e salvas em parquet,
para não precisar recalcular (leva alguns minutos para o dataset todo).
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.color import channel_histograms, to_gray
from src.filters import edge_density, high_freq_energy, laplacian_variance
from src.segmentation import nuclei_mask, tissue_mask


def patch_features(img, cfg_feat):
    """Calcula as features de um patch e devolve um dicionário {nome: valor}. Combina as Aulas 01, 03, 04, 06 e 07.

    - cor: histograma HSV (Aula 01)
    - bordas/nitidez: densidade de bordas Sobel e variância do Laplaciano (Aula 03)
    - textura fina: energia de alta frequência (Aula 04)
    - núcleos: fração de área, número e área média (Aulas 06 e 07)
    """
    f = cfg_feat
    gray = to_gray(img)
    feats = {}

    hist = channel_histograms(img, bins=f["hsv_bins"], space="hsv")
    for i, v in enumerate(hist):
        canal = "HSV"[i // f["hsv_bins"]]
        feats[f"hist_{canal}{i % f['hsv_bins']}"] = v

    feats["sobel_densidade"] = edge_density(gray, f["sobel_threshold"], f["blur_ksize"])
    feats["laplaciano_var"] = laplacian_variance(gray)
    feats["energia_alta_freq"] = high_freq_energy(gray, f["hf_radius"])

    tissue = tissue_mask(img, f["tissue_threshold"], f["tissue_blur_sigma"], f["tissue_kernel"])
    nuclei, areas = nuclei_mask(img, tissue, f["nuclei_kernel"], f["nuclei_min_area"])
    tissue_area = tissue.sum()
    feats["nucleo_fracao_area"] = nuclei.sum() / tissue_area if tissue_area > 0 else 0.0
    feats["nucleo_numero"] = len(areas)
    feats["nucleo_area_media"] = float(areas.mean()) if len(areas) > 0 else 0.0
    return feats


def compute_features(hf_ds, ids, cfg_feat):
    """Calcula as features de uma lista de ids e devolve um DataFrame (coluna 'id' + features). Apoio: laço sobre os patches."""
    rows = []
    for idx in tqdm(ids, desc="features"):
        img = np.asarray(hf_ds[int(idx)]["image"].convert("RGB"))
        rows.append({"id": int(idx), **patch_features(img, cfg_feat)})
    return pd.DataFrame(rows)


def feature_columns(df):
    """Nomes das colunas de features (tudo menos o id e as colunas do manifest). Apoio."""
    skip = {"id", "patient_id", "node_id", "centre_id", "x", "y", "label", "split", "fold"}
    return [c for c in df.columns if c not in skip]


def save_features(df, path):
    """Salva as features em parquet (formato colunar, rápido de ler). Apoio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_features(path, manifest):
    """Lê as features e junta com o manifest pelo id (traz label, fold, centro...). Apoio."""
    return manifest.merge(pd.read_parquet(path), on="id", how="inner")
