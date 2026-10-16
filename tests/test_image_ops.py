"""Testes das operações de imagem com imagens sintéticas (resultado conhecido)."""

import cv2
import numpy as np

from src.color import channel_histograms, equalize_v, rgb_to_hsv
from src.features import patch_features
from src.filters import (edge_density, gaussian_filter, gaussian_lowpass, high_freq_energy,
                         laplacian_variance, sobel_magnitude)
from src.segmentation import extract_border, nuclei_mask, tissue_mask
from src.utils import ROOT, load_config

RNG = np.random.default_rng(0)


def pink_with_nuclei(n=5, radius=6):
    """Fundo rosa (tecido) com n 'núcleos' roxos escuros em posições conhecidas."""
    img = np.zeros((128, 128, 3), np.uint8)
    img[:] = (230, 160, 200)
    for i in range(n):
        cv2.circle(img, (20 + 20 * i, 64), radius, (90, 40, 120), -1)
    return img


def test_histogramas_somam_1():
    h = channel_histograms(RNG.integers(0, 256, (32, 32, 3), dtype=np.uint8), bins=8)
    assert h.shape == (24,)
    assert np.allclose(h.reshape(3, 8).sum(axis=1), 1)


def test_equalizacao_preserva_matiz():
    img = pink_with_nuclei()
    hsv, hsv_eq = rgb_to_hsv(img), rgb_to_hsv(equalize_v(img))
    # o brilho (V) passa a usar a faixa toda...
    assert hsv_eq[..., 2].max() - hsv_eq[..., 2].min() > 200
    # ...e a matiz (H) não muda (exceto onde o pixel virou preto, pois preto não tem matiz)
    visivel = hsv_eq[..., 2] > 50
    assert np.abs(hsv_eq[..., 0].astype(int) - hsv[..., 0])[visivel].mean() < 3


def test_sobel_detecta_degrau():
    gray = np.zeros((32, 32), np.uint8)
    gray[:, 16:] = 255
    mag = sobel_magnitude(gray)
    assert mag[:, 15:17].min() > 500 and mag[:, :10].max() == 0
    assert edge_density(gray, threshold=100) > 0


def test_desfoque_reduz_laplaciano_e_alta_frequencia():
    gray = RNG.integers(0, 256, (64, 64), dtype=np.uint8)
    borrada = gaussian_filter(gray, 7)
    assert laplacian_variance(borrada) < laplacian_variance(gray)
    assert high_freq_energy(borrada, 8) < high_freq_energy(gray, 8)
    assert high_freq_energy(gaussian_lowpass(gray, 8), 8) < high_freq_energy(gray, 8)


def test_alta_frequencia_imagem_constante():
    assert high_freq_energy(np.full((32, 32), 100, np.uint8), 4) == 0.0


def test_mascaras_de_tecido_e_nucleos():
    img = pink_with_nuclei(n=5)
    img[:, 110:] = 245  # faixa de fundo branco à direita
    tissue = tissue_mask(img, threshold=20)
    assert tissue[:, :100].mean() > 0.95 and tissue[:, 120:].mean() < 0.05
    nuclei, areas = nuclei_mask(img, tissue, min_area=15)
    assert len(areas) == 5  # os 5 círculos
    assert extract_border(nuclei).sum() < nuclei.sum()


def test_features_sem_nan():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    f = patch_features(pink_with_nuclei(), cfg["features"])
    assert len(f) == 3 * cfg["features"]["hsv_bins"] + 6
    assert f["nucleo_numero"] == 5
    assert all(np.isfinite(v) for v in f.values())
