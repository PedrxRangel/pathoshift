"""Degradação e restauração de imagens (Entrega 3: robustez a ruído e desfoque).

Por enquanto só as funções (testadas em tests/test_restoration.py), sem experimentos.
"""

import cv2
import numpy as np

from src.filters import gaussian_filter, mean_filter, median_filter

# Intensidade de cada degradação para os níveis 1 a 5
LEVELS = {
    "gaussian": [5, 10, 20, 30, 40],                 # desvio padrão do ruído (0-255)
    "salt_pepper": [0.01, 0.02, 0.05, 0.10, 0.20],   # fração de pixels corrompidos
    "blur": [0.5, 1.0, 1.5, 2.0, 3.0],               # sigma do Gaussiano
}


def add_gaussian_noise(img, sigma, rng=None):
    """Soma ruído gaussiano de média 0 e desvio sigma a cada pixel. Aula 05 – Restauração: ruído gaussiano."""
    rng = rng or np.random.default_rng()
    noisy = img.astype(np.float64) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper(img, amount, rng=None):
    """Troca uma fração `amount` dos pixels por preto (pimenta) ou branco (sal). Aula 05 – Restauração: ruído sal-e-pimenta."""
    rng = rng or np.random.default_rng()
    out = img.copy()
    r = rng.random(img.shape[:2])
    out[r < amount / 2] = 0
    out[(r >= amount / 2) & (r < amount)] = 255
    return out


def add_blur(img, sigma):
    """Desfoque Gaussiano com desvio sigma (simula a lâmina fora de foco). Aula 05 – Restauração: modelo de degradação."""
    return cv2.GaussianBlur(img, (0, 0), sigma)


def degrade(img, kind, level, rng=None):
    """Aplica a degradação `kind` ('gaussian', 'salt_pepper' ou 'blur') no nível 1 a 5. Aula 05 – Restauração: modelo de degradação."""
    value = LEVELS[kind][level - 1]
    if kind == "gaussian":
        return add_gaussian_noise(img, value, rng)
    if kind == "salt_pepper":
        return add_salt_pepper(img, value, rng)
    if kind == "blur":
        return add_blur(img, value)
    raise ValueError(f"Degradação desconhecida: {kind}")


def restore(img, method, k=3):
    """Restaura com filtro da 'median' (mediana), 'mean' (média) ou 'gaussian' (Gaussiano), janela k. Aula 05 – Restauração: filtros de restauração."""
    if method == "median":
        return median_filter(img, k)
    if method == "mean":
        return mean_filter(img, k)
    if method == "gaussian":
        return gaussian_filter(img, k)
    raise ValueError(f"Método de restauração desconhecido: {method}")
