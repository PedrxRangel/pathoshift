"""Cor e histograma: conversões RGB/HSV, histogramas por canal e equalização.

As imagens são arrays NumPy RGB uint8 (H, W, 3).
"""

import cv2
import numpy as np

# Faixa de cada canal no OpenCV: H vai de 0 a 179; S e V de 0 a 255
HSV_RANGES = [(0, 180), (0, 256), (0, 256)]


def rgb_to_hsv(img):
    """Converte RGB em HSV (matiz, saturação, brilho). Aula 01 – Fundamentos: modelos de cor RGB e HSV."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def hsv_to_rgb(hsv):
    """Converte HSV de volta para RGB. Aula 01 – Fundamentos: modelos de cor RGB e HSV."""
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def to_gray(img):
    """RGB em tons de cinza (0,299R + 0,587G + 0,114B). Aula 01 – Fundamentos: operações pontuais."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def channel_histograms(img, bins=8, space="hsv"):
    """Histograma normalizado (soma 1) de cada canal, concatenados. Aula 01 – Fundamentos: histograma."""
    if space == "hsv":
        img, ranges = rgb_to_hsv(img), HSV_RANGES
    else:
        ranges = [(0, 256)] * 3
    hists = []
    for c, (lo, hi) in enumerate(ranges):
        h, _ = np.histogram(img[..., c], bins=bins, range=(lo, hi))
        hists.append(h / h.sum())
    return np.concatenate(hists)


def equalize_v(img):
    """Equaliza o histograma só do canal V (brilho), mantendo H e S, ou seja, a cor. Aula 01 – Fundamentos: equalização de histograma."""
    hsv = rgb_to_hsv(img)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    return hsv_to_rgb(hsv)
