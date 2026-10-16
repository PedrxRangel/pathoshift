"""Segmentação de tecido e de núcleos (limiarização + morfologia).

Observação da EDA: o Otsu POR PATCH no canal S não separa tecido de fundo,
porque a maioria dos patches não tem fundo; as duas classes do Otsu viram
"núcleo x estroma". Por isso a máscara de tecido usa, por padrão, um limiar
global fixo (escolhido pelo vale do histograma de S), e o Otsu fica como opção.
Já para os núcleos o Otsu funciona bem, desde que calculado só dentro do tecido.
"""

import cv2
import numpy as np

from src.color import rgb_to_hsv, to_gray


def otsu_threshold(values):
    """Limiar de Otsu (maximiza a variância entre as 2 classes) de um conjunto de pixels uint8. Aula 06 – Segmentação: limiar de Otsu."""
    t, _ = cv2.threshold(np.asarray(values, np.uint8).reshape(-1, 1), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


def ellipse_kernel(k):
    """Elemento estruturante elíptico k x k. Aula 07 – Morfologia: elemento estruturante."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def open_close(mask, k=5):
    """Abertura (remove pontos soltos) seguida de fecho (tapa buracos). Aula 07 – Morfologia: abertura e fecho."""
    kernel = ellipse_kernel(k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def extract_border(mask, k=3):
    """Borda de uma máscara: máscara menos a sua erosão. Aula 07 – Morfologia: extração de borda."""
    return mask - cv2.erode(mask, ellipse_kernel(k))


def tissue_mask(img, threshold=20, blur_sigma=4, kernel=5, morphology=True):
    """Máscara de tecido (1) x fundo (0): Gaussiano no canal S, limiar e depois abertura e fecho. Aula 06 – Segmentação: limiarização; Aula 07 – Morfologia.

    threshold=None usa o limiar de Otsu do próprio patch (ver observação no topo).
    """
    s = rgb_to_hsv(img)[..., 1]
    s = cv2.GaussianBlur(s, (0, 0), blur_sigma)
    t = otsu_threshold(s) if threshold is None else threshold
    mask = (s > t).astype(np.uint8)
    return open_close(mask, kernel) if morphology else mask


def nuclei_mask(img, tissue, kernel=3, min_area=15, morphology=True):
    """Máscara de núcleos: Otsu no cinza invertido (só pixels de tecido), abertura e componentes conexos. Aula 06 – Segmentação: Otsu; Aula 07 – Morfologia: abertura.

    Os núcleos (hematoxilina) são as regiões mais escuras; invertendo o cinza, viram as mais claras.
    Devolve (máscara, lista de áreas dos núcleos em pixels).
    """
    inv = 255 - cv2.GaussianBlur(to_gray(img), (5, 5), 0)
    if tissue.sum() == 0:
        return np.zeros_like(tissue), np.array([])
    t = otsu_threshold(inv[tissue > 0])  # Otsu só com os pixels de tecido
    mask = ((inv > t) & (tissue > 0)).astype(np.uint8)
    if not morphology:
        return mask, np.array([])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse_kernel(kernel))
    return remove_small_components(mask, min_area)


def remove_small_components(mask, min_area):
    """Rotula os componentes conexos (vizinhança-8) e apaga os menores que min_area. Aula 06 – Segmentação: rotulação de regiões conexas.

    Devolve (máscara limpa, áreas dos componentes que ficaram).
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]  # o rótulo 0 é o fundo
    keep = np.flatnonzero(areas >= min_area) + 1
    clean = np.isin(labels, keep).astype(np.uint8)
    return clean, areas[areas >= min_area]
