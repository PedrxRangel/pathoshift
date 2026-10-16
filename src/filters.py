"""Filtros no domínio do espaço e da frequência.

Funções de nitidez/textura recebem uma imagem em tons de cinza (2D, uint8).
"""

import cv2
import numpy as np


# ---------------------------------------------------------------- domínio do espaço

def mean_filter(img, k=3):
    """Filtro da média com janela k x k (suaviza e borra). Aula 03 – Filtros no espaço: filtro da média."""
    return cv2.blur(img, (k, k))


def gaussian_filter(img, k=5, sigma=0):
    """Filtro Gaussiano k x k (sigma=0: o OpenCV calcula a partir de k). Aula 03 – Filtros no espaço: filtro Gaussiano."""
    return cv2.GaussianBlur(img, (k, k), sigma)


def median_filter(img, k=3):
    """Filtro da mediana k x k (não linear; ótimo contra sal-e-pimenta). Aula 05 – Restauração: filtro da mediana."""
    return cv2.medianBlur(img, k)


def sobel_magnitude(gray):
    """Magnitude do gradiente de Sobel: sqrt(Gx² + Gy²). Aula 03 – Filtros no espaço: Sobel."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def edge_density(gray, threshold=100, blur_ksize=5):
    """Fração de pixels de borda: Gaussiano, Sobel e limiar na magnitude. Aula 03 – Filtros no espaço: Gaussiano + Sobel."""
    mag = sobel_magnitude(gaussian_filter(gray, blur_ksize))
    return float(np.mean(mag > threshold))


def laplacian_variance(gray):
    """Variância do Laplaciano: alta em imagem nítida, baixa em imagem borrada. Aula 03 – Filtros no espaço: Laplaciano."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------- domínio da frequência

def centered_fft(gray):
    """DFT 2D via FFT com a frequência zero deslocada para o centro (fftshift). Aula 04 – Filtros na frequência: DFT/FFT."""
    return np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))


def log_spectrum(gray):
    """Espectro de magnitude log(1 + |F|) centralizado, para visualização. Aula 04 – Filtros na frequência: espectro de Fourier."""
    return np.log1p(np.abs(centered_fft(gray)))


def distance_grid(shape):
    """Matriz D(u, v): distância de cada frequência ao centro do espectro. Aula 04 – Filtros na frequência: D(u, v)."""
    h, w = shape
    u = np.arange(h) - h // 2
    v = np.arange(w) - w // 2
    return np.sqrt(u[:, None] ** 2 + v[None, :] ** 2)


def high_freq_energy(gray, d0=32):
    """Fração da energia |F|² fora do raio D0 (o que um passa-alta ideal deixa passar). Aula 04 – Filtros na frequência: passa-alta ideal.

    A média é subtraída antes, para o termo DC (brilho médio) não dominar a energia.
    """
    g = gray.astype(np.float64)
    power = np.abs(centered_fft(g - g.mean())) ** 2
    total = power.sum()
    if total == 0:
        return 0.0
    return float(power[distance_grid(g.shape) > d0].sum() / total)


def gaussian_lowpass(gray, d0=32):
    """Passa-baixa Gaussiano na frequência: H = exp(-D²/2D0²), depois a transformada inversa. Aula 04 – Filtros na frequência: passa-baixa Gaussiano."""
    F = centered_fft(gray)
    H = np.exp(-(distance_grid(gray.shape) ** 2) / (2 * d0 ** 2))
    out = np.fft.ifft2(np.fft.ifftshift(F * H)).real
    return np.clip(out, 0, 255).astype(np.uint8)
