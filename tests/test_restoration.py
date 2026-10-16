"""Testes das funções de degradação e restauração (Entrega 3)."""

import numpy as np
import pytest

from src.restoration import LEVELS, add_salt_pepper, degrade, restore


def mse(a, b):
    return float(np.mean((a.astype(float) - b.astype(float)) ** 2))


def smooth_image():
    """Imagem suave (gradiente), onde os filtros de restauração quase não estragam nada."""
    x = np.linspace(0, 255, 64)
    gray = np.add.outer(x, x) / 2
    return np.stack([gray, gray * 0.8, gray * 0.6], axis=-1).astype(np.uint8)


@pytest.mark.parametrize("kind", list(LEVELS))
def test_nivel_maior_estraga_mais(kind):
    img = smooth_image() if kind != "blur" else np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    erros = [mse(img, degrade(img, kind, lvl, np.random.default_rng(0))) for lvl in range(1, 6)]
    assert erros == sorted(erros) and erros[0] > 0


def test_sal_e_pimenta_fracao():
    img = np.full((100, 100, 3), 128, np.uint8)
    out = add_salt_pepper(img, 0.1, np.random.default_rng(0))
    corrompidos = np.mean(np.any(out != 128, axis=-1))
    assert 0.08 < corrompidos < 0.12


def test_mediana_vence_media_no_sal_e_pimenta():
    img = smooth_image()
    ruidosa = degrade(img, "salt_pepper", 3, np.random.default_rng(0))
    assert mse(img, restore(ruidosa, "median")) < mse(img, restore(ruidosa, "mean"))
    assert mse(img, restore(ruidosa, "median")) < mse(img, ruidosa)


def test_gaussiano_reduz_ruido_gaussiano():
    img = smooth_image()
    ruidosa = degrade(img, "gaussian", 4, np.random.default_rng(0))
    assert mse(img, restore(ruidosa, "gaussian", 5)) < mse(img, ruidosa)


def test_metodo_invalido():
    with pytest.raises(ValueError):
        restore(smooth_image(), "max")
