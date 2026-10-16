"""Testes dos modelos (sem treinar)."""

import numpy as np
import torch

from src.data import augment
from src.models import SimpleCNN
from src.train import class_weights


def test_simplecnn_formato_da_saida():
    model = SimpleCNN(num_classes=2, image_size=128)
    out = model(torch.rand(4, 3, 128, 128))
    assert out.shape == (4, 2)
    # Flatten: 64 mapas de 28x28 (128 → 126 → 63 → 61 → 30 → 28)
    assert model.classifier[1].in_features == 64 * 28 * 28


def test_peso_de_classe():
    w = class_weights(np.array([0] * 90 + [1] * 10))
    assert torch.allclose(w, torch.tensor([100 / 180, 100 / 20]))


def test_aumentacao_preserva_pixels():
    img = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    out = augment(img)
    assert out.shape == img.shape
    assert sorted(out.ravel()) == sorted(img.ravel())  # só reposiciona os pixels
