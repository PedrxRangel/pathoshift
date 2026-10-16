"""Modelos: CNN simples (igual à do slide da aula) e o baseline clássico (regressão logística).

Código Keras do slide, com a entrada 128x128 e a saída de 2 classes do nosso problema:

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2))
"""

import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class SimpleCNN(nn.Module):
    """CNN Conv → Pooling → Dense, a mesma rede do slide (Keras) escrita em PyTorch. Aula 08 – Classificação: CNN.

    Tamanho da imagem em cada etapa (convolução 3x3 sem padding perde 2 pixels;
    MaxPool 2x2 divide por 2):
        128 → conv 126 → pool 63 → conv 61 → pool 30 → conv 28  ⇒  64 mapas de 28x28
    Diferença do Keras: o PyTorch usa a ordem (canais, altura, largura).
    """

    def __init__(self, num_classes=2, image_size=128):
        super().__init__()
        s = ((image_size - 2) // 2 - 2) // 2 - 2  # lado do mapa final (28 para 128)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),   # Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3))
            nn.ReLU(),
            nn.MaxPool2d(2),                   # MaxPooling2D((2, 2))
            nn.Conv2d(32, 64, kernel_size=3),  # Conv2D(64, (3, 3), activation='relu')
            nn.ReLU(),
            nn.MaxPool2d(2),                   # MaxPooling2D((2, 2))
            nn.Conv2d(64, 64, kernel_size=3),  # Conv2D(64, (3, 3), activation='relu')
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # Flatten()
            nn.Linear(64 * s * s, 64),         # Dense(64, activation='relu')
            nn.ReLU(),
            nn.Linear(64, num_classes),        # Dense(2): logits; o softmax fica dentro da CrossEntropyLoss
        )

    def forward(self, x):
        """Passa a imagem pelas convoluções e depois pela parte densa. Aula 08 – Classificação: CNN."""
        return self.classifier(self.features(x))


def build_logreg(cfg):
    """Baseline clássico: padroniza as features e aplica regressão logística com peso de classe. Aula 08 – Classificação.

    class_weight='balanced' dá mais peso à classe rara (tumor ~10% dos patches).
    """
    lr = cfg["logreg"]
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=lr["C"], max_iter=lr["max_iter"], class_weight="balanced"),
    )
