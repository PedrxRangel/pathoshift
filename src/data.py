"""Carregamento do dataset MiniCAMELYON17 (Hugging Face) e leitura das imagens.

Schema real conferido (revisão fixada no config):
    image (JPEG 256x256 RGB), tumor (0/1), patient_id, node_id, centre_id, patch_x, patch_y
Há 6 centros (0 a 5). Uma lâmina = (patient_id, node_id).
"""

import cv2
import numpy as np
import torch
from datasets import load_dataset

# Colunas de metadados (tudo menos a imagem)
META_COLUMNS = ["tumor", "patient_id", "node_id", "centre_id", "patch_x", "patch_y"]


def load_hf_dataset(cfg):
    """Baixa (na 1ª vez) e abre o dataset do Hugging Face; só existe o split 'train'. Apoio: leitura dos dados."""
    d = cfg["dataset"]
    ds = load_dataset(
        d["hf_name"], split="train", revision=d.get("revision"), cache_dir=d.get("cache_dir")
    )
    missing = set(META_COLUMNS + ["image"]) - set(ds.column_names)
    if missing:
        raise ValueError(f"Colunas esperadas não encontradas no dataset: {sorted(missing)}")
    return ds


def load_metadata(ds):
    """Tabela só com os metadados, sem decodificar as imagens; 'id' = índice da linha no HF. Apoio: leitura dos dados."""
    df = ds.select_columns(META_COLUMNS).to_pandas()
    df.insert(0, "id", np.arange(len(df)))
    return df


def read_image(ds, idx):
    """Lê uma imagem pelo id e devolve um array NumPy RGB (H, W, 3) uint8. Aula 01 – Fundamentos: imagem digital como matriz."""
    return np.asarray(ds[int(idx)]["image"].convert("RGB"))


def resize(img, size):
    """Reduz a resolução para size x size fazendo a média dos pixels vizinhos (INTER_AREA). Aula 01 – Fundamentos: amostragem."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def augment(img):
    """Rotação aleatória de 0/90/180/270° e flip horizontal aleatório (tecido não tem orientação). Aula 03 – Transformações geométricas."""
    img = np.rot90(img, k=int(torch.randint(0, 4, (1,))))
    if torch.rand(1) < 0.5:
        img = np.fliplr(img)
    return np.ascontiguousarray(img)


class PatchDataset(torch.utils.data.Dataset):
    """Dataset do PyTorch: lê o patch, reduz para image_size, aumenta (só no treino) e divide por 255. Aula 08 – Classificação: entrada da CNN."""

    def __init__(self, hf_ds, manifest_df, image_size=128, train=False):
        self.hf_ds = hf_ds
        self.ids = manifest_df["id"].to_numpy()
        self.labels = manifest_df["label"].to_numpy()
        self.image_size = image_size
        self.train = train

    def __len__(self):
        """Número de patches. Apoio."""
        return len(self.ids)

    def __getitem__(self, i):
        """Devolve (tensor 3 x H x W com pixels em [0, 1], rótulo). Aula 08 – Classificação: entrada da CNN."""
        img = resize(read_image(self.hf_ds, self.ids[i]), self.image_size)
        if self.train:
            img = augment(img)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (H, W, C) -> (C, H, W)
        return x, int(self.labels[i])
