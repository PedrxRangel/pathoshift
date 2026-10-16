"""Funções utilitárias: seed global, leitura do config, versões das bibliotecas e tempo.

Nenhuma delas é conteúdo de aula; são apoio para o trabalho ser reprodutível.
"""

import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

# Raiz do repositório (a pasta que contém pathoshift.py)
ROOT = Path(__file__).resolve().parents[1]

# Bibliotecas principais mostradas no `describe`
MAIN_PACKAGES = [
    "torch", "datasets", "numpy", "pandas", "pyarrow", "scikit-learn",
    "opencv-python-headless", "matplotlib", "pyyaml", "tqdm",
]


def resolve_path(path):
    """Caminhos relativos do config são relativos à raiz do repositório. Apoio."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path="configs/default.yaml"):
    """Lê o arquivo YAML de configuração e devolve um dicionário. Apoio."""
    path = Path(path)
    if not path.exists():
        path = resolve_path(path)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Fixa a seed do Python, NumPy e PyTorch para os resultados serem reprodutíveis. Apoio."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Devolve 'cuda' se houver GPU disponível, senão 'cpu'. Apoio."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def package_versions(packages=MAIN_PACKAGES):
    """Dicionário {pacote: versão} das bibliotecas principais. Apoio."""
    versions = {"python": platform.python_version()}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "não instalado"
    return versions


def save_pip_freeze(out_path):
    """Salva a lista de todos os pacotes instalados (equivale ao `pip freeze`, sem precisar do pip). Apoio."""
    lines = sorted(
        f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# python {sys.version.split()[0]} | {platform.platform()}\n"
    out_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return out_path


@contextmanager
def timer(name):
    """Mede o tempo de um bloco de código:  with timer("treino"): ...  Apoio."""
    start = time.time()
    yield
    print(f"[tempo] {name}: {time.time() - start:.1f} s")
