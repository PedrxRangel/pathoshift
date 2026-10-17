"""Funções utilitárias: seed global, leitura do config, versões das bibliotecas e tempo.

Nenhuma delas é conteúdo de aula; são apoio para o trabalho ser reprodutível.
"""

import json
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
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
        # Sem isso a GPU escolhe algoritmos não determinísticos e duas execuções iguais dão números diferentes
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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


def gpu_info():
    """Descrição do dispositivo usado (GPU ou CPU). Apoio."""
    import torch
    if not torch.cuda.is_available():
        return {"dispositivo": "cpu"}
    props = torch.cuda.get_device_properties(0)
    return {"dispositivo": "cuda", "nome": props.name,
            "memoria_gb": round(props.total_memory / 1024 ** 3, 1), "cuda": torch.version.cuda}


def run_info_path(cfg):
    """Caminho de outputs/run_info.json. Apoio."""
    return resolve_path(cfg["paths"]["outputs_dir"]) / "run_info.json"


def load_run_info(cfg):
    """Lê outputs/run_info.json (ou {} se ainda não existe). Apoio."""
    path = run_info_path(cfg)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_run_info(cfg, info):
    """Grava outputs/run_info.json. Apoio."""
    path = run_info_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def log_step(cfg, step, seconds, **extra):
    """Registra em run_info.json o tempo de uma etapa, junto com config, seed, versões e GPU atuais. Apoio."""
    info = load_run_info(cfg)
    info.update(config=cfg, seed=cfg["seed"], versoes=package_versions(),
                gpu=gpu_info(), plataforma=platform.platform())
    info.setdefault("etapas", {})[step] = {
        "segundos": round(seconds, 1), "quando": datetime.now().isoformat(timespec="seconds"), **extra}
    save_run_info(cfg, info)


@contextmanager
def timer(name, cfg=None):
    """Mede o tempo de um bloco (with timer("treino"): ...); com cfg, registra a etapa no run_info.json. Apoio."""
    start = time.time()
    yield
    seconds = time.time() - start
    print(f"[tempo] {name}: {seconds:.1f} s")
    if cfg is not None:
        log_step(cfg, name, seconds)
