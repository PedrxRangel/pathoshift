"""Prepara o ambiente dos notebooks: local (kernel .venv) ou Colab (kernel na VM do Google).

Usa só a biblioteca padrão, porque no Colab este arquivo roda ANTES de instalar as dependências.

- Local: não instala nem baixa nada; só confere se data/ e outputs/ já têm o necessário.
- Colab: instala as dependências (uma vez por sessão), liga outputs/ ao Google Drive,
  baixa o dataset e gera o manifest (prepare) e lê as features do Drive (ou recalcula e
  guarda lá). O clone do repositório é feito na primeira célula do notebook.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

DRIVE_DIR = Path("/content/drive/MyDrive/pathoshift")
MARCA_DEPENDENCIAS = Path("/tmp/pathoshift_dependencias_ok")


def no_colab():
    """True se o kernel está rodando no Colab. Apoio."""
    return "google.colab" in sys.modules


def _rodar(cmd, cwd):
    """Roda um comando mostrando-o antes; para se der erro. Apoio."""
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def _instalar_dependencias(root):
    """pip install do requirements, pulando torch e ipykernel (o Colab já tem versões compatíveis). Apoio."""
    if MARCA_DEPENDENCIAS.exists():
        return
    linhas = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    filtrado = [l for l in linhas if l.strip() and not l.startswith("#")
                and not l.startswith(("torch==", "ipykernel=="))]
    req = Path("/tmp/requirements_colab.txt")
    req.write_text("\n".join(filtrado), encoding="utf-8")
    _rodar([sys.executable, "-m", "pip", "install", "-q", "-r", req], root)
    MARCA_DEPENDENCIAS.touch()
    print("Dependências instaladas. Se o pip trocou a versão do numpy ou do pandas, "
          "reinicie o kernel e rode esta célula de novo.")


def _ligar_outputs_ao_drive(root):
    """Faz outputs/ apontar para MyDrive/pathoshift/outputs (os resultados sobrevivem à queda da sessão). Apoio."""
    destino = DRIVE_DIR / "outputs"
    destino.mkdir(parents=True, exist_ok=True)
    outputs = root / "outputs"
    if outputs.is_symlink():
        return
    if outputs.exists():
        # copia o que já existir localmente para o Drive antes de trocar pela ligação
        shutil.copytree(outputs, destino, dirs_exist_ok=True)
        shutil.rmtree(outputs)
    os.symlink(destino, outputs)


def _features_colab(root):
    """Lê data/features.parquet do Drive; se não existir lá, calcula (~3 min) e guarda no Drive. Apoio."""
    local = root / "data" / "features.parquet"
    no_drive = DRIVE_DIR / "features.parquet"
    if local.exists():
        return
    if no_drive.exists():
        shutil.copy(no_drive, local)
        print("features.parquet copiado do Drive")
    else:
        _rodar([sys.executable, "pathoshift.py", "features"], root)
        shutil.copy(local, no_drive)


def preparar(root, precisa_dados=True, precisa_features=True, precisa_resultados=False):
    """Deixa o notebook pronto para rodar, no Colab ou local, e diz o que está faltando. Apoio."""
    root = Path(root)
    colab = no_colab()
    if colab:
        _instalar_dependencias(root)
        _ligar_outputs_ao_drive(root)
        if precisa_dados and not (root / "data" / "manifest.jsonl").exists():
            _rodar([sys.executable, "pathoshift.py", "prepare"], root)
        if precisa_features:
            _features_colab(root)

    faltando = []
    if precisa_dados and not (root / "data" / "manifest.jsonl").exists():
        faltando.append("data/manifest.jsonl  → python pathoshift.py prepare")
    if precisa_features and not (root / "data" / "features.parquet").exists():
        faltando.append("data/features.parquet → python pathoshift.py features")
    if precisa_resultados and not (root / "outputs" / "metrics.json").exists():
        faltando.append("outputs/metrics.json → rode o pipeline (README, seção 'Rodar tudo')")

    print(f"Ambiente: {'Colab' if colab else 'local'} | raiz do projeto: {root}")
    for f in faltando:
        print("  FALTA:", f)
    if not faltando:
        print("  Tudo pronto.")
    return root
