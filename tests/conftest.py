"""Configuração do pytest: coloca a raiz do repositório no sys.path para `import src...` funcionar."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
