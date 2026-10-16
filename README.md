# PathoShift Challenge

Trabalho de Processamento de Imagens: classificação de patches histopatológicos H&E
(normal × tumor), com análise por lâmina e robustez a mudança de domínio (centro hospitalar).

Tudo no projeto usa **somente o conteúdo das aulas**. Cada função tem uma docstring curta
dizendo o que faz e de qual aula vem o conceito (funções de "apoio" não são conteúdo de aula).

## Mapa das aulas → código

| aula | conteúdo | onde está |
|---|---|---|
| 01 – Fundamentos | RGB/HSV, histograma, equalização, amostragem | `color.py`, `data.resize` |
| 03 – Filtros no espaço | média, Gaussiano, Sobel, Laplaciano, transf. geométricas | `filters.py`, `data.augment` |
| 04 – Filtros na frequência | FFT, espectro, passa-alta ideal, passa-baixa Gaussiano | `filters.py` |
| 05 – Restauração | ruído gaussiano e sal-e-pimenta, mediana, média | `restoration.py`, `filters.median_filter` |
| 06 – Segmentação | limiarização, Otsu, componentes conexos | `segmentation.py` |
| 07 – Morfologia | abertura, fecho, erosão, extração de borda | `segmentation.py` |
| 08 – Classificação | CNN Conv → Pooling → Dense, avaliação | `models.py`, `train.py`, `evaluate.py` |

> **Confiram a numeração das aulas.** Ela foi inferida (Fundamentos = 01, Filtros no espaço = 03,
> Segmentação = 06...). Se estiver diferente, basta um buscar/substituir por "Aula 0X".

## Dados

Dataset `chehablab/MiniCAMELYON17` (Hugging Face, revisão fixada em `configs/default.yaml`).
O schema foi conferido no dataset real. O dataset card cita colunas que não existem (`slide_id`, `stage`)
e diz que as imagens têm 512×512, mas elas têm **256×256**.

| coluna | descrição |
|---|---|
| `image` | patch JPEG 256×256 RGB |
| `tumor` | rótulo 0/1 |
| `patient_id`, `node_id` | uma **lâmina** = (patient_id, node_id) |
| `centre_id` | hospital de origem: **6 centros (0 a 5)** |
| `patch_x`, `patch_y` | posição do patch na lâmina |

São 26.753 patches, 54 pacientes e 83 lâminas. Os tumores estão concentrados nos centros 0 e 2.
Há patches sobrepostos da mesma lâmina e 10 pares de imagens idênticas, sempre do mesmo paciente;
como a divisão é por paciente, isso não vaza informação entre treino e validação.

## Protocolo de avaliação

- **Teste externo = centro 2** (8.100 patches, 4 pacientes com tumor). Usado **uma única vez**, no final.
- **Desenvolvimento = centros 0, 1, 3, 4, 5**, com `StratifiedGroupKFold` de **4 folds** e grupo = paciente.
  Todo fold tem pelo menos um paciente com tumor (≥ 300 patches de tumor por fold).
- Métricas reportadas como **média ± desvio entre os folds**: acurácia, acurácia balanceada,
  F1 macro, AUC, e precisão/recall/F1 por classe, além da matriz de confusão.

## Decisões tomadas a partir dos dados (bons pontos para a defesa)

1. **Máscara de tecido com limiar fixo, não com Otsu por patch.** O Otsu supõe um histograma com
   duas classes. Como a maioria dos patches é só tecido, as duas classes viram "núcleo x estroma",
   e o "tecido" do Otsu é na verdade uma máscara de núcleos. O histograma de S de vários patches mostra
   o fundo em S < 20, então usamos esse limiar global (comparação no `01_eda`).
2. **Otsu dos núcleos calculado só com os pixels de tecido.** Num patch com fundo branco, o Otsu da
   imagem inteira separa fundo × tecido e marca o tecido todo como núcleo.
3. **lr = 1e-4 na CNN.** Com 1e-3 a rede colapsou em 2 dos 4 folds: a loss ficou presa em ln 2 ≈ 0,69,
   ou seja, ela prevê sempre 50/50. Com 1e-4 ela aprende.
4. **Variação enorme entre folds.** O dev tem só 5 pacientes com tumor "de verdade" (300 patches
   cada). Pior: nos pacientes **16 e 17** (e no 41, do centro 2), os 300 patches de tumor vêm de uma
   região de ~300×200 px da lâmina. São patches de 256 px quase sobrepostos, ou seja, **uma única área
   de tecido repetida 300 vezes**. Os folds com recall de tumor zero são exatamente os desses
   pacientes (ver seção 4 do `02_baselines`). Na prática, o dev tem cerca de 3 regiões de tumor diversas.
5. **Mudança de domínio já aparece no dev.** A CNN marca como tumor 91% dos patches normais do centro 5
   e 72% dos do centro 4, enquanto a regressão logística acerta quase todos.

## Estrutura

```
pathoshift.py          CLI: prepare | features | train | evaluate | describe | predict (stub)
src/
  data.py              lê o dataset HF; resize (amostragem); aumentação (rotação 90°/flip)
  manifest.py          data/manifest.jsonl (id, paciente, nó, centro, x, y, label, split, fold)
  splits.py            centro de teste externo + StratifiedGroupKFold por paciente
  color.py             RGB↔HSV, histogramas por canal, equalização do canal V
  filters.py           média, Gaussiano, mediana, Sobel, Laplaciano, FFT, alta frequência, passa-baixa
  segmentation.py      máscara de tecido e de núcleos (limiar/Otsu + morfologia + comp. conexos)
  features.py          30 features por patch → data/features.parquet
  restoration.py       ruídos/desfoque com nível 1–5 e restauração (Entrega 3; sem experimentos)
  models.py            SimpleCNN (a rede do slide) e StandardScaler + LogisticRegression
  train.py             validação cruzada, early stopping, checkpoints, teste final
  evaluate.py          métricas por classe e macro, tabela média ± desvio
  aggregate.py         agregação por lâmina: média, máximo, top-k (pronta; sem experimentos)
  viz.py               grades de exemplos, histogramas, matriz de confusão, erros, heatmap da lâmina
  utils.py             seed, config, versões, tempo
configs/default.yaml   todos os parâmetros
notebooks/             00_colab_setup, 01_eda, 02_baselines
tests/                 splits, operações de imagem, restauração, agregação, modelo
```

## Setup local (Windows)

Precisa de Python 3.10+ (usamos 3.12).

```powershell
python -m venv .venv
.venv\Scripts\activate
# com GPU NVIDIA (o torch do PyPI no Windows é só CPU):
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Com `uv` (mais rápido): `uv venv .venv --python 3.12` e troque `pip install` por `uv pip install`.

```powershell
python pathoshift.py describe                      # config, versões, GPU
python pathoshift.py prepare                       # dataset (~650 MB) + manifest + folds
python pathoshift.py features                      # ~3 min → data/features.parquet
python pathoshift.py train --model logreg          # CV do baseline clássico (segundos)
python pathoshift.py train --model cnn             # CV da CNN (early stopping)
python pathoshift.py train --model cnn --epochs 3  # teste rápido
python pathoshift.py evaluate --model cnn          # métricas por fold e por centro
python -m pytest -q                                # testes
# só no fim do trabalho:
python pathoshift.py train --model logreg --final
python pathoshift.py train --model cnn --final     # usa o nº médio de épocas escolhido na CV
```

Os resultados ficam em `outputs/<modelo>/`: `cv_folds.csv`, `cv_oof.csv`, `cv_history.csv`
(com o tempo de cada época) e os checkpoints `fold<k>.pt`.

## Colab dentro do VS Code

Extensões: **Jupyter** (`ms-toolsai.jupyter`) e **Google Colab** (`Google.colab`).

> O kernel do Colab roda numa **VM do Google**, não na sua máquina. Os arquivos locais não
> existem lá. O código chega no Colab pelo **GitHub**, e os resultados ficam salvos no **Google Drive**.

1. Abra `notebooks/00_colab_setup.ipynb` no VS Code.
2. **Select Kernel** (canto superior direito) → **Colab** → **New Colab Server** → escolha **GPU T4**.
3. Faça login na conta Google quando o navegador abrir e autorize.
4. Preencha `REPO_URL` na primeira célula de código e rode tudo em ordem. O notebook monta o
   Drive, clona o repo, instala as dependências, confere a GPU e roda `describe`, `prepare` e `features`.
5. `outputs/` vira um atalho para `MyDrive/pathoshift/outputs`, então checkpoints e resultados
   não se perdem quando a sessão cai. As features também são guardadas no Drive.

## Fluxo de trabalho do grupo

1. **Editar no VS Code** (local), rodar `python -m pytest -q` e testes rápidos na própria máquina.
2. `git add` / `git commit` / **`git push`** (de preferência em uma branch, depois um PR ou merge na `main`).
3. **No Colab**: rodar de novo a célula 2 do `00_colab_setup` (faz o **`git pull`**) e então treinar.
4. Os números que forem para o relatório devem vir de um commit identificado (anote o hash do `git log`).

Não edite código direto no Colab: as mudanças feitas lá se perdem quando a sessão cai e não
chegam ao GitHub.
