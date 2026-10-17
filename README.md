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
- **Early stopping da CNN sem olhar a validação:** dentro do treino de cada fold, ~1/5 dos pacientes
  (estratificado, com pelo menos um paciente com tumor) formam o **conjunto de parada**. O fold de
  validação só é usado para **medir**.
- Métricas reportadas como **média ± desvio entre os folds**, em três níveis: **por patch** (acurácia,
  acurácia balanceada, F1 macro, AUC, precisão/recall/F1 por classe), **por paciente** (cada paciente
  pesa igual) e **por centro** (centros sem tumor: só especificidade).
- A CNN é reprodutível: `cudnn.deterministic` está ligado, e duas execuções iguais dão os mesmos números.

## Decisões tomadas a partir dos dados (bons pontos para a defesa)

1. **Máscara de tecido com limiar fixo, não com Otsu por patch.** O Otsu supõe um histograma com
   duas classes. Como a maioria dos patches é só tecido, as duas classes viram "núcleo x estroma",
   e o "tecido" do Otsu é na verdade uma máscara de núcleos. O histograma de S de vários patches mostra
   o fundo em S < 20, então usamos esse limiar global (comparação no `01_eda`).
2. **Otsu dos núcleos calculado só com os pixels de tecido.** Num patch com fundo branco, o Otsu da
   imagem inteira separa fundo × tecido e marca o tecido todo como núcleo.
3. **lr = 1e-4 na CNN.** Com 1e-3 o treino foi instável: a loss oscilou e, em algumas execuções,
   ficou presa em ln 2 ≈ 0,69 (a rede responde sempre 50/50). Com 1e-4 a loss cai de forma contínua.
4. **Variação enorme entre folds.** O dev tem só 5 pacientes com tumor "de verdade" (300 patches
   cada). Pior: nos pacientes **16 e 17** (e no 41, do centro 2), os 300 patches de tumor vêm de uma
   região de ~300×200 px da lâmina. São patches de 256 px quase sobrepostos, ou seja, **uma única área
   de tecido repetida 300 vezes**. Na prática, o dev tem cerca de 3 regiões de tumor diversas, e errar um
   desses pacientes derruba o recall de um fold inteiro.
5. **Mudança de domínio já aparece no dev.** Os centros 4 e 5 só têm normais, e a CNN erra muito
   mais neles do que a regressão logística (números atualizados em `docs/RESUMO.md`).

## Estrutura

```
pathoshift.py          CLI: describe | prepare | features | train | predict | evaluate
src/
  data.py              lê o dataset HF; resize (amostragem); aumentação (rotação 90°/flip)
  manifest.py          data/manifest.jsonl (id, paciente, nó, centro, x, y, label, split, fold)
  splits.py            centro de teste externo + StratifiedGroupKFold por paciente + conjunto de parada
  color.py             RGB↔HSV, histogramas por canal, equalização do canal V
  filters.py           média, Gaussiano, mediana, Sobel, Laplaciano, FFT, alta frequência, passa-baixa
  segmentation.py      máscara de tecido e de núcleos (limiar/Otsu + morfologia + comp. conexos)
  features.py          30 features por patch → data/features.parquet
  restoration.py       ruídos/desfoque com nível 1–5 e restauração (Entrega 3; sem experimentos)
  models.py            SimpleCNN (a rede do slide) e StandardScaler + LogisticRegression
  train.py             validação cruzada, early stopping (conjunto de parada), checkpoints, modelo final
  predict.py           monta outputs/predictions.jsonl (CV + teste externo)
  evaluate.py          métricas por patch, por paciente e por centro → outputs/metrics.json
  aggregate.py         agregação por lâmina: média, máximo, top-k (pronta; sem experimentos)
  viz.py               grades, histogramas, matrizes de confusão, erros, curvas, comparação de modelos
  ambiente.py          primeira célula dos notebooks: detecta local × Colab e prepara o que falta
  utils.py             seed, config, versões, tempo, outputs/run_info.json
configs/default.yaml   todos os parâmetros
notebooks/             00_colab_setup, 01_eda, 02_baselines, 03_entrega1_resultados
tests/                 splits, conjunto de parada, operações de imagem, restauração, agregação, métricas, modelo
docs/                  RESUMO, um guia por integrante, relatório da Entrega 1, referência completa
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

### Rodar tudo (na ordem)

```powershell
python pathoshift.py describe                      # config, versões, GPU
python pathoshift.py prepare                       # dataset (~650 MB) + manifest + folds
python pathoshift.py features                      # ~3 min → data/features.parquet
python pathoshift.py train --model logreg          # CV do baseline clássico (< 1 s)
python pathoshift.py train --model cnn             # CV da CNN (até 20 épocas, early stopping)
python pathoshift.py train --model logreg --final  # modelo final no dev inteiro
python pathoshift.py train --model cnn --final
python pathoshift.py predict                       # outputs/predictions.jsonl (só CV)
python pathoshift.py evaluate                      # outputs/metrics.json + outputs/figuras/
python -m pytest -q                                # testes
```

O **teste externo** (`python pathoshift.py predict --teste-externo`, seguido de `evaluate`) **já foi
executado uma vez**, com a configuração congelada. O comando recusa uma segunda execução sem `--forcar`.

Opções úteis: `train --model cnn --epochs 3` (teste rápido) e `features --limit 300` (gera uma
amostra em `data/features_amostra.parquet`, sem tocar no arquivo completo).

### Arquivos de saída

| arquivo | conteúdo |
|---|---|
| `outputs/predictions.jsonl` | uma linha por patch e modelo: `{"id", "modelo", "origem" ("cv" ou "teste_externo"), "fold" (-1 no teste), "prob_tumor", "pred"}`; **sem rótulos** |
| `outputs/metrics.json` | por modelo: `cv` (por patch e por paciente com média, desvio e folds; por centro; detalhe por paciente) e `teste_externo` |
| `outputs/run_info.json` | config, seed, versões, GPU, tempo de cada etapa e o registro do teste externo |
| `outputs/figuras/` | matriz de confusão e erros (FP/FN) de cada modelo, curvas da CNN, comparação dos modelos |
| `outputs/<modelo>/` | `cv_oof.csv`, `cv_folds.csv`, `cv_info.json`, checkpoints, `final.*`, `teste_externo.csv` |

> **Interface de `predict` e `evaluate`:** o grupo não tinha o enunciado em mãos quando ela foi escrita.
> Se o enunciado pedir outro formato, ajuste `src/predict.py::to_records` (linhas do jsonl) e
> `src/evaluate.py::build_metrics` (estrutura do json).

## Notebooks: local ou Colab

Todos os notebooks começam pela **mesma célula de ambiente**, que detecta onde o kernel está rodando:

| | **Local** (kernel `.venv`) | **Colab** (extensão do VS Code) |
|---|---|---|
| código | a pasta do repositório | `git clone`/`git pull` de `REPO_URL` em `/content/pathoshift` |
| dependências | já instaladas no `.venv` | `pip install` do requirements (sem torch/ipykernel), uma vez por sessão |
| dados | usa `data/` (rode `prepare` e `features` antes) | roda `prepare` (download ~1 min); features lidas de `MyDrive/pathoshift/features.parquet` ou recalculadas (~3 min) e guardadas lá |
| resultados | `outputs/` local | `outputs/` vira atalho para `MyDrive/pathoshift/outputs` |

**Abrir localmente:** abra o `.ipynb` no VS Code → **Select Kernel** → **Jupyter Kernel** →
**Python 3.12 (pathoshift .venv)** (ou **Python Environments** → `.venv`).

> **Erro "Running cells with 'Python 3.12.x' requires the ipykernel package":** o VS Code escolheu um
> Python que não é o do projeto. Não instale nada nele: troque o kernel para o `.venv`, como acima. Para o
> kernel aparecer na lista, registre-o uma vez:
> `.venv\Scripts\python.exe -m ipykernel install --user --name pathoshift --display-name "Python 3.12 (pathoshift .venv)"`.

**Abrir no Colab:**
1. Preencha `REPO_URL` na primeira célula (o repositório precisa estar no GitHub).
2. **Select Kernel** → **Colab** → **New Colab Server** → **GPU T4** → faça login no Google.
3. Rode a primeira célula; ela pede para montar o Drive. Se a montagem não funcionar pela extensão,
   use o comando **`Colab: Mount Google Drive to Server`** do VS Code.

| notebook | o que faz | precisa de |
|---|---|---|
| `00_colab_setup` | roda o pipeline pela CLI (só treina com `RODAR_PIPELINE = True`) | dados + features |
| `01_eda` | análise exploratória com as técnicas das aulas | dados + features |
| `02_baselines` | análise da CV: folds, curvas, recall por paciente, erros | dados + resultados |
| `03_entrega1_resultados` | **só carrega** os resultados da Entrega 1 (abre em segundos) | só `outputs/` |

> Os notebooks foram testados com o kernel local. **O modo Colab não foi testado** num runtime real.

## Fluxo de trabalho do grupo

1. **Editar no VS Code** (local), rodar `python -m pytest -q` e testes rápidos na própria máquina.
2. `git add` / `git commit` / **`git push`** (de preferência em uma branch, depois um PR ou merge na `main`).
3. **No Colab**: a primeira célula de qualquer notebook faz o `git pull`; depois é só treinar ou analisar.
4. Os números que forem para o relatório devem vir de um commit identificado (anote o hash do `git log`).

Não edite código direto no Colab: as mudanças feitas lá se perdem quando a sessão cai e não
chegam ao GitHub.
