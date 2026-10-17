# Integrante 1 — Dados e protocolo de avaliação

Nada aqui é conteúdo de uma aula específica: é **metodologia** (como dividir e medir sem se enganar).
Código: `src/data.py`, `src/manifest.py`, `src/splits.py`, `src/utils.py` e `pathoshift.py`.
Leia antes o `docs/RESUMO.md`.

## a) O que você precisa dominar

1. O dataset real e o que diferia do card
2. A EDA que importa: tumor concentrado em poucos pacientes, e de região única
3. O manifest
4. O centro 2 como teste externo (e como garantimos que foi usado uma vez)
5. Divisão por paciente e vazamento
6. StratifiedGroupKFold e o conjunto de parada do early stopping
7. Seed, config e reprodutibilidade
8. Os comandos da CLI e o que cada um gera

## b) Explicação de cada tópico

**1. Dataset** — `src/data.py::load_hf_dataset`. É o `chehablab/MiniCAMELYON17`, com a revisão fixada no config.
As colunas são `image, tumor, patient_id, node_id, centre_id, patch_x, patch_y`, e o código **confere o schema**
e para se faltar alguma. Diferenças em relação ao card: as imagens têm **256×256** (não 512); **não existem**
`slide_id` nem `stage`; e há **6 centros** (0–5), não 5. Por isso uma lâmina é definida como (patient_id, node_id).
![Patches](figuras/01_patches_por_centro.png)

**2. EDA que importa.** O tumor está nos centros 0 (1.500 patches, 5 pacientes) e 2 (1.200, 4 pacientes).
Os centros 1 e 3 têm ~30 patches de tumor cada; os centros 4 e 5, nenhum. Em 3 lâminas (pacientes **16, 17 e
41**), os 300 patches de tumor ficam numa área de ~300×200 px: com patches de 256 px, é quase a mesma imagem
300 vezes. Também há patches sobrepostos e 10 pares idênticos (hash MD5), sempre do mesmo paciente.
![Região única](figuras/11_regiao_unica_tumor.png)

**3. Manifest** — `src/manifest.py`. O `data/manifest.jsonl` tem uma linha por patch com
`id, patient_id, node_id, centre_id, x, y, label, split, fold`. O `id` é a linha no dataset do HF;
`split` é `dev` ou `test_ext`; `fold` vai de 0 a 3 (−1 no teste). Tudo depois disso lê o manifest, e
`summarize` gera as tabelas por centro e por fold.

**4. Centro 2 = teste externo** — `src/splits.py::make_splits`. Deixamos um **hospital inteiro** de fora
para medir a robustez à mudança de domínio. Só os centros 0 e 2 tinham tumor suficiente; com o 2 de fora,
o dev fica com o centro 0. O teste rodou **uma vez**, com `predict --teste-externo`. O `run_info.json`
registra data, hash da config, commit e a declaração de que nada mais será ajustado olhando o centro 2.
Rodar de novo exige `--forcar`.

**5. Por paciente e vazamento.** Vazamento acontece quando a avaliação "vê" algo do treino. Patches do
mesmo paciente são quase iguais (vizinhos, sobrepostos, até idênticos). Se o paciente 17 estiver no treino
e na validação, o modelo acerta por memória. Por isso **cada paciente fica inteiro num fold**, e o
`tests/test_splits.py` confere isso no manifest real.

**6. StratifiedGroupKFold e conjunto de parada** — `assign_folds` e `early_stopping_split`.
- **K-fold:** 4 partes; treina em 3 e valida na outra, quatro vezes.
- **Group:** o grupo é o paciente. **Stratified:** a proporção de tumor fica parecida entre os folds.
- Como há poucos pacientes com tumor, a divisão é **refeita com outra semente** até todo fold ter ≥ 100
  patches de tumor. Passou de primeira (303, 300, 630 e 328).
- **Conjunto de parada:** dentro do treino de cada fold, ~1/5 dos pacientes (com ≥ 100 tumores) são
  separados **só para o early stopping** da CNN. O fold de validação **só mede**. Antes, a parada olhava a
  própria validação, o que deixava a métrica otimista.

**7. Seed, config e reprodutibilidade** — `utils.set_seed`, `configs/default.yaml` e `utils.log_step`.
Todos os parâmetros ficam no YAML (seed 42, folds, limiares, hiperparâmetros). `set_seed` fixa Python,
NumPy e PyTorch e liga `cudnn.deterministic`: duas execuções iguais da CNN deram **números idênticos**
(conferido). Cada comando registra em `outputs/run_info.json` a config, a seed, as versões, a GPU e o tempo.
O `prepare` refeito gerou um manifest **byte a byte igual** ao anterior.

**8. CLI** — `pathoshift.py`:

| comando | lê | gera | tempo (RTX 5060) |
|---|---|---|---|
| `prepare` | dataset HF | `data/manifest.jsonl`, tabelas-resumo | 1,7 s (com cache) |
| `features` | manifest + imagens | `data/features.parquet` | 186 s |
| `train --model logreg` / `cnn` | manifest (+ features) | `outputs/<m>/cv_*`, checkpoints | 0,5 s / 240 s |
| `train --model X --final` | dev inteiro | `outputs/<m>/final.*` | 0,2 s / 94 s |
| `predict [--teste-externo]` | modelos | `outputs/predictions.jsonl` | 0,2 s / 27 s |
| `evaluate` | predições + manifest | `outputs/metrics.json`, `outputs/figuras/` | 5 s |

## c) Decisões que tomamos e por quê

| decisão | por quê | onde |
|---|---|---|
| Conferir o schema antes de usar | o card estava errado (512×512, colunas inexistentes) | `load_hf_dataset` |
| Lâmina = (paciente, nó) | não existe `slide_id` | `manifest.py` |
| Centro 2 como teste externo | precisa ter tumor suficiente; o 0 fica para treinar | `splits.test_centre` |
| 4 folds por paciente | evita vazamento; com 4 folds, cada um ainda tem tumor | `assign_folds` |
| ≥ 100 tumores por fold | força um paciente com tumor "de verdade" em cada fold | `min_tumor_patches_per_fold` |
| Conjunto de parada no treino | o fold de validação só mede, sem otimismo | `early_stopping_split` |
| Teste único com trava | olhar o teste e ajustar invalida o teste | `cmd_predict` (`--forcar`) |
| cudnn determinístico | antes, duas execuções iguais davam AUCs diferentes | `set_seed` |
| `run_info.json` | rastrear config, versões, GPU e tempos de cada resultado | `log_step` |

## d) Perguntas prováveis da defesa

1. **Por que não sortear patches para teste?** Porque patches do mesmo paciente são quase iguais e
   cairiam em treino e teste ao mesmo tempo. A métrica mediria memória, não generalização.
2. **Por que um centro inteiro como teste?** Para medir a robustez a um hospital nunca visto, com outra
   coloração e outro scanner. É o objetivo "mudança de domínio" do trabalho.
3. **Por que o centro 2?** Só os centros 0 e 2 têm tumor suficiente para medir recall. Com o 2 no teste, o
   centro 0 (5 pacientes com tumor) fica para treinar.
4. **O que é vazamento? Dê um exemplo daqui.** É a avaliação usar informação do treino. Exemplo: patches
   vizinhos (ou idênticos) do paciente 17 em treino e validação ao mesmo tempo.
5. **O que faz o StratifiedGroupKFold?** Divide em K partes mantendo cada paciente (grupo) inteiro numa
   parte, com a proporção de tumor parecida entre as partes.
6. **Como garantem tumor em todo fold?** O `assign_folds` refaz a divisão com sementes 42, 43… até todo
   fold ter ≥ 100 patches de tumor. A primeira já serviu.
7. **Por que separar um conjunto de parada?** Se a melhor época for escolhida olhando o fold de validação,
   a métrica fica otimista. Agora a parada usa pacientes do treino, e a validação só mede.
8. **Por que o desvio entre folds é tão grande?** Cada fold depende de 1 ou 2 pacientes com tumor, e os
   pacientes 16 e 17 são de região única (300 patches quase iguais). Errar um zera o recall do fold.
9. **Os resultados são reprodutíveis?** Sim: seed 42, cudnn determinístico, revisão do dataset fixada e
   versões registradas. Conferimos que duas execuções iguais da CNN dão números idênticos.
10. **Como provam que o centro 2 foi usado uma vez?** O `run_info.json` registra a execução (data, hash da
    config, commit), e o comando recusa repetir sem `--forcar`. Ressalva: a exploração inicial dos limiares
    viu imagens (sem rótulos) do centro 2.

## e) O mínimo que você precisa saber das outras partes

- Pré-processamento: HSV; tecido = S > 20 (o Otsu falhou); núcleos = Otsu no cinza invertido só dentro do tecido (integrante 2).
- Features: 24 de histograma HSV + Sobel + Laplaciano + energia de alta frequência + 3 de núcleos → regressão logística balanceada (integrante 3).
- CNN: Conv32→Pool→Conv64→Pool→Conv64→Dense64→Dense2, entrada 128×128 com pixels/255, lr 1e-4 (integrante 4).
- Métricas: acurácia balanceada, F1 macro e AUC, porque "sempre normal" teria 91,6% de acurácia (integrante 4).
- No centro 2, os dois modelos quase só dizem "normal"; a CNN ainda ordena (AUC 0,70) e a regressão logística não (0,42) (integrante 4).
