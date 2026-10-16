# PathoShift — resumo para o grupo

**Problema.** Classificar patches 256×256 de linfonodo corados com H&E em **normal** ou **tumor**
(metástase de mama). Há duas dificuldades: só ~10% dos patches são tumor, e as lâminas vêm de
**6 hospitais (centros)** com cores diferentes (*mudança de domínio*). Dataset: MiniCAMELYON17, com
26.753 patches, 54 pacientes e 83 lâminas; uma lâmina = (paciente, nó).

## Pipeline

```
 HF MiniCAMELYON17 ─ prepare ─► manifest.jsonl  (centro 2 = TESTE EXTERNO; resto = dev em 4 folds POR PACIENTE)
        │
        ├─ features ─► 30 números/patch: histograma HSV · Sobel · var. Laplaciano · energia alta freq. · núcleos
        │              (Fundamentos · Filtros no espaço · Frequência · Segmentação · Morfologia)
        │                 └─► StandardScaler + Regressão logística (class_weight="balanced")
        │
        └─ patch 128×128 ─► SimpleCNN  Conv32→Pool→Conv64→Pool→Conv64→Dense64→Dense2  (Classificação)
                              CrossEntropy ponderada · Adam lr=1e-4 · rotações/flips · early stopping
        │
 train (CV 4 folds) ─► predict ─► predictions.jsonl ─► evaluate ─► metrics.json + figuras
                        (por patch · por paciente · por centro; centro 2 testado UMA vez)
```

## Decisões principais (uma linha cada)

- **Centro 2 como teste externo**, usado uma única vez: só os centros 0 e 2 têm tumor suficiente.
- **Folds por paciente** (StratifiedGroupKFold): patches do mesmo paciente são quase iguais, então misturá-los seria vazamento.
- **Todo fold com tumor** (≥ 100 patches): a divisão é refeita até isso valer; passou na primeira semente.
- **Tecido com S > 20 fixo, não Otsu:** quase todo patch é só tecido, e o Otsu divide em núcleo × estroma.
- **Otsu dos núcleos só dentro do tecido:** na imagem inteira, o Otsu separaria fundo × tecido.
- **Só conteúdo das aulas:** saíram LBP, ResNet18 e normalização Macenko.
- **Peso de classe N/(2·N_c)** nos dois modelos: o dev tem ~92% de normais.
- **lr = 1e-4:** com 1e-3 a CNN ficou instável (loss presa em ln 2 ≈ 0,69 em algumas execuções).
- **Early stopping por um conjunto de parada** (~1/5 dos pacientes do treino de cada fold); o fold de validação só mede.
- **Reprodutível:** seed 42 e `cudnn.deterministic`; config, versões, GPU e tempos ficam em `outputs/run_info.json`.

## Resultados (CV = média ± desvio entre os 4 folds; centro 2 = teste único)

| métrica | Reg. logística CV | SimpleCNN CV | Reg. logística **centro 2** | SimpleCNN **centro 2** |
|---|---|---|---|---|
| acurácia | 0,861 ± 0,006 | 0,834 ± 0,045 | 0,841 | 0,850 |
| acurácia balanceada | 0,632 ± 0,218 | 0,674 ± 0,250 | 0,494 | 0,504 |
| F1 macro | 0,581 ± 0,137 | 0,566 ± 0,134 | 0,457 | 0,471 |
| AUC | 0,770 ± 0,236 | 0,677 ± 0,306 | 0,416 | 0,698 |
| recall tumor | 0,358 ± 0,464 | 0,472 ± 0,544 | 0,000 | 0,012 |
| especificidade | 0,906 ± 0,035 | 0,876 ± 0,087 | 0,988 | 0,996 |
| acc. balanceada **por paciente** | 0,618 ± 0,214 | 0,639 ± 0,198 | 0,491 | 0,504 |

Centros só com normais (CV), especificidade da regressão logística × CNN: **centro 4: 0,987 × 0,577;
centro 5: 1,000 × 0,117**.

**Leitura em quatro pontos:**

1. **Na CV, os dois modelos empatam.** O desvio é enorme porque cada fold depende de 1 ou 2 pacientes
   com tumor, e os pacientes **16 e 17** têm o tumor numa região de ~300×200 px (300 patches quase
   iguais). O paciente 17 foi errado pelos dois modelos.
2. **No centro 2, os dois praticamente só dizem "normal"** (recall ≈ 0). A acurácia de 0,84–0,85 é
   enganosa: é a fração de normais.
3. **A CNN ainda ordena os patches** (AUC 0,70; por paciente, até 0,94): os tumores do centro 2 recebem
   notas maiores, mas abaixo do limiar 0,5. A regressão logística ordena **pior que o acaso** (AUC 0,42):
   as features de cor não transferem para a coloração do centro 2.
4. **A mudança de domínio já aparecia no dev:** a CNN marcou como tumor 42% e 88% dos normais dos centros 4 e 5.

**Custo:** o pipeline inteiro leva ~9,3 min numa RTX 5060 (features 186 s, CV da CNN 240 s, CNN final 94 s,
regressão logística < 1 s). A CNN tem 3,27 milhões de parâmetros.

## Onde estudar

| quem | arquivo |
|---|---|
| Integrante 1 — dados e protocolo | `docs/integrante_1_dados_e_protocolo.md` |
| Integrante 2 — pré-processamento e segmentação | `docs/integrante_2_preprocessamento_e_segmentacao.md` |
| Integrante 3 — filtros, features e baseline | `docs/integrante_3_filtros_features_baseline.md` |
| Integrante 4 — CNN e avaliação | `docs/integrante_4_cnn_e_avaliacao.md` |
| Relatório da Entrega 1 (rascunho) | `docs/relatorio_entrega1.md` |
| Tudo em detalhe (estado antes da Entrega 1) | `docs/referencia/REFERENCIA_COMPLETA.md` |
| Resultados no notebook | `notebooks/03_entrega1_resultados.ipynb` |
