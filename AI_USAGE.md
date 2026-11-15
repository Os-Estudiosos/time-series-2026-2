# Uso de IA generativa — Task 1

> **Rascunho para revisão do grupo.** Este documento é a declaração de
> responsabilidade da equipe. Confiram se ele descreve o que de fato aconteceu e
> ajustem os itens 1 a 3 antes de entregar.

## 1. Ferramentas usadas

- **Claude (Anthropic), via Claude Code** — assistente de linha de comando com
  acesso ao repositório. Usado para escrita de código, revisão do notebook e
  conferência contra o enunciado.

Nenhuma outra ferramenta de IA generativa foi usada nesta entrega.

## 2. Onde ajudou

| Parte do pipeline | Papel da ferramenta |
|---|---|
| Diagnóstico (seção 2) | formatação das figuras, redação dos comentários de leitura dos correlogramas |
| Identificação de ordem (seção 4) | implementação do KPSS iterativo e do grid search de AICc |
| Backtest do `D` | desenho do esquema de origem rolante e do critério de desempate |
| Exportação (seção 7) | conferência do formato exigido por `checks/check_task1.py` |
| Revisão | leitura cruzada do notebook contra o enunciado, item por item |

A escolha metodológica — tratar os zeros de Natal por interpolação, travar `d` e
`D` fora do grid de AICc, decidir o `D` do `HOBBIES` por backtest — foi do grupo.
A ferramenta implementou e mediu; a decisão e a justificativa são nossas.

## 3. Prompts representativos

1. *"Analise o notebook `main.ipynb` com o enunciado em `content/enunciado.pdf` e
   verifique se todos os requisitos estão sendo satisfeitos e se todas as
   proibições estão sendo respeitadas."*
2. *"Planeje a implementação de todos os requisitos. Eu gostaria que tivesse
   validação do `d` por KPSS e do `P` e `Q` por grid search com comparação de
   AICc."*
3. *"O `HOBBIES` dá sinal conflitante para o `D`: força sazonal 0,465 sugere
   `D=0`, mas a ACF em nível nos lags 7/14/21 fica em 0,51-0,55 sem decair. Como
   decidir?"*
4. *"Confirme que `SARIMAXResults` expõe `.aicc` e meça quanto tempo leva um fit
   de `SARIMA(1,0,0)(0,1,1)₇` sobre `store_total`."*

## 4. Erros da IA que o grupo corrigiu

1. **Seleção de ordem por `argmin(aicc)` sem guarda.** A primeira versão do grid
   escolhia o menor AICc direto. Três ajustes em `store_total` convergiram para um
   ótimo degenerado (`llf == 0.0` exato, filtro de Kalman colapsado), com AICc
   artificialmente baixo e **previsão identicamente zero nos 28 dias**. O erro
   passaria no corretor automático, porque as métricas continuariam internamente
   consistentes. Corrigido com a guarda `llf < 0 and converged` e um assert de
   plausibilidade sobre a previsão do modelo vencedor.

2. **Escala do MASE calculada sobre o treino interpolado.** Como o pipeline do
   SARIMA usa a série com os zeros de 25/12 interpolados, a sugestão inicial era
   reaproveitar essa série também na escala do MASE. Isso diverge em até 3,7% da
   definição do enunciado, que usa o `treino.csv` cru. Corrigido: métricas e
   baselines ficam no dado cru, interpolação só no ajuste do SARIMA.

3. **`maxiter` no valor padrão (50).** O grid inicial usava o default de
   `.fit()`, com o qual os modelos de ordem maior não convergem — o AICc fica
   enviesado e o ranking inteiro vira ruído. Corrigido para `maxiter=200`.

## 5. Responsabilidade

O grupo responde integralmente pelo que foi entregue. Todo código gerado com apoio
da ferramenta foi lido, executado e conferido pelos membros; os resultados
numéricos do notebook foram reproduzidos do zero e validados contra
`checks/check_task1.py`. As interpretações estatísticas e as decisões de
modelagem são de responsabilidade dos autores, não da ferramenta.

---

# Uso de IA generativa — Task 2 (features, árvore e rede)

> **Rascunho para revisão do grupo.** Conferir os itens 2 a 4 antes de entregar.

Esta seção cobre as três classes da Task 2 — o `linear` de 01/11 com as features, a
árvore de 08/11 e a rede de 15/11. A seção da Task 1 acima fica como foi entregue.

## 1. Ferramentas usadas

- **Claude (Anthropic), via Claude Code** — com acesso ao repositório. Usado para
  escrever as seções 9 a 12 do `main.ipynb` (features e `linear`, modelos de árvore,
  híbridos, holdout e a classe rede) e conferi-las, item por item, contra o
  `content/enunciado_task2.pdf`.

Nenhuma outra ferramenta de IA generativa foi usada nesta entrega.

## 2. Onde ajudou

| Parte | Papel da ferramenta |
|---|---|
| Features (seção 9) | matriz das 54 colunas: indicadores de dia da semana, seno/cosseno anual e mensal, Natal e véspera, `snap_CA`, indicadores de evento nas duas posições, preço da própria série, lags e médias móveis |
| Padronização e anti-vazamento (seção 9) | média, desvio e vocabulário de eventos calculados só no treino; recursão que devolve a previsão do próprio modelo como lag |
| `linear` e regularização (seções 10 e 10.1) | Ridge com intercepto livre via `lstsq`, seleção de `alpha` em quatro origens históricas, comparativo MQO/Ridge/Lasso/Elastic Net |
| Seleção de modelo (11.1) | backtest de origem rolante com as quatro origens da Task 2, grades das cinco famílias |
| Ajuste e previsão (11.2) | reaproveitamento das features e da recursão das seções 9 e 10 |
| Híbridos (11.2–11.3) | combinação por média simples de SARIMA, linear e árvore |
| Holdout (11.4) | reajuste com treino + validação e exportação de `previsoes_holdout.csv` |
| Métricas | MAPE, sMAPE e viés em `metricas_completas.csv`, sem mudar as colunas de `metricas.csv` |
| Rede: arquiteturas (seção 12) | MLP, RNN e LSTM em `torch`, todas sobre as mesmas 54 features e a mesma recursão; nas recorrentes, a janela da própria série entra pela célula e as covariáveis do dia entram na camada de saída |
| Rede: varredura (12.1) | grades das três famílias e a sondagem de custo que dimensionou a grade antes de rodar os 288 ajustes |
| Rede: variação por semente (12.4) | reajuste da configuração escolhida com cinco sementes, para separar diferença de arquitetura de ruído de inicialização |
| Rede: viés do `FOODS` (12.5) | comparação do viés da rede com o da árvore, ligada à limitação de extrapolação registrada em 11.5 |
| Ambiente | diagnóstico do `torch` e a entrada da build de CPU no `pyproject.toml` |

A escolha das famílias de árvore (árvore de decisão, Random Forest, AdaBoost, Gradient
Boosting e XGBoost), dos três híbridos e das três arquiteturas de rede (MLP, RNN e LSTM)
foi do grupo.

## 3. Prompts representativos

1. *"Crie uma branch nova chamada tree_models, que será o lugar onde faremos a task 3.
   [...] Nós iremos fazer Decision Tree, Random Forest, AdaBoost, Gradient Boosting e
   XGBoost. Nós vamos treinar os modelos com treino.csv, reportar as métricas (MAPE,
   MASE, MAE, RMSE, etc) com validacao.csv para escolher o melhor modelo e fazer as
   previsões para holdout_datas.csv com o melhor modelo escolhido."*
2. *"Vamos fazer mais três modelos: SARIMA + regressão linear, SARIMA + melhor árvore e
   SARIMA + linear + melhor árvore para ver a comparação entre eles, os baselines e os
   modelos crus."*

3. *"No desenvolvimento da regressão, fizemos apenas usando ridge regression. Contudo,
   quero fazer um comparativo entre regressão sem nenhuma regularização, usando ridge
   regression, usando lasso e usando elastic net. Dentre esses modelos, vamos escolher o
   melhor [...]"* — e, no mesmo pedido, as figuras 9 e 15 da Task 1 refeitas.
4. *"Tudo que o `content/enunciado_task2.pdf` pede está implementado em `main.ipynb`?"* —
   usado como conferência do que faltava antes do último domingo.
5. *"Se o enunciado não fala nada sobre qual implementar, vamos implementar as três: MLP,
   LSTM e RNN, e fazer uma comparação. Não esqueça de adicionar varredura de parâmetros em
   todos os modelos."*

## 4. Erros da IA que o grupo corrigiu

1. **Seleção do modelo pelo erro da validação.** O pedido inicial era escolher a melhor
   árvore pelas métricas de `validacao.csv`, e a ferramenta seguiu por esse caminho. O
   enunciado zera o critério de split nesse caso ("a validação publica; não escolhe"). A
   seleção passou para o backtest de origem rolante dentro do treino, nas quatro origens
   que terminam antes de 2016-03-27, e a validação ficou só para medir. A regra valeu para
   as três classes: o `alpha` do linear, os hiperparâmetros das cinco árvores e as 24
   configurações de rede saem todos do backtest. Na rede isso teve duas consequências
   concretas: **nada de early stopping** (parar o treino pelo erro dos 28 dias é o mesmo
   vazamento por outra porta) e a tabela de cinco sementes da seção 12.4 é **medição
   depois do fato** — as linhas entregues são as da semente 2026, fixada antes. Também
   ficou registrado no notebook que o backtest escolheu uma família de árvore que não é a
   melhor na validação, e que não trocamos depois de ver o resultado.

2. **Comentário errado sobre o `torch`, e a reprodução quebrada por baixo.** A seção 0 do
   notebook dizia que o `torch` era "instalado fora do `uv` (build ROCm para GPU AMD)".
   Na verificação, o que havia era uma build **de CPU** no Python 3.12 do sistema,
   enquanto o `.venv` do projeto é 3.14 — `import torch` falhava dentro do ambiente que o
   notebook de fato usa, e o `TORCH_OK` dava `False` em silêncio. Escrever a classe rede
   sobre essa suposição quebraria o comando de reprodução do `README.md` na máquina de
   quem corrige. Corrigido com `torch>=2.14,<2.15` no `pyproject.toml`, apontando para o
   índice CPU-only do PyTorch: o wheel padrão de Linux arrastaria ~2-3 GB de bibliotecas
   CUDA que nada aqui usa.

3. **Reabrir entrega de domingo que já passou.** Duas vezes o caminho mais curto era
   reescrever o que já estava entregue. O Lasso venceu o comparativo de regressões depois
   de 01/11, e a sugestão era promovê-lo a `linear`; as linhas `linear` (Ridge) ficaram
   como foram entregues e o Lasso entrou como `linear_lasso`. E as previsões do holdout
   iriam para `previsoes_validacao.csv`, o que violaria a trava de não haver nenhuma data
   fora de 2016-03-28–2016-04-24; foram para `previsoes_holdout.csv`. A seção 12 segue a
   mesma regra: só acrescenta linhas, e o notebook confere com `assert` que
   `naive_sazonal`, `sarima`, `linear` e `boosting` continuam nas três séries.

## 5. Responsabilidade

Igual à da Task 1: o grupo leu, executou e conferiu todo o código e responde pelas
decisões de modelagem.
