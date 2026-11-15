# Séries Temporais — FGV EMAp (2026/2)

Repositório de entregas do curso de Séries Temporais da graduação em Ciência de
Dados e Inteligência Artificial da Fundação Getulio Vargas.

## Task 1 e Task 2 — Diagnóstico, baselines, SARIMA e as três classes

Previsão de 28 dias (2016-03-28 a 2016-04-24) para três séries diárias da loja
`CA_1` do M5: `store_total`, `FOODS` e `HOBBIES`.

### Como reproduzir

Um comando, a partir da raiz do repositório:

```bash
uv run --frozen jupyter nbconvert --to notebook --inplace --execute main.ipynb
```

Isso lê `dados/`, executa o notebook inteiro e regrava na raiz:

- `previsoes_validacao.csv` — 1932 linhas (`date`, `series`, `modelo`, `yhat`), incluindo `linear`, as quatro regressões, as árvores, as redes e os híbridos
- `metricas.csv` — 69 linhas (`series`, `modelo`, `mae`, `rmse`, `mase`)
- `metricas_completas.csv` — as mesmas 69 linhas com MAPE, sMAPE e viés
- `previsoes_holdout.csv` — 84 linhas, árvore escolhida no holdout
- `figuras/*.png` — diagnóstico e comparação

Tempo de execução: 19 a 24 minutos em 12 núcleos (medido em duas execuções
completas) — ~5 no grid search de AICc, ~3 no backtest das árvores e ~8 no
backtest das redes. Cabe em laptop; nada
exige nuvem nem GPU.

O `torch` entra como dependência do projeto, na build de **CPU**: o
`pyproject.toml` aponta `torch` para o índice `download.pytorch.org/whl/cpu`,
porque o wheel padrão de Linux arrastaria ~2-3 GB de bibliotecas CUDA que nada
aqui usa. O `uv sync` resolve isso sozinho.

Sem `uv`, o equivalente é:

```bash
python -m venv .venv && .venv/bin/pip install -e . \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  && .venv/bin/jupyter nbconvert --to notebook --inplace --execute main.ipynb
```

## Task 2 — comparativo de regressões (seção 10.1)

Sem regularização (MQO), Ridge, Lasso e Elastic Net, com as mesmas features e a
mesma recursão, escolhidos pelo mesmo backtest de origem rolante. O vencedor
(Lasso) é o linear que a seção 11 usa. As linhas `linear` entregues em 01/11 (Ridge)
ficam como estavam; as variantes entram como `linear_ols`, `linear_ridge`,
`linear_lasso` e `linear_elasticnet`, e o SARIMA + Lasso como `sarima_lasso`.

## Task 2, classe árvore (08/11) — árvores e híbridos

Seção 11 do `main.ipynb`. Mesmas 54 features e mesma estratégia recursiva do
linear (seção 9), um ajuste por série, cinco famílias de árvore: árvore de
decisão, Random Forest, AdaBoost, Gradient Boosting e XGBoost.

- **Seleção sem a validação.** Hiperparâmetros e família vencedora saem de um
  backtest de origem rolante nas mesmas quatro origens da Task 2 (todas antes de
  2016-03-27), pelo menor MASE médio. A validação só mede.
- **Nome no CSV.** As cinco famílias entram com o próprio nome; a escolhida
  entra também como `arvore` ou `boosting` (o nome que o check exige).
- **Híbridos.** Média simples das previsões com o melhor linear e a melhor
  árvore: `sarima_lasso` (seção 10.1), `sarima_xgboost` e `sarima_lasso_xgboost`.
- **Holdout.** `previsoes_holdout.csv` (84 linhas) com a família escolhida,
  reajustada com treino + validação até 2016-04-24 e os mesmos hiperparâmetros.
  O `y` do holdout não é usado em lugar nenhum: é o score cego da Task 3.
- `metricas_completas.csv` traz MAE, RMSE, MASE, MAPE, sMAPE e viés de todos os
  modelos; `metricas.csv` mantém as cinco colunas do check.

## Task 2, classe rede (15/11) — MLP, RNN e LSTM

Seção 12 do `main.ipynb`. As mesmas 54 features e a mesma previsão recursiva das
seções 9 a 11, um ajuste por série, três arquiteturas em `torch`:

- **`mlp`** recebe só a linha de features do dia previsto; **`rnn`** e **`lstm`**
  recebem também a janela dos últimos dias da própria série, pela célula
  recorrente, com as covariáveis do dia entrando na camada de saída.
- **Varredura de 24 configurações** (8 por arquitetura), selecionadas pelo mesmo
  backtest de origem rolante das seções 10 e 11 — as quatro origens anteriores a
  2016-03-27. A validação só mede. Treino em batch cheio, sem embaralhar linha e
  **sem early stopping**.
- **Nome no CSV.** As três entram com o próprio nome; a vencedora do backtest
  entra também como `rede`, o nome que o check exige.
- **Seções 12.4 e 12.5** trazem duas leituras extras: a variação por semente, que
  mede quanto da diferença entre as arquiteturas é ruído de inicialização, e o
  viés do `FOODS`, comparado com o da árvore.

### Conferir a nota

O corretor público da Task 2 não está versionado neste repositório. Com o
`checks/check_task2.py` do enunciado copiado para uma pasta `checks/` na raiz:

```bash
.venv/bin/python checks/check_task2.py --ate rede
```

### Estrutura

| Caminho | Conteúdo |
|---|---|
| `main.ipynb` | notebook único: diagnóstico, baselines, SARIMA, as três classes da Task 2, métricas e exportação |
| `dados/` | `treino.csv`, `validacao.csv`, `holdout_datas.csv` (sem `y`) |
| `figuras/` | figuras geradas pelo notebook |
| `content/enunciado.pdf`, `content/enunciado_task2.pdf` | enunciados |
| `AI_USAGE.md` | declaração de uso de IA generativa |

O `y` do holdout não faz parte desta entrega — ele é o score cego da A2.
