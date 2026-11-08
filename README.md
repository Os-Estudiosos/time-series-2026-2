# Séries Temporais — FGV EMAp (2026/2)

Repositório de entregas do curso de Séries Temporais da graduação em Ciência de
Dados e Inteligência Artificial da Fundação Getulio Vargas.

## Task 1 e Task 2 (`linear`) — Diagnóstico, baselines e previsão

Previsão de 28 dias (2016-03-28 a 2016-04-24) para três séries diárias da loja
`CA_1` do M5: `store_total`, `FOODS` e `HOBBIES`.

### Como reproduzir

Um comando, a partir da raiz do repositório:

```bash
uv run --frozen jupyter nbconvert --to notebook --inplace --execute main.ipynb
```

Isso lê `dados/`, executa o notebook inteiro e regrava na raiz:

- `previsoes_validacao.csv` — 1260 linhas (`date`, `series`, `modelo`, `yhat`), incluindo `linear`, as árvores e os híbridos
- `metricas.csv` — 45 linhas (`series`, `modelo`, `mae`, `rmse`, `mase`)
- `metricas_completas.csv` — as mesmas 45 linhas com MAPE, sMAPE e viés
- `previsoes_holdout.csv` — 84 linhas, árvore escolhida no holdout
- `figuras/*.png` — diagnóstico e comparação

Tempo de execução: cerca de 9 minutos, dos quais ~5 são o grid search de AICc e
~3 o backtest das árvores (medido com 2 núcleos; com mais núcleos, menos).
A seleção temporal de `alpha` do linear usa quatro janelas históricas. Cabe em
laptop; nada exige nuvem.

Sem `uv`, o equivalente é:

```bash
python -m venv .venv && .venv/bin/pip install -e . \
  && .venv/bin/jupyter nbconvert --to notebook --inplace --execute main.ipynb
```

## Task 3 (`tree_models`) — modelos de árvore e híbridos

Seção 11 do `main.ipynb`. Mesmas 54 features e mesma estratégia recursiva do
linear (seção 9), um ajuste por série, cinco famílias de árvore: árvore de
decisão, Random Forest, AdaBoost, Gradient Boosting e XGBoost.

- **Seleção sem a validação.** Hiperparâmetros e família vencedora saem de um
  backtest de origem rolante nas mesmas quatro origens da Task 2 (todas antes de
  2016-03-27), pelo menor MASE médio. A validação só mede.
- **Nome no CSV.** As cinco famílias entram com o próprio nome; a escolhida
  entra também como `arvore` ou `boosting` (o nome que o check exige).
- **Híbridos.** Média simples das previsões: `sarima_linear`, `sarima_arvore` e
  `sarima_linear_arvore`.
- **Holdout.** `previsoes_holdout.csv` (84 linhas) com a família escolhida,
  reajustada com treino + validação até 2016-04-24 e os mesmos hiperparâmetros.
- `metricas_completas.csv` traz MAE, RMSE, MASE, MAPE, sMAPE e viés de todos os
  modelos; `metricas.csv` mantém as cinco colunas do check.

### Conferir a nota

```bash
.venv/bin/python checks/check_task1.py --submission . --dados dados
```

### Estrutura

| Caminho | Conteúdo |
|---|---|
| `main.ipynb` | notebook único: diagnóstico, baselines, SARIMA, métricas e exportação |
| `dados/` | `treino.csv`, `validacao.csv`, `holdout_datas.csv` (sem `y`) |
| `figuras/` | figuras geradas pelo notebook |
| `checks/check_task1.py` | corretor público da Task 1 |
| `content/enunciado.pdf` | enunciado |
| `AI_USAGE.md` | declaração de uso de IA generativa |

O `y` do holdout não faz parte desta entrega — ele é o score cego da A2.
