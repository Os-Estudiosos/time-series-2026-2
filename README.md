# Séries Temporais — FGV EMAp (2026/2)

Repositório de entregas do curso de Séries Temporais da graduação em Ciência de
Dados e Inteligência Artificial da Fundação Getulio Vargas.

## Task 1 — Diagnóstico, baselines e ARIMA/SARIMA

Previsão de 28 dias (2016-03-28 a 2016-04-24) para três séries diárias da loja
`CA_1` do M5: `store_total`, `FOODS` e `HOBBIES`.

### Como reproduzir

Um comando, a partir da raiz do repositório:

```bash
uv run --frozen jupyter nbconvert --to notebook --inplace --execute main.ipynb
```

Isso lê `dados/`, executa o notebook inteiro e regrava na raiz:

- `previsoes_validacao.csv` — 420 linhas (`date`, `series`, `modelo`, `yhat`)
- `metricas.csv` — 15 linhas (`series`, `modelo`, `mae`, `rmse`, `mase`)
- `figuras/*.png` — diagnóstico e comparação

Tempo de execução: cerca de 6 minutos, dos quais ~5 são o grid search de AICc.
Cabe em laptop; nada exige nuvem.

Sem `uv`, o equivalente é:

```bash
python -m venv .venv && .venv/bin/pip install -e . \
  && .venv/bin/jupyter nbconvert --to notebook --inplace --execute main.ipynb
```

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
