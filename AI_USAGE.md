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
