# Previsão de Recompra - Online Retail

Projeto final da disciplina de Machine Learning, Centro Universitário SATC. Prevê o valor futuro de recompra de clientes de um varejista online, comparando Regressão Linear Múltipla e Regressão Ridge sobre atributos RFM (Recência, Frequência, Monetário).

Equipe: Thiago Dallo, Vinicius Fabris e Murilo Cambruzzi.

## Tecnologias

- Python 3.11
- pandas, numpy
- scikit-learn (regressão linear, Ridge, GridSearchCV)
- statsmodels (VIF)
- matplotlib
- Jupyter / Google Colab

## O que o projeto faz

- **Split temporal contra data leakage**: separa o histórico do dataset em uma janela de 75% (gera as features) e uma janela de 25% (gera o alvo), sem deixar nenhuma informação futura vazar para o cálculo das features.
- **Engenharia de atributos RFM**: calcula Recência, Frequência, Monetário, Ticket Médio, Tenure e quantidade de itens distintos por cliente, só a partir da janela histórica.
- **Diagnóstico de multicolinearidade**: calcula o VIF de cada feature no conjunto de treino antes de treinar qualquer modelo, para justificar a comparação com um modelo regularizado.
- **Comparação Modelo A x Modelo B**: treina Regressão Linear Múltipla e Regressão Ridge com o mesmo conjunto de features, otimiza o hiperparâmetro `alpha` do Ridge via `GridSearchCV` e compara os dois no conjunto de teste com MAE, RMSE, R² e R² ajustado.

## O problema

O dataset [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) traz um ano de transações de um varejista online do Reino Unido. O projeto separa esse período em uma janela histórica e uma janela futura: a histórica gera as features RFM de cada cliente, e a futura gera o alvo, o valor real que o cliente gastou depois. Essa separação evita data leakage, já que nenhuma informação futura entra no cálculo das features.

O Modelo A (Regressão Linear Múltipla) serve de base. O Modelo B (Regressão Ridge) soma regularização L2 e testa se ela melhora a generalização quando as variáveis RFM se correlacionam entre si.

## O que tem aqui

| Arquivo | Para que serve |
|---------|----------------|
| `Projeto_Final_ML.ipynb` | Notebook completo, pronto para rodar no Google Colab |
| `Projeto_Final_ML_Online_Retail.docx` | Relatório final, estruturado conforme o modelo da disciplina |
| `pipeline.py` | Pipeline equivalente ao notebook, para rodar via terminal |
| `resultados.json` | Métricas e diagnósticos gerados pelo pipeline |
| `figs/` | Gráficos usados no relatório e no notebook |

## Resultados

| Métrica | Modelo A (Linear Múltipla) | Modelo B (Ridge) |
|---------|-----------------------------|-------------------|
| MAE (£) | 463,26 | 444,36 |
| RMSE (£) | 791,00 | 774,05 |
| R² | 0,3723 | 0,3989 |
| R² ajustado | 0,3667 | 0,3935 |

O Modelo B vence em todas as métricas. A análise crítica completa, incluindo o achado sobre o coeficiente de Frequência, está no relatório e na última seção do notebook.

## Como rodar

### Google Colab (recomendado)

Abra `Projeto_Final_ML.ipynb` no Colab e rode todas as células. O notebook baixa o dataset direto da UCI, sem precisar subir nenhum arquivo antes.

### Local

```bash
pip install -r requirements.txt
python pipeline.py
```

O script baixa o dataset na primeira execução (pasta `data/`, ignorada pelo Git) e grava os resultados em `resultados.json`.

## Estrutura de pastas

```
.
├── Projeto_Final_ML.ipynb                  # notebook completo (Colab)
├── Projeto_Final_ML_Online_Retail.docx     # relatório final
├── pipeline.py                             # pipeline em Python
├── resultados.json                         # métricas geradas pelo pipeline
├── requirements.txt
└── figs/                                   # gráficos do relatório e do notebook
```

## Fluxo de uso

1. Abra `Projeto_Final_ML.ipynb` no Google Colab.
2. Rode a célula de imports e a célula de carregamento dos dados, que baixa o dataset direto da UCI.
3. Rode as células de EDA para ver o diagnóstico dos dados brutos (nulos, valores inválidos, distribuição das variáveis).
4. Rode as células de pré-processamento e engenharia de atributos, que geram as features RFM com o split temporal.
5. Rode o diagnóstico de VIF e a padronização, antes de treinar qualquer modelo.
6. Rode o treino do Modelo A e do Modelo B, incluindo a otimização do `alpha` do Ridge.
7. Rode a célula de avaliação para ver a tabela de métricas e os gráficos de real x previsto e de resíduos.
8. Leia a seção **Análise Crítica**, no fim do notebook, para a interpretação dos resultados.

## Próximos passos

O escopo atual cobre só regressão, porque é o único conteúdo já visto em aula até a Aula 6. Quando o professor apresentar Regressão Logística, KNN, SVM e K-Means, o grupo pretende estender o projeto com um problema de classificação (cliente de alto potencial) e uma segmentação de clientes via K-Means sobre as mesmas variáveis RFM.
