# Previsão de Recompra - Online Retail

Prevê, para cada cliente de um varejista online, quanto ele vai gastar nos próximos três meses (regressão) e se ele vai voltar a comprar nesse período (classificação), a partir dos atributos RFM (Recência, Frequência, Monetário). Na regressão, compara Regressão Linear Múltipla, Ridge e KNN Regressor. Na classificação, compara Regressão Logística e KNN com o score RFM manual que o mercado usa sem Machine Learning.

- **EQUIPE**: Thiago Dallo, Vinicius Fabris e Murilo Cambruzzi.
- **DISCIPLINA:** Machine Learning
- **CURSO:** Engenharia de Software - 4ª fase
- **FACULDADE:** UniSATC

## Tecnologias

- Python 3.11
- pandas, numpy
- scikit-learn (regressão linear, Ridge, Regressão Logística, KNN, Pipeline, GridSearchCV)
- statsmodels (VIF)
- matplotlib
- Jupyter / Google Colab

## O que o projeto faz

- **Split temporal contra data leakage**: separa o histórico do dataset em uma janela de 75% (gera as features) e uma janela de 25% (gera os alvos), sem deixar nenhuma informação futura vazar para o cálculo das features.
- **Engenharia de atributos RFM**: calcula Recência, Frequência, Monetário, Ticket Médio, Tenure e quantidade de itens distintos por cliente, só a partir da janela histórica.
- **Diagnóstico de multicolinearidade**: calcula o VIF de cada feature no conjunto de treino antes de treinar qualquer modelo, para justificar a comparação com um modelo regularizado.
- **Regressão, Modelo A x Modelo B**: treina Regressão Linear Múltipla e Regressão Ridge com o mesmo conjunto de features, otimiza o hiperparâmetro `alpha` do Ridge via `GridSearchCV` e compara os dois no conjunto de teste com MAE, RMSE, R² e R² ajustado.
- **Robustez da regressão**: repete a validação cruzada 25 vezes no treino (5 partes x 5 repetições) e inclui o KNN Regressor, para checar se a comparação do teste se sustenta.
- **Classificação, Modelo C x Modelo D**: prevê se o cliente recompra com Regressão Logística e KNN, ambos com log e padronização dentro de um `Pipeline`, hiperparâmetros escolhidos por ROC-AUC na validação cruzada e limiar de decisão escolhido sem olhar o teste.
- **Baselines honestos**: compara os modelos com "todo mundo recompra" e com o score RFM manual em quintis (Hughes, 1994), que é como o mercado resolve o problema sem Machine Learning.
- **Validação temporal**: treina com um trimestre anterior e testa no trimestre seguinte, como o modelo seria usado na prática.

## O problema

O dataset [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) traz um ano de transações de um varejista online do Reino Unido. O projeto separa esse período em uma janela histórica e uma janela futura: a histórica gera as features RFM de cada cliente, e a futura gera os dois alvos. Essa separação evita data leakage, já que nenhuma informação futura entra no cálculo das features.

- **`ValorFuturo`** (regressão): o valor líquido que o cliente gastou na janela futura, com devoluções descontadas.
- **`Recomprou`** (classificação): 1 se o cliente fez pelo menos uma compra na janela futura. Nota que só registra devolução não conta como recompra. 57,5% dos clientes recompram.

O Modelo A (Regressão Linear Múltipla) serve de base da regressão, e o Modelo B (Ridge) soma regularização L2 para lidar com a correlação entre as variáveis RFM. Na classificação, o Modelo C (Regressão Logística) e o Modelo D (KNN) respondem à pergunta de negócio do projeto: quais clientes priorizar em ações comerciais.

## O que tem aqui

| Arquivo | Para que serve |
|---------|----------------|
| `Projeto_Final_ML.ipynb` | Notebook completo, pronto para rodar no Google Colab |
| `Projeto_Final_ML_Online_Retail.docx` | Relatório final, estruturado conforme o modelo da disciplina |
| `pipeline.py` | Pipeline equivalente ao notebook, para rodar via terminal |
| `resultados.json` | Métricas e diagnósticos gerados pelo pipeline |
| `figs/` | Gráficos usados no relatório e no notebook |

## Resultados

### Regressão: quanto o cliente vai gastar

Conjunto de teste (676 clientes):

| Métrica | Modelo A (Linear Múltipla) | Modelo B (Ridge) | KNN Regressor |
|---------|-----------------------------|-------------------|---------------|
| MAE (£) | 463,26 | 444,36 | 429,30 |
| RMSE (£) | 791,00 | 774,05 | 693,28 |
| R² | 0,3723 | 0,3989 | 0,5178 |
| R² ajustado | 0,3667 | 0,3935 | 0,5135 |

Validação cruzada repetida no treino (25 rodadas, média ± desvio):

| Métrica | Modelo A (Linear Múltipla) | Modelo B (Ridge) | KNN Regressor |
|---------|-----------------------------|-------------------|---------------|
| MAE (£) | 654 ± 84 | 631 ± 95 | 633 ± 140 |
| R² | 0,43 ± 0,41 | 0,46 ± 0,36 | 0,42 ± 0,13 |

O Ridge vence o Modelo A em todas as métricas. O KNN Regressor parece o melhor no teste, mas empata com o Ridge na validação cruzada e varia bem mais entre as rodadas: o teste não tem nenhum cliente com gasto futuro acima de £10.000, e o KNN não consegue prever valores acima do maior gasto dos seus vizinhos. O Ridge segue como modelo final da regressão.

### Classificação: o cliente volta a comprar

Conjunto de teste (os mesmos 676 clientes):

| Métrica | Todos recompram | RFM manual (quintis) | Modelo C (Regressão Logística) | Modelo D (KNN) |
|---------|-----------------|----------------------|--------------------------------|----------------|
| Acurácia | 0,571 | 0,678 | 0,683 | 0,676 |
| Precisão | 0,571 | 0,764 | 0,793 | 0,768 |
| Recall | 1,000 | 0,630 | 0,604 | 0,619 |
| Especificidade | 0,000 | 0,741 | 0,790 | 0,752 |
| F1 (classe recomprou) | 0,727 | 0,690 | 0,685 | 0,686 |
| F1-macro | 0,363 | 0,677 | 0,683 | 0,676 |
| ROC-AUC | 0,500 | 0,736 | 0,749 | 0,746 |

- Regressão Logística e KNN ficam tecnicamente empatados: o intervalo de 95% da diferença de AUC (bootstrap no teste) contém zero. A Logística é o modelo final da classificação, por ter o melhor equilíbrio entre as classes e coeficientes interpretáveis.
- O RFM manual chega muito perto (AUC de 0,745 na validação cruzada, contra 0,746 da Logística e 0,749 do KNN). O ganho do Machine Learning aqui é pequeno na ordenação dos clientes. O valor está na probabilidade por cliente, que permite escolher o limiar pelo custo da campanha.
- "Todo mundo recompra" tem o maior F1 da classe positiva e AUC de 0,5. É o motivo de o projeto não avaliar classificação só por acurácia ou pelo F1 de uma classe.
- Na validação temporal (treino no trimestre anterior), as AUCs se mantêm (0,746 da Logística, 0,743 do KNN e 0,738 do RFM manual), mas a taxa de recompra sobe de 49% para 57% com a temporada de Natal. O modelo continua útil, desde que as probabilidades e o limiar sejam recalibrados a cada período.

A análise crítica completa está no relatório e nas seções 12, 13 e 15 do notebook.

## Como rodar

### Google Colab (recomendado)

Abra `Projeto_Final_ML.ipynb` no Colab e rode todas as células. O notebook baixa o dataset direto da UCI, sem precisar subir nenhum arquivo antes. A execução completa leva alguns minutos; a validação cruzada repetida da seção 13 é a parte mais demorada.

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
4. Rode as células de pré-processamento e engenharia de atributos, que geram as features RFM e os dois alvos com o split temporal.
5. Rode o diagnóstico de VIF e a padronização, antes de treinar qualquer modelo.
6. Rode o treino do Modelo A e do Modelo B, incluindo a otimização do `alpha` do Ridge, e a avaliação da regressão.
7. Rode a seção 13 para a validação cruzada repetida e o KNN Regressor.
8. Rode a seção 14 para a classificação: baselines, Regressão Logística, KNN, escolha do limiar, avaliação no teste, interpretação e validação temporal.
9. Leia a seção **Análise Crítica e Conclusão**, no fim do notebook, para a interpretação dos resultados.

## Próximos passos

- Segmentação dos clientes com K-Means sobre as mesmas variáveis RFM, quando o conteúdo for visto em aula.
- Novas variáveis para passar do teto de AUC em torno de 0,75: sazonalidade, categoria dos produtos, país e intervalo médio entre compras.
