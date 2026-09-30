import io
import json
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, balanced_accuracy_score, average_precision_score)
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

RANDOM_STATE = 42

DATA_PATH = 'data/Online Retail.xlsx'
DATA_URL = 'https://archive.ics.uci.edu/static/public/352/online+retail.zip'

if not os.path.exists(DATA_PATH):
    os.makedirs('data', exist_ok=True)
    resp = urllib.request.urlopen(DATA_URL, timeout=60)
    with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
        z.extract('Online Retail.xlsx', 'data')

df = pd.read_excel(DATA_PATH)
n_raw = len(df)

# ------------------------------------------------------------------
# 1. Higiene mínima e checagem de esquema
# ------------------------------------------------------------------
NON_PRODUCT_CODES = {'POST', 'DOT', 'M', 'C2', 'D', 'S', 'BANK CHARGES', 'AMAZONFEE', 'CRUK', 'PADS', 'B'}

df_clean = df.dropna(subset=['CustomerID']).copy()
n_after_customerid = len(df_clean)

df_clean = df_clean[df_clean['UnitPrice'] > 0].copy()
n_after_price = len(df_clean)

df_clean = df_clean[~df_clean['StockCode'].astype(str).isin(NON_PRODUCT_CODES)].copy()
n_after_codes = len(df_clean)

df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

# ------------------------------------------------------------------
# 2. Split temporal e engenharia de atributos (RFM)
# ------------------------------------------------------------------
min_date = df_clean['InvoiceDate'].min()
max_date = df_clean['InvoiceDate'].max()
total_days = (max_date - min_date).days

cutoff = min_date + pd.Timedelta(days=int(total_days * 0.75))

feature_cols = ['Recencia', 'Frequencia', 'Monetario', 'TicketMedio', 'Tenure', 'QtdItensDistintos']


def montar_base(transacoes, corte, fim):
    """Monta uma linha por cliente: features só com compras até `corte`,
    alvos só com compras entre `corte` e `fim`."""
    historico = transacoes[transacoes['InvoiceDate'] <= corte]
    futuro = transacoes[(transacoes['InvoiceDate'] > corte) & (transacoes['InvoiceDate'] <= fim)]

    por_cliente = historico.groupby('CustomerID')
    base = pd.DataFrame({
        'Recencia': (corte - por_cliente['InvoiceDate'].max()).dt.days,
        'Frequencia': por_cliente['InvoiceNo'].nunique(),
        'Monetario': por_cliente['TotalPrice'].sum(),
        'Tenure': (corte - por_cliente['InvoiceDate'].min()).dt.days,
        'QtdItensDistintos': por_cliente['StockCode'].nunique(),
    })
    base['TicketMedio'] = base['Monetario'] / base['Frequencia']

    # alvo de regressão: gasto líquido na janela futura (devoluções descontadas, mínimo zero)
    gasto_futuro = futuro.groupby('CustomerID')['TotalPrice'].sum()
    base['ValorFuturo'] = gasto_futuro.reindex(base.index).fillna(0.0).clip(lower=0)

    # alvo de classificação: pelo menos uma compra (quantidade positiva) na janela futura.
    # Nota só de devolução não conta como recompra.
    clientes_com_compra = futuro.loc[futuro['Quantity'] > 0, 'CustomerID'].unique()
    base['Recomprou'] = base.index.isin(clientes_com_compra).astype(int)

    return base.reset_index()[['CustomerID'] + feature_cols + ['ValorFuturo', 'Recomprou']]


hist = df_clean[df_clean['InvoiceDate'] <= cutoff].copy()
future = df_clean[df_clean['InvoiceDate'] > cutoff].copy()

data = montar_base(df_clean, cutoff, max_date)

n_customers_hist = len(data)

X = data[feature_cols].copy()
y = data['ValorFuturo'].copy()

# clientes que só devolveram na janela futura e clientes que compraram mas devolveram tudo
clientes_futuro = set(future['CustomerID'])
n_so_devolucao = int(((data['Recomprou'] == 0) & data['CustomerID'].isin(clientes_futuro)).sum())
n_recomprou_valor_zero = int(((data['Recomprou'] == 1) & (data['ValorFuturo'] == 0)).sum())

# concentração de receita no histórico: fatia dos 20% de clientes que mais gastaram
monetario_ordenado = data['Monetario'].sort_values(ascending=False)
n_top20 = int(np.ceil(0.2 * len(monetario_ordenado)))
share_top20 = float(monetario_ordenado.iloc[:n_top20].sum() / monetario_ordenado.sum())

eda_stats = {
    'n_raw_rows': int(n_raw),
    'n_after_customerid': int(n_after_customerid),
    'n_after_price_filter': int(n_after_price),
    'n_after_nonproduct_codes': int(n_after_codes),
    'n_customers_historico': int(n_customers_hist),
    'min_date': str(min_date.date()),
    'max_date': str(max_date.date()),
    'cutoff_date': str(cutoff.date()),
    'total_days': int(total_days),
    'n_hist_transactions': int(len(hist)),
    'n_future_transactions': int(len(future)),
    'pct_customers_zero_future': float((data['ValorFuturo'] == 0).mean() * 100),
    'pct_recomprou': float(data['Recomprou'].mean() * 100),
    'n_recomprou': int(data['Recomprou'].sum()),
    'n_so_devolucao_no_futuro': n_so_devolucao,
    'n_recomprou_com_valor_zero': n_recomprou_valor_zero,
    'n_monetario_nao_positivo': int((data['Monetario'] <= 0).sum()),
    'share_receita_top20_clientes': share_top20,
    'feature_describe': X.describe().to_dict(),
    'target_describe': y.describe().to_dict(),
    'medianas_por_classe': data.groupby('Recomprou')[feature_cols].median().to_dict(),
}

# ------------------------------------------------------------------
# 3. Divisão treino/teste (a mesma para regressão e classificação)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
y_cls_train = data.loc[X_train.index, 'Recomprou']
y_cls_test = data.loc[X_test.index, 'Recomprou']

vif_data = pd.DataFrame()
vif_data['feature'] = feature_cols
Xc = X_train.copy()
Xc.insert(0, 'const', 1.0)
vif_data['VIF'] = [variance_inflation_factor(Xc.values, i + 1) for i in range(len(feature_cols))]

# ------------------------------------------------------------------
# 4. Regressão: Modelo A (Linear Múltipla) e Modelo B (Ridge)
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model_a = LinearRegression()
model_a.fit(X_train_s, y_train)
pred_a = model_a.predict(X_test_s)

# o StandardScaler fica dentro do Pipeline: em cada rodada da validação cruzada,
# a escala é aprendida só com as partes de treino daquela rodada
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
param_grid = {'alpha': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}
pipe_ridge = Pipeline([('escala', StandardScaler()), ('ridge', Ridge(random_state=RANDOM_STATE))])
grid = GridSearchCV(pipe_ridge, {'ridge__alpha': param_grid['alpha']}, cv=kf,
                    scoring='neg_mean_absolute_error')
grid.fit(X_train, y_train)
model_b = grid.best_estimator_.named_steps['ridge']
pred_b = grid.predict(X_test)
ridge_best_alpha = grid.best_params_['ridge__alpha']


def metrics(y_true, y_pred, n_features):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'R2_ajustado': r2_adj}


# ------------------------------------------------------------------
# 5. Pré-processamento dos modelos baseados em distância e probabilidade
# ------------------------------------------------------------------
# Frequência, Monetário, Ticket Médio e Itens Distintos têm cauda longa à direita.
# O log com sinal comprime a cauda e aceita os poucos valores negativos (devoluções).
colunas_log = ['Frequencia', 'Monetario', 'TicketMedio', 'QtdItensDistintos']
colunas_sem_log = ['Recencia', 'Tenure']


def log_com_sinal(x):
    return np.sign(x) * np.log1p(np.abs(x))


def preparo(com_log=True, com_escala=True):
    etapas = []
    if com_log:
        etapas.append(('log', ColumnTransformer([
            ('log', FunctionTransformer(log_com_sinal, feature_names_out='one-to-one'), colunas_log),
            ('sem_log', 'passthrough', colunas_sem_log),
        ], verbose_feature_names_out=False)))
    if com_escala:
        etapas.append(('escala', StandardScaler()))
    return etapas


grade_knn = {
    'knn__n_neighbors': [1, 5, 11, 21, 31, 51, 75, 101, 151, 201, 301, 401, 501],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan'],
}

# KNN Regressor (Aula 8): previsão = média do gasto futuro dos K vizinhos
pipe_knn_reg = Pipeline(preparo() + [('knn', KNeighborsRegressor())])
busca_knn_reg = GridSearchCV(pipe_knn_reg, grade_knn, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
busca_knn_reg.fit(X_train, y_train)
pred_knn_reg = busca_knn_reg.predict(X_test)

results = {
    'ModeloA_LinearMultipla': metrics(y_test, pred_a, len(feature_cols)),
    'ModeloB_Ridge': metrics(y_test, pred_b, len(feature_cols)),
    'KNN_Regressor': metrics(y_test, pred_knn_reg, len(feature_cols)),
    'ridge_best_alpha': ridge_best_alpha,
    'ridge_cv_results_alpha': param_grid['alpha'],
    'ridge_cv_mae_mean': list(-grid.cv_results_['mean_test_score']),
    'knn_reg_best_params': {k.replace('knn__', ''): v for k, v in busca_knn_reg.best_params_.items()},
    'coef_modelo_a': dict(zip(feature_cols, model_a.coef_.tolist())),
    'coef_modelo_b': dict(zip(feature_cols, model_b.coef_.tolist())),
    'intercept_a': float(model_a.intercept_),
    'intercept_b': float(model_b.intercept_),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'vif': vif_data.to_dict('records'),
}

resid_a = (y_test.values - pred_a)
resid_b = (y_test.values - pred_b)

# ------------------------------------------------------------------
# 6. Robustez da regressão: validação cruzada repetida, só no treino
# ------------------------------------------------------------------
# O teste tem um único sorteio de 676 clientes. Repetir a validação cruzada
# (5 partes x 5 repetições) mostra o quanto o erro muda de uma partição para outra.
cv_repetida = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
kf_interno = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
modelos_regressao = {
    'Modelo A (Linear Múltipla)': Pipeline([('escala', StandardScaler()), ('linear', LinearRegression())]),
    'Modelo B (Ridge)': GridSearchCV(pipe_ridge, {'ridge__alpha': param_grid['alpha']}, cv=kf_interno,
                                     scoring='neg_mean_absolute_error'),
    'KNN Regressor': GridSearchCV(pipe_knn_reg, grade_knn, cv=kf_interno,
                                  scoring='neg_mean_absolute_error'),
}
metricas_cv = {'MAE': 'neg_mean_absolute_error', 'RMSE': 'neg_root_mean_squared_error', 'R2': 'r2'}
robustez_regressao = {}
for nome, estimador in modelos_regressao.items():
    resultado_cv = cross_validate(estimador, X_train, y_train, cv=cv_repetida, scoring=metricas_cv, n_jobs=-1)
    mae_folds = -resultado_cv['test_MAE']
    rmse_folds = -resultado_cv['test_RMSE']
    r2_folds = resultado_cv['test_R2']
    robustez_regressao[nome] = {
        'MAE_media': float(mae_folds.mean()), 'MAE_dp': float(mae_folds.std()),
        'RMSE_media': float(rmse_folds.mean()), 'RMSE_dp': float(rmse_folds.std()),
        'R2_media': float(r2_folds.mean()), 'R2_dp': float(r2_folds.std()),
        'MAE_folds': mae_folds.tolist(),
    }

# o KNN não extrapola: a previsão nunca passa do maior gasto entre os vizinhos
robustez_regressao['maior_valor_futuro_treino'] = float(y_train.max())
robustez_regressao['maior_valor_futuro_teste'] = float(y_test.max())
robustez_regressao['n_clientes_acima_10mil_treino'] = int((y_train > 10000).sum())
robustez_regressao['n_clientes_acima_10mil_teste'] = int((y_test > 10000).sum())
robustez_regressao['maior_previsao_knn_teste'] = float(pred_knn_reg.max())
robustez_regressao['maior_previsao_ridge_teste'] = float(pred_b.max())

# ------------------------------------------------------------------
# 7. Classificação: o cliente volta a comprar na janela futura?
# ------------------------------------------------------------------
cv_estratificada = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


class ScoreRFM:
    """Método manual de mercado (Hughes, 1994): nota de 1 a 5 em Recência, Frequência
    e Monetário por quintis, somadas em um score de 3 a 15. Os cortes dos quintis e a
    nota de corte da campanha são definidos só com os clientes de treino."""

    def fit(self, X, y):
        self.cortes_ = {col: np.quantile(X[col], [0.2, 0.4, 0.6, 0.8])
                        for col in ['Recencia', 'Frequencia', 'Monetario']}
        score = self.pontuar(X)
        candidatos = range(3, 16)
        self.nota_corte_ = max(candidatos,
                               key=lambda t: balanced_accuracy_score(y, (score >= t).astype(int)))
        return self

    def pontuar(self, X):
        nota_r = 5 - np.searchsorted(self.cortes_['Recencia'], X['Recencia'], side='left')
        nota_f = 1 + np.searchsorted(self.cortes_['Frequencia'], X['Frequencia'], side='left')
        nota_m = 1 + np.searchsorted(self.cortes_['Monetario'], X['Monetario'], side='left')
        return nota_r + nota_f + nota_m

    def predict(self, X):
        return (self.pontuar(X) >= self.nota_corte_).astype(int)


def metricas_classificacao(y_true, y_pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'Acuracia': accuracy_score(y_true, y_pred),
        'Precisao': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred),
        'Especificidade': tn / (tn + fp) if (tn + fp) else 0.0,
        'F1': f1_score(y_true, y_pred),
        'F1_macro': f1_score(y_true, y_pred, average='macro'),
        'ROC_AUC': roc_auc_score(y_true, score) if len(np.unique(score)) > 1 else 0.5,
        # PR-AUC (precisão média): área sob a curva precisão x recall. Sem poder de ordenação,
        # fica igual à proporção de positivos, então a referência aqui é ~0,57 e não 0,5.
        'PR_AUC': average_precision_score(y_true, score) if len(np.unique(score)) > 1 else float(np.mean(y_true)),
        'matriz_confusao': [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


# baseline 1: "todo mundo recompra" (classe majoritária)
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_cls_train)
pred_dummy = dummy.predict(X_test)
score_dummy = dummy.predict_proba(X_test)[:, 1]

# baseline 2: score RFM manual em quintis
rfm_manual = ScoreRFM().fit(X_train, y_cls_train)
pred_rfm = rfm_manual.predict(X_test)
score_rfm = rfm_manual.pontuar(X_test)

# Modelo C: Regressão Logística (Aula 7)
pipe_logistica = Pipeline(preparo() + [('logistica', LogisticRegression(max_iter=1000))])
grade_logistica = {'logistica__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
busca_logistica = GridSearchCV(pipe_logistica, grade_logistica, cv=cv_estratificada,
                               scoring='roc_auc', n_jobs=-1)
busca_logistica.fit(X_train, y_cls_train)
prob_logistica = busca_logistica.predict_proba(X_test)[:, 1]

# Modelo D: KNN (Aula 8)
pipe_knn = Pipeline(preparo() + [('knn', KNeighborsClassifier())])
busca_knn = GridSearchCV(pipe_knn, grade_knn, cv=cv_estratificada, scoring='roc_auc', n_jobs=-1)
busca_knn.fit(X_train, y_cls_train)
prob_knn = busca_knn.predict_proba(X_test)[:, 1]

# Escolha do limiar (Aula 7): nunca olhando o teste. Sem os custos reais da campanha,
# o critério é neutro: o limiar que equilibra o acerto nas duas classes (acurácia
# balanceada) nas probabilidades de validação cruzada do treino. O RFM manual segue
# a mesma regra para a nota de corte, então a comparação fica justa.
limiares_candidatos = np.round(np.arange(0.05, 0.96, 0.01), 2)


def escolher_limiar(y_true, prob):
    acuracias = [balanced_accuracy_score(y_true, (prob >= t).astype(int)) for t in limiares_candidatos]
    return float(limiares_candidatos[int(np.argmax(acuracias))])


prob_cv_logistica = cross_val_predict(busca_logistica.best_estimator_, X_train, y_cls_train,
                                      cv=cv_estratificada, method='predict_proba')[:, 1]
prob_cv_knn = cross_val_predict(busca_knn.best_estimator_, X_train, y_cls_train,
                                cv=cv_estratificada, method='predict_proba')[:, 1]
limiar_logistica = escolher_limiar(y_cls_train, prob_cv_logistica)
limiar_knn = escolher_limiar(y_cls_train, prob_cv_knn)

pred_logistica = (prob_logistica >= limiar_logistica).astype(int)
pred_knn = (prob_knn >= limiar_knn).astype(int)

classificacao_teste = {
    'Baseline: todos recompram': metricas_classificacao(y_cls_test, pred_dummy, score_dummy),
    'Baseline: RFM manual (quintis)': metricas_classificacao(y_cls_test, pred_rfm, score_rfm),
    'Modelo C (Regressão Logística)': metricas_classificacao(y_cls_test, pred_logistica, prob_logistica),
    'Modelo D (KNN)': metricas_classificacao(y_cls_test, pred_knn, prob_knn),
}


def resumo_busca(busca):
    i = busca.best_index_
    return {
        'melhores_parametros': {k.split('__')[-1]: v for k, v in busca.best_params_.items()},
        'auc_cv_media': float(busca.cv_results_['mean_test_score'][i]),
        'auc_cv_dp': float(busca.cv_results_['std_test_score'][i]),
    }


# curva AUC x K (validação cruzada no treino) para as duas formas de voto, na melhor métrica
cv_knn = pd.DataFrame(busca_knn.cv_results_)
melhor_metrica = busca_knn.best_params_['knn__metric']
curva_auc_k = {}
for peso in ['uniform', 'distance']:
    linhas = cv_knn[(cv_knn['param_knn__metric'] == melhor_metrica) & (cv_knn['param_knn__weights'] == peso)]
    linhas = linhas.sort_values('param_knn__n_neighbors')
    curva_auc_k[peso] = {
        'K': [int(k) for k in linhas['param_knn__n_neighbors']],
        'auc_media': linhas['mean_test_score'].tolist(),
        'auc_dp': linhas['std_test_score'].tolist(),
    }

# efeito do pré-processamento no KNN (Aula 8): cada variante é reajustada com a mesma grade
efeito_preprocessamento = {}
for nome, com_log, com_escala in [('Sem pré-processamento', False, False),
                                  ('Só padronização', False, True),
                                  ('Log com sinal + padronização', True, True)]:
    busca = GridSearchCV(Pipeline(preparo(com_log, com_escala) + [('knn', KNeighborsClassifier())]),
                         grade_knn, cv=cv_estratificada, scoring='roc_auc', n_jobs=-1)
    busca.fit(X_train, y_cls_train)
    efeito_preprocessamento[nome] = resumo_busca(busca)

# AUC do RFM manual na mesma validação cruzada do treino (quintis refeitos a cada rodada)
auc_cv_rfm = []
score_cv_rfm = np.zeros(len(X_train))
for idx_tr, idx_va in cv_estratificada.split(X_train, y_cls_train):
    rfm_fold = ScoreRFM().fit(X_train.iloc[idx_tr], y_cls_train.iloc[idx_tr])
    score_cv_rfm[idx_va] = rfm_fold.pontuar(X_train.iloc[idx_va])
    auc_cv_rfm.append(roc_auc_score(y_cls_train.iloc[idx_va], score_cv_rfm[idx_va]))

# diferenças de AUC no teste, com intervalo de 95% por bootstrap (2.000 reamostragens do teste)
rng = np.random.default_rng(RANDOM_STATE)
y_teste_array = y_cls_test.to_numpy()
dif_log_knn, dif_log_rfm = [], []
for _ in range(2000):
    idx = rng.integers(0, len(y_teste_array), len(y_teste_array))
    if len(np.unique(y_teste_array[idx])) < 2:
        continue
    auc_log = roc_auc_score(y_teste_array[idx], prob_logistica[idx])
    dif_log_knn.append(auc_log - roc_auc_score(y_teste_array[idx], prob_knn[idx]))
    dif_log_rfm.append(auc_log - roc_auc_score(y_teste_array[idx], score_rfm[idx]))
ic_diferenca_auc = np.percentile(dif_log_knn, [2.5, 97.5])
ic_diferenca_auc_rfm = np.percentile(dif_log_rfm, [2.5, 97.5])

# troca entre precisão, recall e especificidade conforme o limiar (validação cruzada no treino)
analise_limiar = []
for limiar in limiares_candidatos:
    pred_limiar = (prob_cv_logistica >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_cls_train, pred_limiar, labels=[0, 1]).ravel()
    analise_limiar.append({
        'limiar': float(limiar),
        'precisao': float(tp / (tp + fp)) if (tp + fp) else 0.0,
        'recall': float(tp / (tp + fn)),
        'especificidade': float(tn / (tn + fp)),
        'f1': float(f1_score(y_cls_train, pred_limiar)),
        'pct_clientes_acionados': float(pred_limiar.mean() * 100),
    })

# ------------------------------------------------------------------
# Limiar pelo custo do erro (Aulas 3, 5 e 7): cenário de campanha
# ------------------------------------------------------------------
# A acurácia balanceada trata falso positivo e falso negativo como igualmente ruins.
# Numa campanha real eles custam diferente: contatar quem não ia comprar desperdiça o
# custo do contato (FP), e deixar de contatar quem ia comprar perde o retorno (FN).
# Premissas (hipotéticas, o dataset não traz custo nem margem):
#   - retorno por acerto (VP): £25, cerca de 10% de margem sobre um pedido típico
#     (ticket médio mediano do histórico ≈ £228);
#   - custo por contato: três cenários, de um e-mail com cupom a uma abordagem comercial.
# Lucro da campanha = 25 × VP − custo × (VP + FP). Com probabilidades bem calibradas,
# o limiar ótimo é custo / retorno (Bult e Wansbeek, 1995). O limiar de cada modelo é
# escolhido pelo maior lucro nas previsões de validação cruzada do treino e só depois
# aplicado ao teste, como na escolha anterior.
RETORNO_POR_ACERTO = 25.0
CENARIOS_CUSTO = {'E-mail com cupom': 5.0, 'Catálogo impresso': 12.5, 'Contato comercial': 17.5}


def lucro_campanha(y_true, acionar, custo, retorno=RETORNO_POR_ACERTO):
    y_true = np.asarray(y_true)
    acionar = np.asarray(acionar).astype(bool)
    vp = int((acionar & (y_true == 1)).sum())
    contatados = int(acionar.sum())
    return retorno * vp - custo * contatados, vp, contatados


def melhor_corte(y_true, score, candidatos, custo):
    lucros = [lucro_campanha(y_true, score >= c, custo)[0] for c in candidatos]
    return float(candidatos[int(np.argmax(lucros))]), lucros


notas_rfm = np.arange(3, 16)
estrategias_teste = {
    'Contatar todos': (np.ones(len(y_cls_test)), None, None),
    'RFM manual (quintis)': (score_rfm, score_cv_rfm, notas_rfm),
    'Modelo C (Regressão Logística)': (prob_logistica, prob_cv_logistica, limiares_candidatos),
    'Modelo D (KNN)': (prob_knn, prob_cv_knn, limiares_candidatos),
}
cenarios_lucro = {}
for nome_cenario, custo in CENARIOS_CUSTO.items():
    linha = {'custo_por_contato': custo, 'limiar_teorico': custo / RETORNO_POR_ACERTO, 'estrategias': {}}
    for nome, (score_teste, score_cv, candidatos) in estrategias_teste.items():
        if candidatos is None:
            corte, acionar = None, np.ones(len(y_cls_test), dtype=bool)
        else:
            corte, _ = melhor_corte(y_cls_train, score_cv, candidatos, custo)
            acionar = score_teste >= corte
        lucro, vp, contatados = lucro_campanha(y_cls_test, acionar, custo)
        linha['estrategias'][nome] = {
            'corte': corte,
            'lucro_teste': float(lucro),
            'clientes_contatados': contatados,
            'pct_contatados': float(100 * contatados / len(y_cls_test)),
            'recompras_alcancadas': vp,
            'pct_recompras_alcancadas': float(100 * vp / int(y_cls_test.sum())),
        }
    # referência de teto: contatar só quem de fato recomprou (informação perfeita)
    linha['lucro_teto_informacao_perfeita'] = float((RETORNO_POR_ACERTO - custo) * int(y_cls_test.sum()))
    cenarios_lucro[nome_cenario] = linha

# a diferença de lucro entre Logística e RFM manual é real ou ruído do teste? (bootstrap)
rng_lucro = np.random.default_rng(RANDOM_STATE)
for nome_cenario, custo in CENARIOS_CUSTO.items():
    est = cenarios_lucro[nome_cenario]['estrategias']
    acionar_log = prob_logistica >= est['Modelo C (Regressão Logística)']['corte']
    acionar_rfm = score_rfm >= est['RFM manual (quintis)']['corte']
    difs = []
    for _ in range(2000):
        idx = rng_lucro.integers(0, len(y_teste_array), len(y_teste_array))
        difs.append(lucro_campanha(y_teste_array[idx], acionar_log[idx], custo)[0]
                    - lucro_campanha(y_teste_array[idx], acionar_rfm[idx], custo)[0])
    cenarios_lucro[nome_cenario]['ic95_lucro_logistica_menos_rfm'] = [float(v) for v in np.percentile(difs, [2.5, 97.5])]

# curva de lucro por limiar da Regressão Logística (validação cruzada no treino), por cenário
curva_lucro_logistica = {
    nome_cenario: [float(v / len(y_cls_train))
                   for v in melhor_corte(y_cls_train, prob_cv_logistica, limiares_candidatos, custo)[1]]
    for nome_cenario, custo in CENARIOS_CUSTO.items()
}

# interpretação da Regressão Logística: coeficientes padronizados e razão de chances
modelo_logistica = busca_logistica.best_estimator_
nomes_features = modelo_logistica[:-1].get_feature_names_out()
coef_logistica = modelo_logistica.named_steps['logistica'].coef_[0]
coeficientes_logistica = {nome: {'coef': float(c), 'odds_ratio': float(np.exp(c))}
                          for nome, c in zip(nomes_features, coef_logistica)}

# auditoria de uma decisão do KNN (Aula 8): os vizinhos de um cliente típico do teste
modelo_knn = busca_knn.best_estimator_
transformar = modelo_knn[:-1]
espaco_treino = transformar.transform(X_train)
espaco_teste = transformar.transform(X_test)
cliente_tipico = int(np.argmin(np.linalg.norm(espaco_teste - np.median(espaco_teste, axis=0), axis=1)))
distancias, indices = modelo_knn.named_steps['knn'].kneighbors(espaco_teste[[cliente_tipico]])
vizinhos = X_train.iloc[indices[0]].copy()
vizinhos['Recomprou'] = y_cls_train.iloc[indices[0]].to_numpy()
vizinhos['Distancia'] = distancias[0]
auditoria_knn = {
    'cliente': X_test.iloc[cliente_tipico].to_dict(),
    'recomprou_de_verdade': int(y_cls_test.iloc[cliente_tipico]),
    'prob_prevista': float(prob_knn[cliente_tipico]),
    'classe_prevista': int(prob_knn[cliente_tipico] >= limiar_knn),
    'k': int(modelo_knn.named_steps['knn'].n_neighbors),
    'pct_vizinhos_que_recompraram': float(vizinhos['Recomprou'].mean() * 100),
    'cinco_mais_proximos': vizinhos.head(5).round(3).to_dict('records'),
}

classificacao = {
    'n_train': int(len(y_cls_train)),
    'n_test': int(len(y_cls_test)),
    'pct_positivos_treino': float(y_cls_train.mean() * 100),
    'pct_positivos_teste': float(y_cls_test.mean() * 100),
    'rfm_nota_corte': int(rfm_manual.nota_corte_),
    'limiar_logistica': limiar_logistica,
    'limiar_knn': limiar_knn,
    'logistica': resumo_busca(busca_logistica),
    'knn': resumo_busca(busca_knn),
    'teste': classificacao_teste,
    'diferenca_auc_logistica_menos_knn': float(classificacao_teste['Modelo C (Regressão Logística)']['ROC_AUC']
                                               - classificacao_teste['Modelo D (KNN)']['ROC_AUC']),
    'ic95_diferenca_auc': [float(v) for v in ic_diferenca_auc],
    'diferenca_auc_logistica_menos_rfm': float(classificacao_teste['Modelo C (Regressão Logística)']['ROC_AUC']
                                               - classificacao_teste['Baseline: RFM manual (quintis)']['ROC_AUC']),
    'ic95_diferenca_auc_rfm': [float(v) for v in ic_diferenca_auc_rfm],
    'rfm_auc_cv_media': float(np.mean(auc_cv_rfm)),
    'rfm_auc_cv_dp': float(np.std(auc_cv_rfm)),
    'curva_auc_por_k': curva_auc_k,
    'knn_metrica_da_curva': melhor_metrica,
    'efeito_preprocessamento_knn': efeito_preprocessamento,
    'analise_limiar_logistica_cv': analise_limiar,
    'coeficientes_logistica': coeficientes_logistica,
    'auditoria_knn': auditoria_knn,
    'lucro_campanha': {
        'retorno_por_acerto': RETORNO_POR_ACERTO,
        'cenarios': cenarios_lucro,
        'limiares_da_curva': [float(v) for v in limiares_candidatos],
        'curva_lucro_por_cliente_logistica_cv': curva_lucro_logistica,
    },
}

# ------------------------------------------------------------------
# 8. Validação temporal: treinar no passado e testar no trimestre seguinte
# ------------------------------------------------------------------
# Na prática, o modelo é treinado com o que já aconteceu e usado no período seguinte.
# Aqui o corte volta um horizonte inteiro: features até o corte anterior, alvo entre o
# corte anterior e o corte atual. Os modelos são reajustados só nessa base antiga e
# avaliados na base atual (features até o corte atual, alvo na janela futura).
horizonte = max_date - cutoff
corte_anterior = cutoff - horizonte
base_antiga = montar_base(df_clean, corte_anterior, cutoff)
X_antigo, y_antigo = base_antiga[feature_cols], base_antiga['Recomprou']
X_atual, y_atual = data[feature_cols], data['Recomprou']

busca_log_antiga = GridSearchCV(pipe_logistica, grade_logistica, cv=cv_estratificada,
                                scoring='roc_auc', n_jobs=-1).fit(X_antigo, y_antigo)
busca_knn_antiga = GridSearchCV(pipe_knn, grade_knn, cv=cv_estratificada,
                                scoring='roc_auc', n_jobs=-1).fit(X_antigo, y_antigo)
rfm_antigo = ScoreRFM().fit(X_antigo, y_antigo)

prob_log_temporal = busca_log_antiga.predict_proba(X_atual)[:, 1]
prob_knn_temporal = busca_knn_antiga.predict_proba(X_atual)[:, 1]
limiar_log_antigo = escolher_limiar(y_antigo, cross_val_predict(busca_log_antiga.best_estimator_, X_antigo, y_antigo,
                                                                cv=cv_estratificada, method='predict_proba')[:, 1])
limiar_knn_antigo = escolher_limiar(y_antigo, cross_val_predict(busca_knn_antiga.best_estimator_, X_antigo, y_antigo,
                                                                cv=cv_estratificada, method='predict_proba')[:, 1])
validacao_temporal = {
    'corte_anterior': str(corte_anterior.date()),
    'corte_atual': str(cutoff.date()),
    'horizonte_dias': int(horizonte.days),
    'n_clientes_treino_antigo': int(len(base_antiga)),
    'pct_recomprou_periodo_antigo': float(y_antigo.mean() * 100),
    'n_clientes_teste_atual': int(len(data)),
    'pct_recomprou_periodo_atual': float(y_atual.mean() * 100),
    'tenure_max_treino_antigo': int(X_antigo['Tenure'].max()),
    'tenure_max_teste_atual': int(X_atual['Tenure'].max()),
    'resultados': {
        'Baseline: RFM manual (quintis)': metricas_classificacao(y_atual, rfm_antigo.predict(X_atual),
                                                                 rfm_antigo.pontuar(X_atual)),
        'Modelo C (Regressão Logística)': metricas_classificacao(y_atual, (prob_log_temporal >= limiar_log_antigo).astype(int),
                                                                 prob_log_temporal),
        'Modelo D (KNN)': metricas_classificacao(y_atual, (prob_knn_temporal >= limiar_knn_antigo).astype(int),
                                                 prob_knn_temporal),
    },
    'parametros_logistica': {k.split('__')[-1]: v for k, v in busca_log_antiga.best_params_.items()},
    'parametros_knn': {k.split('__')[-1]: v for k, v in busca_knn_antiga.best_params_.items()},
    'limiar_logistica': limiar_log_antigo,
    'limiar_knn': limiar_knn_antigo,
    'rfm_nota_corte': int(rfm_antigo.nota_corte_),
    'prob_media_prevista_logistica': float(prob_log_temporal.mean() * 100),
}

output = {
    'eda_stats': eda_stats,
    'results': results,
    'residuals_summary': {
        'modelo_a': {'mean': float(resid_a.mean()), 'std': float(resid_a.std())},
        'modelo_b': {'mean': float(resid_b.mean()), 'std': float(resid_b.std())},
    },
    'robustez_regressao': robustez_regressao,
    'classificacao': classificacao,
    'validacao_temporal': validacao_temporal,
}

with open('resultados.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)

resumo = {
    'regressao_teste': {k: v for k, v in results.items() if k in ('ModeloA_LinearMultipla', 'ModeloB_Ridge', 'KNN_Regressor')},
    'regressao_cv_treino': {k: {m: v[m] for m in ('MAE_media', 'MAE_dp', 'R2_media')}
                            for k, v in robustez_regressao.items() if isinstance(v, dict)},
    'classificacao_teste': {k: {m: round(v[m], 4) for m in ('Acuracia', 'F1', 'F1_macro', 'ROC_AUC', 'PR_AUC')}
                            for k, v in classificacao_teste.items()},
    'validacao_temporal_auc': {k: round(v['ROC_AUC'], 4) for k, v in validacao_temporal['resultados'].items()},
    'lucro_teste': {c: {e: round(v['lucro_teste']) for e, v in d['estrategias'].items()}
                    for c, d in cenarios_lucro.items()},
}
print(json.dumps(resumo, indent=2, ensure_ascii=False, default=str))
