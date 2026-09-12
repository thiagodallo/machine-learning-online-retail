import io
import os
import urllib.request
import zipfile

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json

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

NON_PRODUCT_CODES = {'POST', 'DOT', 'M', 'C2', 'D', 'S', 'BANK CHARGES', 'AMAZONFEE', 'CRUK', 'PADS', 'B'}

df_clean = df.dropna(subset=['CustomerID']).copy()
n_after_customerid = len(df_clean)

df_clean = df_clean[df_clean['UnitPrice'] > 0].copy()
n_after_price = len(df_clean)

df_clean = df_clean[~df_clean['StockCode'].astype(str).isin(NON_PRODUCT_CODES)].copy()
n_after_codes = len(df_clean)

df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

min_date = df_clean['InvoiceDate'].min()
max_date = df_clean['InvoiceDate'].max()
total_days = (max_date - min_date).days

cutoff = min_date + pd.Timedelta(days=int(total_days * 0.75))

hist = df_clean[df_clean['InvoiceDate'] <= cutoff].copy()
future = df_clean[df_clean['InvoiceDate'] > cutoff].copy()

ref_date = cutoff

rfm = hist.groupby('CustomerID').agg(
    Recencia=('InvoiceDate', lambda x: (ref_date - x.max()).days),
    Frequencia=('InvoiceNo', 'nunique'),
    Monetario=('TotalPrice', 'sum'),
    TicketMedio=('TotalPrice', lambda x: x.sum() / hist.loc[x.index, 'InvoiceNo'].nunique()),
    Tenure=('InvoiceDate', lambda x: (ref_date - x.min()).days),
    QtdItensDistintos=('StockCode', 'nunique'),
).reset_index()

future_spend = future.groupby('CustomerID')['TotalPrice'].sum().rename('ValorFuturo').reset_index()

data = rfm.merge(future_spend, on='CustomerID', how='left')
data['ValorFuturo'] = data['ValorFuturo'].fillna(0.0)
data['ValorFuturo'] = data['ValorFuturo'].clip(lower=0)

n_customers_hist = len(rfm)
n_customers_final = len(data)

feature_cols = ['Recencia', 'Frequencia', 'Monetario', 'TicketMedio', 'Tenure', 'QtdItensDistintos']
X = data[feature_cols].copy()
y = data['ValorFuturo'].copy()

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
    'feature_describe': X.describe().to_dict(),
    'target_describe': y.describe().to_dict(),
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

vif_data = pd.DataFrame()
vif_data['feature'] = feature_cols
Xc = X_train.copy()
Xc.insert(0, 'const', 1.0)
vif_data['VIF'] = [variance_inflation_factor(Xc.values, i + 1) for i in range(len(feature_cols))]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model_a = LinearRegression()
model_a.fit(X_train_s, y_train)
pred_a = model_a.predict(X_test_s)

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
param_grid = {'alpha': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}
grid = GridSearchCV(Ridge(random_state=RANDOM_STATE), param_grid, cv=kf, scoring='neg_mean_absolute_error')
grid.fit(X_train_s, y_train)
model_b = grid.best_estimator_
pred_b = model_b.predict(X_test_s)


def metrics(y_true, y_pred, n_features):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'R2_ajustado': r2_adj}


results = {
    'ModeloA_LinearMultipla': metrics(y_test, pred_a, len(feature_cols)),
    'ModeloB_Ridge': metrics(y_test, pred_b, len(feature_cols)),
    'ridge_best_alpha': grid.best_params_['alpha'],
    'ridge_cv_results_alpha': param_grid['alpha'],
    'ridge_cv_mae_mean': list(-grid.cv_results_['mean_test_score']),
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

output = {
    'eda_stats': eda_stats,
    'results': results,
    'residuals_summary': {
        'modelo_a': {'mean': float(resid_a.mean()), 'std': float(resid_a.std())},
        'modelo_b': {'mean': float(resid_b.mean()), 'std': float(resid_b.std())},
    },
}

with open('pipeline_output.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)

print(json.dumps({'eda_stats': eda_stats, 'results': {k: v for k, v in results.items() if k in ('ModeloA_LinearMultipla', 'ModeloB_Ridge', 'ridge_best_alpha')}, 'vif': results['vif']}, indent=2, ensure_ascii=False, default=str))
