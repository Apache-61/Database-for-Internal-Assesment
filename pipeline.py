import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
import joblib
import time

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

# -----------------------
# Utilidades I/O y setup
# -----------------------
def ensure_dirs():
    os.makedirs('internas', exist_ok=True)
    os.makedirs('internas/results', exist_ok=True)
    os.makedirs('internas/models', exist_ok=True)

def save_requirements():
    req = """numpy
pandas
scikit-learn
statsmodels
matplotlib
seaborn
joblib
scipy
"""
    with open('internas/requirements.txt','w') as f:
        f.write(req)

# -----------------------
# Preprocesado (FASE 2)
# -----------------------
def preprocess(df):
    # Expect columns: country, year, IDH, PIB, population, unemployment, gdp_growth
    df = df.copy()
    df.sort_values(['country','year'], inplace=True)
    # Ensure numeric types
    for c in ['IDH','PIB','population','unemployment','gdp_growth']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # GDP per capita and log
    df['gdp_pc'] = df['PIB'] / df['population']
    # avoid log(0)
    df['gdp_pc'] = df['gdp_pc'].replace(0, np.nan)
    df['log_gdp_pc'] = np.log(df['gdp_pc'])
    # lags by country
    df['IDH_lag1'] = df.groupby('country')['IDH'].shift(1)
    df['gdp_growth_lag1'] = df.groupby('country')['gdp_growth'].shift(1)
    # drop rows with missing essential values
    df.dropna(subset=['IDH','log_gdp_pc','unemployment','gdp_growth','IDH_lag1','gdp_growth_lag1'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# -----------------------
# EDA (FASE 3) functions
# -----------------------
def eda_report(df):
    print("Dataset head:\n", df.head())
    print("\nSummary by country (IDH, log_gdp_pc):")
    summary = df.groupby('country').agg({'IDH':['mean','std'],'log_gdp_pc':['mean','std']})
    print(summary.head())
    # time series example (top 6 countries)
    countries = df['country'].unique()[:6]
    plt.figure(figsize=(10,6))
    for c in countries:
        tmp = df[df['country']==c]
        plt.plot(tmp['year'], tmp['IDH'], label=c)
    plt.xlabel('Year'); plt.ylabel('IDH'); plt.title('IDH time series (sample countries)')
    plt.legend(); plt.tight_layout()
    plt.savefig('internas/results/IDH_timeseries_sample.png'); plt.close()
    # Correlation and VIF
    X = df[['log_gdp_pc','unemployment','gdp_growth']]
    corr = X.corr()
    sns.heatmap(corr, annot=True)
    plt.title('Correlation matrix'); plt.tight_layout()
    plt.savefig('internas/results/corr_matrix.png'); plt.close()
    # VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.DataFrame({'var': X.columns,
                        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
    print("\nVIF:\n", vif)
    vif.to_csv('internas/results/vif.csv', index=False)

# -----------------------
# Models: wrappers (FASE 5)
# -----------------------
class FERegressor:
    """Fixed effects via within-transformation and OLS (matrix form)."""
    def __init__(self, formula=None):
        self.formula = formula
        self.coef_ = None
        self.intercept_ = None

    def fit(self, df, y_col, X_cols, entity_col='country'):
        # demean by entity
        df_ = df.copy()
        df_[y_col+'_demean'] = df_.groupby(entity_col)[y_col].transform(lambda s: s - s.mean())
        for col in X_cols:
            df_[col+'_demean'] = df_.groupby(entity_col)[col].transform(lambda s: s - s.mean())
        Y = df_[y_col+'_demean'].values
        X = df_[[c+'_demean' for c in X_cols]].values
        X = sm.add_constant(X)
        res = sm.OLS(Y, X).fit()
        self.res = res
        self.coef_ = res.params[1:]
        self.intercept_ = res.params[0]
        return res

    def predict(self, df, X_cols, entity_col='country'):
        # prediction: y_hat = const + X_demeaned * coef
        df_ = df.copy()
        for col in X_cols:
            df_[col+'_demean'] = df_.groupby(entity_col)[col].transform(lambda s: s - s.mean())
        X = df_[[c+'_demean' for c in X_cols]].values
        const = self.intercept_
        yhat = const + X.dot(self.coef_)
        return yhat

def build_pca_ridge_pipeline(n_components=2, alpha=1.0):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ('clf', RidgeClassifier(alpha=alpha))
    ])
    return pipe

def build_logistic():
    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    return Pipeline([('scaler', StandardScaler()), ('clf', logreg)])

def build_mlp():
    mlp = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', alpha=1e-3, max_iter=500, random_state=RANDOM_STATE)
    return Pipeline([('scaler', StandardScaler()), ('clf', mlp)])

# Simple linear softmax classifier (numpy) — SGD or Adam
class LinearSoftmaxClassifier:
    def __init__(self, n_features, n_classes, lr=0.01, epochs=200, batch_size=32, optimizer='sgd', alpha=0.0, verbose=False):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.alpha = alpha
        self.verbose = verbose
        self.W = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)

    def _softmax(self, z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        # y: integer labels 0..K-1
        n, p = X.shape
        K = self.n_classes
        # one-hot
        Y = np.zeros((n, K))
        Y[np.arange(n), y] = 1
        # Adam state
        mW = np.zeros_like(self.W); vW = np.zeros_like(self.W)
        mb = np.zeros_like(self.b); vb = np.zeros_like(self.b)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        t = 0
        for epoch in range(self.epochs):
            idx = np.arange(n)
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                t += 1
                batch_idx = idx[start:start+self.batch_size]
                Xb = X[batch_idx]
                Yb = Y[batch_idx]
                logits = Xb.dot(self.W.T) + self.b
                P = self._softmax(logits)
                grad_W = (P - Yb).T.dot(Xb) / Xb.shape[0] + 2*self.alpha*self.W
                grad_b = (P - Yb).mean(axis=0)
                if self.optimizer == 'sgd':
                    self.W -= self.lr * grad_W
                    self.b -= self.lr * grad_b
                elif self.optimizer == 'adam':
                    # update W
                    mW = beta1*mW + (1-beta1)*grad_W
                    vW = beta2*vW + (1-beta2)*(grad_W**2)
                    mW_hat = mW / (1 - beta1**t)
                    vW_hat = vW / (1 - beta2**t)
                    self.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
                    # update b
                    mb = beta1*mb + (1-beta1)*grad_b
                    vb = beta2*vb + (1-beta2)*(grad_b**2)
                    mb_hat = mb / (1 - beta1**t)
                    vb_hat = vb / (1 - beta2**t)
                    self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)
                else:
                    raise ValueError("optimizer must be 'sgd' or 'adam'")
            if self.verbose and epoch % 50 == 0:
                preds = self.predict(X)
                acc = (preds == y).mean()
                print(f"Epoch {epoch}: acc={acc:.4f}")

    def predict_proba(self, X):
        logits = X.dot(self.W.T) + self.b
        return self._softmax(logits)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# -----------------------
# Validation: rolling-window (FASE 6)
# -----------------------
def rolling_window_evaluate(df, features, label_col, model_builders, window=5, year_col='year'):
    years = sorted(df[year_col].unique())
    results = defaultdict(list)
    preds_storage = []
    for i in range(len(years)-window):
        train_yrs = years[i:i+window]
        test_yrs = [years[i+window]]
        train = df[df[year_col].isin(train_yrs)].copy()
        test = df[df[year_col].isin(test_yrs)].copy()
        X_train = train[features].values
        X_test = test[features].values
        y_train = train[label_col].values
        y_test = test[label_col].values
        for name, builder in model_builders.items():
            start = time.time()
            # if builder is sklearn pipeline or custom with fit/predict
            if name == 'FE':
                fe = FERegressor()
                fe.fit(train, y_col='IDH', X_cols=features, entity_col='country')
                # FE produces continuous predictions — need to convert to classes for IDH_cat
                yhat_cont = fe.predict(test, X_cols=features, entity_col='country')
                # simple mapping: use quantile cuts from train
                q = np.quantile(train['IDH'], [0, 1/3, 2/3, 1.0])
                yhat = np.digitize(yhat_cont, bins=q[1:-1])
            elif name == 'PCA+Ridge':
                pipe = builder
                pipe.fit(X_train, y_train)
                yhat = pipe.predict(X_test)
            elif name == 'MLP':
                pipe = builder
                pipe.fit(X_train, y_train)
                yhat = pipe.predict(X_test)
            elif name == 'LinearSoftmax':
                clf = builder
                clf.fit(X_train, y_train)
                yhat = clf.predict(X_test)
            elif name == 'LogReg':
                pipe = builder
                pipe.fit(X_train, y_train)
                yhat = pipe.predict(X_test)
            else:
                raise ValueError("Unknown model name")
            elapsed = time.time() - start
            f1 = f1_score(y_test, yhat, average='macro')
            prec = precision_score(y_test, yhat, average='macro', zero_division=0)
            rec = recall_score(y_test, yhat, average='macro', zero_division=0)
            cm = confusion_matrix(y_test, yhat)
            results['model'].append(name)
            results['train_years'].append(tuple(train_yrs))
            results['test_year'].append(test_yrs[0])
            results['f1_macro'].append(f1)
            results['precision_macro'].append(prec)
            results['recall_macro'].append(rec)
            results['time_s'].append(elapsed)
            preds_storage.append({'model':name, 'test_year':test_yrs[0], 'y_test':y_test, 'y_pred':yhat})
    res_df = pd.DataFrame(results)
    res_df.to_csv('internas/results/rolling_results.csv', index=False)
    joblib.dump(preds_storage, 'internas/results/preds_storage.joblib')
    return res_df, preds_storage

# -----------------------
# Bootstrap and Permutation tests (FASE 7)
# -----------------------
def bootstrap_ci(y_true, y_pred, metric_func, B=2000, alpha=0.05):
    n = len(y_true)
    scores = []
    rng = np.random.RandomState(RANDOM_STATE)
    idx = np.arange(n)
    for b in range(B):
        sample = rng.choice(idx, size=n, replace=True)
        scores.append(metric_func(y_true[sample], y_pred[sample]))
    lo, hi = np.percentile(scores, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi, np.mean(scores), np.std(scores)

def paired_permutation_test(y_true, yA, yB, metric_func, n_perm=5000):
    # paired permutation by swapping predictions for each paired observation with p=0.5
    rng = np.random.RandomState(RANDOM_STATE)
    obs = metric_func(y_true, yA) - metric_func(y_true, yB)
    n = len(y_true)
    count = 0
    for _ in range(n_perm):
        swap = rng.rand(n) < 0.5
        A = np.where(swap, yB, yA)
        B = np.where(swap, yA, yB)
        diff = metric_func(y_true, A) - metric_func(y_true, B)
        if abs(diff) >= abs(obs):
            count += 1
    pval = count / n_perm
    return obs, pval

# -----------------------
# Main: glue everything
# -----------------------
def main(args):
    ensure_dirs()
    save_requirements()
    if not args.data or not os.path.exists(args.data):
        print("ERROR: data file not found. Please provide --data data_panel_ready.csv with columns: country,year,IDH,PIB,population,unemployment,gdp_growth")
        return
    df = pd.read_csv(args.data)
    print("Raw data shape:", df.shape)
    df = preprocess(df)
    print("Processed shape:", df.shape)
    df.to_csv('internas/data_processed.csv', index=False)
    eda_report(df)

    # Build features/labels: create 3-class label by terciles of IDH (as requested)
    df['IDH_cat'] = pd.qcut(df['IDH'], q=3, labels=[0,1,2])
    features = ['log_gdp_pc', 'unemployment', 'gdp_growth', 'IDH_lag1', 'gdp_growth_lag1']
    label_col = 'IDH_cat'
    # Build models
    model_builders = {
        'PCA+Ridge': build_pca_ridge_pipeline(n_components=2, alpha=1.0),
        'MLP': build_mlp(),
        'LogReg': build_logistic(),
        'LinearSoftmax': LinearSoftmaxClassifier(n_features=len(features), n_classes=3, lr=0.01, epochs=300, batch_size=32, optimizer='adam', alpha=1e-3, verbose=False),
        'FE': 'FE_placeholder' # handled specially in rolling loop
    }

    # For PCA+Ridge, MLP, LogReg builders are actual estimators; LinearSoftmax is instance
    res_df, preds_storage = rolling_window_evaluate(df, features, label_col, model_builders, window=5, year_col='year')
    print("Rolling-window results saved to internas/results/rolling_results.csv")
    print(res_df.groupby('model')['f1_macro'].agg(['mean','std']))

    # Bootstrap CI for each model aggregated across folds (example)
    all_boot = []
    for entry in preds_storage:
        y_test = np.array(entry['y_test'])
        y_pred = np.array(entry['y_pred'])
        lo, hi, mean, sd = bootstrap_ci(y_test, y_pred, lambda a,b: f1_score(a,b,average='macro'), B=500)
        all_boot.append((entry['model'], entry['test_year'], mean, sd, lo, hi))
    pd.DataFrame(all_boot, columns=['model','test_year','mean_f1','sd','ci_lo','ci_hi']).to_csv('internas/results/bootstrap_summary.csv', index=False)

    # Example paired permutation test between two models on the last fold where both exist
    # find last year results for two models
    last_year = max([p['test_year'] for p in preds_storage])
    A = next((p for p in preds_storage if p['model']=='LogReg' and p['test_year']==last_year), None)
    B = next((p for p in preds_storage if p['model']=='PCA+Ridge' and p['test_year']==last_year), None)
    if A and B:
        obs, pval = paired_permutation_test(np.array(A['y_test']), np.array(A['y_pred']), np.array(B['y_pred']), lambda a,b: f1_score(a,b,average='macro'), n_perm=2000)
        print(f"Permutation test between LogReg and PCA+Ridge on year {last_year}: obs_diff={obs:.4f}, p={pval:.4f}")
    else:
        print("Could not find matching fold for permutation example.")

    print("All done. Results in internas/results/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_panel_ready.csv', help='Path to data_panel_ready.csv')
    args = parser.parse_args()
    main(args)