# pipeline.py
import os
import argparse
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

# -----------------------
# Utilidades: directorios y requirements
# -----------------------
def ensure_dirs():
    os.makedirs('internas', exist_ok=True)
    os.makedirs(os.path.join('internas','results'), exist_ok=True)
    os.makedirs(os.path.join('internas','models'), exist_ok=True)

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
# Preprocesado robusto
# -----------------------
def preprocess(df):
    """
    - Normaliza nombres simples
    - Reconcilia unemployment/unemployment.1
    - Calcula gdp_pc y log_gdp_pc si faltan (y evita log(0))
    - Crea lags por país
    - Elimina filas que faltan en columnas esenciales (IDH y log_gdp_pc)
    - Devuelve (df_processed, report)
    """
    report = {'initial_rows': int(df.shape[0]), 'actions': {}}
    df = df.copy()
    # Normalizar nombres whitespace
    df.columns = [c.strip() for c in df.columns]

    # Renombrar variantes simples (si aparecen)
    lower_map = {c.lower().replace(' ','').replace('_',''): c for c in df.columns}
    def find_col(*cands):
        for cand in cands:
            k = cand.lower().replace(' ','').replace('_','')
            if k in lower_map:
                return lower_map[k]
        return None

    # Reconocer unemployment variantes
    unemp_col = find_col('unemployment','unemployment.1','desempleo')
    if unemp_col and unemp_col != 'unemployment':
        df = df.rename(columns={unemp_col:'unemployment'})
        report['actions']['unemployment_renamed_from'] = unemp_col

    # Forzar numéricos en columnas esperadas cuando existan
    maybe_num = ['PIB','population','IDH','unemployment','gdp_pc','log_gdp_pc','gdp_growth','poverty']
    for c in maybe_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Calcular gdp_pc y log_gdp_pc si faltan
    if ('gdp_pc' not in df.columns) and set(['PIB','population']).issubset(df.columns):
        df['gdp_pc'] = df['PIB'] / df['population']
        report['actions']['gdp_pc_computed'] = True
    if 'gdp_pc' in df.columns and 'log_gdp_pc' not in df.columns:
        # evitar log(0)
        df['gdp_pc'] = df['gdp_pc'].replace(0, np.nan)
        df['log_gdp_pc'] = np.log(df['gdp_pc'].where(df['gdp_pc']>0, np.nan))
        report['actions']['log_gdp_pc_computed'] = True

    # Ordenar y crear lags
    if set(['country','year']).issubset(df.columns):
        df = df.sort_values(['country','year']).reset_index(drop=True)
        if 'IDH' in df.columns:
            df['IDH_lag1'] = df.groupby('country')['IDH'].shift(1)
        if 'gdp_growth' in df.columns:
            df['gdp_growth_lag1'] = df.groupby('country')['gdp_growth'].shift(1)

    # Eliminar filas que no tengan columnas esenciales
    essential = []
    if 'IDH' in df.columns: essential.append('IDH')
    if 'log_gdp_pc' in df.columns: essential.append('log_gdp_pc')
    before = int(df.shape[0])
    if essential:
        df = df.dropna(subset=essential).reset_index(drop=True)
    after = int(df.shape[0])
    report['actions']['dropped_for_missing_essentials'] = {'before': before, 'after': after, 'dropped': before-after}
    report['final_rows'] = int(df.shape[0])
    return df, report

# -----------------------
# EDA robusto: series, corr, VIF
# -----------------------
def eda_report(df):
    print("Dataset head:\n", df.head(5).to_string(index=False))
    print("\nSummary by country (IDH, log_gdp_pc):")
    if 'country' in df.columns and 'IDH' in df.columns and 'log_gdp_pc' in df.columns:
        summary = df.groupby('country').agg({'IDH':['mean','std'],'log_gdp_pc':['mean','std']})
        print(summary.head().to_string())
    # Timeseries sample
    if 'year' in df.columns and 'country' in df.columns and 'IDH' in df.columns:
        countries = df['country'].unique()[:6]
        plt.figure(figsize=(10,6))
        for c in countries:
            tmp = df[df['country']==c]
            plt.plot(tmp['year'], tmp['IDH'], label=c)
        plt.xlabel('Year'); plt.ylabel('IDH'); plt.title('IDH time series (sample countries)')
        plt.legend(); plt.tight_layout()
        plt.savefig('internas/results/IDH_timeseries_sample.png'); plt.close()

    # Correlación y VIF (limpio)
    Xcols = [c for c in ['log_gdp_pc','unemployment','gdp_growth'] if c in df.columns]
    if len(Xcols) >= 2:
        X = df[Xcols].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        print("NaNs por columna para corr/VIF:\n", X.isna().sum().to_string())
        X_clean = X.dropna()
        if X_clean.shape[0] < max(5, X.shape[1]+1):
            print("Advertencia: no hay suficientes filas limpias para VIF. Se omite VIF.")
        else:
            corr = X_clean.corr()
            sns.heatmap(corr, annot=True)
            plt.title('Correlation matrix'); plt.tight_layout()
            plt.savefig('internas/results/corr_matrix.png'); plt.close()
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            try:
                vif = pd.DataFrame({'var': X_clean.columns,
                                    'VIF': [variance_inflation_factor(X_clean.values, i) for i in range(X_clean.shape[1])]})
                print("\nVIF:\n", vif.to_string(index=False))
                vif.to_csv('internas/results/vif.csv', index=False)
            except Exception as e:
                print("Error calculando VIF:", e)

# -----------------------
# Guardado de artefactos por fold
# -----------------------
def ensure_model_fold_dir(model_name, fold_idx):
    base = os.path.join('internas', 'results', model_name, f'fold_{fold_idx}')
    os.makedirs(base, exist_ok=True)
    return base

def save_preds_csv(base_dir, df_test, probs=None):
    out_path = os.path.join(base_dir, 'preds.csv')
    df_out = df_test.copy()
    if probs is not None:
        prob_df = pd.DataFrame(probs, columns=[f'prob_class_{i}' for i in range(probs.shape[1])], index=df_out.index)
        df_out = pd.concat([df_out.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)
    df_out.to_csv(out_path, index=False)

def save_loss_csv(base_dir, loss_history):
    out_path = os.path.join(base_dir, 'loss.csv')
    if loss_history is None or len(loss_history) == 0:
        pd.DataFrame(columns=['epoch','train_loss','val_loss','timestamp']).to_csv(out_path, index=False)
        return
    pd.DataFrame(loss_history).to_csv(out_path, index=False)

def save_train_info(base_dir, train_info):
    out_path = os.path.join(base_dir, 'train_info.json')
    with open(out_path, 'w') as f:
        json.dump(train_info, f, indent=2)

def save_model_obj(base_dir, model_obj, name='model.joblib'):
    try:
        joblib.dump(model_obj, os.path.join(base_dir, name))
    except Exception as e:
        print(f"Warning: failed to save model object: {e}")

def try_save_pipeline_components(base_dir, pipeline_obj):
    try:
        if isinstance(pipeline_obj, Pipeline):
            for name, step in pipeline_obj.named_steps.items():
                if hasattr(step, 'get_params') and name in ['scaler','pca']:
                    joblib.dump(step, os.path.join(base_dir, f'{name}.joblib'))
    except Exception as e:
        print("Warning saving pipeline components:", e)

def safe_get_loss_curve(model):
    if hasattr(model, 'loss_curve_'):
        return [{'epoch': i, 'train_loss': float(l), 'val_loss': None, 'timestamp': None} for i, l in enumerate(model.loss_curve_)]
    return []

# -----------------------
# Modelos
# -----------------------
class FERegressor:
    """Fixed-effects (within) OLS regresor. Produce predicciones continuas."""
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.res = None

    def fit(self, df, y_col, X_cols, entity_col='country'):
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
        df_ = df.copy()
        for col in X_cols:
            df_[col+'_demean'] = df_.groupby(entity_col)[col].transform(lambda s: s - s.mean())
        X = df_[[c+'_demean' for c in X_cols]].values
        const = self.intercept_
        yhat = const + X.dot(self.coef_)
        return yhat

def build_pca_ridge_pipeline(n_components=2, alpha=1.0):
    n_components = max(1, int(n_components))
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),   # <-- imputa con mediana (fit en train)
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components)),
        ('clf', RidgeClassifier(alpha=alpha))
    ])
    return pipe

def build_logistic():
    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('clf', logreg)])

def build_mlp():
    mlp = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', alpha=1e-3,
                        max_iter=2000, random_state=RANDOM_STATE, early_stopping=True, n_iter_no_change=30)
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('clf', mlp)])

class LinearSoftmaxClassifier:
    """Implementación simple de softmax lineal con optimizador SGD/Adam."""
    def __init__(self, n_features, n_classes, lr=0.01, epochs=200, batch_size=32, optimizer='adam', alpha=0.0, verbose=False):
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.lr = lr
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.optimizer = optimizer
        self.alpha = alpha
        self.verbose = verbose
        self.W = np.zeros((self.n_classes, self.n_features))
        self.b = np.zeros(self.n_classes)

    def _softmax(self, z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        n, p = X.shape
        K = self.n_classes
        if p != self.n_features:
            # ajustar dimensiones si el usuario cometió error
            self.n_features = p
            self.W = np.zeros((self.n_classes, self.n_features))
        Y = np.zeros((n, K))
        Y[np.arange(n), y] = 1
        mW = np.zeros_like(self.W); vW = np.zeros_like(self.W)
        mb = np.zeros_like(self.b); vb = np.zeros_like(self.b)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        t = 0
        loss_history = []
        start_time = time.time()
        for epoch in range(self.epochs):
            idx = np.arange(n)
            np.random.shuffle(idx)
            epoch_losses = []
            for start in range(0, n, self.batch_size):
                t += 1
                batch_idx = idx[start:start+self.batch_size]
                Xb = X[batch_idx]
                Yb = Y[batch_idx]
                logits = Xb.dot(self.W.T) + self.b
                P = self._softmax(logits)
                train_loss = - np.sum(Yb * np.log(np.clip(P,1e-12,None))) / Xb.shape[0]
                epoch_losses.append(train_loss)
                grad_W = (P - Yb).T.dot(Xb) / Xb.shape[0] + 2*self.alpha*self.W
                grad_b = (P - Yb).mean(axis=0)
                if self.optimizer == 'sgd':
                    self.W -= self.lr * grad_W
                    self.b -= self.lr * grad_b
                elif self.optimizer == 'adam':
                    # Adam updates
                    mW = beta1*mW + (1-beta1)*grad_W
                    vW = beta2*vW + (1-beta2)*(grad_W**2)
                    mW_hat = mW / (1 - beta1**t)
                    vW_hat = vW / (1 - beta2**t)
                    self.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
                    mb = beta1*mb + (1-beta1)*grad_b
                    vb = beta2*vb + (1-beta2)*(grad_b**2)
                    mb_hat = mb / (1 - beta1**t)
                    vb_hat = vb / (1 - beta2**t)
                    self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)
                else:
                    raise ValueError("optimizer must be 'sgd' or 'adam'")
            avg_epoch_loss = np.mean(epoch_losses) if len(epoch_losses)>0 else None
            loss_history.append({'epoch': epoch, 'train_loss': float(avg_epoch_loss), 'val_loss': None, 'timestamp': time.time()})
            if self.verbose and epoch % 50 == 0:
                preds = self.predict(X)
                acc = (preds == y).mean()
                print(f"Epoch {epoch}: acc={acc:.4f}")
        train_time = time.time() - start_time
        self._loss_history = loss_history
        self._train_time = train_time
        return self

    def predict_proba(self, X):
        logits = X.dot(self.W.T) + self.b
        return self._softmax(logits)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def get_loss_history(self):
        return getattr(self, '_loss_history', [])

    def get_train_time(self):
        return getattr(self, '_train_time', None)

# -----------------------
# Rolling-window evaluation (con export de artefactos)
# -----------------------
def save_fold_outputs_all(model_name, fold_idx, df_test, y_true, y_pred, probs, loss_history, train_info, model_obj=None, pipeline_obj=None):
    base = ensure_model_fold_dir(model_name, fold_idx)
    df_out = df_test.copy()
    df_out['y_true'] = y_true
    df_out['y_pred'] = y_pred
    save_preds_csv(base, df_out, probs=probs)
    save_loss_csv(base, loss_history)
    save_train_info(base, train_info)
    if model_obj is not None:
        save_model_obj(base, model_obj, name=f'model_{model_name}_fold{fold_idx}.joblib')
    if pipeline_obj is not None:
        try_save_pipeline_components(base, pipeline_obj)

def rolling_window_evaluate(df, features, label_col, model_builders, window=5, year_col='year'):
    years = sorted(df[year_col].unique())
    results = defaultdict(list)
    preds_storage = []
    folds_meta = []
    fold_idx = 0
    for i in range(len(years)-window):
        train_yrs = years[i:i+window]
        test_yrs = [years[i+window]]
        train = df[df[year_col].isin(train_yrs)].copy()
        test = df[df[year_col].isin(test_yrs)].copy()

        X_train = train[features].values if len(features)>0 else np.empty((len(train),0))
        X_test = test[features].values if len(features)>0 else np.empty((len(test),0))
        y_train = train[label_col].values
        y_test = test[label_col].values

        fold_meta = {
            'fold': int(fold_idx),
            'train_years': [int(y) for y in train_yrs],
            'test_year': int(test_yrs[0])}
        folds_meta.append(fold_meta)


                # --- imputación y escalado por fold (fit sólo en train) ---
        # X_train / X_test matrices pueden estar vacías si no hay features
        if X_train.size == 0:
            Xtr_s = Xte_s = X_train
            imputer_fold = None
            scaler_fold = None
        else:
            # Imputer entrenado en train (mediana) -> evita NaNs para PCA/ML y modelos custom
            imputer_fold = SimpleImputer(strategy='median')
            X_train_imp = imputer_fold.fit_transform(X_train)
            X_test_imp = imputer_fold.transform(X_test)

            # informe rápido (opcional)
            try:
                n_missing_train = np.isnan(X_train).sum()
                n_missing_test = np.isnan(X_test).sum()
                print(f"Fold {fold_idx}: missing in X_train total = {n_missing_train.sum() if hasattr(n_missing_train,'sum') else n_missing_train}, missing in X_test total = {n_missing_test.sum() if hasattr(n_missing_test,'sum') else n_missing_test}")
            except Exception:
                pass

            # scaler por fold (fit sólo en train_imputed)
            scaler_fold = StandardScaler()
            scaler_fold.fit(X_train_imp)
            Xtr_s = scaler_fold.transform(X_train_imp)
            Xte_s = scaler_fold.transform(X_test_imp)

        for name, builder in model_builders.items():
            start = time.time()
            loss_history = []
            train_time = None
            iterations_to_tol = None
            converged = False
            probs = None
            model_obj = None
            pipeline_obj = None
            if name == 'FE':
                train_scaled = train.copy()
                test_scaled = test.copy()
                if len(features) > 0:
                    # usamos imputer + scaler producidos en este fold
                    train_scaled.loc[:, features] = Xtr_s
                    test_scaled.loc[:, features] = Xte_s
                fe = FERegressor()
                res = fe.fit(train_scaled, y_col='IDH', X_cols=features, entity_col='country')
                yhat_cont = fe.predict(test_scaled, X_cols=features, entity_col='country')
                # convertir continuo a clases usando terciles del train IDH
                q = np.quantile(train['IDH'], [0, 1/3, 2/3, 1.0])
                yhat = np.digitize(yhat_cont, bins=q[1:-1])
                probs = None
                model_obj = fe
            else:
                if isinstance(builder, Pipeline):
                    pipeline_obj = builder
                # safety: si por alguna razón hay NaN en X_train, imputa rápido (pipeline ya tiene imputer, así que esto es extra)
                    if np.isnan(X_train).any() or np.isnan(X_test).any():
                        X_train = SimpleImputer(strategy='median').fit_transform(X_train)
                        X_test = SimpleImputer(strategy='median').fit_transform(X_test)
                    pipeline_obj.fit(X_train, y_train)
                    yhat = pipeline_obj.predict(X_test)
                    try:
                        probs = pipeline_obj.predict_proba(X_test)
                    except Exception:
                        probs = None
                    model_obj = pipeline_obj
                    loss_history = safe_get_loss_curve(pipeline_obj.named_steps.get('clf', None))
                    train_time = None
                elif isinstance(builder, LinearSoftmaxClassifier):
                    clf = builder
                    # Xtr_s/Xte_s ya imputados y escalados
                    clf.fit(Xtr_s, y_train)
                    yhat = clf.predict(Xte_s)
                    probs = clf.predict_proba(Xte_s)
                    loss_history = clf.get_loss_history()
                    train_time = clf.get_train_time()
                    model_obj = clf
                else:
                    raise ValueError("Unknown builder type for model: " + str(name))

            elapsed = time.time() - start
            train_time = train_time if train_time is not None else elapsed

            # convergencia heurística
            iterations_to_tol = None
            if loss_history and len(loss_history) > 1:
                eps = 1e-4
                prev = loss_history[0].get('train_loss', None)
                for rec in loss_history[1:]:
                    cur = rec.get('train_loss', None)
                    if prev is not None and cur is not None:
                        if abs(cur - prev) / max(1.0, abs(prev)) < eps:
                            iterations_to_tol = rec.get('epoch')
                            converged = True
                            break
                    prev = cur

            f1 = f1_score(y_test, yhat, average='macro') if len(y_test)>0 else None
            prec = precision_score(y_test, yhat, average='macro', zero_division=0) if len(y_test)>0 else None
            rec = recall_score(y_test, yhat, average='macro', zero_division=0) if len(y_test)>0 else None
            acc = (y_test == yhat).mean() if len(y_test)>0 else None

            results['model'].append(name)
            results['fold'].append(fold_idx)
            results['train_years'].append(tuple(train_yrs))
            results['test_year'].append(test_yrs[0])
            results['f1_macro'].append(f1)
            results['precision_macro'].append(prec)
            results['recall_macro'].append(rec)
            results['accuracy'].append(acc)
            results['time_s'].append(train_time)
            results['iterations_to_tol'].append(iterations_to_tol if iterations_to_tol is not None else 'NA')
            results['converged'].append(converged)

            # Guardar artefactos por fold
            df_test = test.reset_index().rename(columns={'index':'original_index'})
            df_test_for_save = df_test[['original_index','country','year']].copy()
            df_test_for_save.rename(columns={'original_index':'index'}, inplace=True)

            save_fold_outputs_all(
                model_name=name,
                fold_idx=fold_idx,
                df_test=df_test_for_save.assign(y_true=list(y_test)),
                y_true=list(y_test),
                y_pred=list(yhat),
                probs=probs,
                loss_history=loss_history,
                train_info={
                    'n_epochs': len(loss_history) if loss_history else None,
                    'iterations_to_tol': iterations_to_tol,
                    'train_time_s': train_time,
                    'converged': converged,
                    'random_state': RANDOM_STATE
                },
                model_obj=model_obj,
                pipeline_obj=pipeline_obj
            )

            preds_storage.append({'model':name, 'fold':fold_idx, 'test_year':test_yrs[0], 'y_test':list(y_test), 'y_pred':list(yhat)})
        fold_idx += 1

    res_df = pd.DataFrame(results)
    res_df.to_csv('internas/results/rolling_results.csv', index=False)
    joblib.dump(preds_storage, 'internas/results/preds_storage.joblib')
    with open('internas/results/folds_meta.json','w') as f:
        json.dump(folds_meta, f, indent=2)
    return res_df, preds_storage

# -----------------------
# Bootstrap y permutation test
# -----------------------
def bootstrap_ci(y_true, y_pred, metric_func, B=2000, alpha=0.05):
    n = len(y_true)
    scores = []
    rng = np.random.RandomState(RANDOM_STATE)
    idx = np.arange(n)
    for b in range(B):
        sample = rng.choice(idx, size=n, replace=True)
        scores.append(metric_func(np.array(y_true)[sample], np.array(y_pred)[sample]))
    lo, hi = np.percentile(scores, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi, np.mean(scores), np.std(scores)

def paired_permutation_test(y_true, yA, yB, metric_func, n_perm=5000):
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
    pval = (count + 1) / (n_perm + 1)
    return obs, pval

# -----------------------
# Main
# -----------------------
def main(args):
    ensure_dirs()
    save_requirements()
    data_path = args.data if args.data else 'data_panel_ready.csv'
    if not os.path.exists(data_path):
        print("ERROR: data file not found. Provide --data path or place file as data_panel_ready.csv in working dir.")
        return
    df_raw = pd.read_csv(data_path)
    df_raw.to_csv('internas/results/data_raw.csv', index=False)

    # Preprocess
    df_proc, pre_report = preprocess(df_raw)
    df_proc.to_csv('internas/data_processed_imputed.csv', index=False)
    with open('internas/results/preprocess_report.json','w') as f:
        json.dump(pre_report, f, indent=2)

    print("Processed shape:", df_proc.shape)
    eda_report(df_proc)

    # Build features/labels
    if 'IDH' not in df_proc.columns:
        raise ValueError("IDH column required in processed data")
    df_proc['IDH_cat'] = pd.qcut(df_proc['IDH'], q=3, labels=[0,1,2])

    features = [c for c in ['log_gdp_pc','unemployment','gdp_growth','IDH_lag1','gdp_growth_lag1'] if c in df_proc.columns]
    if len(features) == 0:
        print("Advertencia: no hay features detectadas. Revisa preprocess.")
    label_col = 'IDH_cat'

    # Model builders (asegurar n_features >=1 para custom)
    n_feats = max(1, len(features))
    model_builders = {
        'PCA+Ridge': build_pca_ridge_pipeline(n_components=min(2, n_feats), alpha=1.0),
        'MLP': build_mlp(),
        'LogReg': build_logistic(),
        'LinearSoftmax': LinearSoftmaxClassifier(n_features=n_feats, n_classes=3, lr=0.01, epochs=300, batch_size=32, optimizer='adam', alpha=1e-3, verbose=False),
        'FE': 'FE_placeholder'
    }

    # Rolling evaluation
    res_df, preds_storage = rolling_window_evaluate(df_proc, features, label_col, model_builders, window=5, year_col='year')
    print("Rolling-window results saved to internas/results/rolling_results.csv")
    print(res_df.groupby('model')['f1_macro'].agg(['mean','std']))

    # Bootstrap agregado por modelo (concatenando todas las observaciones por modelo)
    all_boot = []
    try:
        prs = joblib.load('internas/results/preds_storage.joblib')
        dfp = pd.DataFrame(prs)
        models = dfp['model'].unique()
        for m in models:
            rows = dfp[dfp['model']==m]
            y_true_all = np.concatenate([np.array(x) for x in rows['y_test']])
            y_pred_all = np.concatenate([np.array(x) for x in rows['y_pred']])
            lo, hi, mean, sd = bootstrap_ci(y_true_all, y_pred_all, lambda a,b: f1_score(a,b,average='macro'), B=500)
            all_boot.append((m, mean, sd, lo, hi))
        pd.DataFrame(all_boot, columns=['model','mean_f1','sd','ci_lo','ci_hi']).to_csv('internas/results/bootstrap_agg_by_model.csv', index=False)
    except Exception as e:
        print("Aviso: fallo en bootstrap agregado:", e)

    print("All done. Results and artifacts in internas/results/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_panel_ready.csv', help='Path to data file (default: data_panel_ready.csv)')
    args = parser.parse_args()
    main(args)