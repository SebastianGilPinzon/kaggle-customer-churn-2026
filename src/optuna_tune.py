"""Optuna hyperparameter tuning for LGB, XGB, CatBoost."""
import pandas as pd, numpy as np, sys, os, json, warnings
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 3  # Fewer folds for faster tuning
N_TRIALS = 50  # Trials per model
VERSION = "v03_optuna"

# === DATA (same as ensemble3) ===
print('Loading...', file=sys.stderr, flush=True)
train = pd.read_csv('data-kaggle/train.csv')
test = pd.read_csv('data-kaggle/test.csv')
sub = pd.read_csv('data-kaggle/sample_submission.csv')

orig = pd.read_csv('data-original/WA_Fn-UseC_-Telco-Customer-Churn.csv')
orig['Churn'] = orig['Churn'].map({'Yes':1,'No':0})
orig.drop('customerID',axis=1,inplace=True)
orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'],errors='coerce')
common = [c for c in train.columns if c in orig.columns]
oa = orig[common].copy()
for c in train.columns:
    if c not in oa.columns: oa[c]=np.nan
train = pd.concat([train, oa[train.columns]], ignore_index=True)

test_ids = test['id'].copy()
train.drop('id',axis=1,errors='ignore',inplace=True)
test.drop('id',axis=1,errors='ignore',inplace=True)
y = train['Churn'].map(lambda x:1 if str(x) in ('Yes','1','1.0') else 0).astype(int)
train.drop('Churn',axis=1,inplace=True)

# === FE (same as ensemble3) ===
for df in [train, test]:
    if df['TotalCharges'].dtype==object:
        df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
    df['AvgSpend']=df['TotalCharges']/(df['tenure']+1)
    df['TenProxy']=df['TotalCharges']/(df['MonthlyCharges']+.01)
    df['ChTen']=df['MonthlyCharges']*df['tenure']
    df['ChTenLog']=np.log1p(df['MonthlyCharges']*df['tenure'])
    df['ChargeRatio']=df['MonthlyCharges']/(df['TotalCharges']+.01)
    svc=['PhoneService','MultipleLines','InternetService','OnlineSecurity',
         'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['nSvc']=sum((df[c].astype(str)=='Yes').astype(int) for c in svc)
    inet=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['nInet']=sum((df[c].astype(str)=='Yes').astype(int) for c in inet)
    df['nSec']=sum((df[c].astype(str)=='Yes').astype(int) for c in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport'])
    df['nStream']=sum((df[c].astype(str)=='Yes').astype(int) for c in ['StreamingTV','StreamingMovies'])
    df['Fiber']=(df['InternetService'].astype(str)=='Fiber optic').astype(int)
    df['DSL']=(df['InternetService'].astype(str)=='DSL').astype(int)
    df['HasInet']=(df['InternetService'].astype(str)!='No').astype(int)
    df['MTM']=(df['Contract'].astype(str)=='Month-to-month').astype(int)
    df['EChk']=(df['PaymentMethod'].astype(str)=='Electronic check').astype(int)
    df['t_x_m']=df['tenure']*df['MonthlyCharges']
    df['MTM_ten']=df['MTM']*df['tenure']
    df['Fib_ch']=df['Fiber']*df['MonthlyCharges']
    df['EC_MTM']=df['EChk']*df['MTM']
    df['Sec_Fib']=df['nSec']*df['Fiber']
    df['Sr_MTM']=df['SeniorCitizen']*df['MTM']
    df['TenBin']=pd.cut(df['tenure'],bins=[-1,6,12,24,48,72,999],labels=[0,1,2,3,4,5]).astype(float)
    df['ChBin']=pd.cut(df['MonthlyCharges'],bins=[-1,20,40,60,80,100,999],labels=[0,1,2,3,4,5]).astype(float)

cats=train.select_dtypes(include=['object','category']).columns.tolist()
for c in cats:
    freq=train[c].value_counts(normalize=True)
    train[f'{c}_freq']=train[c].map(freq).fillna(0)
    test[f'{c}_freq']=test[c].map(freq).fillna(0)
for c in cats:
    le=LabelEncoder()
    le.fit(pd.concat([train[c],test[c]]).astype(str))
    train[c]=le.transform(train[c].astype(str))
    test[c]=le.transform(test[c].astype(str))
train.fillna(-999,inplace=True)
test.fillna(-999,inplace=True)

print(f'Features: {train.shape[1]}', file=sys.stderr, flush=True)

# Fixed folds for all tuning
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(train, y))

# === LIGHTGBM TUNING ===
def lgb_objective(trial):
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'n_estimators': 3000, 'random_state': SEED, 'verbose': -1, 'n_jobs': -1,
    }
    scores = []
    for tr, val in folds:
        m = lgb.LGBMClassifier(**params)
        m.fit(train.iloc[tr], y.iloc[tr], eval_set=[(train.iloc[val], y.iloc[val])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p = m.predict_proba(train.iloc[val])[:, 1]
        scores.append(roc_auc_score(y.iloc[val], p))
    return np.mean(scores)

print('Tuning LightGBM...', file=sys.stderr, flush=True)
lgb_study = optuna.create_study(direction='maximize', study_name='lgb')
lgb_study.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f'LGB best: {lgb_study.best_value:.6f}', file=sys.stderr, flush=True)

# === XGBOOST TUNING ===
def xgb_objective(trial):
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'n_estimators': 3000, 'random_state': SEED,
        'tree_method': 'hist', 'n_jobs': -1, 'verbosity': 0,
        'early_stopping_rounds': 50,
    }
    scores = []
    for tr, val in folds:
        m = xgb.XGBClassifier(**params)
        m.fit(train.iloc[tr], y.iloc[tr], eval_set=[(train.iloc[val], y.iloc[val])], verbose=False)
        p = m.predict_proba(train.iloc[val])[:, 1]
        scores.append(roc_auc_score(y.iloc[val], p))
    return np.mean(scores)

print('Tuning XGBoost...', file=sys.stderr, flush=True)
xgb_study = optuna.create_study(direction='maximize', study_name='xgb')
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f'XGB best: {xgb_study.best_value:.6f}', file=sys.stderr, flush=True)

# === CATBOOST TUNING ===
def cat_objective(trial):
    params = {
        'iterations': 3000,
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2', 0.1, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data', 5, 100),
        'random_seed': SEED, 'verbose': 0, 'eval_metric': 'AUC',
    }
    scores = []
    for tr, val in folds:
        m = CatBoostClassifier(**params)
        m.fit(train.iloc[tr], y.iloc[tr], eval_set=(train.iloc[val], y.iloc[val]),
              early_stopping_rounds=50)
        p = m.predict_proba(train.iloc[val])[:, 1]
        scores.append(roc_auc_score(y.iloc[val], p))
    return np.mean(scores)

print('Tuning CatBoost...', file=sys.stderr, flush=True)
cat_study = optuna.create_study(direction='maximize', study_name='cat')
cat_study.optimize(cat_objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f'CAT best: {cat_study.best_value:.6f}', file=sys.stderr, flush=True)

# === FINAL TRAINING WITH BEST PARAMS (5-fold) ===
print('\nFinal training with best params (5-fold)...', file=sys.stderr, flush=True)
skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_l = np.zeros(len(train)); oof_x = np.zeros(len(train)); oof_c = np.zeros(len(train))
tp_l = np.zeros(len(test)); tp_x = np.zeros(len(test)); tp_c = np.zeros(len(test))

best_lgb = lgb_study.best_params
best_xgb = xgb_study.best_params
best_cat = cat_study.best_params

for fold, (tr, val) in enumerate(skf5.split(train, y)):
    X_tr, X_val = train.iloc[tr], train.iloc[val]
    y_tr, y_val = y.iloc[tr], y.iloc[val]

    # LGB
    m = lgb.LGBMClassifier(objective='binary', metric='auc', boosting_type='gbdt',
        learning_rate=best_lgb['lr'], num_leaves=best_lgb['num_leaves'],
        max_depth=best_lgb['max_depth'], min_child_samples=best_lgb['min_child_samples'],
        subsample=best_lgb['subsample'], colsample_bytree=best_lgb['colsample_bytree'],
        reg_alpha=best_lgb['reg_alpha'], reg_lambda=best_lgb['reg_lambda'],
        n_estimators=5000, random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof_l[val] = m.predict_proba(X_val)[:, 1]
    tp_l += m.predict_proba(test)[:, 1] / 5

    # XGB
    m = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc',
        learning_rate=best_xgb['lr'], max_depth=best_xgb['max_depth'],
        min_child_weight=best_xgb['min_child_weight'],
        subsample=best_xgb['subsample'], colsample_bytree=best_xgb['colsample_bytree'],
        reg_alpha=best_xgb['reg_alpha'], reg_lambda=best_xgb['reg_lambda'],
        n_estimators=5000, random_state=SEED, tree_method='hist',
        n_jobs=-1, verbosity=0, early_stopping_rounds=100)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_x[val] = m.predict_proba(X_val)[:, 1]
    tp_x += m.predict_proba(test)[:, 1] / 5

    # CatBoost
    m = CatBoostClassifier(iterations=5000, learning_rate=best_cat['lr'],
        depth=best_cat['depth'], l2_leaf_reg=best_cat['l2'],
        min_data_in_leaf=best_cat['min_data'],
        random_seed=SEED, verbose=0, eval_metric='AUC')
    m.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
    oof_c[val] = m.predict_proba(X_val)[:, 1]
    tp_c += m.predict_proba(test)[:, 1] / 5

    al = roc_auc_score(y_val, oof_l[val])
    ax = roc_auc_score(y_val, oof_x[val])
    ac = roc_auc_score(y_val, oof_c[val])
    print(f'F{fold+1} L:{al:.6f} X:{ax:.6f} C:{ac:.6f}', file=sys.stderr, flush=True)

# Results
al_ = roc_auc_score(y, oof_l)
ax_ = roc_auc_score(y, oof_x)
ac_ = roc_auc_score(y, oof_c)
ae_ = roc_auc_score(y, (oof_l+oof_x+oof_c)/3)

# Optimize weights
best_a, best_w = 0, (1/3,1/3,1/3)
for w1 in np.arange(0.0, 1.01, 0.05):
    for w2 in np.arange(0.0, 1.01-w1, 0.05):
        w3 = 1-w1-w2
        if w3 < 0: continue
        a = roc_auc_score(y, w1*oof_l + w2*oof_x + w3*oof_c)
        if a > best_a: best_a, best_w = a, (w1, w2, w3)

print(f'\nOOF L:{al_:.6f} X:{ax_:.6f} C:{ac_:.6f} E:{ae_:.6f}', file=sys.stderr, flush=True)
print(f'Optimized w=({best_w[0]:.2f},{best_w[1]:.2f},{best_w[2]:.2f}) AUC={best_a:.6f}', file=sys.stderr, flush=True)

tp_final = best_w[0]*tp_l + best_w[1]*tp_x + best_w[2]*tp_c
os.makedirs('submissions', exist_ok=True)
sub['Churn'] = tp_final
sub.to_csv(f'submissions/{VERSION}.csv', index=False)

results = {
    'cv_lgb': round(al_, 6), 'cv_xgb': round(ax_, 6), 'cv_cat': round(ac_, 6),
    'cv_ens': round(ae_, 6), 'cv_opt': round(best_a, 6),
    'weights': {'l': round(best_w[0],2), 'x': round(best_w[1],2), 'c': round(best_w[2],2)},
    'best_lgb_params': best_lgb, 'best_xgb_params': best_xgb, 'best_cat_params': best_cat,
    'feats': train.shape[1], 'n_trials': N_TRIALS,
}
json.dump(results, open(f'submissions/{VERSION}_results.json', 'w'), indent=2)

print(f'{best_a:.6f}')
print('DONE!', file=sys.stderr, flush=True)
