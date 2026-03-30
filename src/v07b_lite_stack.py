"""v07b: Lighter stacking — 7 GBDT models (no ET/Ridge/HGB to save memory)."""
import pandas as pd, numpy as np, sys, os, json, warnings
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
warnings.filterwarnings('ignore')

VERSION = "v07b_lite_stack"
SEED = 42
N_FOLDS = 5

# === DATA ===
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

# === FE (v04) ===
for df in [train, test]:
    if df['TotalCharges'].dtype==object:
        df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
    df['AvgSpend']=df['TotalCharges']/(df['tenure']+1)
    df['TenProxy']=df['TotalCharges']/(df['MonthlyCharges']+.01)
    df['ChTen']=df['MonthlyCharges']*df['tenure']
    df['ChTenLog']=np.log1p(df['MonthlyCharges']*df['tenure'])
    df['ChargeRatio']=df['MonthlyCharges']/(df['TotalCharges']+.01)
    df['TenureSq']=df['tenure']**2
    df['MonthlyLog']=np.log1p(df['MonthlyCharges'])
    df['TotalLog']=np.log1p(df['TotalCharges'])
    df['ExpectedTotal']=df['tenure']*df['MonthlyCharges']
    df['TotalDiff']=df['TotalCharges']-df['ExpectedTotal']
    df['TotalDiffPct']=df['TotalDiff']/(df['ExpectedTotal']+.01)
    svc=['PhoneService','MultipleLines','InternetService','OnlineSecurity',
         'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['nSvc']=sum((df[c].astype(str)=='Yes').astype(int) for c in svc)
    inet=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['nInet']=sum((df[c].astype(str)=='Yes').astype(int) for c in inet)
    df['nSec']=sum((df[c].astype(str)=='Yes').astype(int) for c in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport'])
    df['nStream']=sum((df[c].astype(str)=='Yes').astype(int) for c in ['StreamingTV','StreamingMovies'])
    df['CostPerService']=df['MonthlyCharges']/(df['nSvc']+1)
    df['Fiber']=(df['InternetService'].astype(str)=='Fiber optic').astype(int)
    df['DSL']=(df['InternetService'].astype(str)=='DSL').astype(int)
    df['HasInet']=(df['InternetService'].astype(str)!='No').astype(int)
    df['MTM']=(df['Contract'].astype(str)=='Month-to-month').astype(int)
    df['EChk']=(df['PaymentMethod'].astype(str)=='Electronic check').astype(int)
    df['AutoPay']=df['PaymentMethod'].astype(str).str.contains('automatic',case=False).astype(int)
    df['t_x_m']=df['tenure']*df['MonthlyCharges']
    df['MTM_ten']=df['MTM']*df['tenure']
    df['Fib_ch']=df['Fiber']*df['MonthlyCharges']
    df['EC_MTM']=df['EChk']*df['MTM']
    df['Sec_Fib']=df['nSec']*df['Fiber']
    df['Sr_MTM']=df['SeniorCitizen']*df['MTM']
    df['FibNoSec']=df['Fiber']*(df['nSec']==0).astype(int)
    df['New_MTM']=(df['tenure']<12).astype(int)*df['MTM']
    df['risk_score']=(df['tenure']<12).astype(int)+df['MTM']+df['Fiber']+df['EChk']+(df['nSec']==0).astype(int)
    df['TenBin']=pd.cut(df['tenure'],bins=[-1,3,6,12,24,48,72,999],labels=[0,1,2,3,4,5,6]).astype(float)
    df['ChBin']=pd.cut(df['MonthlyCharges'],bins=[-1,20,35,50,65,80,95,999],labels=[0,1,2,3,4,5,6]).astype(float)

# Groupby aggs (only 1-way to save memory)
for gc in ['Contract','InternetService','PaymentMethod']:
    gcs=train[gc].astype(str); gct=test[gc].astype(str)
    for ac in ['MonthlyCharges','tenure']:
        means=pd.concat([gcs,train[ac]],axis=1).groupby(gc)[ac].mean()
        train[f'{gc}_{ac}_mean']=gcs.map(means)
        train[f'{gc}_{ac}_diff']=train[ac]-train[f'{gc}_{ac}_mean']
        test[f'{gc}_{ac}_mean']=gct.map(means).fillna(train[ac].mean())
        test[f'{gc}_{ac}_diff']=test[ac]-test[f'{gc}_{ac}_mean']

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

# === 7 GBDT models with diverse hyperparams ===
models = {
    'lgb_a': lambda s: lgb.LGBMClassifier(objective='binary',metric='auc',learning_rate=0.05,num_leaves=63,min_child_samples=20,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=0.5,n_estimators=2000,random_state=s,verbose=-1,n_jobs=-1),
    'lgb_b': lambda s: lgb.LGBMClassifier(objective='binary',metric='auc',learning_rate=0.03,num_leaves=31,min_child_samples=50,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=2.0,n_estimators=2000,random_state=s,verbose=-1,n_jobs=-1),
    'lgb_c': lambda s: lgb.LGBMClassifier(objective='binary',metric='auc',learning_rate=0.02,num_leaves=127,min_child_samples=100,subsample=0.6,colsample_bytree=0.5,reg_alpha=1.0,reg_lambda=5.0,n_estimators=3000,random_state=s,verbose=-1,n_jobs=-1),
    'xgb_a': lambda s: xgb.XGBClassifier(objective='binary:logistic',eval_metric='auc',learning_rate=0.05,max_depth=7,min_child_weight=5,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=1.0,n_estimators=2000,random_state=s,tree_method='hist',n_jobs=-1,verbosity=0,early_stopping_rounds=50),
    'xgb_b': lambda s: xgb.XGBClassifier(objective='binary:logistic',eval_metric='auc',learning_rate=0.03,max_depth=8,min_child_weight=10,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=2.0,n_estimators=2000,random_state=s,tree_method='hist',n_jobs=-1,verbosity=0,early_stopping_rounds=50),
    'cat_a': lambda s: CatBoostClassifier(iterations=1500,learning_rate=0.05,depth=7,l2_leaf_reg=3,random_seed=s,verbose=0,eval_metric='AUC'),
    'cat_b': lambda s: CatBoostClassifier(iterations=1500,learning_rate=0.03,depth=6,l2_leaf_reg=5,random_seed=s,verbose=0,eval_metric='AUC'),
}

# === LEVEL 1 ===
print(f'\n=== LEVEL 1: {len(models)} models ===', file=sys.stderr, flush=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(train, y))
oof_preds = {}; test_preds = {}

for name, model_fn in models.items():
    print(f'  {name}...', file=sys.stderr, flush=True)
    oof = np.zeros(len(train)); tp = np.zeros(len(test))
    for fi, (tr_idx, val_idx) in enumerate(folds):
        m = model_fn(SEED)
        if 'lgb' in name:
            m.fit(train.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(train.iloc[val_idx], y.iloc[val_idx])],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        elif 'xgb' in name:
            m.fit(train.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(train.iloc[val_idx], y.iloc[val_idx])], verbose=False)
        elif 'cat' in name:
            m.fit(train.iloc[tr_idx], y.iloc[tr_idx], eval_set=(train.iloc[val_idx], y.iloc[val_idx]), early_stopping_rounds=50)
        oof[val_idx] = m.predict_proba(train.iloc[val_idx])[:, 1]
        tp += m.predict_proba(test)[:, 1] / N_FOLDS
        del m  # Free memory
    auc = roc_auc_score(y, oof)
    oof_preds[name] = oof; test_preds[name] = tp
    print(f'  {name}: {auc:.6f}', file=sys.stderr, flush=True)

# === Averages ===
oof_avg = np.mean(list(oof_preds.values()), axis=0)
auc_avg = roc_auc_score(y, oof_avg)
print(f'\nAvg OOF: {auc_avg:.6f}', file=sys.stderr, flush=True)

# === LEVEL 2 ===
stack_train = np.column_stack(list(oof_preds.values()))
stack_test = np.column_stack(list(test_preds.values()))
meta_oof = np.zeros(len(train)); meta_test = np.zeros(len(test))
for tr_idx, val_idx in folds:
    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(stack_train[tr_idx], y.iloc[tr_idx])
    meta_oof[val_idx] = meta.predict_proba(stack_train[val_idx])[:, 1]
    meta_test += meta.predict_proba(stack_test)[:, 1] / N_FOLDS
auc_stack = roc_auc_score(y, meta_oof)
print(f'Stack OOF: {auc_stack:.6f}', file=sys.stderr, flush=True)

# Best of stack vs avg
best_a, best_w = 0, 0.5
for w in np.arange(0, 1.01, 0.05):
    a = roc_auc_score(y, w*meta_oof + (1-w)*oof_avg)
    if a > best_a: best_a, best_w = a, w
print(f'Best: {best_w:.2f}*stack + {1-best_w:.2f}*avg = {best_a:.6f}', file=sys.stderr, flush=True)

final = best_w * meta_test + (1-best_w) * np.mean(list(test_preds.values()), axis=0)

# Optimized weight among top models
best_opt, best_ow = 0, None
for w1 in np.arange(0, 0.51, 0.05):
    for w2 in np.arange(0, 0.51, 0.05):
        for w3 in np.arange(0, 0.51, 0.05):
            rem = 1.0 - w1 - w2 - w3
            if rem < 0 or rem > 0.5: continue
            # w1=lgb_a, w2=xgb_a, w3=cat_a, rem=rest avg
            blend = w1*oof_preds['lgb_a'] + w2*oof_preds['xgb_a'] + w3*oof_preds['cat_a'] + rem*oof_avg
            a = roc_auc_score(y, blend)
            if a > best_opt: best_opt, best_ow = a, (w1,w2,w3,rem)
if best_ow:
    print(f'Opt weights: lgb={best_ow[0]:.2f} xgb={best_ow[1]:.2f} cat={best_ow[2]:.2f} avg={best_ow[3]:.2f} AUC={best_opt:.6f}', file=sys.stderr, flush=True)

os.makedirs('submissions', exist_ok=True)
sub['Churn'] = final
sub.to_csv(f'submissions/{VERSION}.csv', index=False)
sub2 = sub.copy(); sub2['Churn'] = oof_avg  # wrong - should be test avg
sub2['Churn'] = np.mean(list(test_preds.values()), axis=0)
sub2.to_csv(f'submissions/{VERSION}_avg.csv', index=False)

results = {'model_aucs': {k: round(roc_auc_score(y,v),6) for k,v in oof_preds.items()},
           'avg': round(auc_avg,6), 'stack': round(auc_stack,6), 'best_blend': round(best_a,6),
           'feats': train.shape[1]}
json.dump(results, open(f'submissions/{VERSION}_results.json', 'w'), indent=2)
print(f'\nSaved. DONE!', file=sys.stderr, flush=True)
