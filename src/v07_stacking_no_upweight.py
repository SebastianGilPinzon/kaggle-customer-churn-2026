"""
v07: 10-Model Stacking SIN upweight — fix de v06.
Usa features v04 (134) + 10 modelos diversos + LogReg meta-learner.
NO upweight de datos originales (causó drop de 0.91426 a 0.913).
"""
import pandas as pd, numpy as np, sys, os, json, warnings
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from scipy.stats import rankdata
warnings.filterwarnings('ignore')

VERSION = "v07_stacking_noup"
SEED = 42
N_FOLDS = 5

# === DATA (same as v04 — NO upweight) ===
print('Loading...', file=sys.stderr, flush=True)
train = pd.read_csv('data-kaggle/train.csv')
test = pd.read_csv('data-kaggle/test.csv')
sub = pd.read_csv('data-kaggle/sample_submission.csv')

# Merge original WITHOUT upweight
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

# === FEATURE ENGINEERING (v04 exhaustive) ===
for df in [train, test]:
    if df['TotalCharges'].dtype==object:
        df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
    df['AvgSpend']=df['TotalCharges']/(df['tenure']+1)
    df['TenProxy']=df['TotalCharges']/(df['MonthlyCharges']+.01)
    df['ChTen']=df['MonthlyCharges']*df['tenure']
    df['ChTenLog']=np.log1p(df['MonthlyCharges']*df['tenure'])
    df['ChargeRatio']=df['MonthlyCharges']/(df['TotalCharges']+.01)
    df['TenureSq']=df['tenure']**2
    df['MonthlySq']=df['MonthlyCharges']**2
    df['TenureLog']=np.log1p(df['tenure'])
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
    df['nNoInet']=sum((df[c].astype(str)=='No internet service').astype(int) for c in inet)
    df['nNo']=sum((df[c].astype(str)=='No').astype(int) for c in svc)
    df['CostPerService']=df['MonthlyCharges']/(df['nSvc']+1)
    df['CostPerInetService']=df['MonthlyCharges']/(df['nInet']+1)
    df['Fiber']=(df['InternetService'].astype(str)=='Fiber optic').astype(int)
    df['DSL']=(df['InternetService'].astype(str)=='DSL').astype(int)
    df['HasInet']=(df['InternetService'].astype(str)!='No').astype(int)
    df['MTM']=(df['Contract'].astype(str)=='Month-to-month').astype(int)
    df['OneYr']=(df['Contract'].astype(str)=='One year').astype(int)
    df['TwoYr']=(df['Contract'].astype(str)=='Two year').astype(int)
    df['EChk']=(df['PaymentMethod'].astype(str)=='Electronic check').astype(int)
    df['AutoPay']=df['PaymentMethod'].astype(str).str.contains('automatic',case=False).astype(int)
    for flag in ['MTM','Fiber','EChk','AutoPay','DSL','HasInet']:
        if flag in df.columns:
            df[f'{flag}_ten']=df[flag]*df['tenure']
            df[f'{flag}_mch']=df[flag]*df['MonthlyCharges']
            df[f'{flag}_tch']=df[flag]*df['TotalCharges']
    df['EC_MTM']=df['EChk']*df['MTM']
    df['Fib_MTM']=df['Fiber']*df['MTM']
    df['Sr_MTM']=df['SeniorCitizen']*df['MTM']
    df['Sr_Fib']=df['SeniorCitizen']*df['Fiber']
    df['Sec_Fib']=df['nSec']*df['Fiber']
    df['Auto_TwoYr']=df['AutoPay']*df['TwoYr']
    df['FibNoSec']=df['Fiber']*(df['nSec']==0).astype(int)
    df['MTM_HighCharge']=df['MTM']*(df['MonthlyCharges']>70).astype(int)
    df['New_MTM']=(df['tenure']<12).astype(int)*df['MTM']
    df['Sr_Fib_EC']=df['SeniorCitizen']*df['Fiber']*df['EChk']
    df['risk_score']=(df['tenure']<12).astype(int)+df['MTM']+df['Fiber']+df['EChk']+(df['nSec']==0).astype(int)
    df['TenBin']=pd.cut(df['tenure'],bins=[-1,3,6,12,24,48,72,999],labels=[0,1,2,3,4,5,6]).astype(float)
    df['ChBin']=pd.cut(df['MonthlyCharges'],bins=[-1,20,35,50,65,80,95,999],labels=[0,1,2,3,4,5,6]).astype(float)
    df['TotalBin']=pd.cut(df['TotalCharges'],bins=[-1,500,1000,2000,4000,6000,999999],labels=[0,1,2,3,4,5]).astype(float)

# Groupby aggs
group_cols=['Contract','InternetService','PaymentMethod']
agg_cols=['MonthlyCharges','TotalCharges','tenure']
for gc in group_cols:
    gcs=train[gc].astype(str); gct=test[gc].astype(str)
    for ac in agg_cols:
        grp=pd.concat([gcs,train[ac]],axis=1).groupby(gc)[ac]
        means=grp.mean(); stds=grp.std().fillna(0)
        train[f'{gc}_{ac}_mean']=gcs.map(means)
        train[f'{gc}_{ac}_std']=gcs.map(stds)
        train[f'{gc}_{ac}_diff']=train[ac]-train[f'{gc}_{ac}_mean']
        test[f'{gc}_{ac}_mean']=gct.map(means).fillna(train[ac].mean())
        test[f'{gc}_{ac}_std']=gct.map(stds).fillna(0)
        test[f'{gc}_{ac}_diff']=test[ac]-test[f'{gc}_{ac}_mean']

for g1,g2 in [('Contract','InternetService'),('Contract','PaymentMethod'),('InternetService','PaymentMethod')]:
    key_tr=train[g1].astype(str)+'_'+train[g2].astype(str)
    key_te=test[g1].astype(str)+'_'+test[g2].astype(str)
    for ac in ['MonthlyCharges','tenure']:
        grp=pd.DataFrame({'key':key_tr,ac:train[ac]}).groupby('key')[ac]
        means=grp.mean()
        train[f'{g1}_{g2}_{ac}_mean']=key_tr.map(means)
        test[f'{g1}_{g2}_{ac}_mean']=key_te.map(means).fillna(train[ac].mean())
        train[f'{g1}_{g2}_{ac}_diff']=train[ac]-train[f'{g1}_{g2}_{ac}_mean']
        test[f'{g1}_{g2}_{ac}_diff']=test[ac]-test[f'{g1}_{g2}_{ac}_mean']

# Freq encoding + label encode
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

# === 10 DIVERSE MODELS ===
models = {
    'lgb_v1': lambda s: lgb.LGBMClassifier(objective='binary',metric='auc',learning_rate=0.05,num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=1.0,n_estimators=3000,random_state=s,verbose=-1,n_jobs=-1),
    'lgb_v2': lambda s: lgb.LGBMClassifier(objective='binary',metric='auc',learning_rate=0.03,num_leaves=127,min_child_samples=50,subsample=0.7,colsample_bytree=0.5,reg_alpha=0.5,reg_lambda=0.1,n_estimators=3000,random_state=s,verbose=-1,n_jobs=-1),
    'lgb_v3': lambda s: lgb.LGBMClassifier(objective='binary',metric='auc',learning_rate=0.02,num_leaves=63,min_child_samples=100,subsample=0.6,colsample_bytree=0.6,reg_alpha=1.0,reg_lambda=5.0,n_estimators=3000,random_state=s,verbose=-1,n_jobs=-1),
    'xgb_v1': lambda s: xgb.XGBClassifier(objective='binary:logistic',eval_metric='auc',learning_rate=0.05,max_depth=6,min_child_weight=5,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=1.0,n_estimators=3000,random_state=s,tree_method='hist',n_jobs=-1,verbosity=0,early_stopping_rounds=50),
    'xgb_v2': lambda s: xgb.XGBClassifier(objective='binary:logistic',eval_metric='auc',learning_rate=0.03,max_depth=8,min_child_weight=10,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=2.0,n_estimators=3000,random_state=s,tree_method='hist',n_jobs=-1,verbosity=0,early_stopping_rounds=50),
    'cat_v1': lambda s: CatBoostClassifier(iterations=2000,learning_rate=0.05,depth=6,l2_leaf_reg=3,random_seed=s,verbose=0,eval_metric='AUC'),
    'cat_v2': lambda s: CatBoostClassifier(iterations=2000,learning_rate=0.03,depth=8,l2_leaf_reg=5,random_seed=s,verbose=0,eval_metric='AUC'),
    'hgb': lambda s: HistGradientBoostingClassifier(max_iter=1000,max_leaf_nodes=63,learning_rate=0.05,min_samples_leaf=50,random_state=s),
    'et': lambda s: ExtraTreesClassifier(n_estimators=500,max_depth=20,min_samples_leaf=50,random_state=s,n_jobs=-1),
    'ridge': lambda s: LogisticRegression(C=1.0,solver='lbfgs',max_iter=500,random_state=s),
}

# === LEVEL 1 ===
print(f'\n=== LEVEL 1: {len(models)} models ===', file=sys.stderr, flush=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(train, y))

oof_preds = {}
test_preds = {}

scaler = StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index=train.index)
test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)

for name, model_fn in models.items():
    print(f'  {name}...', file=sys.stderr, flush=True)
    oof = np.zeros(len(train))
    tp = np.zeros(len(test))
    use_scaled = name == 'ridge'
    tr_data = train_scaled if use_scaled else train
    te_data = test_scaled if use_scaled else test

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr, X_val = tr_data.iloc[tr_idx], tr_data.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        m = model_fn(SEED)

        if 'lgb' in name:
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        elif 'xgb' in name:
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        elif 'cat' in name:
            m.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
        else:
            m.fit(X_tr, y_tr)

        if hasattr(m, 'predict_proba'):
            oof[val_idx] = m.predict_proba(X_val)[:, 1]
            tp += m.predict_proba(te_data)[:, 1] / N_FOLDS
        else:
            oof[val_idx] = m.decision_function(X_val)
            tp += m.decision_function(te_data) / N_FOLDS

    auc = roc_auc_score(y, oof)
    oof_preds[name] = oof
    test_preds[name] = tp
    print(f'  {name}: OOF AUC = {auc:.6f}', file=sys.stderr, flush=True)

# === Averages ===
oof_avg = np.mean(list(oof_preds.values()), axis=0)
auc_avg = roc_auc_score(y, oof_avg)
print(f'\nSimple avg OOF: {auc_avg:.6f}', file=sys.stderr, flush=True)

rank_avg_oof = np.mean([rankdata(p)/len(p) for p in oof_preds.values()], axis=0)
auc_rank = roc_auc_score(y, rank_avg_oof)
print(f'Rank avg OOF: {auc_rank:.6f}', file=sys.stderr, flush=True)

# === LEVEL 2: Meta-learner ===
print('\n=== LEVEL 2: Stacking ===', file=sys.stderr, flush=True)
stack_train = np.column_stack(list(oof_preds.values()))
stack_test = np.column_stack(list(test_preds.values()))

meta_oof = np.zeros(len(train))
meta_test = np.zeros(len(test))
for tr_idx, val_idx in folds:
    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(stack_train[tr_idx], y.iloc[tr_idx])
    meta_oof[val_idx] = meta.predict_proba(stack_train[val_idx])[:, 1]
    meta_test += meta.predict_proba(stack_test)[:, 1] / N_FOLDS
auc_stack = roc_auc_score(y, meta_oof)
print(f'Stacking OOF: {auc_stack:.6f}', file=sys.stderr, flush=True)

# === Best blend ===
best_a, best_w = 0, 1.0
for w in np.arange(0, 1.01, 0.05):
    blend = w * meta_oof + (1-w) * oof_avg
    a = roc_auc_score(y, blend)
    if a > best_a: best_a, best_w = a, w
print(f'Best blend: {best_w:.2f}*stack + {1-best_w:.2f}*avg = {best_a:.6f}', file=sys.stderr, flush=True)

final = best_w * meta_test + (1-best_w) * np.mean(list(test_preds.values()), axis=0)

# Also try: just top 3 GBDT weighted average (like v04)
gbdt_names = ['lgb_v1','lgb_v2','lgb_v3','xgb_v1','xgb_v2','cat_v1','cat_v2']
gbdt_oofs = [oof_preds[n] for n in gbdt_names]
gbdt_tests = [test_preds[n] for n in gbdt_names]
gbdt_avg_oof = np.mean(gbdt_oofs, axis=0)
gbdt_avg_test = np.mean(gbdt_tests, axis=0)
auc_gbdt = roc_auc_score(y, gbdt_avg_oof)
print(f'GBDT-only avg OOF: {auc_gbdt:.6f}', file=sys.stderr, flush=True)

# Save all versions
os.makedirs('submissions', exist_ok=True)
sub['Churn'] = final
sub.to_csv(f'submissions/{VERSION}.csv', index=False)
sub2 = sub.copy(); sub2['Churn'] = gbdt_avg_test
sub2.to_csv(f'submissions/{VERSION}_gbdt.csv', index=False)
sub3 = sub.copy(); sub3['Churn'] = np.mean(list(test_preds.values()), axis=0)
sub3.to_csv(f'submissions/{VERSION}_allavg.csv', index=False)

results = {
    'model_aucs': {k: round(roc_auc_score(y, v), 6) for k, v in oof_preds.items()},
    'simple_avg': round(auc_avg, 6), 'rank_avg': round(auc_rank, 6),
    'stacking': round(auc_stack, 6), 'best_blend': round(best_a, 6),
    'gbdt_avg': round(auc_gbdt, 6), 'blend_w': round(best_w, 2),
    'feats': train.shape[1],
}
json.dump(results, open(f'submissions/{VERSION}_results.json', 'w'), indent=2)
print(f'\nSaved submissions/{VERSION}.csv', file=sys.stderr, flush=True)
print('DONE!', file=sys.stderr, flush=True)
