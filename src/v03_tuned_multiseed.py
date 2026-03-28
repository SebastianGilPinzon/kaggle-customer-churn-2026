"""v03: Hand-tuned params + multi-seed averaging (3 seeds) + target encoding."""
import pandas as pd, numpy as np, sys, os, json, warnings
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

VERSION = "v03_tuned_multiseed"
BASE_SEED = 42
N_SEEDS = 3
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

# === FE ===
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

# Target encoding with KFold (prevents leakage)
cats = train.select_dtypes(include=['object','category']).columns.tolist()
global_mean = y.mean()

for col in cats:
    te_train = pd.Series(np.nan, index=train.index)
    skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=BASE_SEED)
    for tr_idx, val_idx in skf_te.split(train, y):
        means = y.iloc[tr_idx].groupby(train[col].iloc[tr_idx].astype(str)).agg(['mean','count'])
        smooth = (means['mean']*means['count'] + global_mean*10) / (means['count']+10)
        te_train.iloc[val_idx] = train[col].iloc[val_idx].astype(str).map(smooth)
    train[f'{col}_te'] = te_train.fillna(global_mean)

    means_full = y.groupby(train[col].astype(str)).agg(['mean','count'])
    smooth_full = (means_full['mean']*means_full['count'] + global_mean*10) / (means_full['count']+10)
    test[f'{col}_te'] = test[col].astype(str).map(smooth_full).fillna(global_mean)

# Freq encoding + label encode
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

# === TUNED PARAMS (from similar PS competitions) ===
lgb_p = dict(objective='binary', metric='auc', boosting_type='gbdt',
    learning_rate=0.03, num_leaves=48, max_depth=8, min_child_samples=30,
    subsample=0.75, colsample_bytree=0.65, reg_alpha=0.3, reg_lambda=1.5,
    n_estimators=5000, verbose=-1, n_jobs=-1)

xgb_p = dict(objective='binary:logistic', eval_metric='auc',
    learning_rate=0.03, max_depth=8, min_child_weight=10,
    subsample=0.75, colsample_bytree=0.65, reg_alpha=0.3, reg_lambda=2.0,
    n_estimators=5000, tree_method='hist', n_jobs=-1, verbosity=0,
    early_stopping_rounds=100)

cat_p = dict(iterations=5000, learning_rate=0.03, depth=8,
    l2_leaf_reg=5, min_data_in_leaf=30, verbose=0, eval_metric='AUC')

# === MULTI-SEED TRAINING ===
seeds = [BASE_SEED, BASE_SEED+1000, BASE_SEED+2000]
all_test = []

for si, seed in enumerate(seeds):
    print(f'\nSeed {si+1}/{N_SEEDS}: {seed}', file=sys.stderr, flush=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_l=np.zeros(len(train)); oof_x=np.zeros(len(train)); oof_c=np.zeros(len(train))
    tp_l=np.zeros(len(test)); tp_x=np.zeros(len(test)); tp_c=np.zeros(len(test))

    for fold,(tr,val) in enumerate(skf.split(train,y)):
        X_tr,X_val = train.iloc[tr], train.iloc[val]
        y_tr,y_val = y.iloc[tr], y.iloc[val]

        m=lgb.LGBMClassifier(**lgb_p, random_state=seed)
        m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],
              callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
        oof_l[val]=m.predict_proba(X_val)[:,1]; tp_l+=m.predict_proba(test)[:,1]/N_FOLDS

        m=xgb.XGBClassifier(**xgb_p, random_state=seed)
        m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
        oof_x[val]=m.predict_proba(X_val)[:,1]; tp_x+=m.predict_proba(test)[:,1]/N_FOLDS

        m=CatBoostClassifier(**cat_p, random_seed=seed)
        m.fit(X_tr,y_tr,eval_set=(X_val,y_val),early_stopping_rounds=100)
        oof_c[val]=m.predict_proba(X_val)[:,1]; tp_c+=m.predict_proba(test)[:,1]/N_FOLDS

        print(f'  F{fold+1} L:{roc_auc_score(y_val,oof_l[val]):.6f} '
              f'X:{roc_auc_score(y_val,oof_x[val]):.6f} '
              f'C:{roc_auc_score(y_val,oof_c[val]):.6f}', file=sys.stderr, flush=True)

    al=roc_auc_score(y,oof_l); ax=roc_auc_score(y,oof_x); ac=roc_auc_score(y,oof_c)
    ae=roc_auc_score(y,(oof_l+oof_x+oof_c)/3)
    print(f'Seed {seed} -> L:{al:.6f} X:{ax:.6f} C:{ac:.6f} E:{ae:.6f}', file=sys.stderr, flush=True)

    # Optimized blend for this seed
    best_a,best_w=0,(1/3,1/3,1/3)
    for w1 in np.arange(0,1.01,.1):
        for w2 in np.arange(0,1.01-w1,.1):
            w3=1-w1-w2
            if w3<0: continue
            a=roc_auc_score(y,w1*oof_l+w2*oof_x+w3*oof_c)
            if a>best_a: best_a,best_w=a,(w1,w2,w3)

    tp_seed = best_w[0]*tp_l + best_w[1]*tp_x + best_w[2]*tp_c
    all_test.append(tp_seed)
    print(f'Seed {seed} opt: w=({best_w[0]:.1f},{best_w[1]:.1f},{best_w[2]:.1f}) AUC={best_a:.6f}', file=sys.stderr, flush=True)

# Average across seeds
tp_final = np.mean(all_test, axis=0)

os.makedirs('submissions', exist_ok=True)
sub['Churn'] = tp_final
sub.to_csv(f'submissions/{VERSION}.csv', index=False)
json.dump({'version': VERSION, 'n_seeds': N_SEEDS, 'feats': train.shape[1]},
          open(f'submissions/{VERSION}_results.json', 'w'), indent=2)

print(f'\nSaved: submissions/{VERSION}.csv', file=sys.stderr, flush=True)
print(f'Range: [{tp_final.min():.4f}, {tp_final.max():.4f}], Mean: {tp_final.mean():.4f}', file=sys.stderr, flush=True)
print('DONE!', file=sys.stderr, flush=True)
