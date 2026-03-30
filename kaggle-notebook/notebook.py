"""
v08: Multi-seed v04 on Kaggle (faster CPUs).
5 seeds × 5 folds × 3 models = 75 trainings.
"""
import pandas as pd, numpy as np, os, json, warnings, time
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

start = time.time()
VERSION = "v08_multiseed_v04"
N_FOLDS = 5
N_SEEDS = 5

# === DATA ===
print('Loading...')
train = pd.read_csv('/kaggle/input/playground-series-s6e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s6e3/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s6e3/sample_submission.csv')

# Original dataset
orig = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
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

print(f'Train: {train.shape}, Test: {test.shape}')

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

# Groupby aggregations
for gc in ['Contract','InternetService','PaymentMethod']:
    gcs=train[gc].astype(str); gct=test[gc].astype(str)
    for ac in ['MonthlyCharges','TotalCharges','tenure']:
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
        means=pd.DataFrame({'key':key_tr,ac:train[ac]}).groupby('key')[ac].mean()
        train[f'{g1}_{g2}_{ac}_mean']=key_tr.map(means)
        test[f'{g1}_{g2}_{ac}_mean']=key_te.map(means).fillna(train[ac].mean())
        train[f'{g1}_{g2}_{ac}_diff']=train[ac]-train[f'{g1}_{g2}_{ac}_mean']
        test[f'{g1}_{g2}_{ac}_diff']=test[ac]-test[f'{g1}_{g2}_{ac}_mean']

# Encode
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

print(f'Features: {train.shape[1]}')

# === MULTI-SEED TRAINING ===
seeds = [42, 1042, 2042, 3042, 4042]
all_test = []

for si, seed in enumerate(seeds):
    print(f'\nSeed {si+1}/{N_SEEDS}: {seed}')
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_l=np.zeros(len(train)); oof_x=np.zeros(len(train)); oof_c=np.zeros(len(train))
    tp_l=np.zeros(len(test)); tp_x=np.zeros(len(test)); tp_c=np.zeros(len(test))

    for fold,(tr,val) in enumerate(skf.split(train,y)):
        X_tr,X_val=train.iloc[tr],train.iloc[val]
        y_tr,y_val=y.iloc[tr],y.iloc[val]

        m=lgb.LGBMClassifier(objective='binary',metric='auc',learning_rate=0.05,
            num_leaves=63,min_child_samples=20,subsample=0.8,colsample_bytree=0.7,
            reg_alpha=0.1,reg_lambda=0.5,n_estimators=3000,random_state=seed,verbose=-1,n_jobs=-1)
        m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
        oof_l[val]=m.predict_proba(X_val)[:,1]; tp_l+=m.predict_proba(test)[:,1]/N_FOLDS
        del m

        m=xgb.XGBClassifier(objective='binary:logistic',eval_metric='auc',learning_rate=0.05,
            max_depth=7,min_child_weight=5,subsample=0.8,colsample_bytree=0.7,
            reg_alpha=0.1,reg_lambda=1.0,n_estimators=3000,random_state=seed,
            tree_method='hist',n_jobs=-1,verbosity=0,early_stopping_rounds=50)
        m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
        oof_x[val]=m.predict_proba(X_val)[:,1]; tp_x+=m.predict_proba(test)[:,1]/N_FOLDS
        del m

        m=CatBoostClassifier(iterations=2000,learning_rate=0.05,depth=7,l2_leaf_reg=3,
            random_seed=seed,verbose=0,eval_metric='AUC')
        m.fit(X_tr,y_tr,eval_set=(X_val,y_val),early_stopping_rounds=50)
        oof_c[val]=m.predict_proba(X_val)[:,1]; tp_c+=m.predict_proba(test)[:,1]/N_FOLDS
        del m

        al=roc_auc_score(y_val,oof_l[val])
        ax=roc_auc_score(y_val,oof_x[val])
        ac=roc_auc_score(y_val,oof_c[val])
        print(f'  F{fold+1} L:{al:.6f} X:{ax:.6f} C:{ac:.6f}')

    # v04 optimized weights
    tp_seed = 0.20*tp_l + 0.30*tp_x + 0.50*tp_c
    all_test.append(tp_seed)

    al=roc_auc_score(y,oof_l); ax=roc_auc_score(y,oof_x); ac=roc_auc_score(y,oof_c)
    ae=roc_auc_score(y,0.20*oof_l+0.30*oof_x+0.50*oof_c)
    print(f'Seed {seed} -> L:{al:.6f} X:{ax:.6f} C:{ac:.6f} Ens:{ae:.6f}')

# Average across seeds
final = np.mean(all_test, axis=0)
sub['Churn'] = final
sub.to_csv('submission.csv', index=False)

elapsed = time.time() - start
print(f'\nDone in {elapsed/60:.1f} min')
print(f'Range: [{final.min():.4f}, {final.max():.4f}], Mean: {final.mean():.4f}')
print(f'Saved: submission.csv')
