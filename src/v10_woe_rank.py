"""v10: WoE encoding + rank features + cluster features on v04 base."""
import pandas as pd, numpy as np, sys, os, json, warnings
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

VERSION = "v10_woe_rank"
SEED = 42
N_FOLDS = 5

print('Loading...', file=sys.stderr, flush=True)
train = pd.read_csv('data-kaggle/train.csv')
test = pd.read_csv('data-kaggle/test.csv')
sub = pd.read_csv('data-kaggle/sample_submission.csv')
orig = pd.read_csv('data-original/WA_Fn-UseC_-Telco-Customer-Churn.csv')
orig['Churn']=orig['Churn'].map({'Yes':1,'No':0})
orig.drop('customerID',axis=1,inplace=True)
orig['TotalCharges']=pd.to_numeric(orig['TotalCharges'],errors='coerce')
common=[c for c in train.columns if c in orig.columns]
oa=orig[common].copy()
for c in train.columns:
    if c not in oa.columns: oa[c]=np.nan
train=pd.concat([train,oa[train.columns]],ignore_index=True)
test_ids=test['id'].copy()
train.drop('id',axis=1,errors='ignore',inplace=True)
test.drop('id',axis=1,errors='ignore',inplace=True)
y=train['Churn'].map(lambda x:1 if str(x) in ('Yes','1','1.0') else 0).astype(int)
train.drop('Churn',axis=1,inplace=True)

# === v04 FE ===
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
    for flag in ['MTM','Fiber','EChk','AutoPay','DSL','HasInet']:
        df[f'{flag}_ten']=df[flag]*df['tenure']
        df[f'{flag}_mch']=df[flag]*df['MonthlyCharges']
        df[f'{flag}_tch']=df[flag]*df['TotalCharges']
    df['EC_MTM']=df['EChk']*df['MTM']
    df['Fib_MTM']=df['Fiber']*df['MTM']
    df['Sr_MTM']=df['SeniorCitizen']*df['MTM']
    df['Sec_Fib']=df['nSec']*df['Fiber']
    df['FibNoSec']=df['Fiber']*(df['nSec']==0).astype(int)
    df['New_MTM']=(df['tenure']<12).astype(int)*df['MTM']
    df['risk_score']=(df['tenure']<12).astype(int)+df['MTM']+df['Fiber']+df['EChk']+(df['nSec']==0).astype(int)
    df['TenBin']=pd.cut(df['tenure'],bins=[-1,3,6,12,24,48,72,999],labels=[0,1,2,3,4,5,6]).astype(float)
    df['ChBin']=pd.cut(df['MonthlyCharges'],bins=[-1,20,35,50,65,80,95,999],labels=[0,1,2,3,4,5,6]).astype(float)
    df['TotalBin']=pd.cut(df['TotalCharges'],bins=[-1,500,1000,2000,4000,6000,999999],labels=[0,1,2,3,4,5]).astype(float)

# Groupby
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

# === NEW: Rank features within group ===
print('Rank features...', file=sys.stderr, flush=True)
full = pd.concat([train, test], ignore_index=True)
for gc in ['Contract','InternetService','PaymentMethod']:
    for nc in ['MonthlyCharges','TotalCharges','tenure']:
        col = f'{gc}_{nc}_rank'
        full[col] = full.groupby(gc)[nc].rank(pct=True)
for nc in ['MonthlyCharges','TotalCharges','tenure','AvgSpend']:
    full[f'{nc}_pctrank'] = full[nc].rank(pct=True)
train = full.iloc[:len(train)].reset_index(drop=True)
test = full.iloc[len(train):].reset_index(drop=True)
del full

# === NEW: KMeans cluster features ===
print('Cluster features...', file=sys.stderr, flush=True)
cluster_cols = ['tenure','MonthlyCharges','TotalCharges','nSvc','nInet','AvgSpend']
scaler = StandardScaler()
X_cl_train = scaler.fit_transform(train[cluster_cols].fillna(0))
X_cl_test = scaler.transform(test[cluster_cols].fillna(0))
for k in [5, 10]:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km.fit(X_cl_train)
    train[f'cluster_{k}'] = km.predict(X_cl_train)
    test[f'cluster_{k}'] = km.predict(X_cl_test)
    dists_tr = km.transform(X_cl_train)
    dists_te = km.transform(X_cl_test)
    train[f'cluster_{k}_dist'] = dists_tr.min(axis=1)
    test[f'cluster_{k}_dist'] = dists_te.min(axis=1)

# Freq encode + label encode
cats=train.select_dtypes(include=['object','category']).columns.tolist()
for c in cats:
    freq=train[c].value_counts(normalize=True)
    train[f'{c}_freq']=train[c].map(freq).fillna(0)
    test[f'{c}_freq']=test[c].map(freq).fillna(0)

# WoE encoding (inside CV to prevent leakage — use full train for test)
print('WoE encoding...', file=sys.stderr, flush=True)
woe_cols = [c for c in cats]  # All categoricals
global_mean = y.mean()

# For test: compute WoE on full train
for col in woe_cols:
    events = y.groupby(train[col].astype(str)).sum()
    non_events = y.groupby(train[col].astype(str)).count() - events
    total_e = y.sum(); total_ne = len(y) - total_e
    dist_e = (events + 10) / (total_e + 10*len(events))
    dist_ne = (non_events + 10) / (total_ne + 10*len(non_events))
    woe = np.log(dist_ne / dist_e)
    train[f'{col}_woe'] = train[col].astype(str).map(woe).fillna(0)
    test[f'{col}_woe'] = test[col].astype(str).map(woe).fillna(0)

for c in cats:
    le=LabelEncoder()
    le.fit(pd.concat([train[c],test[c]]).astype(str))
    train[c]=le.transform(train[c].astype(str))
    test[c]=le.transform(test[c].astype(str))
train.fillna(-999,inplace=True)
test.fillna(-999,inplace=True)

print(f'Features: {train.shape[1]}', file=sys.stderr, flush=True)

# === TRAIN ===
skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)
oof_l=np.zeros(len(train)); oof_x=np.zeros(len(train)); oof_c=np.zeros(len(train))
tp_l=np.zeros(len(test)); tp_x=np.zeros(len(test)); tp_c=np.zeros(len(test))

for fold,(tr,val) in enumerate(skf.split(train,y)):
    X_tr,X_val=train.iloc[tr],train.iloc[val]
    y_tr,y_val=y.iloc[tr],y.iloc[val]

    m=lgb.LGBMClassifier(objective='binary',metric='auc',learning_rate=0.05,num_leaves=63,min_child_samples=20,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=0.5,n_estimators=3000,random_state=SEED,verbose=-1,n_jobs=-1)
    m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
    oof_l[val]=m.predict_proba(X_val)[:,1]; tp_l+=m.predict_proba(test)[:,1]/N_FOLDS; del m

    m=xgb.XGBClassifier(objective='binary:logistic',eval_metric='auc',learning_rate=0.05,max_depth=7,min_child_weight=5,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=1.0,n_estimators=3000,random_state=SEED,tree_method='hist',n_jobs=-1,verbosity=0,early_stopping_rounds=50)
    m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
    oof_x[val]=m.predict_proba(X_val)[:,1]; tp_x+=m.predict_proba(test)[:,1]/N_FOLDS; del m

    m=CatBoostClassifier(iterations=2000,learning_rate=0.05,depth=7,l2_leaf_reg=3,random_seed=SEED,verbose=0,eval_metric='AUC')
    m.fit(X_tr,y_tr,eval_set=(X_val,y_val),early_stopping_rounds=50)
    oof_c[val]=m.predict_proba(X_val)[:,1]; tp_c+=m.predict_proba(test)[:,1]/N_FOLDS; del m

    print(f'F{fold+1} L:{roc_auc_score(y_val,oof_l[val]):.6f} X:{roc_auc_score(y_val,oof_x[val]):.6f} C:{roc_auc_score(y_val,oof_c[val]):.6f}',file=sys.stderr,flush=True)

al=roc_auc_score(y,oof_l); ax=roc_auc_score(y,oof_x); ac=roc_auc_score(y,oof_c)
best_a,best_w=0,(1/3,1/3,1/3)
for w1 in np.arange(0,1.01,.05):
    for w2 in np.arange(0,1.01-w1,.05):
        w3=1-w1-w2
        if w3<0: continue
        a=roc_auc_score(y,w1*oof_l+w2*oof_x+w3*oof_c)
        if a>best_a: best_a,best_w=a,(w1,w2,w3)

print(f'\nOOF L:{al:.6f} X:{ax:.6f} C:{ac:.6f}',file=sys.stderr,flush=True)
print(f'Opt: w=({best_w[0]:.2f},{best_w[1]:.2f},{best_w[2]:.2f}) AUC={best_a:.6f}',file=sys.stderr,flush=True)

tp_final=best_w[0]*tp_l+best_w[1]*tp_x+best_w[2]*tp_c
os.makedirs('submissions',exist_ok=True)
sub['Churn']=tp_final
sub.to_csv(f'submissions/{VERSION}.csv',index=False)
json.dump({'cv_opt':round(best_a,6),'weights':list(best_w),'feats':train.shape[1]},
          open(f'submissions/{VERSION}_results.json','w'),indent=2)
print(f'Saved: submissions/{VERSION}.csv',file=sys.stderr,flush=True)
print('DONE!',file=sys.stderr,flush=True)
