"""v14: PARADIGM SHIFT - 4 feature sets x 3 models = 12 base + hill climbing."""
import pandas as pd, numpy as np, sys, os, json, warnings
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')

VERSION = "v14_paradigm_shift"
SEED = 42
N_FOLDS = 5

print('Loading...', file=sys.stderr, flush=True)
train_raw = pd.read_csv('data-kaggle/train.csv')
test_raw = pd.read_csv('data-kaggle/test.csv')
sub = pd.read_csv('data-kaggle/sample_submission.csv')
test_ids = test_raw['id'].copy()
train_raw.drop('id', axis=1, errors='ignore', inplace=True)
test_raw.drop('id', axis=1, errors='ignore', inplace=True)
y = train_raw['Churn'].map(lambda x: 1 if str(x) in ('Yes','1','1.0') else 0).astype(int)
train_raw.drop('Churn', axis=1, inplace=True)

for df in [train_raw, test_raw]:
    if df['TotalCharges'].dtype == object:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

cats_raw = train_raw.select_dtypes(include=['object','category']).columns.tolist()

def enc(tr, te):
    tr = tr.copy(); te = te.copy()
    for c in cats_raw:
        if c in tr.columns and tr[c].dtype == object:
            le = LabelEncoder()
            le.fit(pd.concat([tr[c], te[c]]).astype(str))
            tr[c] = le.transform(tr[c].astype(str))
            te[c] = le.transform(te[c].astype(str))
    tr.fillna(-999, inplace=True); te.fillna(-999, inplace=True)
    return tr, te

# SET A: Minimal
def set_a(tr, te):
    tr=tr.copy(); te=te.copy()
    for df in [tr,te]:
        df['AvgSpend']=df['TotalCharges']/(df['tenure']+1)
        df['ChTen']=df['MonthlyCharges']*df['tenure']
        svc=['PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
        df['nSvc']=sum((df[c].astype(str)=='Yes').astype(int) for c in svc)
        df['MTM']=(df['Contract'].astype(str)=='Month-to-month').astype(int)
        df['EChk']=(df['PaymentMethod'].astype(str)=='Electronic check').astype(int)
    return enc(tr,te)

# SET B: Groupby-heavy
def set_b(tr, te):
    tr=tr.copy(); te=te.copy()
    for df in [tr,te]:
        df['AvgSpend']=df['TotalCharges']/(df['tenure']+1)
        df['ChTen']=df['MonthlyCharges']*df['tenure']
    for gc in ['Contract','InternetService','PaymentMethod','gender','Partner','Dependents','PaperlessBilling']:
        gcs=tr[gc].astype(str); gct=te[gc].astype(str)
        for ac in ['MonthlyCharges','TotalCharges','tenure']:
            grp=pd.concat([gcs,tr[ac]],axis=1).groupby(gc)[ac]
            m=grp.mean(); s=grp.std().fillna(0); md=grp.median()
            tr[f'{gc}_{ac}_mean']=gcs.map(m).fillna(0)
            tr[f'{gc}_{ac}_std']=gcs.map(s).fillna(0)
            tr[f'{gc}_{ac}_med']=gcs.map(md).fillna(0)
            tr[f'{gc}_{ac}_diff']=tr[ac]-tr[f'{gc}_{ac}_mean']
            te[f'{gc}_{ac}_mean']=gct.map(m).fillna(0)
            te[f'{gc}_{ac}_std']=gct.map(s).fillna(0)
            te[f'{gc}_{ac}_med']=gct.map(md).fillna(0)
            te[f'{gc}_{ac}_diff']=te[ac]-te[f'{gc}_{ac}_mean']
    for g1,g2 in [('Contract','InternetService'),('Contract','PaymentMethod'),('InternetService','PaymentMethod')]:
        kt=tr[g1].astype(str)+'_'+tr[g2].astype(str)
        ke=te[g1].astype(str)+'_'+te[g2].astype(str)
        for ac in ['MonthlyCharges','tenure']:
            ms=pd.DataFrame({'k':kt,ac:tr[ac]}).groupby('k')[ac].mean()
            tr[f'{g1}_{g2}_{ac}_m']=kt.map(ms); te[f'{g1}_{g2}_{ac}_m']=ke.map(ms).fillna(tr[ac].mean())
    return enc(tr,te)

# SET C: Interaction-heavy
def set_c(tr, te):
    tr=tr.copy(); te=te.copy()
    for df in [tr,te]:
        df['AvgSpend']=df['TotalCharges']/(df['tenure']+1)
        df['TenProxy']=df['TotalCharges']/(df['MonthlyCharges']+.01)
        df['ChTen']=df['MonthlyCharges']*df['tenure']
        df['ChargeRatio']=df['MonthlyCharges']/(df['TotalCharges']+.01)
        df['ExpTotal']=df['tenure']*df['MonthlyCharges']
        df['TotalDiff']=df['TotalCharges']-df['ExpTotal']
        svc=['PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
        df['nSvc']=sum((df[c].astype(str)=='Yes').astype(int) for c in svc)
        df['nSec']=sum((df[c].astype(str)=='Yes').astype(int) for c in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport'])
        df['Fiber']=(df['InternetService'].astype(str)=='Fiber optic').astype(int)
        df['MTM']=(df['Contract'].astype(str)=='Month-to-month').astype(int)
        df['EChk']=(df['PaymentMethod'].astype(str)=='Electronic check').astype(int)
        df['Auto']=df['PaymentMethod'].astype(str).str.contains('automatic',case=False).astype(int)
        for f in ['MTM','Fiber','EChk','Auto']:
            df[f'{f}_t']=df[f]*df['tenure']; df[f'{f}_m']=df[f]*df['MonthlyCharges']
        df['EC_MTM']=df['EChk']*df['MTM']; df['Fib_MTM']=df['Fiber']*df['MTM']
        df['Sr_MTM']=df['SeniorCitizen']*df['MTM']; df['Sec_Fib']=df['nSec']*df['Fiber']
        df['FibNoSec']=df['Fiber']*(df['nSec']==0).astype(int)
        df['NewMTM']=(df['tenure']<12).astype(int)*df['MTM']
        df['risk']=(df['tenure']<12).astype(int)+df['MTM']+df['Fiber']+df['EChk']+(df['nSec']==0).astype(int)
        df['CpS']=df['MonthlyCharges']/(df['nSvc']+1)
    return enc(tr,te)

# SET D: Full v04
def set_d(tr, te):
    tr=tr.copy(); te=te.copy()
    for df in [tr,te]:
        df['AvgSpend']=df['TotalCharges']/(df['tenure']+1)
        df['TenProxy']=df['TotalCharges']/(df['MonthlyCharges']+.01)
        df['ChTen']=df['MonthlyCharges']*df['tenure']
        df['ChTenLog']=np.log1p(df['MonthlyCharges']*df['tenure'])
        df['ChargeRatio']=df['MonthlyCharges']/(df['TotalCharges']+.01)
        df['TenSq']=df['tenure']**2; df['MonSq']=df['MonthlyCharges']**2
        df['ExpTotal']=df['tenure']*df['MonthlyCharges']
        df['TotalDiff']=df['TotalCharges']-df['ExpTotal']
        df['TotalDiffPct']=df['TotalDiff']/(df['ExpTotal']+.01)
        svc=['PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
        df['nSvc']=sum((df[c].astype(str)=='Yes').astype(int) for c in svc)
        inet=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
        df['nInet']=sum((df[c].astype(str)=='Yes').astype(int) for c in inet)
        df['nSec']=sum((df[c].astype(str)=='Yes').astype(int) for c in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport'])
        df['CpS']=df['MonthlyCharges']/(df['nSvc']+1)
        df['Fiber']=(df['InternetService'].astype(str)=='Fiber optic').astype(int)
        df['DSL']=(df['InternetService'].astype(str)=='DSL').astype(int)
        df['HasInet']=(df['InternetService'].astype(str)!='No').astype(int)
        df['MTM']=(df['Contract'].astype(str)=='Month-to-month').astype(int)
        df['EChk']=(df['PaymentMethod'].astype(str)=='Electronic check').astype(int)
        df['Auto']=df['PaymentMethod'].astype(str).str.contains('automatic',case=False).astype(int)
        for f in ['MTM','Fiber','EChk','Auto','DSL','HasInet']:
            df[f'{f}_t']=df[f]*df['tenure']; df[f'{f}_m']=df[f]*df['MonthlyCharges']
        df['EC_MTM']=df['EChk']*df['MTM']; df['Sec_Fib']=df['nSec']*df['Fiber']
        df['FibNoSec']=df['Fiber']*(df['nSec']==0).astype(int)
        df['NewMTM']=(df['tenure']<12).astype(int)*df['MTM']
        df['risk']=(df['tenure']<12).astype(int)+df['MTM']+df['Fiber']+df['EChk']+(df['nSec']==0).astype(int)
        df['TenBin']=pd.cut(df['tenure'],bins=[-1,3,6,12,24,48,72,999],labels=[0,1,2,3,4,5,6]).astype(float)
        df['ChBin']=pd.cut(df['MonthlyCharges'],bins=[-1,20,35,50,65,80,95,999],labels=[0,1,2,3,4,5,6]).astype(float)
    for gc in ['Contract','InternetService','PaymentMethod']:
        gcs=tr[gc].astype(str); gct=te[gc].astype(str)
        for ac in ['MonthlyCharges','TotalCharges','tenure']:
            ms=pd.concat([gcs,tr[ac]],axis=1).groupby(gc)[ac].mean()
            tr[f'{gc}_{ac}_m']=gcs.map(ms); tr[f'{gc}_{ac}_d']=tr[ac]-tr[f'{gc}_{ac}_m']
            te[f'{gc}_{ac}_m']=gct.map(ms).fillna(tr[ac].mean()); te[f'{gc}_{ac}_d']=te[ac]-te[f'{gc}_{ac}_m']
    for c in cats_raw:
        freq=tr[c].value_counts(normalize=True)
        tr[f'{c}_freq']=tr[c].map(freq).fillna(0); te[f'{c}_freq']=te[c].map(freq).fillna(0)
    return enc(tr,te)

# BUILD
print('Building feature sets...', file=sys.stderr, flush=True)
sets = {'A':set_a(train_raw,test_raw), 'B':set_b(train_raw,test_raw),
        'C':set_c(train_raw,test_raw), 'D':set_d(train_raw,test_raw)}
for n,(tr,te) in sets.items():
    print(f'  Set {n}: {tr.shape[1]} feats', file=sys.stderr, flush=True)

# TRAIN 12 MODELS
skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)
folds=list(skf.split(train_raw,y))
oof_all={}; test_all={}

cfgs = {
    'lgb': lambda s: lgb.LGBMClassifier(objective='binary',metric='auc',learning_rate=0.05,num_leaves=63,min_child_samples=20,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=0.5,n_estimators=3000,random_state=s,verbose=-1,n_jobs=-1),
    'xgb': lambda s: xgb.XGBClassifier(objective='binary:logistic',eval_metric='auc',learning_rate=0.05,max_depth=7,min_child_weight=5,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=1.0,n_estimators=3000,random_state=s,tree_method='hist',n_jobs=-1,verbosity=0,early_stopping_rounds=50),
    'cat': lambda s: CatBoostClassifier(iterations=2000,learning_rate=0.05,depth=7,l2_leaf_reg=3,random_seed=s,verbose=0,eval_metric='AUC'),
}

print(f'\nTraining 12 models...', file=sys.stderr, flush=True)
for sn,(trd,ted) in sets.items():
    for mn,mfn in cfgs.items():
        k=f'{mn}_{sn}'
        print(f'  {k} ({trd.shape[1]}f)...', file=sys.stderr, flush=True)
        oof=np.zeros(len(trd)); tp=np.zeros(len(ted))
        for fi,(tri,vai) in enumerate(folds):
            m=mfn(SEED)
            if 'lgb' in mn:
                m.fit(trd.iloc[tri],y.iloc[tri],eval_set=[(trd.iloc[vai],y.iloc[vai])],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
            elif 'xgb' in mn:
                m.fit(trd.iloc[tri],y.iloc[tri],eval_set=[(trd.iloc[vai],y.iloc[vai])],verbose=False)
            else:
                m.fit(trd.iloc[tri],y.iloc[tri],eval_set=(trd.iloc[vai],y.iloc[vai]),early_stopping_rounds=50)
            oof[vai]=m.predict_proba(trd.iloc[vai])[:,1]; tp+=m.predict_proba(ted)[:,1]/N_FOLDS; del m
        a=roc_auc_score(y,oof); oof_all[k]=oof; test_all[k]=tp
        print(f'    {k}: {a:.6f}', file=sys.stderr, flush=True)

# HILL CLIMBING
print('\nHill Climbing...', file=sys.stderr, flush=True)
names=list(oof_all.keys())
om=np.column_stack([oof_all[k] for k in names])
tm=np.column_stack([test_all[k] for k in names])
bi=np.argmax([roc_auc_score(y,oof_all[k]) for k in names])
sel=[bi]; cb=om[:,bi].copy(); ca=roc_auc_score(y,cb)
print(f'Start: {names[bi]} = {ca:.6f}', file=sys.stderr, flush=True)
for _ in range(11):
    bg=0; bx=-1
    for i in range(len(names)):
        nb=(cb*len(sel)+om[:,i])/(len(sel)+1); na=roc_auc_score(y,nb)
        if na-ca>bg: bg=na-ca; bx=i
    if bx>=0 and bg>0.00001:
        sel.append(bx); cb=np.mean(om[:,sel],axis=1); ca=roc_auc_score(y,cb)
        print(f'  +{names[bx]}: {ca:.6f} (+{bg:.6f})', file=sys.stderr, flush=True)
    else: break
print(f'HC: {ca:.6f} ({len(sel)} models)', file=sys.stderr, flush=True)
thc=np.mean(tm[:,sel],axis=1)

# RIDGE META
print('\nRidge Meta...', file=sys.stderr, flush=True)
mo=np.zeros(len(y)); mt=np.zeros(len(test_raw))
for tri,vai in folds:
    m=LogisticRegression(C=1.0,max_iter=1000); m.fit(om[tri],y.iloc[tri])
    mo[vai]=m.predict_proba(om[vai])[:,1]; mt+=m.predict_proba(tm)[:,1]/N_FOLDS
am=roc_auc_score(y,mo)
print(f'Ridge: {am:.6f}', file=sys.stderr, flush=True)

# AVG
aa=roc_auc_score(y,np.mean(om,axis=1))
print(f'Avg: {aa:.6f}', file=sys.stderr, flush=True)

# BEST
r={'hc':(ca,thc),'ridge':(am,mt),'avg':(aa,np.mean(tm,axis=1))}
bn=max(r,key=lambda k:r[k][0]); ba,bt=r[bn]
print(f'\nBest: {bn} = {ba:.6f}', file=sys.stderr, flush=True)

os.makedirs('submissions',exist_ok=True)
sub['Churn']=bt; sub.to_csv(f'submissions/{VERSION}.csv',index=False)
for n,(a,p) in r.items():
    s=sub.copy(); s['Churn']=p; s.to_csv(f'submissions/{VERSION}_{n}.csv',index=False)
json.dump({'hc':round(ca,6),'ridge':round(am,6),'avg':round(aa,6),'best':bn,
           'models':{k:round(roc_auc_score(y,v),6) for k,v in oof_all.items()},
           'selected':[names[i] for i in sel]},
          open(f'submissions/{VERSION}_results.json','w'),indent=2)
print('DONE!', file=sys.stderr, flush=True)
