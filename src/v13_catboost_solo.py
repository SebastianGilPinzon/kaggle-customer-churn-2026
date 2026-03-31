import pandas as pd, numpy as np, sys, os, json, warnings
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')
VERSION="v13_catboost_solo"
SEED=42; N_FOLDS=5; N_SEEDS=5
train=pd.read_csv('data-kaggle/train.csv')
test=pd.read_csv('data-kaggle/test.csv')
sub=pd.read_csv('data-kaggle/sample_submission.csv')
test_ids=test['id'].copy()
train.drop('id',axis=1,errors='ignore',inplace=True)
test.drop('id',axis=1,errors='ignore',inplace=True)
y=train['Churn'].map(lambda x:1 if str(x) in ('Yes','1','1.0') else 0).astype(int)
train.drop('Churn',axis=1,inplace=True)
for df in [train,test]:
    if df['TotalCharges'].dtype==object: df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
    df['AvgSpend']=df['TotalCharges']/(df['tenure']+1)
    df['TenProxy']=df['TotalCharges']/(df['MonthlyCharges']+.01)
    df['ChTen']=df['MonthlyCharges']*df['tenure']
    df['ChTenLog']=np.log1p(df['MonthlyCharges']*df['tenure'])
    df['ChargeRatio']=df['MonthlyCharges']/(df['TotalCharges']+.01)
    df['ExpectedTotal']=df['tenure']*df['MonthlyCharges']
    df['TotalDiff']=df['TotalCharges']-df['ExpectedTotal']
    svc=['PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['nSvc']=sum((df[c].astype(str)=='Yes').astype(int) for c in svc)
    df['nSec']=sum((df[c].astype(str)=='Yes').astype(int) for c in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport'])
    df['Fiber']=(df['InternetService'].astype(str)=='Fiber optic').astype(int)
    df['MTM']=(df['Contract'].astype(str)=='Month-to-month').astype(int)
    df['EChk']=(df['PaymentMethod'].astype(str)=='Electronic check').astype(int)
    df['AutoPay']=df['PaymentMethod'].astype(str).str.contains('automatic',case=False).astype(int)
    for flag in ['MTM','Fiber','EChk','AutoPay']:
        df[f'{flag}_ten']=df[flag]*df['tenure']
        df[f'{flag}_mch']=df[flag]*df['MonthlyCharges']
    df['EC_MTM']=df['EChk']*df['MTM']
    df['Sec_Fib']=df['nSec']*df['Fiber']
    df['FibNoSec']=df['Fiber']*(df['nSec']==0).astype(int)
    df['New_MTM']=(df['tenure']<12).astype(int)*df['MTM']
    df['risk_score']=(df['tenure']<12).astype(int)+df['MTM']+df['Fiber']+df['EChk']+(df['nSec']==0).astype(int)
cats=train.select_dtypes(include=['object','category']).columns.tolist()
for c in cats:
    le=LabelEncoder(); le.fit(pd.concat([train[c],test[c]]).astype(str))
    train[c]=le.transform(train[c].astype(str)); test[c]=le.transform(test[c].astype(str))
train.fillna(-999,inplace=True); test.fillna(-999,inplace=True)
print(f'Features: {train.shape[1]}',file=sys.stderr,flush=True)
seeds=[42,1042,2042,3042,4042]; all_test=[]
for si,seed in enumerate(seeds):
    print(f'Seed {si+1}/{N_SEEDS}: {seed}',file=sys.stderr,flush=True)
    skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=seed)
    oof=np.zeros(len(train)); tp=np.zeros(len(test))
    for fold,(tr,val) in enumerate(skf.split(train,y)):
        m=CatBoostClassifier(iterations=3000,learning_rate=0.03,depth=8,l2_leaf_reg=5,random_seed=seed,verbose=0,eval_metric='AUC',min_data_in_leaf=30)
        m.fit(train.iloc[tr],y.iloc[tr],eval_set=(train.iloc[val],y.iloc[val]),early_stopping_rounds=100)
        oof[val]=m.predict_proba(train.iloc[val])[:,1]; tp+=m.predict_proba(test)[:,1]/N_FOLDS; del m
    auc=roc_auc_score(y,oof)
    print(f'  AUC={auc:.6f}',file=sys.stderr,flush=True)
    all_test.append(tp)
final=np.mean(all_test,axis=0)
sub['Churn']=final; sub.to_csv(f'submissions/{VERSION}.csv',index=False)
json.dump({'version':VERSION},open(f'submissions/{VERSION}_results.json','w'))
print('DONE!',file=sys.stderr,flush=True)
