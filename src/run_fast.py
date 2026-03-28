"""Ultra-fast baseline — fewer trees, fewer folds."""
import pandas as pd, numpy as np, lightgbm as lgb, sys, os, json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data-kaggle/train.csv')
test = pd.read_csv('data-kaggle/test.csv')
sub = pd.read_csv('data-kaggle/sample_submission.csv')

# Add original
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

# FE
for df in [train, test]:
    if df['TotalCharges'].dtype==object:
        df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
    df['AvgSpend']=df['TotalCharges']/(df['tenure']+1)
    df['TenProxy']=df['TotalCharges']/(df['MonthlyCharges']+.01)
    df['ChTen']=df['MonthlyCharges']*df['tenure']
    svc=['PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['nSvc']=sum((df[c].astype(str)=='Yes').astype(int) for c in svc)
    df['Fiber']=(df['InternetService'].astype(str)=='Fiber optic').astype(int)
    df['MTM']=(df['Contract'].astype(str)=='Month-to-month').astype(int)
    df['EChk']=(df['PaymentMethod'].astype(str)=='Electronic check').astype(int)
    df['t_x_m']=df['tenure']*df['MonthlyCharges']

cats=train.select_dtypes(include=['object','category']).columns.tolist()
for c in cats:
    le=LabelEncoder()
    le.fit(pd.concat([train[c],test[c]]).astype(str))
    train[c]=le.transform(train[c].astype(str))
    test[c]=le.transform(test[c].astype(str))
train.fillna(-999,inplace=True)
test.fillna(-999,inplace=True)

print(f'Shape: {train.shape}',file=sys.stderr,flush=True)

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
oof=np.zeros(len(train))
tp=np.zeros(len(test))

for fold,(tr,val) in enumerate(skf.split(train,y)):
    m=lgb.LGBMClassifier(objective='binary',metric='auc',learning_rate=0.1,
        num_leaves=31,subsample=0.8,colsample_bytree=0.8,
        n_estimators=500,random_state=42,verbose=-1,n_jobs=-1)
    m.fit(train.iloc[tr],y.iloc[tr],eval_set=[(train.iloc[val],y.iloc[val])],
          callbacks=[lgb.early_stopping(30),lgb.log_evaluation(0)])
    oof[val]=m.predict_proba(train.iloc[val])[:,1]
    tp+=m.predict_proba(test)[:,1]/5
    a=roc_auc_score(y.iloc[val],oof[val])
    print(f'F{fold+1}:{a:.6f}(iter={m.best_iteration_})',file=sys.stderr,flush=True)

cv=roc_auc_score(y,oof)
print(f'OOF:{cv:.6f}',file=sys.stderr,flush=True)

os.makedirs('submissions',exist_ok=True)
sub['Churn']=tp
sub.to_csv('submissions/v01_lgb_quick.csv',index=False)
json.dump({'cv':round(cv,6),'feats':train.shape[1]},open('submissions/v01_lgb_quick_results.json','w'))
print(f'{cv:.6f}')

imp=pd.DataFrame({'f':train.columns,'i':m.feature_importances_}).sort_values('i',ascending=False)
print(imp.head(10).to_string(index=False),file=sys.stderr,flush=True)
