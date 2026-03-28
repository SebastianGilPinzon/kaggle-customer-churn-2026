"""3-model ensemble: LGB + XGB + CatBoost, optimized weights."""
import pandas as pd, numpy as np, sys, os, json, warnings
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

VERSION = "v02_ensemble3"
SEED = 42

# Load + merge
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

# Advanced FE
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

    # Interactions
    df['t_x_m']=df['tenure']*df['MonthlyCharges']
    df['MTM_ten']=df['MTM']*df['tenure']
    df['Fib_ch']=df['Fiber']*df['MonthlyCharges']
    df['EC_MTM']=df['EChk']*df['MTM']
    df['Sec_Fib']=df['nSec']*df['Fiber']
    df['Sr_MTM']=df['SeniorCitizen']*df['MTM']

    # Bins
    df['TenBin']=pd.cut(df['tenure'],bins=[-1,6,12,24,48,72,999],labels=[0,1,2,3,4,5]).astype(float)
    df['ChBin']=pd.cut(df['MonthlyCharges'],bins=[-1,20,40,60,80,100,999],labels=[0,1,2,3,4,5]).astype(float)

# Frequency encoding
cats=train.select_dtypes(include=['object','category']).columns.tolist()
for c in cats:
    freq=train[c].value_counts(normalize=True)
    train[f'{c}_freq']=train[c].map(freq).fillna(0)
    test[f'{c}_freq']=test[c].map(freq).fillna(0)

# Label encode
for c in cats:
    le=LabelEncoder()
    le.fit(pd.concat([train[c],test[c]]).astype(str))
    train[c]=le.transform(train[c].astype(str))
    test[c]=le.transform(test[c].astype(str))

train.fillna(-999,inplace=True)
test.fillna(-999,inplace=True)

print(f'Features: {train.shape[1]}', file=sys.stderr, flush=True)

# CV
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
oof_l=np.zeros(len(train)); oof_x=np.zeros(len(train)); oof_c=np.zeros(len(train))
tp_l=np.zeros(len(test)); tp_x=np.zeros(len(test)); tp_c=np.zeros(len(test))

for fold,(tr,val) in enumerate(skf.split(train,y)):
    X_tr,X_val=train.iloc[tr],train.iloc[val]
    y_tr,y_val=y.iloc[tr],y.iloc[val]

    # LGB
    m=lgb.LGBMClassifier(objective='binary',metric='auc',learning_rate=0.05,
        num_leaves=63,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,
        reg_alpha=0.1,reg_lambda=0.5,n_estimators=2000,random_state=SEED,verbose=-1,n_jobs=-1)
    m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
    oof_l[val]=m.predict_proba(X_val)[:,1]; tp_l+=m.predict_proba(test)[:,1]/5
    al=roc_auc_score(y_val,oof_l[val])

    # XGB
    m=xgb.XGBClassifier(objective='binary:logistic',eval_metric='auc',learning_rate=0.05,
        max_depth=7,min_child_weight=5,subsample=0.8,colsample_bytree=0.8,
        reg_alpha=0.1,reg_lambda=1.0,n_estimators=2000,random_state=SEED,
        tree_method='hist',n_jobs=-1,verbosity=0,early_stopping_rounds=50)
    m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
    oof_x[val]=m.predict_proba(X_val)[:,1]; tp_x+=m.predict_proba(test)[:,1]/5
    ax=roc_auc_score(y_val,oof_x[val])

    # CatBoost
    m=CatBoostClassifier(iterations=2000,learning_rate=0.05,depth=7,l2_leaf_reg=3,
        random_seed=SEED,verbose=0,eval_metric='AUC')
    m.fit(X_tr,y_tr,eval_set=(X_val,y_val),early_stopping_rounds=50)
    oof_c[val]=m.predict_proba(X_val)[:,1]; tp_c+=m.predict_proba(test)[:,1]/5
    ac=roc_auc_score(y_val,oof_c[val])

    print(f'F{fold+1} L:{al:.6f} X:{ax:.6f} C:{ac:.6f}',file=sys.stderr,flush=True)

al_=roc_auc_score(y,oof_l); ax_=roc_auc_score(y,oof_x); ac_=roc_auc_score(y,oof_c)
ae_=roc_auc_score(y,(oof_l+oof_x+oof_c)/3)
print(f'OOF L:{al_:.6f} X:{ax_:.6f} C:{ac_:.6f} E:{ae_:.6f}',file=sys.stderr,flush=True)

# Optimize weights
best_a,best_w=0,(1/3,1/3,1/3)
for w1 in np.arange(0.1,0.8,0.05):
    for w2 in np.arange(0.1,0.8-w1,0.05):
        w3=1-w1-w2
        if w3<0.05: continue
        a=roc_auc_score(y,w1*oof_l+w2*oof_x+w3*oof_c)
        if a>best_a: best_a,best_w=a,(w1,w2,w3)

print(f'Best w=({best_w[0]:.2f},{best_w[1]:.2f},{best_w[2]:.2f}) AUC={best_a:.6f}',file=sys.stderr,flush=True)

tp_final=best_w[0]*tp_l+best_w[1]*tp_x+best_w[2]*tp_c
os.makedirs('submissions',exist_ok=True)
sub['Churn']=tp_final
sub.to_csv(f'submissions/{VERSION}.csv',index=False)

# Also save equal-weight version
sub2=sub.copy(); sub2['Churn']=(tp_l+tp_x+tp_c)/3
sub2.to_csv(f'submissions/{VERSION}_equal.csv',index=False)

json.dump({'cv_lgb':round(al_,6),'cv_xgb':round(ax_,6),'cv_cat':round(ac_,6),
           'cv_ens':round(ae_,6),'cv_opt':round(best_a,6),
           'weights':{'l':round(best_w[0],2),'x':round(best_w[1],2),'c':round(best_w[2],2)},
           'feats':train.shape[1]},
          open(f'submissions/{VERSION}_results.json','w'),indent=2)
print(f'{best_a:.6f}')
print('DONE!',file=sys.stderr,flush=True)
