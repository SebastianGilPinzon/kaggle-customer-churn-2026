"""v04: Exhaustive feature engineering + groupby aggregations."""
import pandas as pd, numpy as np, sys, os, json, warnings
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
warnings.filterwarnings('ignore')

VERSION = "v04_exhaustive_fe"
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

print(f'Train: {train.shape}', file=sys.stderr, flush=True)

# === EXHAUSTIVE FEATURE ENGINEERING ===
def exhaustive_fe(df):
    df = df.copy()
    if df['TotalCharges'].dtype==object:
        df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')

    # --- Basic ratios ---
    df['AvgSpend'] = df['TotalCharges']/(df['tenure']+1)
    df['TenProxy'] = df['TotalCharges']/(df['MonthlyCharges']+.01)
    df['ChTen'] = df['MonthlyCharges']*df['tenure']
    df['ChTenLog'] = np.log1p(df['MonthlyCharges']*df['tenure'])
    df['ChargeRatio'] = df['MonthlyCharges']/(df['TotalCharges']+.01)
    df['TenureSq'] = df['tenure']**2
    df['MonthlySq'] = df['MonthlyCharges']**2
    df['TenureLog'] = np.log1p(df['tenure'])
    df['MonthlyLog'] = np.log1p(df['MonthlyCharges'])
    df['TotalLog'] = np.log1p(df['TotalCharges'])

    # Expected total vs actual (how much they "should" have paid)
    df['ExpectedTotal'] = df['tenure'] * df['MonthlyCharges']
    df['TotalDiff'] = df['TotalCharges'] - df['ExpectedTotal']
    df['TotalDiffPct'] = df['TotalDiff'] / (df['ExpectedTotal']+.01)

    # --- Service counts ---
    svc=['PhoneService','MultipleLines','InternetService','OnlineSecurity',
         'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['nSvc']=sum((df[c].astype(str)=='Yes').astype(int) for c in svc)
    inet=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['nInet']=sum((df[c].astype(str)=='Yes').astype(int) for c in inet)
    df['nSec']=sum((df[c].astype(str)=='Yes').astype(int) for c in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport'])
    df['nStream']=sum((df[c].astype(str)=='Yes').astype(int) for c in ['StreamingTV','StreamingMovies'])
    df['nNoInet']=sum((df[c].astype(str)=='No internet service').astype(int) for c in inet)
    df['nNo']=sum((df[c].astype(str)=='No').astype(int) for c in svc)

    # Cost per service
    df['CostPerService'] = df['MonthlyCharges'] / (df['nSvc']+1)
    df['CostPerInetService'] = df['MonthlyCharges'] / (df['nInet']+1)

    # --- Flags ---
    df['Fiber']=(df['InternetService'].astype(str)=='Fiber optic').astype(int)
    df['DSL']=(df['InternetService'].astype(str)=='DSL').astype(int)
    df['HasInet']=(df['InternetService'].astype(str)!='No').astype(int)
    df['MTM']=(df['Contract'].astype(str)=='Month-to-month').astype(int)
    df['OneYr']=(df['Contract'].astype(str)=='One year').astype(int)
    df['TwoYr']=(df['Contract'].astype(str)=='Two year').astype(int)
    df['EChk']=(df['PaymentMethod'].astype(str)=='Electronic check').astype(int)
    df['AutoPay']=df['PaymentMethod'].astype(str).str.contains('automatic',case=False).astype(int)

    # --- Interactions (numeric x flag) ---
    for flag in ['MTM','Fiber','EChk','AutoPay','DSL','HasInet']:
        if flag in df.columns:
            df[f'{flag}_ten'] = df[flag] * df['tenure']
            df[f'{flag}_mch'] = df[flag] * df['MonthlyCharges']
            df[f'{flag}_tch'] = df[flag] * df['TotalCharges']

    # Flag x flag
    df['EC_MTM'] = df['EChk']*df['MTM']
    df['Fib_MTM'] = df['Fiber']*df['MTM']
    df['Sr_MTM'] = df['SeniorCitizen']*df['MTM']
    df['Sr_Fib'] = df['SeniorCitizen']*df['Fiber']
    df['Sec_Fib'] = df['nSec']*df['Fiber']
    df['Auto_TwoYr'] = df['AutoPay']*df['TwoYr']
    df['NoInet_MTM'] = (df['HasInet']==0).astype(int)*df['MTM']

    # Fiber without security = risky customer
    df['FibNoSec'] = df['Fiber'] * (df['nSec']==0).astype(int)
    # MTM with high charges = likely to churn
    df['MTM_HighCharge'] = df['MTM'] * (df['MonthlyCharges']>70).astype(int)
    # New customer with MTM
    df['New_MTM'] = (df['tenure']<12).astype(int) * df['MTM']
    # Senior with fiber and electronic check
    df['Sr_Fib_EC'] = df['SeniorCitizen'] * df['Fiber'] * df['EChk']

    # --- Bins ---
    df['TenBin']=pd.cut(df['tenure'],bins=[-1,3,6,12,24,48,72,999],labels=[0,1,2,3,4,5,6]).astype(float)
    df['ChBin']=pd.cut(df['MonthlyCharges'],bins=[-1,20,35,50,65,80,95,999],labels=[0,1,2,3,4,5,6]).astype(float)
    df['TotalBin']=pd.cut(df['TotalCharges'],bins=[-1,500,1000,2000,4000,6000,999999],labels=[0,1,2,3,4,5]).astype(float)

    return df

train = exhaustive_fe(train)
test = exhaustive_fe(test)

# --- Groupby aggregations ---
print('Groupby aggregations...', file=sys.stderr, flush=True)
group_cols = ['Contract','InternetService','PaymentMethod']
agg_cols = ['MonthlyCharges','TotalCharges','tenure']

for gc in group_cols:
    gc_str = train[gc].astype(str) if train[gc].dtype==object else train[gc]
    gc_str_test = test[gc].astype(str) if test[gc].dtype==object else test[gc]
    for ac in agg_cols:
        grp = pd.concat([gc_str, train[ac]], axis=1).groupby(gc)[ac]
        means = grp.mean()
        stds = grp.std().fillna(0)
        train[f'{gc}_{ac}_mean'] = gc_str.map(means)
        train[f'{gc}_{ac}_std'] = gc_str.map(stds)
        train[f'{gc}_{ac}_diff'] = train[ac] - train[f'{gc}_{ac}_mean']
        test[f'{gc}_{ac}_mean'] = gc_str_test.map(means).fillna(train[ac].mean())
        test[f'{gc}_{ac}_std'] = gc_str_test.map(stds).fillna(0)
        test[f'{gc}_{ac}_diff'] = test[ac] - test[f'{gc}_{ac}_mean']

# 2-way groupby
for g1, g2 in [('Contract','InternetService'),('Contract','PaymentMethod'),('InternetService','PaymentMethod')]:
    g1s = train[g1].astype(str); g2s = train[g2].astype(str)
    g1t = test[g1].astype(str); g2t = test[g2].astype(str)
    key_tr = g1s + '_' + g2s
    key_te = g1t + '_' + g2t
    for ac in ['MonthlyCharges','tenure']:
        grp = pd.DataFrame({'key':key_tr, ac:train[ac]}).groupby('key')[ac]
        means = grp.mean()
        train[f'{g1}_{g2}_{ac}_mean'] = key_tr.map(means)
        test[f'{g1}_{g2}_{ac}_mean'] = key_te.map(means).fillna(train[ac].mean())
        train[f'{g1}_{g2}_{ac}_diff'] = train[ac] - train[f'{g1}_{g2}_{ac}_mean']
        test[f'{g1}_{g2}_{ac}_diff'] = test[ac] - test[f'{g1}_{g2}_{ac}_mean']

# Freq encoding
cats = train.select_dtypes(include=['object','category']).columns.tolist()
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

# === TRAINING ===
skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)
oof_l=np.zeros(len(train)); oof_x=np.zeros(len(train)); oof_c=np.zeros(len(train))
tp_l=np.zeros(len(test)); tp_x=np.zeros(len(test)); tp_c=np.zeros(len(test))

for fold,(tr,val) in enumerate(skf.split(train,y)):
    X_tr,X_val=train.iloc[tr],train.iloc[val]
    y_tr,y_val=y.iloc[tr],y.iloc[val]

    m=lgb.LGBMClassifier(objective='binary',metric='auc',learning_rate=0.05,
        num_leaves=63,min_child_samples=20,subsample=0.8,colsample_bytree=0.7,
        reg_alpha=0.1,reg_lambda=0.5,n_estimators=3000,random_state=SEED,verbose=-1,n_jobs=-1)
    m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
    oof_l[val]=m.predict_proba(X_val)[:,1]; tp_l+=m.predict_proba(test)[:,1]/N_FOLDS

    m=xgb.XGBClassifier(objective='binary:logistic',eval_metric='auc',learning_rate=0.05,
        max_depth=7,min_child_weight=5,subsample=0.8,colsample_bytree=0.7,
        reg_alpha=0.1,reg_lambda=1.0,n_estimators=3000,random_state=SEED,
        tree_method='hist',n_jobs=-1,verbosity=0,early_stopping_rounds=50)
    m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
    oof_x[val]=m.predict_proba(X_val)[:,1]; tp_x+=m.predict_proba(test)[:,1]/N_FOLDS

    m=CatBoostClassifier(iterations=3000,learning_rate=0.05,depth=7,l2_leaf_reg=3,
        random_seed=SEED,verbose=0,eval_metric='AUC')
    m.fit(X_tr,y_tr,eval_set=(X_val,y_val),early_stopping_rounds=50)
    oof_c[val]=m.predict_proba(X_val)[:,1]; tp_c+=m.predict_proba(test)[:,1]/N_FOLDS

    print(f'F{fold+1} L:{roc_auc_score(y_val,oof_l[val]):.6f} X:{roc_auc_score(y_val,oof_x[val]):.6f} C:{roc_auc_score(y_val,oof_c[val]):.6f}',file=sys.stderr,flush=True)

al=roc_auc_score(y,oof_l); ax=roc_auc_score(y,oof_x); ac=roc_auc_score(y,oof_c)
ae=roc_auc_score(y,(oof_l+oof_x+oof_c)/3)
print(f'OOF L:{al:.6f} X:{ax:.6f} C:{ac:.6f} E:{ae:.6f}',file=sys.stderr,flush=True)

# Optimize
best_a,best_w=0,(1/3,1/3,1/3)
for w1 in np.arange(0,1.01,.05):
    for w2 in np.arange(0,1.01-w1,.05):
        w3=1-w1-w2
        if w3<0: continue
        a=roc_auc_score(y,w1*oof_l+w2*oof_x+w3*oof_c)
        if a>best_a: best_a,best_w=a,(w1,w2,w3)
print(f'Opt w=({best_w[0]:.2f},{best_w[1]:.2f},{best_w[2]:.2f}) AUC={best_a:.6f}',file=sys.stderr,flush=True)

tp_final=best_w[0]*tp_l+best_w[1]*tp_x+best_w[2]*tp_c
os.makedirs('submissions',exist_ok=True)
sub['Churn']=tp_final
sub.to_csv(f'submissions/{VERSION}.csv',index=False)
json.dump({'cv_lgb':round(al,6),'cv_xgb':round(ax,6),'cv_cat':round(ac,6),
           'cv_ens':round(ae,6),'cv_opt':round(best_a,6),
           'weights':{'l':round(best_w[0],2),'x':round(best_w[1],2),'c':round(best_w[2],2)},
           'feats':train.shape[1]},
          open(f'submissions/{VERSION}_results.json','w'),indent=2)
print(f'\nSaved: submissions/{VERSION}.csv',file=sys.stderr,flush=True)
print('DONE!',file=sys.stderr,flush=True)
