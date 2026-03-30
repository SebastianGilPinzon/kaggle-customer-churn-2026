"""
v06: 10-Model Stacking Pipeline + Upweighted Original Data
Based on expert panel recommendations:
- 10 diverse base models (3 GBDT types × multiple configs + non-GBDT)
- Upweight original 7K rows (sample_weight=10)
- LogisticRegression meta-learner
- Rank averaging as fallback
- Feature selection: top 30 features only for some models
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

VERSION = "v06_stacking"
SEED = 42
N_FOLDS = 5

# === DATA ===
print('Loading...', file=sys.stderr, flush=True)
train = pd.read_csv('data-kaggle/train.csv')
test = pd.read_csv('data-kaggle/test.csv')
sub = pd.read_csv('data-kaggle/sample_submission.csv')

n_synthetic = len(train)

# Merge original with marker
orig = pd.read_csv('data-original/WA_Fn-UseC_-Telco-Customer-Churn.csv')
orig['Churn'] = orig['Churn'].map({'Yes':1,'No':0})
orig.drop('customerID',axis=1,inplace=True)
orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'],errors='coerce')
common = [c for c in train.columns if c in orig.columns]
oa = orig[common].copy()
for c in train.columns:
    if c not in oa.columns: oa[c]=np.nan
train = pd.concat([train, oa[train.columns]], ignore_index=True)

# Sample weights: upweight original rows 10x
sample_weights = np.ones(len(train))
sample_weights[n_synthetic:] = 10.0
print(f'Original rows: {len(train)-n_synthetic}, weight=10x', file=sys.stderr, flush=True)

test_ids = test['id'].copy()
train.drop('id',axis=1,errors='ignore',inplace=True)
test.drop('id',axis=1,errors='ignore',inplace=True)
y = train['Churn'].map(lambda x:1 if str(x) in ('Yes','1','1.0') else 0).astype(int)
train.drop('Churn',axis=1,inplace=True)

# === FEATURE ENGINEERING (v02 level — 56 features, proven) ===
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
    df['AutoPay']=df['PaymentMethod'].astype(str).str.contains('automatic',case=False).astype(int)
    df['t_x_m']=df['tenure']*df['MonthlyCharges']
    df['MTM_ten']=df['MTM']*df['tenure']
    df['Fib_ch']=df['Fiber']*df['MonthlyCharges']
    df['EC_MTM']=df['EChk']*df['MTM']
    df['Sec_Fib']=df['nSec']*df['Fiber']
    df['Sr_MTM']=df['SeniorCitizen']*df['MTM']
    df['TenBin']=pd.cut(df['tenure'],bins=[-1,6,12,24,48,72,999],labels=[0,1,2,3,4,5]).astype(float)
    df['ChBin']=pd.cut(df['MonthlyCharges'],bins=[-1,20,40,60,80,100,999],labels=[0,1,2,3,4,5]).astype(float)
    # Expert-recommended: rank within group
    df['ExpectedTotal']=df['tenure']*df['MonthlyCharges']
    df['TotalDiff']=df['TotalCharges']-df['ExpectedTotal']
    df['FibNoSec']=df['Fiber']*(df['nSec']==0).astype(int)
    df['New_MTM']=(df['tenure']<12).astype(int)*df['MTM']
    df['risk_score']=(df['tenure']<12).astype(int)+df['MTM']+df['Fiber']+df['EChk']+(df['nSec']==0).astype(int)

# Freq encoding
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

all_features = train.columns.tolist()
print(f'Features: {len(all_features)}', file=sys.stderr, flush=True)

# === DEFINE 10 DIVERSE MODELS ===
models = {
    'lgb_v1': lambda seed: lgb.LGBMClassifier(
        objective='binary',metric='auc',learning_rate=0.05,num_leaves=31,
        min_child_samples=20,subsample=0.8,colsample_bytree=0.7,
        reg_alpha=0.1,reg_lambda=1.0,n_estimators=3000,random_state=seed,verbose=-1,n_jobs=-1),
    'lgb_v2': lambda seed: lgb.LGBMClassifier(
        objective='binary',metric='auc',learning_rate=0.03,num_leaves=127,
        min_child_samples=50,subsample=0.7,colsample_bytree=0.5,
        reg_alpha=0.5,reg_lambda=0.1,n_estimators=3000,random_state=seed,verbose=-1,n_jobs=-1),
    'lgb_dart': lambda seed: lgb.LGBMClassifier(
        objective='binary',metric='auc',learning_rate=0.05,num_leaves=63,
        boosting_type='dart',drop_rate=0.1,skip_drop=0.5,
        n_estimators=500,random_state=seed,verbose=-1,n_jobs=-1),
    'xgb_v1': lambda seed: xgb.XGBClassifier(
        objective='binary:logistic',eval_metric='auc',learning_rate=0.05,
        max_depth=6,min_child_weight=5,subsample=0.8,colsample_bytree=0.7,
        reg_alpha=0.1,reg_lambda=1.0,n_estimators=3000,random_state=seed,
        tree_method='hist',n_jobs=-1,verbosity=0,early_stopping_rounds=50),
    'xgb_v2': lambda seed: xgb.XGBClassifier(
        objective='binary:logistic',eval_metric='auc',learning_rate=0.03,
        max_depth=8,min_child_weight=10,subsample=0.7,colsample_bytree=0.6,
        reg_alpha=0.5,reg_lambda=2.0,n_estimators=3000,random_state=seed,
        tree_method='hist',n_jobs=-1,verbosity=0,early_stopping_rounds=50),
    'cat_v1': lambda seed: CatBoostClassifier(
        iterations=2000,learning_rate=0.05,depth=6,l2_leaf_reg=3,
        random_seed=seed,verbose=0,eval_metric='AUC',boosting_type='Plain'),
    'cat_v2': lambda seed: CatBoostClassifier(
        iterations=2000,learning_rate=0.03,depth=8,l2_leaf_reg=5,
        random_seed=seed,verbose=0,eval_metric='AUC',boosting_type='Plain'),
    'hgb': lambda seed: HistGradientBoostingClassifier(
        max_iter=1000,max_leaf_nodes=63,learning_rate=0.05,
        min_samples_leaf=50,random_state=seed),
    'et': lambda seed: ExtraTreesClassifier(
        n_estimators=500,max_depth=20,min_samples_leaf=50,
        random_state=seed,n_jobs=-1),
    'ridge': lambda seed: LogisticRegression(
        C=1.0,solver='lbfgs',max_iter=500,random_state=seed),
}

# === LEVEL 1: Generate OOF predictions ===
print(f'\n=== LEVEL 1: {len(models)} models ===', file=sys.stderr, flush=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(train, y))

oof_preds = {}  # model_name -> oof array
test_preds = {} # model_name -> test array

# For Ridge: need scaled features
scaler = StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index=train.index)
test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)

for name, model_fn in models.items():
    print(f'\n  Training {name}...', file=sys.stderr, flush=True)
    oof = np.zeros(len(train))
    tp = np.zeros(len(test))

    use_scaled = name == 'ridge'
    tr_data = train_scaled if use_scaled else train
    te_data = test_scaled if use_scaled else test

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr, X_val = tr_data.iloc[tr_idx], tr_data.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        sw_tr = sample_weights[tr_idx]

        m = model_fn(SEED)

        if 'lgb' in name:
            m.fit(X_tr, y_tr, sample_weight=sw_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        elif 'xgb' in name:
            m.fit(X_tr, y_tr, sample_weight=sw_tr,
                  eval_set=[(X_val, y_val)], verbose=False)
        elif 'cat' in name:
            m.fit(X_tr, y_tr, sample_weight=sw_tr,
                  eval_set=(X_val, y_val), early_stopping_rounds=50)
        elif name == 'hgb':
            m.fit(X_tr, y_tr, sample_weight=sw_tr)
        elif name == 'et':
            m.fit(X_tr, y_tr, sample_weight=sw_tr)
        elif name == 'ridge':
            m.fit(X_tr, y_tr, sample_weight=sw_tr)

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

# === Check correlations ===
print('\n=== Model Correlation Matrix ===', file=sys.stderr, flush=True)
oof_df = pd.DataFrame(oof_preds)
corr = oof_df.corr()
print(corr.to_string(), file=sys.stderr, flush=True)

# === Simple weighted average (baseline) ===
simple_avg = np.mean(list(test_preds.values()), axis=0)
oof_avg = np.mean(list(oof_preds.values()), axis=0)
auc_avg = roc_auc_score(y, oof_avg)
print(f'\nSimple average OOF AUC: {auc_avg:.6f}', file=sys.stderr, flush=True)

# === Rank average ===
def rank_average(preds_dict):
    ranked = [rankdata(p)/len(p) for p in preds_dict.values()]
    return np.mean(ranked, axis=0)

rank_avg_test = rank_average(test_preds)
rank_avg_oof = rank_average(oof_preds)
auc_rank = roc_auc_score(y, rank_avg_oof)
print(f'Rank average OOF AUC: {auc_rank:.6f}', file=sys.stderr, flush=True)

# === LEVEL 2: Meta-learner (LogisticRegression) ===
print('\n=== LEVEL 2: Stacking with LogisticRegression ===', file=sys.stderr, flush=True)
stack_train = np.column_stack(list(oof_preds.values()))
stack_test = np.column_stack(list(test_preds.values()))

meta_oof = np.zeros(len(train))
meta_test = np.zeros(len(test))

for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    meta = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    meta.fit(stack_train[tr_idx], y.iloc[tr_idx])
    meta_oof[val_idx] = meta.predict_proba(stack_train[val_idx])[:, 1]
    meta_test += meta.predict_proba(stack_test)[:, 1] / N_FOLDS

auc_stack = roc_auc_score(y, meta_oof)
print(f'Stacking OOF AUC: {auc_stack:.6f}', file=sys.stderr, flush=True)

# === Final blend: best of stacking vs rank avg ===
# Try blends
best_blend_auc = 0
best_blend_w = 1.0
for w in np.arange(0, 1.01, 0.05):
    blend_oof = w * meta_oof + (1-w) * rank_avg_oof
    auc_b = roc_auc_score(y, blend_oof)
    if auc_b > best_blend_auc:
        best_blend_auc = auc_b
        best_blend_w = w

print(f'Best blend: {best_blend_w:.2f}*stack + {1-best_blend_w:.2f}*rank = {best_blend_auc:.6f}', file=sys.stderr, flush=True)

final_test = best_blend_w * meta_test + (1-best_blend_w) * rank_avg_test

# === SAVE ===
os.makedirs('submissions', exist_ok=True)
sub['Churn'] = final_test
sub.to_csv(f'submissions/{VERSION}.csv', index=False)

# Also save stacking-only and rank-only versions
sub2 = sub.copy(); sub2['Churn'] = meta_test
sub2.to_csv(f'submissions/{VERSION}_stack.csv', index=False)
sub3 = sub.copy(); sub3['Churn'] = rank_avg_test
sub3.to_csv(f'submissions/{VERSION}_rank.csv', index=False)

results = {
    'version': VERSION,
    'n_models': len(models),
    'model_aucs': {k: round(roc_auc_score(y, v), 6) for k, v in oof_preds.items()},
    'simple_avg_auc': round(auc_avg, 6),
    'rank_avg_auc': round(auc_rank, 6),
    'stacking_auc': round(auc_stack, 6),
    'best_blend_auc': round(best_blend_auc, 6),
    'blend_weight': round(best_blend_w, 2),
    'feats': len(all_features),
}
json.dump(results, open(f'submissions/{VERSION}_results.json', 'w'), indent=2)

print(f'\nSaved: submissions/{VERSION}.csv', file=sys.stderr, flush=True)
print(f'Range: [{final_test.min():.4f}, {final_test.max():.4f}], Mean: {final_test.mean():.4f}', file=sys.stderr, flush=True)
print('DONE!', file=sys.stderr, flush=True)
