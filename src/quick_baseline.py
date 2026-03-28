"""Quick baseline — LightGBM only, fast execution."""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import os, json, warnings
from datetime import datetime

warnings.filterwarnings('ignore')
SEED = 42
N_FOLDS = 5
VERSION = "v01_lgb_quick"

print(f"Loading data...")
train = pd.read_csv("data-kaggle/train.csv")
test = pd.read_csv("data-kaggle/test.csv")
sample_sub = pd.read_csv("data-kaggle/sample_submission.csv")

# Merge original
orig = pd.read_csv("data-original/WA_Fn-UseC_-Telco-Customer-Churn.csv")
orig['Churn'] = orig['Churn'].map({'Yes': 1, 'No': 0})
orig = orig.drop('customerID', axis=1)
orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce')
common = [c for c in train.columns if c in orig.columns]
orig_a = orig[common].copy()
for c in train.columns:
    if c not in orig_a.columns:
        orig_a[c] = np.nan
train = pd.concat([train, orig_a[train.columns]], ignore_index=True)

# Prep
test_ids = test['id'].copy()
train.drop('id', axis=1, errors='ignore', inplace=True)
test.drop('id', axis=1, errors='ignore', inplace=True)
y = train['Churn'].map(lambda x: 1 if str(x) in ('Yes','1','1.0') else 0).astype(int)
train.drop('Churn', axis=1, inplace=True)

print(f"Train: {train.shape}, Test: {test.shape}, Churn: {y.mean():.4f}")

# Feature engineering
def fe(df):
    df = df.copy()
    if df['TotalCharges'].dtype == object:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['TenureProxy'] = df['TotalCharges'] / (df['MonthlyCharges'] + 0.01)
    df['ChargesTenure'] = df['MonthlyCharges'] * df['tenure']

    svc = ['PhoneService','MultipleLines','InternetService','OnlineSecurity',
           'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['NumServices'] = sum((df[c].astype(str)=='Yes').astype(int) for c in svc if c in df.columns)

    if 'tenure' in df.columns:
        df['TenureBin'] = pd.cut(df['tenure'], bins=[-1,6,12,24,48,72,999], labels=[0,1,2,3,4,5]).astype(float)
    if 'MonthlyCharges' in df.columns:
        df['ChargesBin'] = pd.cut(df['MonthlyCharges'], bins=[-1,20,40,60,80,100,999], labels=[0,1,2,3,4,5]).astype(float)

    df['HasInternet'] = (df['InternetService'].astype(str)!='No').astype(int)
    df['HasFiber'] = (df['InternetService'].astype(str)=='Fiber optic').astype(int)
    df['IsMonthToMonth'] = (df['Contract'].astype(str)=='Month-to-month').astype(int)
    df['IsElectronicCheck'] = (df['PaymentMethod'].astype(str)=='Electronic check').astype(int)
    df['tenure_x_monthly'] = df['tenure'] * df['MonthlyCharges']
    df['MTM_x_tenure'] = df['IsMonthToMonth'] * df['tenure']

    return df

train = fe(train)
test = fe(test)

# Label encode
cat_cols = train.select_dtypes(include=['object','category']).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]]).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

train.fillna(-999, inplace=True)
test.fillna(-999, inplace=True)

print(f"Features: {train.shape[1]}")

# Train LightGBM
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
test_pred = np.zeros(len(test))

params = dict(objective='binary', metric='auc', boosting_type='gbdt',
              learning_rate=0.05, num_leaves=31, min_child_samples=20,
              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
              n_estimators=2000, random_state=SEED, verbose=-1, n_jobs=-1)

for fold, (tr, val) in enumerate(skf.split(train, y)):
    print(f"Fold {fold+1}...", end=" ", flush=True)
    m = lgb.LGBMClassifier(**params)
    m.fit(train.iloc[tr], y.iloc[tr], eval_set=[(train.iloc[val], y.iloc[val])],
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof[val] = m.predict_proba(train.iloc[val])[:, 1]
    test_pred += m.predict_proba(test)[:, 1] / N_FOLDS
    print(f"AUC={roc_auc_score(y.iloc[val], oof[val]):.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"\nOOF AUC: {cv_auc:.6f}")

# Save
os.makedirs("submissions", exist_ok=True)
sub = sample_sub.copy()
sub['Churn'] = test_pred
sub.to_csv(f"submissions/{VERSION}.csv", index=False)
print(f"Saved: submissions/{VERSION}.csv")
print(f"Range: [{test_pred.min():.4f}, {test_pred.max():.4f}], Mean: {test_pred.mean():.4f}")

# Feature importance
imp = pd.DataFrame({'feature': train.columns, 'importance': m.feature_importances_})
imp = imp.sort_values('importance', ascending=False)
print(f"\nTop 15 features:")
print(imp.head(15).to_string(index=False))

json.dump({'version': VERSION, 'cv_auc': round(cv_auc, 6), 'n_features': train.shape[1],
           'timestamp': datetime.now().isoformat()},
          open(f"submissions/{VERSION}_results.json", 'w'), indent=2)
print("\nDONE!")
