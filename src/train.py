import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score
import warnings
import os
import joblib
warnings.filterwarnings('ignore')


def load_data(file_path):
    data = pd.read_csv(file_path)
    print(f"✓ Loaded {len(data):,} transactions from {file_path}")
    return data

def eda(data):
    print(data.head())
    data.info()
    
    missing = data.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")
    
    print(data.describe())
    
    counts = data['isFraud'].value_counts()
    print(counts)
    print(f"Fraud Rate: {counts[1]/len(data)*100:.2f}% | Imbalance Ratio: 1:{counts[0]//counts[1]}")
    
    fraud_by_type = data.groupby('type')['isFraud'].agg(['sum','count','mean'])
    fraud_by_type['mean'] *= 100
    fraud_by_type = fraud_by_type.rename(columns={'sum':'Fraud_Count','count':'Total_Count','mean':'Fraud_Rate'}).sort_values('Fraud_Rate', ascending=False)
    print(fraud_by_type)
    
    return data

def create_fraud_features(df):
    df = df.copy()
    
    # Balance diff
    df["diffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["diffDest"] = df["oldbalanceDest"] - df["newbalanceDest"]
    
    # Zero balance indicators
    df['is_balance_zeroed'] = (df['newbalanceOrig'] == 0).astype(int)
    df['dest_balance_zeroed'] = (df['newbalanceDest'] == 0).astype(int)
    
    # Exact transfer
    df['is_exact_balance_transfer'] = (df['amount'] == df['oldbalanceOrg']).astype(int)
    
    # Amount ratios
    df['amount_to_oldbalance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['amount_to_newbalance_ratio'] = df['amount'] / (df['newbalanceOrig'] + 1)
    
    # Round number detection
    df['is_round_100k'] = (df['amount'] % 100000 == 0).astype(int)
    df['is_round_50k'] = (df['amount'] % 50000 == 0).astype(int)
    df['is_round_10k'] = (df['amount'] % 10000 == 0).astype(int)
    
    # Destination balance pattern
    df['dest_balance_increase_ratio'] = (df['newbalanceDest'] - df['oldbalanceDest']) / (df['oldbalanceDest'] + 1)
    
    # One-hot type
    df = pd.concat([df, pd.get_dummies(df['type'], prefix='type')], axis=1)
    
    # Interaction features
    df['cashout_and_zeroed'] = df.get('type_CASH_OUT',0) * df['is_balance_zeroed']
    df['transfer_and_zeroed'] = df.get('type_TRANSFER',0) * df['is_balance_zeroed']
    df['cashout_and_exact'] = df.get('type_CASH_OUT',0) * df['is_exact_balance_transfer']
    
    # Time-based
    df['is_early_transaction'] = (df['step'] < 100).astype(int)
    df['is_late_transaction'] = (df['step'] > 600).astype(int)
    
    # Original zero balance
    df['orig_had_zero_balance'] = (df['oldbalanceOrg'] == 0).astype(int)
    df['dest_had_zero_balance'] = (df['oldbalanceDest'] == 0).astype(int)
    
    print(f"✓ Created {df.shape[1]-5} fraud features")  # assuming first 5 are original cols
    return df

def preprocess(df, scaler=None, is_train=True):
    df = df.drop(columns=["nameOrig","nameDest","isFlaggedFraud"], errors='ignore')
    df = create_fraud_features(df)
    df = df.drop(columns=['type'], errors='ignore')
    
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]
    
    feature_names = X.columns.tolist()
    
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("✓ Scaler fitted on training data")
    else:
        X_scaled = scaler.transform(X)
        print("✓ Test/val data scaled")
    
    return X_scaled, y, scaler, feature_names

def train_models(X_train, y_train):
    models = {}
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    print("✓ Random Forest trained")
    return models

def evaluate_models(models, X_test, y_test, feature_names):
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:,1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        y_pred = (y_prob >= 0.5).astype(int)
        
        print(f"\n=== {name} Metrics ===")
        print(f"PR-AUC: {pr_auc:.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    
    print("\n=== CREATED FEATURES ===")
    print(feature_names)

if __name__ == "__main__":
    data = load_data("data.csv")
    data = eda(data)
    
    train_df, test_df = train_test_split(data, test_size=0.2, stratify=data['isFraud'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['isFraud'], random_state=42)
    
    X_train, y_train, scaler, feature_names = preprocess(train_df, is_train=True)
    X_val, y_val, _, _ = preprocess(val_df, scaler=scaler, is_train=False)
    X_test, y_test, _, _ = preprocess(test_df, scaler=scaler, is_train=False)
    
    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test, feature_names)

    os.makedirs("model", exist_ok=True)
    joblib.dump(models['RandomForest'], "model/rf_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(feature_names, "model/feature_names.pkl")
    print("✓ Feature names saved to model/feature_names.pkl")
    print("✓ Random Forest model saved to model/rf_model.pkl")
    print("✓ Scaler saved to model/scaler.pkl")