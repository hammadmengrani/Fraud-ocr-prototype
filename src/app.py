from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import joblib
from ocr_pipeline import get_receipt_data
from train import create_fraud_features

app = FastAPI(title="Fraud + OCR API")

rf_model = joblib.load("model/rf_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")  

class Transaction(BaseModel):
    transaction: dict
    receipt_path: str

@app.post("/score")
def score(transaction: Transaction):
    try:
        df = pd.DataFrame([transaction.transaction])
        df = df.drop(columns=["isFlaggedFraud"], errors='ignore')
        df = create_fraud_features(df)
        df = df.drop(columns=['nameOrig', 'nameDest', 'type', 'isFraud'], errors='ignore')

        # Ensure all training features exist
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns exactly as in training
        df = df[feature_names]

        # Scale features
        X_scaled = scaler.transform(df)

        # Fraud prediction
        fraud_score = rf_model.predict_proba(X_scaled)[:,1][0]

        # OCR extraction
        if os.path.exists(transaction.receipt_path):
            merchant_name, total = get_receipt_data(transaction.receipt_path)
        else:
            merchant_name, total = "Unknown", None

        return {
            "fraud_score": round(float(fraud_score), 4),
            "merchant_name": merchant_name,
            "total": total
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing transaction: {str(e)}")
