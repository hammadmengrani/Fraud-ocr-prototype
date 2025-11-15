Mini-Gateway Fraud & OCR Prototype

1. Project Overview

This project is a prototype for a mini-gateway, combining a real-time fraud detection model with an OCR pipeline for receipt verification.

The primary deliverables include:

A fraud detection model trained on transaction data.

An OCR pipeline to extract data from receipt images.

A REST API (FastAPI) to serve the model and OCR logic.

A Dockerized deployment for the service.

2. Dataset & Feature Engineering

The model was trained on a transactional dataset.

Base Features: The model uses base features like transaction type, amount, and balances (oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest).

Engineered Features: To capture the actual fraud signals, new features were created, as the base features alone are not predictive. These were critical for the model's high performance:

diffOrig: The difference between the original balance and the new balance for the originator (oldbalanceOrg - newbalanceOrig).

diffDest: The difference for the destination (oldbalanceDest - newbalanceDest).

is_balance_zeroed: A flag (1 or 0) if the originator's account was emptied (newbalanceOrig == 0).

amount_to_oldbalance_ratio: The ratio of the transaction amount to the oldbalanceOrg.

cashout_and_zeroed: An interaction feature that flags (1) if the transaction is CASH_OUT and the originator's balance was zeroed. This is a very strong fraud indicator.

3. Model Training

Several models were evaluated to handle the non-linear fraud patterns and severe class imbalance (1:800).

Models Tested: Logistic Regression, XGBoost, LightGBM (LGB), and Random Forest (RF).

Justification: While most models (except LightGBM) achieved similar high PR-AUC scores (~0.99) due to the strong engineered features, Random Forest was selected as the final model. The choice was based on its robustness in handling non-linear patterns (which our features rely on) and its simple, effective method for managing class imbalance (via the class_weight='balanced' parameter).

Saved Artifacts: The training pipeline (src/train.py) produces three essential artifacts, which are stored in the /models directory:

rf_model.pkl: The trained Random Forest classifier.

scaler.pkl: The StandardScaler object (for transforming live data).

feature_names.pkl: The list of feature names (to ensure column order).

4. OCR Pipeline

The OCR pipeline is implemented in src/ocr_pipeline.py.

OCR Choice: It uses the external ocr.space API, which handles image processing tasks like auto-deskewing and text extraction. A simple heuristic is used to find the merchant name (prioritizing early, non-numeric lines), and a regex pattern is used to find the total amount.

Data & Key Access: The dataset (data.csv) and a pre-configured .env file (containing the OCR_API_KEY) are provided in the following Google Drive link:

[https://drive.google.com/file/d/1bZOVmLkiizD3ijlmXt3X-TcBChouGd4y/view?usp=sharing]

Dependency: This pipeline requires a valid API key. You must create a .env file in the project root with your key (or use the one from the Google Drive link):

# .env file
OCR_API_KEY=YOUR_API_KEY_HERE


5. REST API

The service is exposed via a FastAPI app defined in src/app.py.

Endpoint: POST /score

Input JSON:

{
  "transaction": {
    "step": 309,
    "type": "CASH_OUT",
    "amount": 3490916.0,
    "nameOrig": "C1526528147",
    "oldbalanceOrg": 3490916.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C138172146",
    "oldbalanceDest": 112756.43,
    "newbalanceDest": 3603672.42
  },
  "receipt_path": "./images/image_1.jpg"
}


Output JSON:

{
  "fraud_score": 0.9987,
  "merchant_name": "Example Merchant",
  "total": 3490916.0
}


6. Docker Setup

Follow these steps in order to run the service.

Step 1: Run Local Training (One-time Setup)

This is a mandatory first step. The Docker container expects the trained model files (.pkl) to exist before building. Run the training script locally one time. This will also allow you to verify the model metrics (see Section 7).

# This will create the /models directory with .pkl files
python src/train.py


Step 2: Run Docker Service

Now that the models/ folder exists, you can build and run the service.

Run with Docker Compose (Recommended)

This method automatically builds the image (copying the models/ folder) and runs the container.

# Build and run the service in detached mode
docker-compose up --build -d


Run Manually (Alternative)

# 1. Build the image (Ensure models/ exists first!)
docker build -t fraud-ocr-api .

# 2. Run the image, passing the .env file
docker run -p 8000:8000 --env-file .env -d fraud-ocr-api


The service will be available at http://localhost:8000.

Note: If the OCR_API_KEY is missing or invalid, the service will still run. The fraud_score will be calculated, but merchant_name and total will return null or placeholder values.

7. Evaluation Metrics

PR-AUC: 0.9992 (achieved by Random Forest).

Precision (Test Set): 0.9994

Recall (Test Set): 0.9970

F1-Score (Test Set): 0.9982

Reproducing Metrics

The metrics above represent the model's performance on the test split. To reproduce these metrics, simply run the training script as described in Step 1 of the Docker Setup:

python src/train.py


The script will print the final PR-AUC, Precision, Recall, and F1-Score to the console.