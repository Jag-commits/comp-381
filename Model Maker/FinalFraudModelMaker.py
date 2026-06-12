import joblib
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def load_and_prepare(filepath: str):
    """
    Loads CSV, encodes categoricals, scales all features.

    Features used:
        amount, transaction_type, merchant_category, country, hour

    Excluded:
        transaction_id, user_id - no point, do nothing
        device_risk_score -  synthetic artifact that guarantees 100% accuracy
        ip_risk_score - synthetic artifact that guarantees 100% accuracy

    Returns:
        X_scaled: scaled feature array (n_samples, n_features)
        y: ground truth labels - used only for evaluation
        df_raw: original DataFrame - needed for contextual detector
    """
    df = pd.read_csv(filepath)
    y  = df["is_fraud"].values

    feature_cols = ["amount", "transaction_type", "merchant_category", "country", "hour"]
    X_raw = df[feature_cols].copy()

    encoders = {}
    for col in ["transaction_type", "merchant_category", "country"]:
        le = LabelEncoder()
        X_raw[col] = le.fit_transform(X_raw[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, y, df, scaler, encoders


def build_splits(X, y, test_size=0.25, random_state=42):
    
    all_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=random_state, stratify=y
    )
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx



def _report(name: str, anomaly_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Prints metrics and returns a result dictionary."""
    acc  = accuracy_score(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred)
    rep  = classification_report(y_true, y_pred,
                                 target_names=["Normal", "Anomaly"],
                                 zero_division=0)
    f1   = f1_score(y_true, y_pred,        average="binary", zero_division=0)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec  = recall_score(y_true, y_pred,    average="binary", zero_division=0)

    print(f"\n{'='*58}")
    print(f"  [{anomaly_type}]  {name}")
    print(f"{'='*58}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (anomaly class)")
    print(f"  Recall    : {rec:.4f}  (anomaly class)")
    print(f"  F1 Score  : {f1:.4f}  (anomaly class)")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                 Normal  Anomaly")
    print(f"  Actual Normal  {cm[0,0]:>6}  {cm[0,1]:>7}")
    print(f"  Actual Anomaly {cm[1,0]:>6}  {cm[1,1]:>7}")
    print(f"\n  Classification Report:\n{rep}")

    return {
        "name": name, "type": anomaly_type,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "confusion_matrix": cm,
    }


def run_logistic_regression(X_train, X_test, y_train, y_test) -> dict:
    
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    _report("Logistic Regression (Supervised)", "SUPERVISED", y_test, y_pred)
    return model

def run_supervised_rf(X_train, X_test, y_train, y_test) -> dict:
    """
    Supervised Random Forest.
    Trained on labelled data (75%), tested on 25%.
    class_weight='balanced' for the 95/5 class imbalance
    """
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    _report("Random Forest (Supervised)", "SUPERVISED", y_test, y_pred)
    return model



def save_artifacts(name, model, scaler, encoders, folder="saved_models"):
    os.makedirs(folder, exist_ok=True)

    joblib.dump(model, f"{folder}/{name}_model.pkl")
    joblib.dump(scaler, f"{folder}/{name}_scaler.pkl")
    joblib.dump(encoders, f"{folder}/{name}_encoders.pkl")

    print(f"Saved {name} artifacts to '{folder}/'")


if __name__ == "__main__":
    DATA_PATH = r"Model Maker\Dataset\synthetic_fraud_dataset.csv"

    print("Wait a few minutes, it's loading the data")
    X, y, df_raw, scaler, encoders = load_and_prepare(DATA_PATH)
    X_train, X_test, y_train, y_test, train_idx, test_idx = build_splits(X, y)
    
    

    # Supervised RF
    rf_model = run_supervised_rf(X_train, X_test, y_train, y_test)
    save_artifacts("rf_supervised", rf_model, scaler, encoders)

    logistical_regression_model = run_logistic_regression(X_train, X_test, y_train, y_test)
    save_artifacts("logistical_regression", logistical_regression_model, scaler, encoders)
    



    print(f"Split  : 75% train / 25% test (stratified)")
    print(f"Train  : {len(y_train)} rows  |  Test : {len(y_test)} rows")
    print(f"Fraud in test : {y_test.sum()} / {len(y_test)} ({y_test.mean()*100:.1f}%)\n")

    #results = []

   
    #results.append(run_supervised_rf(X_train, X_test, y_train, y_test))
    #results.append(run_logistic_regression(X_train, X_test, y_train, y_test))