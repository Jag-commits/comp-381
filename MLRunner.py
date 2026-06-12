import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class MLRunner:
   

    ALGO_CONFIG = {
        "option_1": {"prefix": "rf_supervised", "supervised": True},
        "option_2": {"prefix": "logistical_regression","supervised": True},
    }

    ALGO_LABELS = {
        "option_1": "Random Forest",
        "option_2": "Logistical Regression",
    }

    DATASET_FOLDERS = {
        "Finance Dataset": "finance",
    }

    OUTPUT_PATHS = {
        "Finance Dataset":"anomalies_finance.csv",
    }

    CLEAN_PATHS ={
        "Finance Dataset":"clean_finance.csv"
    }



    @classmethod
    def run(cls, dataset_name: str, algorithm_selection: str, df: pd.DataFrame) -> dict:
        
        try:
            # Check if the dataset is already specified, it only works with the same variables
            if dataset_name not in cls.DATASET_FOLDERS:
                return cls.Error_(f"Unknown dataset: '{dataset_name}'")

            if algorithm_selection not in cls.ALGO_CONFIG:
                return cls.Error_("Please select an algorithm before running.")

            # Modular design to find the saved models
            algo   = cls.ALGO_CONFIG[algorithm_selection]
            prefix = algo["prefix"]
            folder = f"saved_models/{cls.DATASET_FOLDERS[dataset_name]}"

            # Load stored artefacts
            model    = joblib.load(f"{folder}/{prefix}_model.pkl")
            scaler   = joblib.load(f"{folder}/{prefix}_scaler.pkl")
            encoders = joblib.load(f"{folder}/{prefix}_encoders.pkl")

            # Prepare features using the correct pipeline for this dataset
            X, df_out = cls.prepareFinance(df, encoders)

            # Apply stored scaler
            X_scaled = scaler.transform(X)

            # Predict
            raw_preds = model.predict(X_scaled)
            if algo["supervised"]:
                y_pred = raw_preds
            else:
                y_pred = (raw_preds == -1).astype(int)

            df_out = df_out.copy()
            df_out["anomaly"] = y_pred
            clean_df     = df_out[df_out["anomaly"] == 0]
            anomalies_df  = df_out[df_out["anomaly"] == 1]
            
            anomaly_count = len(anomalies_df)
            anomaly_pct   = (anomaly_count / len(df_out)) * 100

            output_path = cls.OUTPUT_PATHS[dataset_name]
            clean_path = cls.CLEAN_PATHS[dataset_name]

            anomalies_df.to_csv(output_path, index=True)
            clean_df.to_csv(clean_path, index=True)
            sample_rows = anomalies_df.head(5).to_string(index=True)

            # Useful information to pull out from the dataset results
            return {
                "algo_label":    cls.ALGO_LABELS[algorithm_selection],
                "anomaly_count": anomaly_count,
                "anomaly_pct":   round(anomaly_pct, 2),
                "output_path":   output_path,
                "sample_rows":   sample_rows,
                "anomalies_df":   anomalies_df,
                "error":         None,
            }

        except Exception as e:
            return cls.Error_(str(e))

    # Helper function to encode the variables properly
    @classmethod
    def prepareFinance(cls, df: pd.DataFrame, encoders: dict):
        feature_cols = ["amount", "transaction_type", "merchant_category", "country", "hour"]
        X_raw = df[feature_cols].copy()

        for col in ["transaction_type", "merchant_category", "country"]:
            le = encoders[col]
            X_raw[col] = le.transform(X_raw[col].astype(str))

        return X_raw, df

    @staticmethod
    def Error_(msg: str) -> dict:
        return {
            "algo_label":    "",
            "anomaly_count": 0,
            "anomaly_pct":   0.0,
            "output_path":   "",
            "sample_rows":   "",
            "anomalies_df": pd.DataFrame(),
            "error":         msg,
        }