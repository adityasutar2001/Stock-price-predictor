# src/train.py
import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from src.utils import load_config, load_dataframe, ensure_dir, save_scaler
from src.model import get_models
import joblib

# IMPORT ALL THING FOR TRAINING 



def main():
    cfg = load_config()
    ticker = cfg["data"]["ticker"]
    interval = cfg["data"].get("interval", "1d")
    feat_path = f'{cfg["paths"]["processed"]}/{ticker}_{interval}_features.csv'
    df = load_dataframe(feat_path)

 # Features and target
    y = df["y_next_close"].values
    feature_cols = [c for c in df.columns if c not in ["y_next_close"]]
    X = df[feature_cols].values

  # Train-test split by time
    test_size = int(len(df) * cfg["train"]["test_size"])
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

      # Scaling (only features)
    scaler = None
    if cfg["train"].get("scale_features", True):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        save_scaler(scaler, "data/processed/scaler.joblib")

    ensure_dir("models")

 # Train each model
    for mb in get_models():
        mb.model.fit(X_train, y_train)
        preds = mb.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        joblib.dump(mb.model, f"models/{mb.name}.joblib")
        print(f"[train] {mb.name}: MAE={mae:.4f}  saved -> models/{mb.name}.joblib")

# Save columns order for inference
    joblib.dump(feature_cols, "models/feature_cols.joblib")
    print("[train] Saved feature columns.")

if __name__ == "__main__":
    main()
