# predict.py(src.PY)

import argparse # for
import joblib
import numpy as np
from src.utils import load_config, load_dataframe, get_scaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days_ahead", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config()
    ticker = cfg["data"]["ticker"]
    interval = cfg["data"].get("interval", "1d")
    feat_path = f'{cfg["paths"]["processed"]}/{ticker}_{interval}_features.csv'
    df = load_dataframe(feat_path)

    feature_cols = joblib.load("models/feature_cols.joblib")
    X_latest = df[feature_cols].iloc[[-1]].values
 # PREDICT TOMAROWS CLOSING PRICE

    try:
        scaler = get_scaler("data/processed/scaler.joblib")
        X_latest = scaler.transform(X_latest)
    except Exception:
        pass

    # model
    model = joblib.load("models/RandomForest.joblib")
    pred = float(model.predict(X_latest)[0])
    print(f"[predict] Predicted next-day Close for {ticker}: {pred:.2f}")

if __name__ == "__main__":
    main()
