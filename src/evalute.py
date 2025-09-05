import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from src.utils import load_config, load_dataframe, get_scaler, plot_predictions

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def main():
    cfg = load_config()
    ticker = cfg["data"]["ticker"]
    interval = cfg["data"].get("interval", "1d")
    feat_path = f'{cfg["paths"]["processed"]}/{ticker}_{interval}_features.csv'
    df = load_dataframe(feat_path)

    y = df["y_next_close"].values
    feature_cols = joblib.load("models/feature_cols.joblib")
    X = df[feature_cols].values

    test_size = int(len(df) * cfg["train"]["test_size"])
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    scaler = None
    try:
        scaler = get_scaler("data/processed/scaler.joblib")
        X_test = scaler.transform(X_test)
    except Exception:
        pass

    for name in ["LinearRegression", "RandomForest"]:
        model = joblib.load(f"models/{name}.joblib")
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mape_val = mape(y_test, preds)
        print(f"[evaluate] {name}  MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape_val:.2f}%")
        plot_predictions(df.index[-test_size:], y_test, preds, title=f"{name}: Pred vs Actual")

if __name__ == "__main__":
    main()
