# src/lstm_model.py
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from utils import load_config, load_dataframe

def make_sequences(X, y, window=30):
    Xs, ys = [], []
    for i in range(len(X)-window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

def train_lstm():
    cfg = load_config()
    ticker = cfg["data"]["ticker"]
    interval = cfg["data"].get("interval", "1d")
    feat_path = f'{cfg["paths"]["processed"]}/{ticker}_{interval}_features.csv'
    df = load_dataframe(feat_path)

    feature_cols = [c for c in df.columns if c not in ["y_next_close"]]
    X = df[feature_cols].values
    y = df["y_next_close"].values

    # scale features for neural net
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    window = 30
    Xs, ys = make_sequences(X, y, window=window)

    test_size = int(len(ys) * cfg["train"]["test_size"])
    X_train, X_test = Xs[:-test_size], Xs[-test_size:]
    y_train, y_test = ys[:-test_size], ys[-test_size:]

    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(window, X_train.shape[-1])),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mae")
    cb = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[cb], verbose=1)

    model.save("models/LSTM.keras")
    joblib.dump(feature_cols, "models/feature_cols.joblib")
    joblib.dump(scaler, "data/processed/scaler_lstm.joblib")
    print("[lstm] Saved model and artifacts.")

if __name__ == "__main__":
    train_lstm()
