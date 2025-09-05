# src/features.py
#this Python script is a feature engineering module for financial time series data (likely stock prices). It calculates various technical indicators and features from raw price data to prepare it for machine learning models.

import numpy as np
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from src.utils import load_config, load_dataframe, save_dataframe, ensure_dir

def add_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["features"]
    target_col = cfg["features"]["target"]# decide which column to predict
    
    out = df.copy()

    out["Close"] = pd.to_numeric(out["Close"], errors="coerce") 
    out = out.dropna(subset=["Close"])


         # 
    out["sma_5"] = SMAIndicator(close=out["Close"], window=5).sma_indicator()#1 week
    out["sma_10"] = SMAIndicator(close=out["Close"], window=10).sma_indicator()  
    out["ema_12"] = EMAIndicator(close=out["Close"], window=12).ema_indicator() 
    out["ema_26"] = EMAIndicator(close=out["Close"], window=26).ema_indicator()

    # RSI
    out["rsi_14"] = RSIIndicator(close=out["Close"], window=14).rsi()

    # MACD
    macd = MACD(close=out["Close"], window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal() # Trend direction and momentum

    # Bollinger Bands
    bb = BollingerBands(close=out["Close"], window=20, window_dev=2) # for one month(window= 20)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()

    # Simple returns,rolling volatility
    out["returns_1d"] = out["Close"].pct_change()
    out["volatility_14d"] = out["returns_1d"].rolling(14).std() * np.sqrt(252)

    # Target-next-day Close
    out["y_next_close"] = out[target_col].shift(-1) # FOR TOMARROW PREDICTIONS

    out = out.dropna() # final clean data set
    return out
 # in new out data there is all things add as folder (SMA,EMA,tomarrow prediction, volatility, macd
    
    
    
    
def main():
    cfg = load_config()
    ticker = cfg["data"]["ticker"]
    interval = cfg["data"].get("interval", "1d")

    raw_path = f'{cfg["paths"]["raw"]}/{ticker}_{interval}.csv'
    df = load_dataframe(raw_path)

    feat_df = add_features(df, cfg)
    save_path = f'{cfg["paths"]["processed"]}/{ticker}_{interval}_features.csv'
    save_dataframe(feat_df, save_path)
    print(f"[features] Saved: {save_path}  rows={len(feat_df)}  cols={len(feat_df.columns)}")

if __name__ == "__main__":
    main()
