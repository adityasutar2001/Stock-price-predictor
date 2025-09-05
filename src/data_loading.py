# src.data_loading.py





import yfinance as yf #data
from src.utils import load_config, ensure_dir, save_dataframe

# this code handles 
def main():
    cfg = load_config()
    raw_dir = cfg["paths"]["raw"]
    ensure_dir(raw_dir)

    ticker = cfg["data"]["ticker"]
    start = cfg["data"]["start"]
    end = cfg["data"]["end"]
    interval = cfg["data"].get("interval", "1d")

    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}. Check symbol or dates.")

    save_dataframe(df, f"{raw_dir}/{ticker}_{interval}.csv")
    print(f"[data_loading] Saved: {raw_dir}/{ticker}_{interval}.csv  rows={len(df)}")

if __name__ == "__main__":
    main()
