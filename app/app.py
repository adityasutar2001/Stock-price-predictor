# app/app.py
from flask import Flask, render_template_string
import joblib
from utils import load_config, load_dataframe, get_scaler

HTML = """
<!doctype html>
<title>Stock Price Predictor</title>
<div style="max-width:700px;margin:40px auto;font-family:system-ui;">
  <h2>Stock Price Predictor</h2>
  <p>Model: RandomForest (change in code if you prefer LR)</p>
  <pre>{{ msg }}</pre>
</div>
"""

app = Flask(__name__)

@app.route("/")
def index():
    cfg = load_config()
    ticker = cfg["data"]["ticker"]
    interval = cfg["data"].get("interval", "1d")
    feat_path = f'{cfg["paths"]["processed"]}/{ticker}_{interval}_features.csv'
    df = load_dataframe(feat_path)

    feature_cols = joblib.load("models/feature_cols.joblib")
    X_latest = df[feature_cols].iloc[[-1]].values

    try:
        scaler = get_scaler("data/processed/scaler.joblib")
        X_latest = scaler.transform(X_latest)
    except Exception:
        pass

    model = joblib.load("models/RandomForest.joblib")
    pred = float(model.predict(X_latest)[0])
    msg = f"Ticker: {ticker}\nLatest features date: {df.index[-1].date()}\nPredicted next-day Close: {pred:.2f}"
    return render_template_string(HTML, msg=msg)

if __name__ == "__main__":
    app.run(debug=True)
#Simple Flask web app.  