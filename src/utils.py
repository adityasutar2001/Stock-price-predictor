# utils.py
 # thiis file is reusable in the file functions like reading config,loading CSV,scaling, and plotting are present
    
import os #gives accessto creating folders, handling files.

import yaml
import joblib #  for saving model
import pandas as pd
import matplotlib.pyplot as plt # for visualization to get idea and clearity
from sklearn.preprocessing import StandardScaler


def load_config(path: str = "config.yaml") -> dict:     # Reads config.yml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) # converts into dictionary for easy use

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    # exist_ok=True > it goes silently if folder alreedy exit(okey if alredy exisit)
    # function makes directory from given path 
    
def save_dataframe(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path)) #os.path.dirname> extracts directory from full path
    df.to_csv(path, index=True)
    # save file as csv 

def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)

def get_scaler(path: str):
    return joblib.load(path)# Retrieve scaler later to apply the same scaling to new data.

def save_scaler(scaler: StandardScaler, path: str):
    ensure_dir(os.path.dirname(path))
    joblib.dump(scaler, path)

def plot_predictions(dates, y_true, y_pred, title="Predictions vs Actuals"):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_true, label="Actual")
    plt.plot(dates, y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()
