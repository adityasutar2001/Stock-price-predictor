# src.model.py
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

@dataclass
class ModelBundle:
    name: str
    model: object

def get_models():
    return [
        ModelBundle(name="LinearRegression", model=LinearRegression()),
        ModelBundle(name="RandomForest", model=RandomForestRegressor(n_estimators=400, max_depth=8, random_state=42)),
    ]
 # Function that returns both models for training process. random forest and linear regression