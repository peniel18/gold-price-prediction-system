from gold_prediction.logging.logger import logging 
from gold_prediction.exception.exception import CustomException 
import pandas as pd 
import optuna 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import Optional, Union 
from xgboost import XGBRegressor 




def get_parameters(
    model_fn: Optional[Union[
        LinearRegression,
        Lasso, 
        XGBRegressor,
        RandomForestRegressor,
        DecisionTreeRegressor
    ]],
    trial: optuna.trial.Trial
) -> dict[str, str | int | float ]:
    """
    Defines a range of parameters for a specific model 

    Returns: 
        dict: the parameters to be optimised 
    
    """

    if model_fn == LinearRegression: 
        return {}
    
    elif model_fn == Lasso: 
        return {}


    elif model_fn == RandomForestRegressor:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4)
        }

    elif model_fn == DecisionTreeRegressor:
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4)
        }

    elif model_fn == XGBRegressor:
        return {
            "objective": "reg:absoluteerror",
            "eta": trial.suggest_float("eta", 0.1, 1),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "alpha": trial.suggest_float("alpha", 0, 2),
            "subsample": trial.suggest_int("subsample", 0.1, 1)
        }

    else: 
        raise ValueError(f"Parameters not defined for model type: {model_fn}")