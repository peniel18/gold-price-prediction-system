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
        lr_params = {}
        return lr_params
    elif model_fn == Lasso: 
        lasso_params = {
            
        }