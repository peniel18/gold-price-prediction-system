from gold_prediction.logging.logger import logging 
from gold_prediction.exception.exception import CustomException 
from gold_prediction.utils.utility_functions import ParametersTracker
import pandas as pd 
import optuna 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import Optional, Union 
from sklearn.model_selection import TimeSeriesSplit
import numpy as np 
import sys 
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

    Args: 
        model_fn: The model class to get parameters for 
        trial: Optuna Trial object for parameter suggestion 

    Returns: 
        dict: the parameters to be optimised 
    
    """

    if model_fn == LinearRegression: 
        return {}
    
    elif model_fn == Lasso: 
        return {} # add alpha 


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
    



def optimise_hyperparameters(
        model_fn: Optional[Union[
            LinearRegression, 
            Lasso, 
            DecisionTreeRegressor, 
            RandomForestRegressor, 
            XGBRegressor
        ]], 
        num_of_trials, 
        X: pd.DataFrame, 
        y: pd.Series

)-> dict[str, str | int | float ]:
    models_and_tags: dict[callable, str] = {
       LinearRegression: "LinearRegression", 
        Lasso: "lasso",
        XGBRegressor: "XGBoost", 
        DecisionTreeRegressor: "DescisionTreeRegressor", 
        RandomForestRegressor: "RandomForest" 

    }
    assert model_fn in models_and_tags.keys()
    model_name = models_and_tags[model_fn]



    def objective(trial: optuna.trial.Trial) -> float:
        try:
            logging.info(f"Tuning Hyperparameters of model: {model_name}")
            hyperparameters = get_parameters(model_fn=model_fn, trial=trial)        
            tss = TimeSeriesSplit(n_split=5)
            model = model_fn(**hyperparameters)
            error_scores = []

            for train_idx, val_idx in tss.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train) 
                yHat = model.predict(X_val)
                error = mean_squared_error(y_true=y_val, y_pred=yHat)
                error_scores.append(error)

            avg_score = np.mean(error_scores)
            return avg_score
        except Exception as e: 
            logging.error(f"Error occurred for the objective function")
            raise CustomException(e, sys)
        


    study = optuna.create_study(study_name="study", direction="minimize")
    study.optimize(func=objective, n_trials=num_of_trials)
    best_hyperparameters = study.best_params
    best_metric = study.best_value


    # log params and experiments to mlflow 



    return best_hyperparameters

    
