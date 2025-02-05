from gold_prediction.logging.logger import logging
from gold_prediction.components.hyperparameter_tuning import get_parameters, optimise_hyperparameters
from gold_prediction.exception.exception import CustomException
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Optional, Type, List
from sklearn.metrics import mean_squared_error
import pandas as pd 
import numpy as np
from xgboost import XGBRegressor
import hopsworks 
from dataclasses import dataclass
import os



load_dotenv()



class ModelTrainer: 
    def __init__(self, 
                 ModelTrainerConfig, 
                 tune_hyperparameters: bool | None = True):
        self.ModelTrainerConfg = ModelTrainerConfig
        self.HOPSWORKS_API = os.getenv("HOPSWORKS_API_KEY")
        self.Hopswork_project = hopsworks.login(
            api_key_value = self.HOPSWORKS_API
        )
        self.tune_hyperparameters = tune_hyperparameters

    def get_training_data(self, feature_store, name: str, description) -> pd.DataFrame:
        """
         Retrieve training data from feature store and split into train/validation sets.
    
        Args:
        feature_store: The feature store instance
            name: Name of the feature group
            descripiton: Description of the feature view 
        
        Returns:
            tuple: (X_train, y_train, X_valid, y_valid) arrays
        
        """
        try: 
            # get a feature view that already exist
            logging.info("Get Feature View on Hopsworks ")
            feature_group = feature_store.get_feature_group(name=name)
            feature_view = feature_store.get_feature_view(name=description)
            data = feature_view.training_data(
                description=description, 
                #version=1
            )
            return data
        except: 
            # create a new feature view if it doesnt exist
            logging.info("Create a new Feature View on hopsworks")
            feature_group = feature_store.get_feature_group(name=name)
            columns_to_query =  feature_group.select_all()
            feature_view = feature_store.create_feature_view(
                name=description, 
                #labels=["close"], 
                #version=4, 
                query=columns_to_query
            )
            data = feature_view.training_data(
                description=description
            )
            return data

    def save_model_locally(self):
        pass 

    def register_models_on_hopswork(self):
        pass  

    def get_model(self, model_name: str) -> LinearRegression | Lasso | XGBRegressor | DecisionTreeRegressor | RandomForestRegressor:
        """
        Args: 
            model_name: name of the model 
        
        Returns: 
            A model function 
        """
        models = {
            "LinearRegression": LinearRegression, 
            "lasso": Lasso,
            "XGBoost": XGBRegressor, 
            "DescisionTreeRegressor": DecisionTreeRegressor, 
            "RandomForest" : RandomForestRegressor
        }

        if model_name.lower() in models.keys():
            return models[model_name.lower()]
        else: 
            raise KeyError(f"Model {model_name} is not available")



    def PrepareTrainingData(self, data: Tuple[pd.DataFrame]) -> pd.DataFrame:
        df = data[0]
        # features 
        columns = ['close', 'agg_mean', 'agg_max', 'agg_std', 'agg_min', 'kurt',
                     'skewness', 'month', 'year', 'day', 'dayofweek', 'is_weekend', 'dayofyear', 'quarter']
        #columns = ["close", "month", "year", "day", "dayOfweek", "is_weekend", "dayOfweek", "dayOfyear", "quarter"]
        ds = df[columns]
    

        return ds 
 


    def train(self, model_name: str):
        model_fn = self.get_model(model_name=model_name)
        # get data hopsworks
        feature_store = self.Hopswork_project.get_feature_store()
        ds = self.get_training_data(
            feature_store=feature_store, 
            name = "gold_prediction_train_data",
            description="gold_train_fv"
        )   
        ds = self.PrepareTrainingData(ds)
        features = list(ds.columns)
        print(features)
        features.remove("close")
    
        target = "close"

        # time series split 
        tss = TimeSeriesSplit(n_splits=5)
        ds.sort_index()

        # train 

        if not self.tune_hyperparameters:
            fold = 0 
            preds = []
            scores = []
            logging.info("Training model with default parameters")
            for train_idx, val_idx in tss.split(ds):
                train = ds.iloc[train_idx]
                test = ds.iloc[val_idx]

                X_train, y_train = train[features], train[target]
                X_val, y_val = test[features] , test[target]

                model = model_fn() 
                model.fit(X_train, y_train)
                yHat = model.predict(X_val)
                preds.append(yHat)
                errors = mean_squared_error(y_val, yHat)
                scores.append(errors)

            # track preds and scores during training 
            print(np.average(scores))
            print(preds)
        else: 
            logging(f"Tuning parameters of {model_name}")
            
            #tuned_model_parameters = optimise_hyperparameter()
            model_hyperparameters = optimise_hyperparameters(
                model_fn=model_fn, 
                num_of_trials=None, 
                X=X_train, 
                y=y_val
            )




        


    def InitiateModelTrainer(self):
        self.train(model_name="lasso")


@dataclass 
class config: 
    pass 

if __name__ == "__main__":
    params = None 
    modelTrainer = ModelTrainer(ModelTrainerConfig=config, tune_hyperparameters=params)
    modelTrainer.InitiateModelTrainer()