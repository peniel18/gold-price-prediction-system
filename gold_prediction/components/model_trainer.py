from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from typing import Tuple, OPtional
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

    def get_training_data(self, feature_store, name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
         Retrieve training data from feature store and split into train/validation sets.
    
        Args:
        feature_store: The feature store instance
            name: Name of the feature group
        
        Returns:
            tuple: (X_train, y_train, X_valid, y_valid) arrays
        
        
        """
        try: 
            # get a feature view that already exist
            logging.info("Get Feature View on Hopsworks ")
            feature_group = feature_store.get_feature_group(name=name)
            feature_view = feature_store.get_feature_view(name=name)
            features_df, label_df = feature_view.training_data(
                description=name 
            )
            return features_df,  label_df
        except: 
            # create a new feature view if it doesnt exist
            logging.info("Create a new Feature View on hopsworks")
            feature_group = feature_store.create_feature_group(name=name)
            columns_to_query =  feature_group.select_all()
            feature_view = feature_store.create_feature_view(
                name=name, 
                labels=["close"], 
               # version=4, 
                query=columns_to_query
            )
            features_df, label_df = feature_view.training_data(
                description=name
            )
            return features_df,  label_df

    

    
    def save_model_locally(self):
        pass 

    def register_models_on_hopswork(self):
        pass  

    def get_model(self, model_name: str) -> LinearRegression | Lasso | XGBRegressor | DecisionTreeRegressor | RandomForestRegressor:
        """
        Args: 
            model_name: name of the model 
        
        Returns: 
            Optional
        """
        models = {
            "Linear Regression": LinearRegression, 
            "lasso": Lasso,
            "XGBoost": XGBRegressor, 
            "DescisionTreeRegressor": DecisionTreeRegressor, 
            "RandomForest" : RandomForestRegressor
        }

        if model_name.lower() in models.keys():
            return models[model_name.lower()]
        else: 
            raise KeyError(f"Model {model_name} is not available")


    
    def train(self):
        model_fn = self.get_model()
        train_features, train_label = self.get_training_data(

        )

        





    def InitiateModelTrainer(self):
        feature_store = self.Hopswork_project.get_feature_store()
        features, label = self.get_training_data(
            feature_store=feature_store, 
            name = "gold_price_prediction_train_data",
        )
        print(features, label)
        print(type(features))



@dataclass 
class config: 
    pass 

if __name__ == "__main__":
    params = None 
    modelTrainer = ModelTrainer(ModelTrainerConfig=config, tune_hyperparameters=params)
    modelTrainer.InitiateModelTrainer()