from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException
from dotenv import load_dotenv
from typing import Tuple
import pandas as pd 
import numpy as np 
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

    def get_training_data(self, feature_store, name: str):
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
            logging.info("Creating a feature view on hopsworks")
            feature_group = feature_store.get_feature_group(name=name)
            columns_to_query = feature_group.select_all()
            train_feature_view = feature_store.get_feature_view(name=name)
            features_df, label_df = train_feature_view.training_data(
                description="gold_price_prediction_train_data"
            )
            return features_df,  label_df
        except: 
            # create a new feature view if it doesnt exist
            feature_group = feature_store.create_feature_group(name=name)
            columns_to_query =  feature_group.select_except(features=["close"])
            train_feature_view = feature_store.create_feature_view(
                name=name, 
                labels=["close"], 
                version=4, 
                query=columns_to_query
            )

    

    def get_validation_data(self, feature_store, description):
        #test_feature_view = feature_store
        pass 

    def save_model_locally(self):
        pass 

    def register_models_on_hopswork(self):
        pass  
    
    def train(self):
        pass 





    def InitiateModelTrainer(self):
        feature_store = self.Hopswork_project.get_feature_store()
        features, label = self.get_training_data(
            feature_store=feature_store, 
            name = "gold_price_prediction_train_data",
        )
        print(features, label)



@dataclass 
class config: 
    pass 

if __name__ == "__main__":
    params = None 
    modelTrainer = ModelTrainer(ModelTrainerConfig=config, tune_hyperparameters=params)
    modelTrainer.InitiateModelTrainer()