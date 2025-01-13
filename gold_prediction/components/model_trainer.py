from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException
from dotenv import load_dotenv
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

    def get_training_data(self, feature_store, name: str) -> None:

        try: 
            columns_to_query = feature_store.select_all()
            print(columns_to_query)
            logging.info("Getting Data from Hopsoworks")
            train_feature_view = feature_store.get_feature_view(
                name=None, 
                labels=["Close"], 
                query = columns_to_query
            )
            X_train, y_train, X_valid, y_vali = feature_store.train_test_spilt(test_size=0.2)
        except: 
            columns_to_query = feature_store.select_all()
            train_feature_view = feature_store.create_feature_view(
                name=name, 
                labels=["Close"], 
                query=columns_to_query
            )
        print(X_train)


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
        self.get_training_data(
            feature_store=feature_store, 
            name = "gold_price_prediction_train_data",
        )



@dataclass 
class config: 
    pass 

if __name__ == "__main__":
    params = None 
    modelTrainer = ModelTrainer(ModelTrainerConfig=config, tune_hyperparameters=params)
    modelTrainer.InitiateModelTrainer()