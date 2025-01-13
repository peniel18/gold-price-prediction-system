from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException
from dotenv import load_dotenv
import pandas as pd 
import numpy as np 
import hopsworks 
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
            logging.info("Getting Data from Hopsoworks")
            train_feature_view = feature_store.get_feature_view(
                name=None, 
                labels=["Close"], 
            )
            X_train, y_train, _, _ = feature_store.train_test_spilt(test_size=0.2)
        except: 
            train_feature_view = feature_store.create_feature_view(
                name=name, 
                labels=["Close"]
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
        pass 