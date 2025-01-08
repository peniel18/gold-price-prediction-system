from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException




class ModelTrainer: 
    def __init__(self, 
                 ModelTrainerConfig, 
                 tune_hyperparameters: bool | None = True):
        self.ModelTrainerConfg = ModelTrainerConfig
        self.tune_hyperparameters = tune_hyperparameters

    def get_training_data(self):
        return None 

    def save_model_locally(self):
        pass 

    def register_models_on_hopswork(self):
        pass  
    
    def train(self):
        pass 





    def InitiateModelTrainer(self):
        pass 