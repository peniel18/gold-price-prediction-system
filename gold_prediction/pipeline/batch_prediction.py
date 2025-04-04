from gold_prediction.exception.exception import CustomException
from gold_prediction.logging.logger import logging 
from gold_prediction.utils.utility_functions import load_model
import hopsworks
from pathlib import Path
### use hopsworks batch prediction feature 
# 1. get data for batch predictions eg. get dates and create features based on this 
# 2. batch predictions from model 
# 3. send predictions to a cloud store 
# 4. build an api to fetch predictions upon demand



class BatchPredictionsPipeline: 
    def __init__(self):
        pass 


    def get_predictions_data(self):
        """
        computes data from dates for predictions 
        
        
        """
        
        
        pass 

    
    def load_model_from_model_registry(self, hopsworks_project, model_version: int = 1):
        """
        Loads model from Hopsworks model registry
        
        Args:
            hopworks_project: 

            model_version: 

        """
        model_registry = hopsworks_project.get_model_registry()
        model_registry_reference = model_registry.get_model(name="", version=model_version)
        model_dir = model_registry_reference.download()
        model_path = Path(model_dir)
        model = load_model(model_path)
        return model 

    
    def predictions(model, data): 
        pass 


    def save_predictions(self): 
        """
        Save batch predictions on cloud 
        
        """
    
