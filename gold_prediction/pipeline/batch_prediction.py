from gold_prediction.exception.exception import CustomException
from gold_prediction.logging.logger import logging 
from gold_prediction.utils.utility_functions import load_model
import hopsworks
from pathlib import Path
import os 


### use hopsworks batch prediction feature 
# 1. get data for batch predictions eg. get dates and create features based on this 
# 2. batch predictions from model 
# 3. send predictions to a cloud store 
# 4. build an api to fetch predictions upon demand


class BatchPredictionsPipeline: 
    def __init__(self, BatchPredsConfig):
        self.BatchPredsConfig = BatchPredsConfig
        self.hopsworks_project = hopsworks.login(
            api_key_file=os.getenv("HOPSWORKS_API_KEY")
        )


    def get_predictions_data(self, name, description):
        """
        Gets data from feature store for batch predictions 
       
        Args: 
            name (str): 
            description (str): 
        
        """
        try: 
            feature_store = self.hopsworks_project.get_feature_store()
            feature_view = feature_store.get_feature_view(name=description)
            data = feature_view.get_batch_data(
                start_time = None , 
                end_time = None 
            )
            return data 
        except: 
            logging.info("Creating a Feature View for batch predictions data")
            feature_store = self.hopsworks_project.get_feature_store()
            feature_group = feature_store.get_feature_group(name=name)
            columns_to_query = feature_group.select_all()
            feature_view = feature_store.create_feature_view(
                 name = description, 
                 query=columns_to_query
            )

            data = feature_view.get_batch_data(
                start_time = None , 
                end_time = None
            )
            return data



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
        model_path = Path(model_dir) + "lasso.pkl"
        model = load_model(model_path)
        return model 

    
    def predictions(self, model, data): 
        pass 


    def save_predictions(self): 
        """
        Save batch predictions on cloud 
        
        """

    def InitializeBatchPredictionsPipeline(self): 
        data = self.get_predictions_data(
            name="batch_predictions_data", 
            description="batch predictions data for batch predicitions"
        )
    
