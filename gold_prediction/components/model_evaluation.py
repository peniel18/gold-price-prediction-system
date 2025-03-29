from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException
import os 
import hopsworks
import sys
import pandas as pd 


class ModelEvaluation:
    def __init__(self, ModelEvaluationConfig):
        self.ModelEvaluationConfig = ModelEvaluationConfig
        self.HOPSWORKS_API = os.getenv("HOPSWORKS_API_KEY")
        self.HOPSWORKS_PROJECT = hopsworks.login(
            api_key_value=self.HOPSWORKS_API
        )
        

    def load_model(self, name: str = "gold_model"): 
        """
        Loads model from hopsworks model registry 

        Args: 
            name(str): name of the model 
        
        """
        model_registry = self.HOPSWORKS_PROJECT.get_model_registry()
        model = model_registry.get_model(name=name, version=None)
        return model 
    
    
    def get_inference_data(self) -> pd.DataFrame:
        """
        Retrieves test data from feature store 

        Args: 
            name

            desc



        """
        try: 
            logging.info("Get Feature View on Hopsworks")
            feature_store = self.HOPSWORKS_PROJECT.get_feature_store()
            feature_view = feature_store.get_feature_view(name = 'gold_price_prediction_test_data')
            data = feature_view.get_training_data()
            return data
        except Exception as e: 
            logging.error("Error Occured during reteriving inference data")
            raise CustomException(e, sys)



    def InitializeModelEvaluation(self):
        model = self.load_model()
        print(model)
        data = self.get_inference_data()

        print(model)
        print(data)


if __name__ == "__main__":
    modelEvaluation = ModelEvaluation(ModelEvaluationConfig=None)
    modelEvaluation.InitializeModelEvaluation()