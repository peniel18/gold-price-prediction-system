from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException
import os 
import hopsworks



class ModelEvaluation:
    def __init__(self, ModelEvaluationConfig):
        self.ModelEvaluationConfig = ModelEvaluationConfig
        self.HOPSWORKS_API = os.getenv("HOPSWORKS_API_KEY")
        self.HOPSWORKS_PROJECT = hopsworks.login(
            api_key_value=self.HOPSWORKS_API
        )
        

    def load_model(self, name: str): 
        """
        Loads model from hopsworks model registry 

        Args: 
            name(str): name of the model 
        
        """
        model_registry = self.HOPSWORKS_PROJECT.get_model_registry()
        model = model_registry.get_model(name=name, version=None)
        return model 
    
    
    def get_inference_data(self):
        pass 


    def InitializeModelEvaluation(self):
        model = self.load_model(name=None)


if __name__ == "__main__":
    modelEvaluation = ModelEvaluation(ModelEvaluationConfig=None)
    modelEvaluation