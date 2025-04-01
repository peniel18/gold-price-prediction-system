from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException
import os 
import hopsworks
from sklearn import metrics 
import sys
from typing import Tuple
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
        model = model.download()
        return model 
    
    
    def get_inference_data(self) -> pd.DataFrame:
        """
        Retrieves test data from feature store 

        Args: 


        Raises: 
            CustomException

        """
        try: 
            
            logging.info("Get Feature View on Hopsworks")
            # get a feature view 
            feature_store = self.HOPSWORKS_PROJECT.get_feature_store()
            feature_view = feature_store.get_feature_view(name="gold_prediction_test_data")
            data = feature_view.training_data(
                descrption = "gold_prediction_test_data"
            )
            return data 
        except: 
            # create a feature view 
            logging.info("Create a New feature View")
            feature_store = self.HOPSWORKS_PROJECT.get_feature_store()
            feature_group = feature_store.get_feature_group(name="gold_prediction_test_data")
            columns_to_query = feature_group.select_all()
            feature_view = feature_store.create_feature_view(
                name = "gold_prediction_test_data",
                query=columns_to_query

            )
            data = feature_view.training_data(
                description="gold_prediction_test_data"
            )
            return data 
        

    def prepare_data_for_inference(self, data: Tuple[pd.DataFrame]) -> pd.DataFrame:
        df = data[0]
        columns = [
            'close', 'agg_mean', 'agg_max', 'agg_std', 'agg_min', 'kurt',
            'skewness', 'month', 'year', 'day', 'dayofweek', 'is_weekend', 'dayofyear', 'quarter'
            ]
        
        return df[columns]
    

    def model_inference(self, data:pd.DataFrame) -> dict: 
        data = self.get_inference_data()
        df = self.prepare_data_for_inference(data)
        model = self.load_model()

        yTest = df["close"]
        XTest = df.drop("close", axis="columns")
        
        yHat = model.predict(XTest)

        mse = metrics.mean_squared_error(yTest, yHat)
        rmse = metrics.root_mean_squared_error(yTest, yHat)
        mape = metrics.mean_absolute_percentage_error(yTest, yHat)

        metrics = {
            "Mean Squared Error": mse, 
            "Root Mean Squared Error": rmse, 
            "Mean Absolute Percentage Error" : mape
        }

        return metrics


    def log_metrics(self):
        pass 

    def InitializeModelEvaluation(self):
        model = self.load_model()
        print(model)
        data = self.get_inference_data()
        print(data)
        metrics = self.model_inference(data)
        print(metrics)


if __name__ == "__main__":
    modelEvaluation = ModelEvaluation(ModelEvaluationConfig=None)
    modelEvaluation.InitializeModelEvaluation()