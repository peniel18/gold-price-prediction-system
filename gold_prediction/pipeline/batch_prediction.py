from gold_prediction.exception.exception import CustomException
from gold_prediction.logging.logger import logging 
from gold_prediction.utils.utility_functions import load_model
import hopsworks
from pathlib import Path
import os 
from omegaconf import OmegaConf
from dotenv import load_dotenv
import pandas as pd 
import boto3 

load_dotenv()
### use hopsworks batch prediction feature 
# 1. get data for batch predictions eg. get dates and create features based on this 
# 2. batch predictions from model 
# 3. send predictions to a cloud store 
# 4. build an api to fetch predictions upon demand


class BatchPredictionsPipeline: 
    def __init__(self, BatchPredsConfig):
        self.BatchPredsConfig = BatchPredsConfig
        self.HOPSWORKS_API = os.getenv("HOPSWORKS_API_KEY")
        self.hopsworks_project = hopsworks.login(
            api_key_value=self.HOPSWORKS_API
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
            feature_view = feature_store.get_feature_view(name=name)
            data = feature_view.get_batch_data(
               # start_time = "2025-02-01", 
               # end_time = "2025-05-31" 
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
               # start_time = "2025-02-01" , 
               # end_time = "2025-05-31"
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
        model_registry_reference = model_registry.get_model(name="gold_model", version=model_version)
        model_dir = model_registry_reference.download()
        print(model_dir)
        model_path = Path(model_dir) / "lasso"
        model = load_model(model_path)
        return model 

    
    def make_predictions(self, model, data): 
        columns = ['agg_mean', 'agg_max', 'agg_std', 'agg_min', 'kurt',
                     'skewness', 'month', 'year', 'day', 'dayofweek', 'is_weekend', 'dayofyear', 'quarter']
        dates = data["dates"]
        print(dates)
        df = data[columns]
        y_preds = model.predict(df)
        print(y_preds)
        print(len(y_preds))
        predictions = pd.DataFrame({
            "dates" : dates, 
            "predictions" : y_preds
        })
        predictions.to_csv(self.BatchPredsConfig.PredictionsPath, index=False)
        return predictions


    def save_predictions_on_cloud(self, file_path: str, bucket_name: str, s3_key): 
        """
        Save batch predictions csv on aws s3 bucket 
        
        Args: 
            file_path (str): the path of the csv file 
            bucket_name (str): name of the s3 bucket 
        """
        s3_bucket = boto3.client("s3")
        s3_bucket.upload_file(file_path, bucket_name, s3_key)



    def InitializeBatchPredictionsPipeline(self): 
        data = self.get_predictions_data(
            name="batch_predictions_data", 
            description="batch_predictions_data_for_batch_predicitions"
        )
        # get model from hopsworks 
        model = self.load_model_from_model_registry(self.hopsworks_project)
        print(model)
        self.make_predictions(model=model, data=data)
        self.save_predictions_on_cloud(
            file_path=self.BatchPredsConfig.PredictionsPath, 
            bucket_name=self.BatchPredsConfig.BUCKET_NAME, 
            s3_key=os.getenv("AWS_S3_KEY")
        )




if __name__ == "__main__":
    BatchPredsConfig = OmegaConf.load("configs/batch_features_pipeline.yaml")
    batchPipeline = BatchPredictionsPipeline(
        BatchPredsConfig=BatchPredsConfig, 
    )
    batchPipeline.InitializeBatchPredictionsPipeline()