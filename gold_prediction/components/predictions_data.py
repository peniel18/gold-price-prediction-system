from gold_prediction.exception.exception import CustomException
from gold_prediction.logging.logger import logging
from omegaconf import OmegaConf
import hopsworks
import pandas as pd 
from typing import List
import os 
import sys



class PredictionsFeatures: 
    def __init__(self, PredictionFeatureConfig):
        self.PredictionFeatureConfig = PredictionFeatureConfig
        self.HOPSWORKS_API = os.getenv("HOPSWORKS_API_KEY")
        self.hopsworks_project = hopsworks.login(api_key_value=HOPSWORKS_API)


    def make_predictions_data(start_date=None, end_date=None) -> pd.DataFrame:
        """
        Create training data based on start and end dates

        Args: 
            start_date: Start date in 'YYYY-MM-DD' format or datetime object.
            end_date: End date in 'YYYY-MM-DD' format or datetime object.
        Returns: 
            pd.DataFrame: Data with computed features.
        
        """
    

        if start_date is None:
            start_date = "2025-02-01"
        if end_date is None:
            end_date = "2025-07-31"
        # change this 
        dates = pd.date_range(start=start_date, end=end_date)
        df = pd.DataFrame()
        df["dates"] = pd.to_datetime(dates)

        df.index = pd.to_datetime(df['dates'])
    
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['dayofyear'] = df.index.dayofyear
        df['quarter'] = df.index.quarter

        features = [
            "month", "year", "day", "dayofweek", "is_weekend",
            "dayofyear", "quarter" , 
        ]

        df["agg_mean"] = df[features].mean(axis=1)
        df["agg_max"] = df[features].max(axis=1)
        df["agg_std"] = df[features].std(axis=1)
        df["agg_min"] = df[features].min(axis=1)
        df["kurt"] = df[features].kurt(axis=1)
        df["skewness"] = df[features].skew(axis=1)
        

        return df 





    def create_or_get_feature_group(self, 
                                    name, 
                                    description, 
                                    primary_key: List[str]):
        """
        Create or Gets feature group from hopworks 

        Args: 
            name(str): name of the feature store 
            description (str): 
            primary_key: 
        
        
        """
        try: 
            logging.info("Getting feature store for predictions features")
            feature_store = self.hopsworks_project.get_feature_store()
            feature_group = feature_store.get_feature_group(
                name = name, 
                description = description, 
                primary_key = primary_key
            )
            return feature_group 
        except: 
            # create a new feature group if it does not exist 
            logging.info("Creating feature store")
            feature_store = self.hopsworks_project.get_feature_store()
            feature_group = feature_store.create_feature_group(
                name=name, 
                description=description, 
                primary_key=primary_key
            )
            return feature_group


    def store_data_on_feature_store(self, data):
        try: 
            logging.info("Inserting data into feature store")
            feature_group = self.create_or_get_feature_group(
                name=self.PredictionFeatureConfig.feature_group_name, 
                description=self.PredictionFeatureConfig.feature_group_description, 
                primary_key=["dates"]
            )
            feature_group.insert(data)
        except Exception as e: 
            logging.error("Error Occurred during data ingestion of batch features")
            raise CustomException(e, sys)


    def IntializeFeatures(self):
        data = self.make_predictions_data(None, None)
        self.store_data_on_feature_store(data)



if __name__ == "__main__":
    config = OmegaConf.load("configs/batch_features_pipeline.yaml")
    PredictionsFeatures = PredictionsFeatures(config)
    PredictionsFeatures.IntializeFeatures()