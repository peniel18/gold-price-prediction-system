from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException
import pandas as pd 
import sys
import numpy as np 
import os 
from dotenv import load_dotenv 
from omegaconf import OmegaConf
import hopsworks 

load_dotenv()
HOPSWORKS_API = os.getenv("HOPSWORKS_API_KEY")


class DataTransformation:
    def __init__(self, dataTransformationConfig):
        self.dataTransConfig = dataTransformationConfig
        self.Hopswork_project = hopsworks.login(
            api_key_value = HOPSWORKS_API
        )


    def label_encode(self, df: pd.DataFrame, feature: str) -> pd.Series:
        """
        Encode daysOfWeek column to numerical values
        """
        if feature not in df.columns:
            raise KeyError(f"Column {feature} not found in DataFrame")
            

        daysOfweek = {
                'Sunday':0,
                'Monday':1,
                'Tuesday' : 2,
                'Wednesday' : 3,
                'Thursday' : 4,
                'Friday' : 5,
                'Saturday' :6
            }

    
        return df[feature].map(daysOfweek)


    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame: 
        logging.info("Feature engineering has started")
        try: 
            y = df.Close 
            features = ['High', 'Low', 'Open', 'Volume']
            X = df[features]
            # aggregations 
            df["Agg_mean"] = df[features].mean(axis=1)
            df["Agg_max"] = df[features].max(axis=1)
            df["Agg_std"] = df[features].std(axis=1)
            df["Agg_min"] = df[features].min(axis=1)
            df["Kurt"] = df[features].kurt(axis=1)
            df["skewness"] = df[features].skew(axis=1)
            
            # lag and time features 
            df["month"]  = df["Date"].dt.month
            df["year"]  = df["Date"].dt.month
            df["day"] = df["Date"].dt.day
            df["dayOfweek"] = df["Date"].dt.day_name()
            df["dayOfweek"] = self.label_encode(df, "dayOfweek")
            df['is_weekend'] = np.where(df['dayOfweek'].isin(['Sunday', 'Saturday']), 1, 0)
            df["dayOfyear"] = df["Date"].dt.dayofyear
            # df["weekOfyear"] = df["Date"].dt.weekofyear
            df["quarter"] = df["Date"].dt.quarter

            df["lag_1"] = df["Close"].shift(1)
            df["lag_2"] = df["Close"].shift(2)
            df["lag_3"] = df["Close"].shift(3)
            df.dropna(inplace=True)
             
            
            logging.info("Features Succesfully engineered")
            return df 
        
        except Exception as e: 
            logging.error("Error Occurred during engineering features") 
            raise CustomException(e, sys)
        

    def data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        try: 
            logging.info("Data Cleaning has started")
            # drop the first two rows 
            df = df.iloc[2:].reset_index(drop=True)
            # rename the date column from price to date 
            df.rename(columns={"Price" : "Date"}, inplace=True)
            df.dropna(inplace=True)
            # change the dtypes of the columns from object to floats
            data_dtypes = {
                "Close" : "float", 
                "Open" : "float", 
                "High" : "float", 
                "Low" : "float", 
                "Volume" : "float"
            }

            df = df.astype(data_dtypes)
            df["Date"]  = pd.to_datetime(df["Date"])
            logging.info("Data Cleaning is succesfully completed")
            return df 

        except Exception as e: 
            logging.error("Error occurred during data cleaning")
            raise CustomException(e, sys)


    def IntializeDataTransformation(self):
        """
        Initialize Data Transformation Process 
        
        """
        try: 
            logging.info("Data Transformation has started")
            trainData = pd.read_csv(self.dataTransConfig.train_data.path)
            testData = pd.read_csv(self.dataTransConfig.test_data.path)

            # clean data 
            trainData = self.data_cleaning(trainData)
            testData = self.data_cleaning(testData) 
            # create features 
            trainData = self.generate_features(trainData)
            testData = self.generate_features(testData)

            print(trainData.columns)
            # hopswork feature store 
            feature_store = self.Hopswork_project.get_feature_store()
            train_feature_group = feature_store.get_or_create_feature_group(
                name = "gold_price_prediction_train_data", 
                version = 1, 
                primary_key = ["Date"], 
                #online = None, 
                description = "Gold Price Prediction Features"
            )

            test_feature_group = feature_store.get_or_create_feature_group(
                name = "gold_price_prediction_test_data", 
                version=1, 
                description="Gold price dataset",
                primary_key = ["Date"]

            )
            train_feature_group.insert(trainData)
            test_feature_group.insert(testData)
            # update features descriptions 



            # save features to artifacts folder 


        except Exception as e: 
            logging.error("Error Occurred during data transformation ")
            raise CustomException(e, sys)



if __name__ == "__main__":
    config = OmegaConf.load("configs/data_transformation.yaml")
    dataTrans = DataTransformation(dataTransformationConfig=config)
    dataTrans.IntializeDataTransformation()