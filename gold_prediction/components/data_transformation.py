from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException
import pandas as pd 
import sys
import numpy as np 
import os 
from dotenv import load_dotenv 
from omegaconf import OmegaConf
import hopsworks 
from typing import List 

load_dotenv()



class DataTransformation:
    def __init__(self, dataTransformationConfig):
        self.dataTransConfig = dataTransformationConfig
        self.HOPSWORKS_API = os.getenv("HOPSWORKS_API_KEY")
        self.Hopswork_project = hopsworks.login(
            api_key_value = self.HOPSWORKS_API
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
            df["year"]  = df["Date"].dt.year
            df["day"] = df["Date"].dt.day
            df["dayOfweek"] = df["Date"].dt.day_name()
            # df["dayOfweek"] = self.label_encode(df, "dayOfweek")
            df['is_weekend'] = np.where(df['dayOfweek'].isin(['Sunday', 'Saturday']), 1, 0)
            df["dayOfweek"] = self.label_encode(df, "dayOfweek")
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
    
    def create_or_get_feature_group(self, 
                                    feature_store, 
                                    name, 
                                    version, 
                                    description, 
                                    primary_key: List[str]):
        try: 
            logging.info("Getting Hopsworks Feature Group")
            feature_group = feature_store.get_feature_group(
                name=name, 
                #version=version, 
                description=description, 
                primary_key=primary_key
            )
            return feature_group
        except:
            logging.info("Creating Hopworks Feature Group")
            # create a feature group for the first time 
            feature_group = feature_store.create_feature_group(
                name=name, 
                #version=version, 
                description=description, 
                primary_key=primary_key, 
            )
            logging.info("Feature Group Succesfully Created")
            return feature_group 




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
            
            train_feature_group = self.create_or_get_feature_group(
                feature_store=feature_store, 
                name = "gold_price_prediction_train_data", 
                version = 2, 
                primary_key = ["Date"],  
                description = "Gold Price Prediction Features" 
            )

            test_feature_group = self.create_or_get_feature_group(
                feature_store=feature_store, 
                name = "gold_price_prediction_test_data", 
                version=2, 
                description="Gold price dataset",
                primary_key = ["Date"]

            )
            
            
            
            
            
            train_feature_group.insert(trainData)
            test_feature_group.insert(testData)
            # update features descriptions 
            data_dictionary = {
                # Time-Related Features
                'Date': 'The timestamp or date of the recorded data point',
                'month': 'The month number (1-12) extracted from the date',
                'year': 'The year extracted from the date',
                'day': 'The day of the month (1-31)',
                'day_week_name': 'Name of the day of the week (Monday-Sunday)',
                'is_weekend': 'Boolean indicator for whether the day is a weekend (0 for weekday, 1 for weekend)',
                'dayOfyear': 'The day number within the year (1-365/366)',
                'quarter': 'The quarter of the year (1-4)',
                
                # Price Features
                'Close': 'The closing price of gold for the given day',
                'High': 'The highest price of gold reached during the trading day',
                'Low': 'The lowest price of gold reached during the trading day',
                'Open': 'The opening price of gold for the trading day',
                'Volume': 'The total trading volume of gold for that day',
                
                # Statistical Features
                'Agg_mean': 'Mean price calculated over a specific window period',
                'Agg_max': 'Maximum price over a specific window period',
                'Agg_std': 'Standard deviation of prices over a specific window period',
                'Agg_min': 'Minimum price over a specific window period',
                'Kurt': 'Kurtosis measure, indicating the "tailedness" of the price distribution',
                'skewness': 'Measure of asymmetry in the price distribution',
                
                # Lagged Features
                'lag_1': 'Price value from 1 time period ago',
                'lag_2': 'Price value from 2 time periods ago',
                'lag_3': 'Price value from 3 time periods ago'
            }





            # save features to artifacts folder 


        except Exception as e: 
            logging.error("Error Occurred during data transformation ")
            raise CustomException(e, sys)



if __name__ == "__main__":
    config = OmegaConf.load("configs/data_transformation.yaml")
    dataTrans = DataTransformation(dataTransformationConfig=config)
    dataTrans.IntializeDataTransformation()