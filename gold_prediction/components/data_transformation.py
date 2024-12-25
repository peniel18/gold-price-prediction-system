from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException
import pandas as pd 
from gold_prediction.utils.utility_functions import read_data




class DataTransformation:
    def __init__(self, dataTransformationConfig):
        self.dataTransConfig = dataTransformationConfig
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame: 
        pass 


    def data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        # drop the first two rows 
        df.iloc[2:].reset_index(drop=True)
        # rename the date column from price to date 
        df.rename(columns={"Price" : "Date"}, inplace=True)
        # change the dtypes of the columns 
        return df 

    def StartDataTransformation(self):
        pass 