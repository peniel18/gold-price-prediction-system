from gold_prediction.exception.exception import CustomException
from gold_prediction.logging.logger import logging
import hopsworks
import pandas as pd 



class PredictionsFeatures: 

    def make_predictions_data(self, start_date, end_date):
        """
        Create training data based on start and end dates


        Args: 
            start_date 
            end_date 
        
        """

        start_date = None # start from feburary, 2025 
        end_date = None # july, 2025 

        # change this 
        start_date = pd.to_datetime(start_date)




        pass 


    def create_or_get_feature_store(self):
        pass 


    def store_data_on_feature_store(self):
        pass 


    def IntializeFeatures(self):
        pass 



