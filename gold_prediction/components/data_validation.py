from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException 
from evidently.dashboard import Dashboard 
from scipy.stats import ks_2samp
import sys 
import pandas as pd 
import json 




# validate columns 
# calculate data drift 
# validate the number of features created
class DataValidation:
    def __init__(self, dataValidationConfig):
        self.dataValidationConfig = dataValidationConfig 

    def evaluate_data_drift(self, base_dataset, incoming_dataset, threshold):
        """
        Calculates Data drift for incoming data 

        Returns: 
            the p-value of statistical test performed for each feature


        """
        try: 
            logging.info("Checking data drift")
            data_drift_report = {}
            dataDrift: bool = False 
            for column in base_dataset.columns: 
                base = base_dataset[column]
                current = incoming_dataset[column]
                ks_stat, p_value = ks_2samp(base, current)
                dataDrift = ks_stat > threshold 
                # log drifts of each colum to mlflow 

            return dataDrift

        except Exception as e: 
            raise CustomException(e, sys)

    def validateFeatureColumns(self):
        pass 

    def InitiateDataValidaton(self):
        pass 


            