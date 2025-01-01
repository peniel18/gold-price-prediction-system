from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException 
import sys 
import pandas as pd 




# validate columns 
# calculate data drift 
# validate the number of features created
class DataValidation:
    def __init__(self, dataValidationConfig):
        self.dataValidationConfig = dataValidationConfig 

    def check_data_drift(self):
        pass 
    
    def validateFeatureColumns(self):
        pass 

    def InitiateDataValidaton(self):
        pass 


            