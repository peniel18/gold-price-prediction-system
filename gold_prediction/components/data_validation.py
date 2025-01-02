from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException 
from evidently.dashboard import Dashboard 
import sys 
import pandas as pd 
import json 




# validate columns 
# calculate data drift 
# validate the number of features created
class DataValidation:
    def __init__(self, dataValidationConfig):
        self.dataValidationConfig = dataValidationConfig 

    def evaluate_data_drift(self, reference, production, column_mapping):
        """
        Calculates Data drift for incoming data 

        Returns: 
            the p-value of statistical test performed for each feature


        """
        
        data_drift_profile = Profile(sections=[DataDriftProfileSection])
        data_drift_profile.calculate(reference, production, column_mapping=column_mapping)
        drift_report = data_drift_profile.json()
        json_report = json.load(drift_report)

        drifts = []
        for feature in column_mapping["numerical_features"] + column_mapping["categorical_features"]: 
            drifts.append((feature, json_report['data_drift']['data']['metrics'][feature]['p_value']))

        return drifts



    def validateFeatureColumns(self):
        pass 

    def InitiateDataValidaton(self):
        pass 


            