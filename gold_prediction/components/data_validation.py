from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException 
#from evidently.dashboard import Dashboard 
from scipy.stats import ks_2samp
from omegaconf import OmegaConf
from typing import List, Tuple
import sys 
import pandas as pd 
import json 




# validate columns 
# calculate data drift 
# validate the number of features created
class DataValidation:
    def __init__(self, dataValidationConfig):
        self.dataValidationConfig = dataValidationConfig 

    def evaluate_data_drift(self,
                            base_dataset: pd.DataFrame,
                            incoming_dataset: pd.DataFrame,
                             threshold: float = 0.05 
                             ) -> List[dict]:
        """
        Calculates Data drift for incoming data 

        Args: 
            base_dataset: Reference dataset used as baseline 
            incoming_dataset: New Dataset to compare against base
            threshold: Threshold for KS Statistic to determine drift

        Returns: 
            the p-value of statistical test performed for each feature


        """
        try: 
            logging.info("Checking data drift")
            data_drift_report: List[dict] = []
            dataDrift: bool = False 
            for column in base_dataset.columns: 
                base = base_dataset[column]
                current = incoming_dataset[column]
                ks_stat, p_value = ks_2samp(base, current)
                # we reject the null hypothesis, there is statistical evidence of drift
                dataDrift = p_value <= threshold 
                # log drifts of each colum to mlflow 
                column_drift: dict = {}
                column_drift["column"] = column
                column_drift["P-value"] = p_value
                column_drift["KS statistic"] = ks_stat
                column_drift["drift detected"] = dataDrift

                data_drift_report.append(column_drift)

            # log experiments to mlflow 




            return data_drift_report
        
        except Exception as e: 
            logging.error("Error occurred during evaulation of drifts")
            raise CustomException(e, sys)


    def validateFeatureColumns(self):

        return None 

    def InitiateDataValidaton(self):
        trainData = pd.read_csv(self.dataValidationConfig.train_data.path)
        testData =  pd.read_csv(self.dataValidationConfig.test_data.path)

        DataDriftReport = self.evaluate_data_drift(
            base_dataset=trainData, 
            incoming_dataset=testData, 
        )
        print(DataDriftReport)


        

if __name__ == "__main__":
    dataValConfig = OmegaConf.load("configs/data_validation.yaml")
    dataValidation = DataValidation(dataValidationConfig=dataValConfig)
    dataValidation.InitiateDataValidaton()