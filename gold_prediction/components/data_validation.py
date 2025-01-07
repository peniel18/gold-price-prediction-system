from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException 
from scipy.stats import ks_2samp
from omegaconf import OmegaConf, DictConfig
from typing import List
from urllib.parse import urlparse 
import dagshub
import sys 
import pandas as pd 
import mlflow 



class DataValidation:
    def __init__(self, dataValidationConfig: DictConfig):
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
                column_drift["p-value"] = p_value
                column_drift["KS statistic"] = ks_stat
                column_drift["drift detected"] = dataDrift

                data_drift_report.append(column_drift)

            return data_drift_report
        
        except Exception as e: 
            logging.error("Error occurred during evaulation of drifts")
            raise CustomException(e, sys)


    def track_data_drift_with_mlflow(self, drift_results: List[dict], MLFLOW_URI: str) -> None: 
        """
        Pushes data drifts reports to dagshub for experiment tracking 
        
        Params: 
            drift_results: Data drifts of each feature column 

            MLFLOW_URI: mlflow tracking uri from dagshub

        Returns: 
            None

        """
        try:
            logging.info("Pushing data drifts to dagshub")
        
            dagshub.init(repo_owner='peniel18', repo_name='gold-price-prediction-system', mlflow=True)
            mlflow.set_registry_uri(MLFLOW_URI)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            mlflow.set_experiment("Data Drift Report")

            for dataDrifts in drift_results:
                column = dataDrifts["column"]
                p_value = dataDrifts["p-value"]
                ks_statistic = dataDrifts["KS statistic"]
                detected = dataDrifts["drift detected"]

                with mlflow.start_run(run_name=column):
                    # log data drifts params 
                    mlflow.log_metric(f"{column}_p_value", p_value)
                    mlflow.log_metric(f"{column}_ks_statistic", ks_statistic)
                    mlflow.log_metric(f"{column}_drift_detected", int(detected))
                    mlflow.log_param("feature_column", column)
        except Exception as e: 
            logging.error("Error Occurred during pushing data drifts")
            raise CustomException(e, sys)
            
             
 

    def validateFeatureColumns(self, data: pd.DataFrame):
        """
        Validates Columns against Baseline Columns 
        
        """
        logging.info("Validation features and column names")
        columnsConfig = self.dataValidationConfig["Columns"]
        expected_columns = list(columnsConfig.keys())
        actual_columns = data.columns
        
        logging.info(f"Expected Columns: {expected_columns}")
        logging.info(f"Actual Columns: {actual_columns}")

        number_features_match = bool(len(expected_columns) == len(actual_columns))
        # validate column names 
        missing_columns = [col for col in expected_columns if col not in actual_columns]
    
        if missing_columns: 
            logging.info(f"Missing Columns Detected: {', '.join(missing_columns)}")
        else: 
            logging.info("All expected columns are present")

        if not number_features_match: 
            logging.warning(f"Column count mismatch: Expected {len(expected_columns)} columns but found {len(actual_columns)}")
        else:
            logging.info("Column count matches expected configuration")

        logging.info("Feature and Columns Validated Succesfully")
        return number_features_match

    def InitiateDataValidaton(self):
        try: 
            logging.info("Data Validation has started")
            trainData = pd.read_csv(self.dataValidationConfig.train_data.path)
            testData =  pd.read_csv(self.dataValidationConfig.test_data.path)

            DataDriftReport = self.evaluate_data_drift(
                base_dataset=trainData, 
                incoming_dataset=testData, 
            )
            # log drifts results 
            MLFLOW_URI = "https://dagshub.com/peniel18/gold-price-prediction-system.mlflow"
            self.track_data_drift_with_mlflow(
                drift_results=DataDriftReport,
                MLFLOW_URI=MLFLOW_URI
            )

            columnValidationStatus = self.validateFeatureColumns(trainData)
            logging.info(f"Columns and feature Validation Status: {columnValidationStatus}")
            


        except Exception as e: 
            logging.error("Error Occurred during Data Validation")
            raise CustomException(e, sys)


        

if __name__ == "__main__":
    dataValConfig = OmegaConf.load("configs/data_validation.yaml")
    dataValidation = DataValidation(dataValidationConfig=dataValConfig)
    dataValidation.InitiateDataValidaton()