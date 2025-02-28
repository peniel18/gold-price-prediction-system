from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException 
from gold_prediction.utils.utility_functions import save_dataframe_object
import sys
import pandas as pd 
from omegaconf import OmegaConf
import yfinance as yf 




class DataIngestion: 
    def __init__(self, dataIngestionConfig) -> None:
        self.dataIngestionConfig = dataIngestionConfig

    def get_data_from_yfinance(self, ticker: str, 
                               train_start_date: str, 
                               train_end_date: str, 
                               test_start_date: str, 
                               test_end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        gets gold data from yfinance using start dates and end dates 
        
        Args: 

        Returns: 
            A tuple of train and test data sets 

        Raises
            CustomException error if ticker or dates are invalid 
        
        """
    
        try:        
            train_data = yf.download(
                "GC=F", 
                period="3y"
            ) 
            print(train_data)
            print(ticker)
            print(train_start_date)
            print(test_start_date)
            test_data = yf.download(
                ticker, 
                start=test_start_date,
                end=test_end_date
            )
            # check if any of the data is empty 
            if train_data.empty or test_data.empty:
                raise ValueError("No data found for ticker")

            return train_data, test_data
        except Exception as e: 
            logging.error("Error occurred in function get_data_from_yfinance")
            raise CustomException(e, sys)


    def InitializeDataIngestion(self):
        trianData, testData = self.get_data_from_yfinance(
            ticker=self.dataIngestionConfig.DataIngestion.gold_ticker, 
            train_start_date=self.dataIngestionConfig.DataIngestion.train_start_date, 
            train_end_date=self.dataIngestionConfig.DataIngestion.train_end_date, 
            test_start_date=self.dataIngestionConfig.DataIngestion.test_start_date, 
            test_end_date=self.dataIngestionConfig.DataIngestion.test_end_date                   
        )
        # save train object to artifacts 
        save_dataframe_object(
            data_object=trianData, 
            path=self.dataIngestionConfig.Data_paths.train_data_path,
            filename=self.dataIngestionConfig.Data_paths.train_file_name,
        )
        # save test data objects to artifacts 
        save_dataframe_object(
            data_object=testData, 
            path=self.dataIngestionConfig.Data_paths.test_data_path,
            filename=self.dataIngestionConfig.Data_paths.test_file_name, 
        )



if __name__ == "__main__":
    data_ingestion_config = OmegaConf.load("configs/data_ingestion.yaml")
    dataIngestion = DataIngestion(dataIngestionConfig=data_ingestion_config)
    dataIngestion.InitializeDataIngestion()