from gold_prediction.logging.logger import logging
from gold_prediction.exception.exception import CustomException 
import sys
#import pandas as pd 
from omegaconf import OmegaConf


data_ingestion_config = OmegaConf.load("configs/data_ingestion.yaml")
print(data_ingestion_config)


class DataIngestion: 
    def __init__(self, dataIngestionConfig) -> None:
        self.dataIngestionConfig = dataIngestionConfig

    def InitiateDataIngestion(self):
        pass 
