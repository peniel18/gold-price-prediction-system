from gold_prediction.exception.exception import CustomException
from gold_prediction.logging.logger import logging
from gold_prediction.components.data_ingestion import DataIngestion
from gold_prediction.components.data_transformation import DataTransformation
from gold_prediction.components.model_trainer import ModelTrainer
from gold_prediction.components.data_validation import DataValidation
import sys
from omegaconf import OmegaConf



class TrainingPipeline:
    def __init__(self, TrainingPipelineConfig):
        self.TrainingPipelineConfig = TrainingPipelineConfig


    def dataIngestion(self):
        try:
            dataIngestConfig = OmegaConf.load("configs/data_ingestion.yaml")
            dataIngestion = DataIngestion(
                dataIngestionConfig=dataIngestConfig
            )
            dataIngestion.InitializeDataIngestion()
        except Exception as e: 
            logging.error("Error Occured in TrainingPipeline at Data Ingestion Stage")
            raise CustomException(e, sys)
        
    
    def dataTransformation(self):
        try: 
            dataTransConfig = OmegaConf.load("configs/data_transformation.yaml")
            dataTransformation = DataTransformation(dataTransformationConfig=dataTransConfig)
            dataTransformation.IntializeDataTransformation()
        except Exception as e: 
            logging.error("Error Occured in TrainingPipeline at dataTransformation Stage")
            raise CustomException(e, sys)
        

    def dataValidation(self):
        try: 
            dataValConfig = OmegaConf.load("configs/data_validation.yaml")
            dataValidation = DataValidation(dataValidationConfig=dataValConfig)
            dataValidation.InitializeDataValidaton()
        except Exception as e: 
            logging.error("Error Occured in TrainingPipeline at Data Validation Stage")
            raise CustomException(e, sys)
        



    def modelTrainer(self):
        try: 
            modelTrainerConfig = OmegaConf.load("configs/model_trainer.yaml")
            modelTrainer = ModelTrainer(ModelTrainerConfig=modelTrainerConfig, tune_hyperparameters=True)
            modelTrainer.InitializeModelTrainer()
        except Exception as e:
            logging.error("Error Occured in TrainingPipeline at Data Validation Stage")
            raise CustomException(e, sys)
        

    def InitailizeTrainingPipeline(self):
        self.dataIngestion()
        self.dataTransformation()
        self.dataValidation()
        self.modelTrainer()
        print("Training Successful")



if __name__ == "__main__":
    config = None 
    training_pipeline = TrainingPipeline(TrainingPipelineConfig=config)
    training_pipeline.InitailizeTrainingPipeline()