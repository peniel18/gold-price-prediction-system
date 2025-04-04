from gold_prediction.exception.exception import CustomException 
from gold_prediction.logging.logger import logging
from gold_prediction.components.model_evaluation import ModelEvaluation
from omegaconf import OmegaConf



class InferencePipeline: 
    def __init__(self, InferencePipelineConfig):
        self.InferencePipelineConfig = InferencePipelineConfig


    def model_inference(self):
        modelInferenceConfig =  OmegaConf.load("configs/model_evaluation.yaml")
        modelEvaluation = ModelEvaluation(ModelEvaluationConfig=modelInferenceConfig)
        modelEvaluation.InitializeModelEvaluation()


    def IntializeInferencePipeline(self):
        logging.info("Inference Pipeline Running")
        self.model_inference()


if __name__ == "__main__":
    InferencePipeline(InferencePipelineConfig=None).IntializeInferencePipeline()