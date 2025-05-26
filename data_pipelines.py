from prefect import flow, task 
from gold_prediction.pipeline.training_pipeline import  TrainingPipeline
from gold_prediction.pipeline.inference_pipeline import InferencePipeline
from gold_prediction.pipeline.batch_prediction import BatchPredictionsPipeline
from omegaconf import OmegaConf 


@task 
def run_training_pipeline():
    # Load the configuration for the training pipeline
    config = None
    training_pipeline = TrainingPipeline(TrainingPipelineConfig=config)
    training_pipeline.InitailizeTrainingPipeline()


@task
def run_inference_pipeline():
    # load the configuration for the inference pipeline
    inference_pipeline_config = OmegaConf.load("configs/inference_pipeline.yaml")
    # Initialize the inference pipeline with the loaded configuration
    inference_pipeline = InferencePipeline(InferencePipelineConfig=inference_pipeline_config)
    inference_pipeline.InitializeInferencePipeline()


@task
def run_inference_pipeline_batch():
    batch_prediction_pipeline_config = OmegaConf.load("configs/batch_prediction.yaml")  
    # Initialize the batch prediction pipeline with the loaded configuration
    batch_prediction_pipeline = BatchPredictionsPipeline(BatchPredsConfig=batch_prediction_pipeline_config)
    batch_prediction_pipeline.InitializeBatchPredictionPipeline()


@flow(name="main_flow")
def main_flow():
    run_training_pipeline()
    run_inference_pipeline()
    run_inference_pipeline_batch()


if __name__ == "__main__":
    main_flow()


