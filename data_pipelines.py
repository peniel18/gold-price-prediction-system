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
    inference_pipeline = InferencePipeline(InferencePipelineConfig=None)
    inference_pipeline.IntializeInferencePipeline()


@task
def run_batch_prediction_pipeline():
    BatchPredsConfig = OmegaConf.load("configs/batch_features_pipeline.yaml")
    batch_prediction_pipeline = BatchPredictionsPipeline(BatchPredsConfig=BatchPredsConfig)
    batch_prediction_pipeline.InitializeBatchPredictionsPipeline()


@flow(name="main_flow")
def main_flow():
    run_training_pipeline()
    run_inference_pipeline()
    run_batch_prediction_pipeline()


if __name__ == "__main__":
    main_flow()


