from gold_prediction.logging.logger import logging
from gold_prediction.components.hyperparameter_tuning import get_parameters, optimise_hyperparameters
from gold_prediction.exception.exception import CustomException
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Optional, Type, List
from omegaconf import OmegaConf
from sklearn.metrics import mean_squared_error
import pandas as pd 
import numpy as np
from xgboost import XGBRegressor
import hopsworks
import dagshub 
from urllib.parse import urlparse
import joblib
import mlflow
import sys
from dataclasses import dataclass
import os



load_dotenv()



class ModelTrainer: 
    def __init__(self, ModelTrainerConfig, tune_hyperparameters: bool | None = True):
            self.ModelTrainerConfig = ModelTrainerConfig
            self.HOPSWORKS_API = os.getenv("HOPSWORKS_API_KEY")
            self.Hopswork_project = hopsworks.login(
            api_key_value = self.HOPSWORKS_API
            )
            self.tune_hyperparameters = tune_hyperparameters

    def get_training_data(self, feature_store, name: str, description) -> pd.DataFrame:
        """
         Retrieve training data from feature store and split into train/validation sets.
    
        Args:
        feature_store: The feature store instance
            name: Name of the feature group
            descripiton: Description of the feature view 
        
        Returns:
            tuple: 
        
        """
        try: 
            # get a feature view that already exists
            logging.info("Get Feature View on Hopsworks ")
            feature_group = feature_store.get_feature_group(name=name)
            feature_view = feature_store.get_feature_view(name=description)
            data = feature_view.training_data(
                description=description, 
                #version=1
            )
            return data
        except: 
            # create a new feature view if it doesnt exist
            logging.info("Create a new Feature View on hopsworks")
            feature_group = feature_store.get_feature_group(name=name)
            columns_to_query =  feature_group.select_all()
            feature_view = feature_store.create_feature_view(
                name=description, 
                #labels=["close"], 
                #version=4, 
                query=columns_to_query
            )
            data = feature_view.training_data(
                description=description
            )
            return data
        

    def save_model_locally(self, model, model_name) -> str:
        """
        Save the trained model locally as a pickle file 

        Args: 
            model: the model object to be saved 
            model_name: the name of the model to be saved

        """ 
        try: 
            os.makedirs(self.ModelTrainerConfig.model_artifacts.path, exist_ok=True)
            path = os.path.join(self.ModelTrainerConfig.model_artifacts.path, model_name)
            joblib.dump(model, path)

            return path 
        except Exception as e: 
            logging.error(f"Error saving trained model at path:{path}")
            raise CustomException(e, sys)


    def register_models_on_hopswork(self, model_path: str, metric: dict, description: str):
        """
        Push model to Hopsworks model registery 

        Args: 
            model_path: trained model path 
            metric: Error metric of the trained model 
            description: Description of the feature view 
        
        """
        try: 
            model_registry = self.Hopswork_project.get_model_registry()
            feature_store = self.Hopswork_project.get_feature_store()
            feature_view = feature_store.get_feature_view(name=description)
            skl_model = model_registry.python.create_model(
                name = "gold prediction", 
                metrics = metric, 
                feature_view = feature_view, 
            )
            skl_model.save(model_path)
        # create input scheme 
        except Exception as e: 
            raise CustomException(e, sys)


    def get_model(self, model_name: str) -> LinearRegression | Lasso | XGBRegressor | DecisionTreeRegressor | RandomForestRegressor:
        """
        Args: 
            model_name: name of the model 
        
        Returns: 
            A model function 
        """
        models = {
            "LinearRegression": LinearRegression, 
            "lasso": Lasso,
            "XGBoost": XGBRegressor, 
            "DescisionTreeRegressor": DecisionTreeRegressor, 
            "RandomForest" : RandomForestRegressor
        }

        if model_name.lower() in models.keys():
            return models[model_name.lower()]
        else: 
            raise KeyError(f"Model {model_name} is not available")

    def track_model_parameters_with_mlflow(self, model,
                                            parameters: dict, 
                                            loss_metric: float,
                                            experiment: str) -> None:
        """
        Track training and inference metrics using mlflow and dagshub
        

        Args: 
            model: 
            parameters: 
            loss_metric: 
            experiment: Name of the experiement, eg Training Metrics and Model 
        """

        
        # dagshub init here 7
        dagshub.init(repo_owner='peniel18', repo_name='gold-price-prediction-system', mlflow=True)
        mlflow.set_experiment(experiment)
        with mlflow.start_run() as run:
            # log hyperparamters 
            mlflow.log_params(parameters)
            # log training meterics 
            mlflow.log_metric("MSE", loss_metric)
            mlflow.sklearn.log_model(model, "model")


    def PrepareTrainingData(self, data: Tuple[pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare the training data from hopsworks feature store 

        Args: 
            data (Tuple): data from the feature store
        """
        df = data[0]
        # features 
        columns = ['close', 'agg_mean', 'agg_max', 'agg_std', 'agg_min', 'kurt',
                     'skewness', 'month', 'year', 'day', 'dayofweek', 'is_weekend', 'dayofyear', 'quarter']
        #columns = ["close", "month", "year", "day", "dayOfweek", "is_weekend", "dayOfweek", "dayOfyear", "quarter"]
        ds = df[columns]
    
        return ds 
 


    def train(self, model_name: str) -> object:
        """
        Train model and register models to hopsworks model registry 

        Args: 
            model_name: the model of the model to train 
        """
        try: 
            model_fn = self.get_model(model_name=model_name)
            # get data hopsworks
            feature_store = self.Hopswork_project.get_feature_store()
            ds = self.get_training_data(
                feature_store=feature_store, 
                name = "gold_prediction_train_data",
                description="gold_train_fv"
            )   
            ds = self.PrepareTrainingData(ds)
            features = list(ds.columns)
            print(features)
            features.remove("close")
        
            target = "close"

            # time series split 
            tss = TimeSeriesSplit(n_splits=5)
            ds.sort_index()

            fold = 0 
            preds = []
            scores = []

            if not self.tune_hyperparameters:
                logging.info("Training model with default parameters")
                for train_idx, val_idx in tss.split(ds):
                    train = ds.iloc[train_idx]
                    test = ds.iloc[val_idx]

                    X_train, y_train = train[features], train[target]
                    X_val, y_val = test[features] , test[target]

                    model = model_fn() 
                    model.fit(X_train, y_train)
                    yHat = model.predict(X_val)
                    preds.append(yHat)
                    errors = mean_squared_error(y_val, yHat)
                    scores.append(errors)

                # track preds and scores during training 
                print(np.average(scores))
                print(preds)

                metric = {"MSE": scores[0]}

                
                model_save_path = self.save_model_locally(
                    model=model, 
                    model_name=model_name
                )

                self.register_models_on_hopswork(
                    model_path=model_save_path, 
                    metric=metric
                )

                self.track_model_parameters_with_mlflow(
                    model=None, 
                    parameters=None, 
                    experiment="Training Metrics and Models",
                    loss_metric=None 
                )
                return model 
            else: 
                logging(f"Tuning parameters of {model_name}")
                
                #tuned_model_parameters = optimise_hyperparameter()
                model_hyperparameters = optimise_hyperparameters(
                    model_fn=model_fn, 
                    num_of_trials=20, 
                    X=X_train, 
                    y=y_val
                )

                logging.info("Training model with tuned hyperparameters")
                for train_idx, val_idx in tss.split(ds):
                    train = ds.iloc[train_idx]
                    test = ds.iloc[val_idx]

                    X_train, y_train = train[features], train[target]
                    X_val, y_val = test[features] , test[target]

                    model = model_fn(**model_hyperparameters)
                    model.fit(X_train, y_train)
                    yHat = model.predict(X_val)
                    preds.append(yHat)
                    errors = mean_squared_error(y_val, yHat)
                    scores.append(errors)

                print(np.average(scores))

                model_save_path = self.save_model_locally(
                    model=model, 
                    model_name=model_name
                )

                self.track_model_parameters_with_mlflow(
                    model=None,
                    parameters=None, 
                    experiment="Training Metrics and model(Tuned Hyperparameter)", 
                    loss_metric=None 
                )
                return model 
        except Exception as e: 
            logging.info("Error Occured during training model")
            raise CustomException(e, sys)

    def InitiateModelTrainer(self):
        model = self.train(model_name="lasso")
        self.register_models_on_hopswork(model_registry=None)


@dataclass 
class config: 
    pass 

if __name__ == "__main__":
    modelTrainerConfig = OmegaConf.load("configs/model_trainer.yaml")
    modelTrainer = ModelTrainer(ModelTrainerConfig=modelTrainerConfig, tune_hyperparameters=None)
    modelTrainer.InitiateModelTrainer()