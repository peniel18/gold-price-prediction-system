import os 
from omegaconf import OmegaConf
from pathlib import Path 
import pandas as pd
from urllib.parse import urlparse 
import mlflow



def save_dataframe_object(data_object: pd.DataFrame,
                          path: str, 
                          filename: str, 
                          index: bool = True) -> None: 
    path = Path(path)
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, filename)
    data_object.to_csv(file_path, index=index)

    

def read_data(filepath: str) -> pd.DataFrame:
    data = pd.read_csv(filepath)
    return data



def ParametersTracker(model, params: dict, name: str):
    mlflow.set_tracking_uri("")
    tracking_url_type_store = urlparse(mlflow.get_tracking_ur)

