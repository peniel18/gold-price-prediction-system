import os 
from omegaconf import OmegaConf
from pathlib import Path 
import pandas as pd



def save_dataframe_object(data_object: pd.DataFrame, path: str, filename: str) -> None: 
    file_path = os.path.join(path, filename)
    data_object.to_csv(file_path, index=False)