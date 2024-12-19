import os 
from pathlib import Path 

project_name = "gold_prediction"

list_of_files = [
    f"{project_name}/components/data_ingestion.py", 
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py", 
    f"{project_name}/components/data_transformation.py", 
    f"{project_name}/components/model_trainer.py", 
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/utility_functions.py",
    f"{project_name}/exception/__init__.py", 
    f"{project_name}/exception/exception.py", 
    f"{project_name}/logging/__init__.py"
    "app.py", 
    "notebooks/experiments.ipynb", 
    ".env", 
    "Artifacts", 
    "setup.py"
]




for filepath in list_of_files: 
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath) 

    if filedir != "": 
        os.makedirs(filedir, exist_ok=True)
        #logging.info(f"Creating directory: {filedir} for file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open (filepath, "w") as f: 
            pass # create an empty file