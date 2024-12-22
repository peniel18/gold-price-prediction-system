import logging 
import os 
from datetime import datetime 

log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)
log_file_path = os.path.join(logs_path, log_file)
Format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"


logging.basicConfig(
    filename=log_file_path, 
    format=Format, 
    level=logging.INFO
)

