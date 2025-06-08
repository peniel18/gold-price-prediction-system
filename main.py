from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request       
from gold_prediction.logging.logger import logging
from boto3 import client
from botocore.exceptions import ClientError
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd 
import os
from dotenv import load_dotenv
load_dotenv()  

app = FastAPI(
    title="Gold Batch Predictions API", 
    version="1.0.0"
)


def get_s3_client(): 
    try: 
        return client("s3")
    except Exception as e: 
        logging.error(f"Error occurred get S3 client: {e}")
        raise HTTPException(status_code=500, detail="Failed to get s3 client")



def load_predictions_from_s3_bucket(bucket_name: str, s3_key: str) -> pd.DataFrame: 
    try: 
        s3_client = get_s3_client()
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        # read csv data from the response
        data = response['Body'].read().decode('utf-8')
        df = pd.read_csv(pd.compat.StringIO(data), parse_dates=["dates"])
        logging.info("Successfully loaded predictions from S3 bucket")
        return df
    except ClientError as e: 
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logging.error(f"File not found in S3 bucket: {s3_key}")
            raise HTTPException(status_code=404, detail="File not found in S3 bucket")
        elif error_code == 'NoSuchBucket':
            logging.error(f"Bucket not found: {bucket_name}")
            raise HTTPException(status_code=404, detail="Bucket not found")
        elif error_code == 'AccessDenied':
            logging.error(f"Access denied to bucket: {bucket_name}")
            raise HTTPException(status_code=403, detail="Access denied to bucket")
        else:
            logging.error(f"Error occurred while accessing S3 bucket: {e}")
            raise HTTPException(status_code=500, detail="Failed to access S3 bucket")
    except Exception as e:
        logging.error(f"Error occurred while loading predictions from S3 bucket: {e}")
        raise HTTPException(status_code=500, detail="Failed to load predictions from S3 bucket")
    


# pydantic model for request/response
class PredictionRequest(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: Optional[int] = 7  # default to 7 days


class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]  # List of dictionaries with date and prediction
    total_count: int    
    timestamp: str 
    returned_count: int
    metadata: Dict[str, Any] = {}


class PredictionsConfig: 
    BUCKET_NAME =  os.getenv("BUCKET_NAME")
    S3_KEY = os.getenv("AWS_S3_KEY")
    MAX_LIMIT = 30  


config = PredictionsConfig()



@app.get("/predictions", response_model=PredictionResponse)
async def get_predictions(request: Request,
                          bucket_name: str = config.BUCKET_NAME,
                          s3_key: str = config.S3_KEY,
                          limit: Optional[int] = 7,
                          offset: Optional[int] = 0,
): 
    """Gets predictions from S3 bucket within a date range or limit"""

    if limit > config.MAX_LIMIT:
        raise HTTPException(status_code=400, detail=f"Limit cannot exceed {config.MAX_LIMIT} days")     
    
    df = load_predictions_from_s3_bucket(bucket_name, s3_key)
    total_count = len(df)

    if limit: 
        df = df.head(limit)

    predictions = df.to_dict(orient="records")

    return PredictionResponse(
        predictions=predictions,
        total_count=total_count,
        timestamp=datetime.now().isoformat(),
        returned_count=len(predictions),
        metadata={
            "bucket_name": bucket_name,
            "s3_key": s3_key,
            "limit": limit,
            "offset": offset
        }
    )




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)