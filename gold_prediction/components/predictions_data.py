from gold_prediction.exception.exception import CustomException
from gold_prediction.logging.logger import logging
import hopsworks
import pandas as pd 



class PredictionsFeatures: 

    def make_predictions_data(start_date=None, end_date=None) -> pd.DataFrame:
        """
        Create training data based on start and end dates


        Args: 
            start_date: Start date in 'YYYY-MM-DD' format or datetime object.
            end_date: End date in 'YYYY-MM-DD' format or datetime object.
        Returns: 
            pd.DataFrame: Data with computed features.
        
        """
    

        if start_date is None:
            start_date = "2025-02-01"
        if end_date is None:
            end_date = "2025-07-31"
        # change this 
        dates = pd.date_range(start=start_date, end=end_date)
        df = pd.DataFrame()
        df["dates"] = pd.to_datetime(dates)

        df.index = pd.to_datetime(df['dates'])
    
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['dayofyear'] = df.index.dayofyear
        df['quarter'] = df.index.quarter

        features = [
            "month", "year", "day", "dayofweek", "is_weekend",
            "dayofyear", "quarter" , 
        ]

        df["Agg_mean"] = df[features].mean(axis=1)
        df["Agg_max"] = df[features].max(axis=1)
        df["Agg_std"] = df[features].std(axis=1)
        df["Agg_min"] = df[features].min(axis=1)
        df["Kurt"] = df[features].kurt(axis=1)
        df["skewness"] = df[features].skew(axis=1)
        
       
        return df 





    def create_or_get_feature_store(self):
        try: 
            pass 
        except: 
            pass 


    def store_data_on_feature_store(self):
        pass 


    def IntializeFeatures(self):
        pass 



