# Gold Price Prediction System

A serverless solution to predict gold prices using real-time data from Yahoo Finance API, AWS S3 for storage, and Prefect for workflow orchestration.

## Features

- Real-time gold price data ingestion from Yahoo Finance
- Machine learning-based price prediction
- Data and model artifact storage on AWS S3
- Automated data pipelines and workflow management with Prefect
- FastAPI backend for serving predictions

## Prerequisites

- Python 3.12+
- AWS account and S3 bucket
- AWS CLI configured (`aws configure`)
- Prefect installed and configured

## Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/peniel18/gold-price-prediction-system.git
   cd gold-price-prediction-system
   ```

2. **Install uv package manager: **
   ``` sh 
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create and activate a virtual environment:**
   ```sh
   uv venv 
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. **Install dependencies:**
   ```sh
   uv pip install -r requirements.txt
   ```

5. **Set up your `PYTHONPATH`:**
   ```sh
   export PYTHONPATH="$PYTHONPATH:/path/to/your/project"
   ```

6. **Configure AWS S3:**
   - Install AWS CLI: `uv pip install awscli`
   - Configure credentials:
     ```sh
     aws configure
     ```
   - Ensure your S3 bucket exists and update your configs with the bucket name.

7. **Configure Prefect:**
   - Install Prefect: `uv pip install prefect`
   - Set up Prefect backend/cloud if needed.
   - Register and run flows as described in your pipeline scripts.

## Usage

- **Run the main FastAPI server:**
  ```sh
  python main.py
  ```

- **Run Prefect flows:**
  - Register and execute flows as defined in your Prefect pipeline scripts.

## Project Structure

- `main.py` – FastAPI app entry point
- `gold_prediction/` – Core ML and pipeline code
- `configs/` – YAML configuration files
- `Artifacts/` – Data and model artifacts (synced with S3)
- `logs/` – Log files

## Notes

- All data and model artifacts are stored in your configured AWS S3 bucket.
- Prefect orchestrates data ingestion, transformation, and model training pipelines.
- Update configuration files in `configs/` as needed for your environment.

## License

This project is licensed under the MIT License.

