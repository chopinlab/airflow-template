# Scripts Directory

Reusable Python scripts for ML operations:

## Structure
- `data_processing/` - Data extraction, transformation scripts
- `training/` - Model training and evaluation scripts  
- `inference/` - Batch and real-time prediction scripts
- `monitoring/` - Model performance monitoring scripts
- `utils/` - Common utility functions and helpers

## Script Categories
- **ETL Scripts**: Data pipeline automation
- **Training Scripts**: Model training with various algorithms
- **Evaluation Scripts**: Model validation and testing
- **Deployment Scripts**: Model serving and API deployment
- **Monitoring Scripts**: Performance tracking and alerting

## Usage in Airflow
- Scripts called via PythonOperator or BashOperator
- Parameterized for different environments
- Integrated with logging and error handling