# Data Directory

This directory manages data assets for ML pipelines:

## Structure
- `raw/` - Raw, unprocessed data from various sources
- `processed/` - Cleaned and transformed data ready for ML
- `features/` - Feature engineered datasets
- `splits/` - Train/validation/test dataset splits
- `external/` - External reference data (lookup tables, etc.)

## Data Pipeline Flow
1. **Raw Data** → Extract from sources (databases, APIs, files)
2. **Processed Data** → Clean, validate, transform
3. **Features** → Engineer features for ML models  
4. **Splits** → Create train/validation/test datasets

## Data Validation
- Use Great Expectations for data quality checks
- Store expectation suites in `expectations/` subdirectory
- Validation results logged to Airflow task logs