# Models Directory

This directory contains trained ML models and related files:

## Structure
- `trained/` - Production-ready trained models
- `experiments/` - Experimental models and checkpoints
- `configs/` - Model configuration files
- `metrics/` - Model performance metrics and validation results

## Naming Convention
- Models: `{model_type}_{version}_{date}.pkl`
- Configs: `{model_type}_config_{version}.yaml`
- Metrics: `{model_type}_metrics_{version}.json`

## Usage
Models are typically loaded and used in Airflow DAGs for:
- Batch prediction tasks
- Model validation and testing
- A/B testing comparisons