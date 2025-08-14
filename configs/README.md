# Configs Directory

Configuration files for ML experiments and deployments:

## Structure
- `experiments/` - Experiment configurations (hyperparameters, etc.)
- `models/` - Model architecture and training configurations
- `data/` - Data processing pipeline configurations
- `deployment/` - Model serving and deployment configurations

## File Types
- `*.yaml` - Main configuration files
- `*.json` - Parameter configurations
- `*.env` - Environment-specific settings

## Usage in Airflow
- Configurations loaded dynamically in DAGs
- Version controlled experiment tracking
- Environment-specific deployments (dev/staging/prod)