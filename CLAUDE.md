# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

This is a newly initialized Airflow template repository that currently contains:
- An empty README.md file
- Basic git repository structure
- Claude Code settings configuration

## Current State

The repository is in its initial state with no commits yet and minimal file structure. This appears to be a template or starting point for Apache Airflow projects.

## Configuration

- Claude Code permissions are configured in `.claude/settings.local.json` with specific bash command allowances
- The repository uses git for version control but has no commit history yet

## Development Setup

### Prerequisites
- Docker and Docker Compose installed
- Basic understanding of Apache Airflow

### Getting Started

1. **Create required directories**:
   ```bash
   mkdir -p dags logs plugins config
   ```

2. **Set AIRFLOW_UID (Linux/WSL)**:
   ```bash
   echo -e "AIRFLOW_UID=$(id -u)" > .env
   ```

3. **Start Airflow**:
   ```bash
   docker-compose up
   ```

4. **Access Services**:
   - Airflow Web UI: http://localhost:8080 (admin/admin)
   - MLflow Web UI: http://localhost:5000
   - Model Serving API: http://localhost:5001

### Common Commands

```bash
# Start services (includes MLflow)
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Execute airflow commands
docker-compose exec airflow airflow --help

# View MLflow logs
docker-compose logs -f mlflow
```

### Directory Structure
- `dags/` - Put your DAG files here
- `logs/` - Airflow execution logs
- `plugins/` - Custom plugins
- `configs/` - Configuration files
- `models/` - Trained ML models
- `data/` - Raw and processed datasets
- `artifacts/` - MLflow artifacts and outputs
- `mlflow/` - MLflow tracking data
- `notebooks/` - Jupyter notebooks
- `scripts/` - Reusable Python scripts
- `sql/` - SQL queries

### MLflow Integration

The setup includes a complete MLOps pipeline with separated concerns:

**Services:**
- **MLflow Server**: http://localhost:5000 - Experiment tracking and model registry
- **MLflow Training**: Background container for running training jobs
- **MLflow Serving**: http://localhost:5001 - FastAPI model serving service
- **Backend Store**: PostgreSQL database
- **Artifact Store**: Local filesystem (`./artifacts`)

**Architecture:**
- **Airflow**: Orchestrates the ML pipeline workflow
- **MLflow Training Container**: Executes actual model training and inference
- **MLflow Serving Container**: Serves models via REST API
- **MLflow Server**: Manages experiments, runs, and model registry

**Example Usage:**
```python
import mlflow
mlflow.set_tracking_uri("http://mlflow:5000")

with mlflow.start_run():
    mlflow.log_param("param1", value)
    mlflow.log_metric("metric1", value)
    mlflow.sklearn.log_model(model, "model")
```

**Sample DAGs:**
- `hello_world_dag.py` - Basic Airflow functionality
- `mlflow_orchestration_pipeline.py` - MLflow-based ML pipeline orchestration
- `simple_image_classifier.py` - Simple image classification example

**MLflow Projects:**
- `mlflow-projects/image-classification/` - Complete image classification project
  - `train.py` - Model training script
  - `inference.py` - Model inference script
  - `serve.py` - FastAPI model serving service
  - `MLproject` - MLflow project configuration
  - `conda.yaml` - Environment dependencies

## Notes

This template repository will likely be expanded with Airflow-specific structure including:
- DAGs directory for workflow definitions
- Plugins directory for custom operators
- Configuration files for Airflow setup
- Requirements files for Python dependencies
- Docker configuration for containerized deployment