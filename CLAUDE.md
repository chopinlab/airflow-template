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

The setup includes MLflow for experiment tracking and model management:

**Services:**
- **MLflow Server**: http://localhost:5000
- **Backend Store**: PostgreSQL database
- **Artifact Store**: Local filesystem (`./artifacts`)

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
- `mlflow_example_dag.py` - Complete ML pipeline with MLflow tracking

## Notes

This template repository will likely be expanded with Airflow-specific structure including:
- DAGs directory for workflow definitions
- Plugins directory for custom operators
- Configuration files for Airflow setup
- Requirements files for Python dependencies
- Docker configuration for containerized deployment