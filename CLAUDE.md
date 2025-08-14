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

4. **Access Airflow**:
   - Web UI: http://localhost:8080
   - Default credentials: admin/admin

### Common Commands

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Execute airflow commands
docker-compose exec airflow airflow --help
```

### Directory Structure
- `dags/` - Put your DAG files here
- `logs/` - Airflow execution logs
- `plugins/` - Custom plugins
- `config/` - Configuration files

## Notes

This template repository will likely be expanded with Airflow-specific structure including:
- DAGs directory for workflow definitions
- Plugins directory for custom operators
- Configuration files for Airflow setup
- Requirements files for Python dependencies
- Docker configuration for containerized deployment