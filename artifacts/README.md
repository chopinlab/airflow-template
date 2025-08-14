# Artifacts Directory

ML artifacts and outputs from training pipelines:

## Structure
- `models/` - Serialized model files and weights
- `plots/` - Generated visualizations and charts
- `reports/` - Training reports and summaries
- `logs/` - Detailed training and validation logs
- `metadata/` - Model metadata and lineage information

## Artifact Management
- Automated cleanup of old artifacts
- Version tracking with MLflow integration
- Automated archival to object storage
- Metadata tracking for model governance

## Integration
- Linked with MLflow Model Registry
- Accessible through Airflow Variables/XCom
- Monitored for drift and performance degradation