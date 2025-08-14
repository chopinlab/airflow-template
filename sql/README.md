# SQL Directory

SQL queries for data extraction and transformation:

## Structure
- `extract/` - Data extraction queries from source systems
- `transform/` - Data transformation and feature engineering
- `load/` - Data loading and aggregation queries
- `validation/` - Data quality validation queries
- `monitoring/` - Data pipeline monitoring queries

## Naming Convention
- `extract_{source}_{table}.sql`
- `transform_{feature_name}.sql`
- `validate_{dataset}_quality.sql`

## Usage in Airflow
- Executed via PostgresOperator, SQLExecuteQueryOperator
- Templated with Jinja2 for dynamic parameters
- Integrated with data quality checks