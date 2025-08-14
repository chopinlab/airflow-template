# Notebooks Directory

Jupyter notebooks for exploration, analysis, and prototyping:

## Structure
- `exploratory/` - Initial data exploration and analysis
- `experiments/` - Model experimentation and hyperparameter tuning
- `validation/` - Model validation and performance analysis
- `reports/` - Generated reports and visualizations

## Naming Convention
- `01_exploration_{dataset}_YYYYMMDD.ipynb`
- `02_experiment_{model_type}_YYYYMMDD.ipynb`
- `03_validation_{model}_YYYYMMDD.ipynb`

## Integration with Airflow
- Notebooks can be executed via PapermillOperator
- Results stored as outputs with parameters
- Automated report generation through DAGs