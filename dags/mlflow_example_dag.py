from datetime import datetime, timedelta
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mlflow_example_dag',
    default_args=default_args,
    description='MLflow experiment tracking example',
    schedule=timedelta(days=1),
    catchup=False,
    tags=['mlflow', 'machine-learning', 'example'],
)

def generate_sample_data(**context):
    """Generate sample classification dataset"""
    print("Generating sample dataset...")
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save to data directory
    df.to_csv('/opt/airflow/data/sample_dataset.csv', index=False)
    print(f"Dataset saved with shape: {df.shape}")
    
    return f"Generated dataset with {len(df)} samples"

def train_model(**context):
    """Train ML model with MLflow tracking"""
    print("Starting model training with MLflow...")
    
    # Set MLflow tracking URI (set in docker-compose)
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    
    # Set experiment
    experiment_name = "airflow_ml_pipeline"
    mlflow.set_experiment(experiment_name)
    
    # Load data
    df = pd.read_csv('/opt/airflow/data/sample_dataset.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"airflow_run_{context['ds']}") as run:
        # Model parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('train_samples', len(X_train))
        mlflow.log_metric('test_samples', len(X_test))
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "random_forest_model",
            registered_model_name="AirflowRandomForest"
        )
        
        # Log dataset info
        mlflow.log_param('dataset_shape', str(df.shape))
        mlflow.log_param('feature_count', X.shape[1])
        
        # Save model artifact locally
        model_path = f'/opt/airflow/models/rf_model_{context["ds"]}.pkl'
        import joblib
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, "local_models")
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"MLflow Run ID: {run.info.run_id}")
        
        return {
            'run_id': run.info.run_id,
            'accuracy': accuracy,
            'model_path': model_path
        }

def evaluate_model(**context):
    """Evaluate and validate the trained model"""
    print("Evaluating model performance...")
    
    # Get previous task output
    ti = context['ti']
    train_results = ti.xcom_pull(task_ids='train_model')
    
    print(f"Retrieved model with accuracy: {train_results['accuracy']:.4f}")
    
    # Load the model
    import joblib
    model = joblib.load(train_results['model_path'])
    
    # Load test data
    df = pd.read_csv('/opt/airflow/data/sample_dataset.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Make predictions on full dataset for validation
    y_pred = model.predict(X)
    full_accuracy = accuracy_score(y, y_pred)
    
    print(f"Full dataset accuracy: {full_accuracy:.4f}")
    
    # Model validation logic
    if train_results['accuracy'] > 0.8:
        status = "APPROVED"
        print("✅ Model performance is satisfactory")
    else:
        status = "REJECTED"
        print("❌ Model performance below threshold")
    
    return {
        'validation_status': status,
        'full_accuracy': full_accuracy,
        'run_id': train_results['run_id']
    }

# Define tasks
generate_data_task = PythonOperator(
    task_id='generate_data',
    python_callable=generate_sample_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

# Task dependencies
generate_data_task >> train_model_task >> evaluate_model_task