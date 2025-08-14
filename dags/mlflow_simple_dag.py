from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os

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
    'mlflow_simple_integration',
    default_args=default_args,
    description='Simple MLflow integration with Airflow using Python operators',
    schedule=timedelta(days=1),
    catchup=False,
    tags=['mlflow', 'simple', 'integration'],
)

def create_experiment(**context):
    """Create MLflow experiment"""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    client = MlflowClient()
    experiment_name = "airflow_simple_demo"
    
    try:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location="/mlflow/artifacts"
        )
        print(f"✅ Created experiment '{experiment_name}' with ID: {experiment_id}")
        return experiment_id
    except Exception as e:
        if "already exists" in str(e):
            experiment = client.get_experiment_by_name(experiment_name)
            print(f"✅ Experiment '{experiment_name}' already exists")
            return experiment.experiment_id
        else:
            raise

def train_and_log_model(**context):
    """Train model and log to MLflow"""
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("airflow_simple_demo")
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"simple_demo_{context['ds']}") as run:
        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        # Log parameters
        mlflow.log_params({
            'n_estimators': 100,
            'max_depth': 8,
            'random_state': 42,
            'n_features': X.shape[1],
            'n_samples': len(X)
        })
        
        # Log metrics
        mlflow.log_metrics({
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        })
        
        # Log model
        model_info = mlflow.sklearn.log_model(
            model,
            "random_forest_model",
            registered_model_name="SimpleAirflowModel"
        )
        
        print(f"✅ Model logged: {model_info.model_uri}")
        print(f"📊 Train accuracy: {train_accuracy:.4f}")
        print(f"📊 Test accuracy: {test_accuracy:.4f}")
        print(f"🏃 Run ID: {run.info.run_id}")
        
        # Push results to XCom
        context['ti'].xcom_push(key='test_accuracy', value=test_accuracy)
        context['ti'].xcom_push(key='run_id', value=run.info.run_id)
        
        return {
            'test_accuracy': test_accuracy,
            'run_id': run.info.run_id,
            'model_uri': model_info.model_uri
        }

def list_experiments(**context):
    """List all MLflow experiments"""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    
    experiments = client.search_experiments()
    
    print("📋 Available experiments:")
    for exp in experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")
    
    return [{'name': exp.name, 'id': exp.experiment_id} for exp in experiments]

def check_model_registry(**context):
    """Check registered models"""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    
    try:
        models = client.search_registered_models()
        
        print("🏷️  Registered models:")
        for model in models:
            print(f"  - {model.name}")
            
            # Get latest versions
            latest_versions = client.get_latest_versions(model.name)
            for version in latest_versions:
                print(f"    Version {version.version} ({version.current_stage})")
        
        return [{'name': model.name} for model in models]
    except Exception as e:
        print(f"ℹ️  No registered models found or error: {e}")
        return []

def promote_model(**context):
    """Promote model based on performance"""
    test_accuracy = context['ti'].xcom_pull(task_ids='train_and_log', key='test_accuracy')
    
    if test_accuracy is None:
        print("⚠️  No test accuracy found, skipping promotion")
        return
    
    import mlflow
    from mlflow.tracking import MlflowClient
    
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    
    model_name = "SimpleAirflowModel"
    
    try:
        # Get latest model version
        latest_versions = client.search_model_versions(f"name='{model_name}'")
        if not latest_versions:
            print(f"⚠️  No versions found for model {model_name}")
            return
        
        latest_version = max(latest_versions, key=lambda x: int(x.version))
        
        # Promotion logic
        if test_accuracy > 0.85:
            stage = "Production"
            print(f"🎉 Excellent performance ({test_accuracy:.4f})! Promoting to Production")
        elif test_accuracy > 0.75:
            stage = "Staging" 
            print(f"👍 Good performance ({test_accuracy:.4f})! Promoting to Staging")
        else:
            print(f"⚠️  Poor performance ({test_accuracy:.4f}). No promotion.")
            return
        
        # Transition model stage
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage=stage
        )
        
        print(f"✅ Model {model_name} v{latest_version.version} promoted to {stage}")
        
        return {
            'model_name': model_name,
            'version': latest_version.version,
            'stage': stage,
            'accuracy': test_accuracy
        }
        
    except Exception as e:
        print(f"❌ Error promoting model: {e}")
        return None

# Define tasks
create_experiment_task = PythonOperator(
    task_id='create_experiment',
    python_callable=create_experiment,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_and_log',
    python_callable=train_and_log_model,
    dag=dag
)

list_experiments_task = PythonOperator(
    task_id='list_experiments',
    python_callable=list_experiments,
    dag=dag
)

check_registry_task = PythonOperator(
    task_id='check_model_registry',
    python_callable=check_model_registry,
    dag=dag
)

promote_task = PythonOperator(
    task_id='promote_model',
    python_callable=promote_model,
    dag=dag
)

# Define dependencies
create_experiment_task >> train_task >> [list_experiments_task, check_registry_task] >> promote_task