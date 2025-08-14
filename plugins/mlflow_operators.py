"""
Custom MLflow Operators for Airflow
"""
from typing import Any, Dict, Optional
import mlflow
from airflow.models import BaseOperator
from airflow.utils.context import Context


class MLflowStartRunOperator(BaseOperator):
    """
    Operator to start an MLflow run
    """
    
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: str = "http://mlflow:5000",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
    
    def execute(self, context: Context) -> str:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        run_name = self.run_name or f"airflow_run_{context['ds']}"
        run = mlflow.start_run(run_name=run_name)
        
        # Store run_id for downstream tasks
        context['ti'].xcom_push(key='mlflow_run_id', value=run.info.run_id)
        
        self.log.info(f"Started MLflow run: {run.info.run_id}")
        return run.info.run_id


class MLflowLogMetricsOperator(BaseOperator):
    """
    Operator to log metrics to MLflow
    """
    
    def __init__(
        self,
        metrics: Dict[str, Any],
        run_id_task_id: Optional[str] = None,
        tracking_uri: str = "http://mlflow:5000",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.metrics = metrics
        self.run_id_task_id = run_id_task_id
        self.tracking_uri = tracking_uri
    
    def execute(self, context: Context) -> None:
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Get run_id from previous task or start new run
        if self.run_id_task_id:
            run_id = context['ti'].xcom_pull(
                task_ids=self.run_id_task_id, 
                key='mlflow_run_id'
            )
            with mlflow.start_run(run_id=run_id):
                for metric_name, metric_value in self.metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
        else:
            for metric_name, metric_value in self.metrics.items():
                mlflow.log_metric(metric_name, metric_value)
        
        self.log.info(f"Logged metrics: {self.metrics}")


class MLflowLogModelOperator(BaseOperator):
    """
    Operator to log model to MLflow
    """
    
    def __init__(
        self,
        model_path: str,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        run_id_task_id: Optional[str] = None,
        tracking_uri: str = "http://mlflow:5000",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.artifact_path = artifact_path
        self.registered_model_name = registered_model_name
        self.run_id_task_id = run_id_task_id
        self.tracking_uri = tracking_uri
    
    def execute(self, context: Context) -> None:
        import joblib
        
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Load model
        model = joblib.load(self.model_path)
        
        # Get run_id from previous task
        if self.run_id_task_id:
            run_id = context['ti'].xcom_pull(
                task_ids=self.run_id_task_id, 
                key='mlflow_run_id'
            )
            with mlflow.start_run(run_id=run_id):
                mlflow.sklearn.log_model(
                    model,
                    self.artifact_path,
                    registered_model_name=self.registered_model_name
                )
        
        self.log.info(f"Logged model: {self.model_path}")


class MLflowTransitionModelOperator(BaseOperator):
    """
    Operator to transition model stage in MLflow Model Registry
    """
    
    def __init__(
        self,
        model_name: str,
        version: str,
        stage: str,  # "Staging", "Production", "Archived"
        tracking_uri: str = "http://mlflow:5000",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.version = version
        self.stage = stage
        self.tracking_uri = tracking_uri
    
    def execute(self, context: Context) -> None:
        from mlflow.tracking import MlflowClient
        
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient()
        
        client.transition_model_version_stage(
            name=self.model_name,
            version=self.version,
            stage=self.stage
        )
        
        self.log.info(f"Transitioned {self.model_name} v{self.version} to {self.stage}")


class MLflowDeleteRunsOperator(BaseOperator):
    """
    Operator to clean up old MLflow runs
    """
    
    def __init__(
        self,
        experiment_name: str,
        max_runs: int = 100,
        tracking_uri: str = "http://mlflow:5000",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.experiment_name = experiment_name
        self.max_runs = max_runs
        self.tracking_uri = tracking_uri
    
    def execute(self, context: Context) -> None:
        from mlflow.tracking import MlflowClient
        
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient()
        
        experiment = client.get_experiment_by_name(self.experiment_name)
        if not experiment:
            self.log.warning(f"Experiment {self.experiment_name} not found")
            return
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        if len(runs) > self.max_runs:
            runs_to_delete = runs[self.max_runs:]
            for run in runs_to_delete:
                client.delete_run(run.info.run_id)
                self.log.info(f"Deleted run: {run.info.run_id}")
        
        self.log.info(f"Cleanup completed. Kept {min(len(runs), self.max_runs)} runs")