from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.datasets import Dataset
import os
import json
import mlflow
import subprocess
import requests

# Dataset 정의
training_data_dataset = Dataset("file:///data/train/")
model_dataset = Dataset("mlflow://models/image-classification-cnn/latest")
inference_results_dataset = Dataset("file:///data/inference_results.json")

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mlflow_orchestration_pipeline',
    default_args=default_args,
    description='MLflow-based ML pipeline orchestration with manual BentoML deployment',
    schedule=timedelta(hours=6),
    catchup=False,
    tags=['mlflow', 'pytorch', 'image-classification', 'orchestration', 'bentoml'],
)

def generate_sample_data(**context):
    """샘플 데이터 생성"""
    print("🎨 Generating sample images...")
    result = subprocess.run([
        'python', '/opt/airflow/data/generate_sample_data.py'
    ], capture_output=True, text=True, cwd='/opt/airflow')
    
    if result.returncode == 0:
        print("✅ Sample data generated successfully")
    else:
        raise Exception(f"Data generation failed: {result.stderr}")

def trigger_mlflow_training(**context):
    """MLflow 훈련 작업 트리거"""
    print("🚀 Triggering MLflow training...")
    
    # MLflow 설정
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    
    # MLflow Training 서비스가 준비될 때까지 대기
    import time
    
    max_retries = 30  # 최대 15분 대기
    for attempt in range(max_retries):
        try:
            health_response = requests.get("http://mlflow-training:8000/health", timeout=10)
            if health_response.status_code == 200:
                print("✅ MLflow Training service is ready!")
                break
        except requests.exceptions.ConnectionError:
            pass
        
        print(f"⏳ Waiting for MLflow Training service... (attempt {attempt + 1}/{max_retries})")
        time.sleep(30)  # 30초 대기
    else:
        raise Exception("MLflow Training service did not start within 15 minutes")
    
    try:
        training_request = {"data_path": "/data", "epochs": 10, "learning_rate": 0.001, "batch_size": 4}
        response = requests.post("http://mlflow-training:8000/train", json=training_request, timeout=1800)
        
        if response.status_code == 200:
            result_data = response.json()
            print(f"✅ MLflow training completed successfully: {result_data}")
            
            # XCom에 결과 저장
            context['ti'].xcom_push(key='training_status', value='success')
            context['ti'].xcom_push(key='run_id', value=result_data.get('run_id'))
            
            return result_data
        else:
            raise Exception(f"Training failed: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.Timeout:
        raise Exception("Training timed out after 30 minutes")
    except requests.exceptions.ConnectionError:
        raise Exception("Could not connect to MLflow training service")
    except Exception as e:
        raise Exception(f"Training execution error: {str(e)}")

def get_latest_model_info(**context):
    """최신 모델 정보 조회"""
    print("📋 Getting latest model information...")
    
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    
    try:
        client = mlflow.tracking.MlflowClient()
        model_name = "image-classification-cnn"
        
        # 최신 모델 버전 조회
        latest_versions = client.get_latest_versions(
            model_name, 
            stages=["Production", "Staging", "None"]
        )
        
        if latest_versions:
            latest_version = latest_versions[0]
            model_uri = f"models:/{model_name}/{latest_version.version}"
            
            # 모델 정보
            model_info = {
                'model_name': model_name,
                'version': latest_version.version,
                'stage': latest_version.current_stage,
                'model_uri': model_uri,
                'run_id': latest_version.run_id
            }
            
            print(f"📝 Latest model: {model_uri}")
            
            # XCom에 저장
            context['ti'].xcom_push(key='model_info', value=model_info)
            
            return model_info
        else:
            raise Exception("No model versions found")
    except Exception as e:
        raise Exception(f"Failed to get model info: {str(e)}")

def trigger_mlflow_inference(**context):
    """MLflow 추론 작업 트리거"""
    print("🔮 Triggering MLflow inference...")
    
    model_info = context['ti'].xcom_pull(task_ids='get_model_info', key='model_info')
    
    if not model_info:
        raise Exception("Model information not available")
    
    try:
        response = requests.post(
            "http://mlflow-training:8000/inference",
            params={"model_uri": model_info['model_uri'], "data_path": "/data/test/images"},
            timeout=600
        )
        
        if response.status_code == 200:
            result_data = response.json()
            print(f"✅ MLflow inference completed successfully: {result_data}")
            summary = result_data.get('results', {})
            context['ti'].xcom_push(key='inference_summary', value=summary)
        else:
            raise Exception(f"Inference failed: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.Timeout:
        raise Exception("Inference timed out after 10 minutes")
    except requests.exceptions.ConnectionError:
        raise Exception("Could not connect to MLflow inference service")
    except Exception as e:
        raise Exception(f"Inference execution error: {str(e)}")

def validate_pipeline(**context):
    """파이프라인 결과 검증 (배포 제외)"""
    print("✅ Validating pipeline results...")
    training_status = context['ti'].xcom_pull(task_ids='trigger_training', key='training_status')
    model_info = context['ti'].xcom_pull(task_ids='get_model_info', key='model_info')
    inference_summary = context['ti'].xcom_pull(task_ids='trigger_inference', key='inference_summary')
    
    validation_results = {
        'training_completed': training_status == 'success',
        'model_available': model_info is not None,
        'inference_completed': inference_summary is not None,
        'pipeline_success': True
    }
    
    if not all(k for k in [validation_results['training_completed'], validation_results['model_available'], validation_results['inference_completed']]):
        validation_results['pipeline_success'] = False
        print(f"❌ Pipeline validation failed. Results: {validation_results}")
        raise Exception("Pipeline validation failed")
    
    print("🎉 Pre-deployment validation successful!")
    return validation_results

# Task 정의
generate_data_task = PythonOperator(
    task_id='generate_sample_data',
    python_callable=generate_sample_data,
    outlets=[training_data_dataset],
    dag=dag,
)

train_task = PythonOperator(
    task_id='trigger_training',
    python_callable=trigger_mlflow_training,
    inlets=[training_data_dataset],
    outlets=[model_dataset],
    dag=dag,
)

get_model_task = PythonOperator(
    task_id='get_model_info',
    python_callable=get_latest_model_info,
    dag=dag,
)

inference_task = PythonOperator(
    task_id='trigger_inference',
    python_callable=trigger_mlflow_inference,
    inlets=[model_dataset],
    outlets=[inference_results_dataset],
    dag=dag,
)

manual_deployment_instructions = BashOperator(
    task_id='manual_deployment_instructions',
    bash_command=(
        "echo '\n' && "
        "echo '################################################################################' && "
        "echo '✅ Model training and validation complete.' && "
        "echo '👉 To deploy this model with BentoML, please run the following command in your shell:' && "
        "echo '   docker-compose exec airflow bash -c \"python /opt/airflow/scripts/deploy_with_bentoml.py\"' && "
        "echo '################################################################################' && "
        "echo '\n'"
    ),
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_pipeline',
    python_callable=validate_pipeline,
    dag=dag,
)

# Task 의존성 정의
generate_data_task >> train_task >> get_model_task >> inference_task >> validate_task >> manual_deployment_instructions