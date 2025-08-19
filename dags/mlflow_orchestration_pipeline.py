from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.datasets import Dataset
import os
import json
import mlflow

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
    
    try:
        # Python 모듈로 직접 import해서 실행
        import sys
        sys.path.append('/opt/airflow/data')
        
        # generate_sample_data 모듈 실행
        import generate_sample_data
        
        # 모듈에 main 함수가 있다면 실행, 없다면 스크립트 자체가 실행됨
        if hasattr(generate_sample_data, 'main'):
            generate_sample_data.main()
        
        print("✅ Sample data generated successfully")
        
    except ImportError as e:
        print(f"⚠️ Could not import data generation script: {e}")
        # 폴백: 간단한 데이터 디렉토리 확인
        data_dir = '/opt/airflow/data/train/images'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        print("✅ Data directories ensured to exist")
        
    except Exception as e:
        raise Exception(f"Data generation failed: {str(e)}")

def trigger_mlflow_training(**context):
    """MLflow Projects를 통한 훈련 작업 실행"""
    print("🚀 Running MLflow Project for image classification training...")
    
    # MLflow 설정
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    
    try:
        # MLflow Projects 실행 - 이미지 분류 훈련
        print("📊 Starting MLflow Project execution...")
        
        submitted_run = mlflow.run(
            uri="/opt/airflow/mlflow-projects/image-classification",
            entry_point="train",
            parameters={
                "data_path": "/data",
                "epochs": 10,
                "learning_rate": 0.001,
                "batch_size": 4
            },
            experiment_name="image-classification",
            synchronous=True,  # 동기 실행으로 완료까지 대기
            backend="docker",
            backend_config={
                "image": "mlflow-pytorch:latest",
                "volumes": {
                    "/opt/airflow/mlflow-projects": "/mlflow-projects",
                    "/opt/airflow/data": "/data"
                }
            }
        )
        
        run_id = submitted_run.run_id
        print(f"✅ MLflow Project training completed successfully!")
        print(f"🏃 Run ID: {run_id}")
        
        # 실행 결과 조회
        client = mlflow.tracking.MlflowClient()
        run_info = client.get_run(run_id)
        
        # 메트릭 조회
        final_accuracy = run_info.data.metrics.get('accuracy', 0)
        print(f"📊 Final accuracy: {final_accuracy:.4f}")
        
        # XCom에 결과 저장
        context['ti'].xcom_push(key='training_status', value='success')
        context['ti'].xcom_push(key='run_id', value=run_id)
        context['ti'].xcom_push(key='accuracy', value=final_accuracy)
        
        return {
            'status': 'success',
            'run_id': run_id,
            'accuracy': final_accuracy
        }
        
    except Exception as e:
        print(f"❌ MLflow Project execution failed: {str(e)}")
        context['ti'].xcom_push(key='training_status', value='failed')
        raise Exception(f"MLflow Project training failed: {str(e)}")

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
    """MLflow Projects를 통한 추론 작업 실행"""
    print("🔮 Running MLflow Project for image classification inference...")
    
    model_info = context['ti'].xcom_pull(task_ids='get_model_info', key='model_info')
    
    if not model_info:
        raise Exception("Model information not available")
    
    # MLflow 설정
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    
    try:
        print(f"🎯 Using model: {model_info['model_uri']}")
        
        # MLflow Projects 실행 - 이미지 분류 추론
        submitted_run = mlflow.run(
            uri="/opt/airflow/mlflow-projects/image-classification",
            entry_point="inference",
            parameters={
                "model_uri": model_info['model_uri'],
                "data_path": "/data/test",
                "output_path": "/data/inference_results.json"
            },
            experiment_name="image-classification-inference",
            synchronous=True,
            backend="docker",
            backend_config={
                "image": "mlflow-pytorch:latest",
                "volumes": {
                    "/opt/airflow/mlflow-projects": "/mlflow-projects",
                    "/opt/airflow/data": "/data"
                }
            }
        )
        
        run_id = submitted_run.run_id
        print(f"✅ MLflow Project inference completed successfully!")
        print(f"🏃 Inference Run ID: {run_id}")
        
        # 추론 결과 파일 읽기
        try:
            import json
            with open('/opt/airflow/data/inference_results.json', 'r') as f:
                inference_results = json.load(f)
            
            summary = {
                'total_samples': inference_results.get('total_samples', 0),
                'accuracy': inference_results.get('accuracy', 0),
                'run_id': run_id
            }
            
            print(f"📊 Inference summary: {summary}")
            context['ti'].xcom_push(key='inference_summary', value=summary)
            
            return summary
            
        except Exception as e:
            print(f"⚠️ Could not read inference results file: {e}")
            # 기본 결과 반환
            summary = {'run_id': run_id, 'status': 'completed'}
            context['ti'].xcom_push(key='inference_summary', value=summary)
            return summary
        
    except Exception as e:
        print(f"❌ MLflow Project inference failed: {str(e)}")
        raise Exception(f"MLflow Project inference failed: {str(e)}")

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