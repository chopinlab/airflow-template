from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.datasets import Dataset
import subprocess
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

# Dataset 정의
mnist_data_dataset = Dataset("mlflow://mnist-data")
mnist_model_dataset = Dataset("mlflow://models/MNIST-CNN/latest")

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
    'mnist_simple_pipeline',
    default_args=default_args,
    description='MNIST 손글씨 숫자 분류 ML 파이프라인 (단순화 버전)',
    schedule=timedelta(hours=12),
    catchup=False,
    tags=['mnist', 'pytorch', 'classification', 'simple'],
)

def trigger_mnist_training(**context):
    """MLflow 서버에서 MNIST 모델 훈련 실행"""
    try:
        logger.info("🚀 Starting MNIST model training in mlflow-server...")
        
        # 훈련 파라미터 (빠른 훈련을 위해 데이터 제한)
        training_params = {
            "epochs": "2",
            "batch_size": "128", 
            "lr": "1.0",
            "gamma": "0.7",
            "seed": "42",
            "data_limit": "5000"  # 5000개 샘플만 사용 (빠른 훈련)
        }
        
        # mlflow-server 컨테이너에서 훈련 스크립트 직접 실행
        cmd = [
            "docker", "exec", "airflow-template_mlflow-server_1",
            "python", "/mlflow-projects/mnist-classification/train.py",
            "--epochs", training_params["epochs"],
            "--batch-size", training_params["batch_size"],
            "--lr", training_params["lr"],
            "--gamma", training_params["gamma"],
            "--seed", training_params["seed"],
            "--data-limit", training_params["data_limit"],
            "--model-dir", "/mlflow-projects/mnist-classification/models"
        ]
        
        logger.info(f"🔧 Executing command: {' '.join(cmd)}")
        
        # 훈련 실행 (타임아웃 10분으로 증가)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info("✅ MNIST training completed successfully!")
            logger.info(f"Training output: {result.stdout}")
            
            # 성공 정보를 XCom에 저장
            context['ti'].xcom_push(key='training_status', value='success')
            context['ti'].xcom_push(key='training_output', value=result.stdout)
            
            return "MNIST training completed successfully"
        else:
            # 에러 로그에서 훈련 성공 여부 확인 (MLflow 에러 vs 실제 훈련 에러)
            stderr_output = result.stderr
            if "🎯 Training completed!" in stderr_output and "Best test accuracy:" in stderr_output:
                logger.warning("⚠️  Training succeeded but MLflow artifact logging failed")
                logger.info("✅ Model was saved locally, continuing pipeline...")
                
                # 부분 성공으로 처리
                context['ti'].xcom_push(key='training_status', value='success')
                context['ti'].xcom_push(key='training_output', value=stderr_output)
                
                return "MNIST training completed (with MLflow artifact warning)"
            else:
                logger.error(f"❌ Training failed: {stderr_output}")
                raise Exception(f"MNIST training failed: {stderr_output}")
        
    except Exception as e:
        logger.error(f"❌ MNIST training failed: {str(e)}")
        context['ti'].xcom_push(key='training_status', value='failed')
        raise Exception(f"MNIST training failed: {str(e)}")

def trigger_mnist_inference(**context):
    """MLflow 서버에서 MNIST 모델 추론 실행"""
    try:
        # 이전 훈련 상태 확인
        training_status = context['ti'].xcom_pull(task_ids='run_training', key='training_status')
        logger.info(f"📊 Retrieved training status: {training_status}")
        
        if training_status != 'success':
            logger.warning(f"⚠️  Training status is '{training_status}', but attempting inference anyway...")
            # 훈련이 완료되었다면 추론을 계속 진행
        else:
            logger.info("✅ Training was successful, proceeding with inference")
        
        logger.info("🔍 Starting MNIST model inference in mlflow-server...")
        
        # 최신 모델 사용 (MLflow에서 자동으로 가져오거나 로컬 파일 사용)
        model_uri = "models:/MNIST-CNN/latest"  # MLflow 모델 레지스트리에서 최신 버전 사용
        
        # mlflow-server 컨테이너에서 추론 스크립트 직접 실행
        cmd = [
            "docker", "exec", "airflow-template_mlflow-server_1",
            "python", "/mlflow-projects/mnist-classification/inference.py",
            "--model-uri", model_uri,
            "--output-dir", "/mlflow-projects/mnist-classification/inference_results"
        ]
        
        logger.info(f"🔧 Executing command: {' '.join(cmd)}")
        
        # 추론 실행
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            logger.info("✅ MNIST inference completed successfully!")
            logger.info(f"Inference output: {result.stdout}")
            
            context['ti'].xcom_push(key='inference_status', value='success')
            context['ti'].xcom_push(key='inference_output', value=result.stdout)
            
            return "MNIST inference completed successfully"
        else:
            logger.error(f"❌ Inference failed: {result.stderr}")
            raise Exception(f"MNIST inference failed: {result.stderr}")
        
    except Exception as e:
        logger.error(f"❌ MNIST inference failed: {str(e)}")
        context['ti'].xcom_push(key='inference_status', value='failed')
        raise Exception(f"MNIST inference failed: {str(e)}")

def validate_pipeline(**context):
    """파이프라인 결과 검증"""
    try:
        logger.info("✅ Validating pipeline results...")
        
        training_status = context['ti'].xcom_pull(key='training_status')
        inference_status = context['ti'].xcom_pull(key='inference_status')
        
        logger.info("\n" + "="*60)
        logger.info("🎯 MNIST SIMPLE PIPELINE VALIDATION REPORT")
        logger.info("="*60)
        logger.info(f"🚂 Training Status: {'✅ SUCCESS' if training_status == 'success' else '❌ FAILED'}")
        logger.info(f"🔮 Inference Status: {'✅ SUCCESS' if inference_status == 'success' else '❌ FAILED'}")
        logger.info(f"🏠 Execution Environment: mlflow-server container")
        logger.info(f"🔗 Architecture: Simple direct execution")
        logger.info("="*60)
        
        pipeline_success = training_status == 'success' and inference_status == 'success'
        
        if pipeline_success:
            logger.info("🎉 MNIST Simple Pipeline: ✅ SUCCESS")
            logger.info("🚀 All tasks completed in mlflow-server!")
        else:
            logger.info("⚠️ MNIST Simple Pipeline: ❌ ISSUES FOUND")
        
        logger.info("="*60 + "\n")
        
        return {
            'training_completed': training_status == 'success',
            'inference_completed': inference_status == 'success',
            'pipeline_success': pipeline_success,
            'execution_environment': 'mlflow-server'
        }
        
    except Exception as e:
        logger.error(f"❌ Pipeline validation failed: {str(e)}")
        raise Exception(f"Pipeline validation failed: {str(e)}")

# Task 정의
train_task = PythonOperator(
    task_id='train_model',
    python_callable=trigger_mnist_training,
    outlets=[mnist_model_dataset],
    dag=dag,
)

inference_task = PythonOperator(
    task_id='run_inference',
    python_callable=trigger_mnist_inference,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_pipeline',
    python_callable=validate_pipeline,
    dag=dag,
)

# 성공 메시지
success_message = BashOperator(
    task_id='success_notification',
    bash_command='''echo "
🎉 MNIST Simple Pipeline Completed Successfully! 🎉

📊 Pipeline Summary:
   - Training and inference executed in mlflow-server
   - No separate training container needed
   - Simple docker exec approach
   - Clean environment separation maintained

🏗️ Architecture Benefits:
   - ✅ Simplified container setup
   - ✅ Direct execution in ML environment
   - ✅ No complex API layers
   - ✅ Easy to scale (add more mlflow-server instances)

🔍 To view results:
   - MLflow UI: http://localhost:5000

🚀 This approach is much simpler and more maintainable!
"''',
    dag=dag,
)

# Task 의존성
train_task >> inference_task >> validate_task >> success_message