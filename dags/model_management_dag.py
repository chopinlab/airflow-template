from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import mlflow
from mlflow.tracking import MlflowClient

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
    'model_management_dag',
    default_args=default_args,
    description='Complete MLflow model management pipeline',
    schedule=timedelta(hours=6),  # 6시간마다 모델 관리
    catchup=False,
    tags=['mlflow', 'model-management', 'mlops'],
)

def list_all_models(**context):
    """모든 등록된 모델 조회"""
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    
    models = client.search_registered_models()
    model_info = []
    
    for model in models:
        latest_versions = client.get_latest_versions(model.name)
        model_info.append({
            'name': model.name,
            'description': model.description,
            'versions': len(latest_versions),
            'latest_version': latest_versions[0].version if latest_versions else None,
            'current_stage': latest_versions[0].current_stage if latest_versions else None
        })
    
    print(f"Found {len(models)} registered models:")
    for info in model_info:
        print(f"- {info['name']}: v{info['latest_version']} ({info['current_stage']})")
    
    context['ti'].xcom_push(key='models', value=model_info)
    return model_info

def validate_staging_models(**context):
    """Staging 상태 모델들 검증"""
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    
    models = client.search_registered_models()
    validation_results = []
    
    for model in models:
        versions = client.get_latest_versions(model.name, stages=["Staging"])
        
        for version in versions:
            # 모델 메트릭 조회
            run = client.get_run(version.run_id)
            accuracy = run.data.metrics.get('accuracy', 0)
            
            # 검증 규칙
            is_valid = accuracy > 0.8
            
            validation_results.append({
                'model_name': model.name,
                'version': version.version,
                'accuracy': accuracy,
                'is_valid': is_valid,
                'run_id': version.run_id
            })
            
            print(f"Model {model.name} v{version.version}: accuracy={accuracy:.4f}, valid={is_valid}")
    
    context['ti'].xcom_push(key='validation_results', value=validation_results)
    return validation_results

def decide_promotion_action(**context):
    """모델 승격 여부 결정"""
    validation_results = context['ti'].xcom_pull(task_ids='validate_staging_models', key='validation_results')
    
    valid_models = [r for r in validation_results if r['is_valid']]
    
    if valid_models:
        print(f"Found {len(valid_models)} models ready for production")
        return 'promote_to_production'
    else:
        print("No models meet production criteria")
        return 'skip_promotion'

def promote_models_to_production(**context):
    """검증된 모델들을 Production으로 승격"""
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    
    validation_results = context['ti'].xcom_pull(task_ids='validate_staging_models', key='validation_results')
    valid_models = [r for r in validation_results if r['is_valid']]
    
    promoted_models = []
    
    for model_info in valid_models:
        # 기존 Production 모델을 Archived로 변경
        prod_versions = client.get_latest_versions(model_info['model_name'], stages=["Production"])
        for old_version in prod_versions:
            client.transition_model_version_stage(
                name=model_info['model_name'],
                version=old_version.version,
                stage="Archived"
            )
            print(f"Archived old production model: {model_info['model_name']} v{old_version.version}")
        
        # 새 모델을 Production으로 승격
        client.transition_model_version_stage(
            name=model_info['model_name'],
            version=model_info['version'],
            stage="Production"
        )
        
        promoted_models.append(model_info)
        print(f"Promoted to production: {model_info['model_name']} v{model_info['version']}")
    
    context['ti'].xcom_push(key='promoted_models', value=promoted_models)
    return promoted_models

def cleanup_old_model_versions(**context):
    """오래된 모델 버전 정리"""
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    
    models = client.search_registered_models()
    cleanup_results = []
    
    for model in models:
        all_versions = client.search_model_versions(f"name='{model.name}'")
        
        # Archived 상태이고 30일 이상 된 버전들 삭제
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for version in all_versions:
            if (version.current_stage == "Archived" and 
                datetime.fromtimestamp(version.creation_timestamp / 1000) < cutoff_date):
                
                client.delete_model_version(model.name, version.version)
                cleanup_results.append(f"{model.name} v{version.version}")
                print(f"Deleted old version: {model.name} v{version.version}")
    
    print(f"Cleaned up {len(cleanup_results)} old model versions")
    return cleanup_results

def generate_model_report(**context):
    """모델 현황 보고서 생성"""
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    
    models = client.search_registered_models()
    report = {
        'total_models': len(models),
        'production_models': 0,
        'staging_models': 0,
        'model_details': []
    }
    
    for model in models:
        prod_versions = client.get_latest_versions(model.name, stages=["Production"])
        staging_versions = client.get_latest_versions(model.name, stages=["Staging"])
        
        if prod_versions:
            report['production_models'] += 1
        if staging_versions:
            report['staging_models'] += 1
        
        model_detail = {
            'name': model.name,
            'production_version': prod_versions[0].version if prod_versions else None,
            'staging_version': staging_versions[0].version if staging_versions else None,
            'total_versions': len(client.search_model_versions(f"name='{model.name}'"))
        }
        report['model_details'].append(model_detail)
    
    print("\n=== MODEL MANAGEMENT REPORT ===")
    print(f"Total Models: {report['total_models']}")
    print(f"Production Models: {report['production_models']}")
    print(f"Staging Models: {report['staging_models']}")
    print("\nModel Details:")
    for detail in report['model_details']:
        print(f"- {detail['name']}: Prod v{detail['production_version']}, "
              f"Staging v{detail['staging_version']}, Total: {detail['total_versions']}")
    
    return report

def monitor_model_performance(**context):
    """Production 모델 성능 모니터링"""
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    
    models = client.search_registered_models()
    performance_alerts = []
    
    for model in models:
        prod_versions = client.get_latest_versions(model.name, stages=["Production"])
        
        for version in prod_versions:
            run = client.get_run(version.run_id)
            accuracy = run.data.metrics.get('accuracy', 0)
            
            # 성능 임계값 체크
            if accuracy < 0.75:  # 75% 미만이면 알림
                performance_alerts.append({
                    'model_name': model.name,
                    'version': version.version,
                    'accuracy': accuracy,
                    'alert_type': 'LOW_PERFORMANCE'
                })
                print(f"⚠️  Performance Alert: {model.name} v{version.version} accuracy={accuracy:.4f}")
    
    if performance_alerts:
        print(f"Found {len(performance_alerts)} performance alerts")
        # 여기서 Slack/Email 알림 발송 가능
    else:
        print("✅ All production models performing well")
    
    return performance_alerts

# Tasks 정의
list_models_task = PythonOperator(
    task_id='list_all_models',
    python_callable=list_all_models,
    dag=dag
)

validate_models_task = PythonOperator(
    task_id='validate_staging_models',
    python_callable=validate_staging_models,
    dag=dag
)

decide_promotion_task = BranchPythonOperator(
    task_id='decide_promotion_action',
    python_callable=decide_promotion_action,
    dag=dag
)

promote_task = PythonOperator(
    task_id='promote_to_production',
    python_callable=promote_models_to_production,
    dag=dag
)

skip_promotion_task = EmptyOperator(
    task_id='skip_promotion',
    dag=dag
)

cleanup_task = PythonOperator(
    task_id='cleanup_old_versions',
    python_callable=cleanup_old_model_versions,
    dag=dag,
    trigger_rule='none_failed_min_one_success'  # 어떤 브랜치든 성공하면 실행
)

report_task = PythonOperator(
    task_id='generate_model_report',
    python_callable=generate_model_report,
    dag=dag
)

monitor_task = PythonOperator(
    task_id='monitor_model_performance',
    python_callable=monitor_model_performance,
    dag=dag
)

# Task dependencies
list_models_task >> validate_models_task >> decide_promotion_task
decide_promotion_task >> [promote_task, skip_promotion_task]
[promote_task, skip_promotion_task] >> cleanup_task >> [report_task, monitor_task]