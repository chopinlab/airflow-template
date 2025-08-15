import bentoml
import mlflow
import os

def deploy_with_bentoml():
    """MLflow 모델을 BentoML로 배포하는 스크립트"""
    
    # MLflow 추적 URI 설정
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000')
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"✅ MLflow Tracking URI: {mlflow_tracking_uri}")

    model_name = "image-classification-cnn"
    bento_model_name = "image_classifier_bento"

    print(f"🔎 Finding the latest version of model '{model_name}'...")

    try:
        # MLflow에서 최신 모델 정보 가져오기
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
        model_uri = f"models:/{model_name}/{latest_version.version}"
        
        print(f"✅ Found model: {model_uri} (version: {latest_version.version})")

        # MLflow 모델을 BentoML로 임포트
        print(f"🚀 Importing model into BentoML as '{bento_model_name}'...")
        bento_model = bentoml.mlflow.import_model(
            bento_model_name,
            model_uri=model_uri,
            signatures={
                "__call__": {"batchable": False}
            }
        )
        
        print(f"✅ Successfully imported model to BentoML!")
        print(f"   BentoML model tag: {bento_model.tag}")
        print(f"👉 You can now create a 'service.py' and 'bentofile.yaml' to serve this model.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("   Please ensure the model exists in MLflow and that the Airflow container can connect to the MLflow server.")

if __name__ == "__main__":
    deploy_with_bentoml()
