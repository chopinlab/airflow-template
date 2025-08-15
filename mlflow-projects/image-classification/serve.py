import os
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import numpy as np
import mlflow
import mlflow.pytorch
import uvicorn
from io import BytesIO

app = FastAPI(title="Image Classification API", version="1.0.0")

# 글로벌 모델 변수
model = None
model_version = None

class ModelManager:
    def __init__(self):
        self.current_model = None
        self.current_version = None
        self.idx_to_class = {0: 'cat', 1: 'dog'}
    
    async def load_latest_model(self):
        """최신 모델 로드"""
        try:
            # MLflow에서 최신 모델 가져오기
            client = mlflow.tracking.MlflowClient()
            model_name = "image-classification-cnn"
            
            # 최신 버전 가져오기
            latest_version = client.get_latest_versions(
                model_name, 
                stages=["Production", "Staging", "None"]
            )[0]
            
            model_uri = f"models:/{model_name}/{latest_version.version}"
            
            # 모델이 변경된 경우에만 다시 로드
            if self.current_version != latest_version.version:
                print(f"🔄 Loading model version {latest_version.version}...")
                self.current_model = mlflow.pytorch.load_model(model_uri)
                self.current_model.eval()
                self.current_version = latest_version.version
                print(f"✅ Model loaded successfully: {model_uri}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_image(self, image_bytes):
        """이미지 전처리"""
        try:
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            # 64x64로 리사이즈 (훈련 시와 동일하게)
            image = image.resize((64, 64))
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # BCHW 형식
            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")
    
    async def predict(self, image_bytes):
        """예측 수행"""
        if self.current_model is None:
            await self.load_latest_model()
            if self.current_model is None:
                raise HTTPException(status_code=503, detail="Model not available")
        
        try:
            # 이미지 전처리
            image_tensor = self.preprocess_image(image_bytes)
            
            # 예측
            with torch.no_grad():
                outputs = self.current_model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()
            
            predicted_class = self.idx_to_class[predicted_class_idx]
            
            return {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'probabilities': {
                    'cat': float(probabilities[0][0]),
                    'dog': float(probabilities[0][1])
                },
                'model_version': self.current_version
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# 모델 매니저 인스턴스
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    print("🚀 Starting Image Classification API...")
    
    # MLflow tracking URI 설정
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    
    # 최신 모델 로드 시도
    await model_manager.load_latest_model()

@app.get("/")
async def root():
    """헬스체크"""
    return {
        "message": "Image Classification API", 
        "status": "running",
        "model_loaded": model_manager.current_model is not None,
        "model_version": model_manager.current_version
    }

@app.get("/health")
async def health_check():
    """상세 헬스체크"""
    model_status = model_manager.current_model is not None
    return {
        "status": "healthy" if model_status else "unhealthy",
        "model_loaded": model_status,
        "model_version": model_manager.current_version,
        "mlflow_uri": os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """이미지 분류 예측"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # 파일 읽기
        image_bytes = await file.read()
        
        # 예측 수행
        result = await model_manager.predict(image_bytes)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            **result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.post("/reload-model")
async def reload_model():
    """모델 수동 재로드"""
    try:
        success = await model_manager.load_latest_model()
        if success:
            return {
                "success": True, 
                "message": "Model reloaded successfully",
                "model_version": model_manager.current_version
            }
        else:
            raise HTTPException(status_code=503, detail="Failed to reload model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {e}")

@app.get("/model-info")
async def get_model_info():
    """현재 모델 정보"""
    return {
        "model_loaded": model_manager.current_model is not None,
        "model_version": model_manager.current_version,
        "classes": list(model_manager.idx_to_class.values()),
        "mlflow_tracking_uri": os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    }

if __name__ == "__main__":
    uvicorn.run(
        "serve:app", 
        host="0.0.0.0", 
        port=5001, 
        reload=True,
        log_level="info"
    )