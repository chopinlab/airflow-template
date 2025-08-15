from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import json
import asyncio
import os

app = FastAPI(title="MLflow Training API", version="1.0.0")

class TrainingRequest(BaseModel):
    data_path: str = "/data"
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 4

@app.get("/health")
async def health_check():
    """헬스체크"""
    return {"status": "healthy", "service": "MLflow Training API"}

@app.post("/train")
async def trigger_training(request: TrainingRequest):
    """훈련 트리거"""
    try:
        print(f"🚀 Starting training with params: {request.dict()}")
        
        # 훈련 스크립트 실행
        result = subprocess.run([
            'python', 'train.py',
            '--data_path', request.data_path,
            '--epochs', str(request.epochs),
            '--learning_rate', str(request.learning_rate),
            '--batch_size', str(request.batch_size)
        ], capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            # 성공적인 훈련 결과 파싱
            lines = result.stdout.split('\n')
            run_id = None
            for line in lines:
                if 'Run ID:' in line:
                    run_id = line.split('Run ID:')[-1].strip()
                    break
            
            return {
                "status": "success",
                "message": "Training completed successfully",
                "run_id": run_id,
                "stdout": result.stdout[-1000:],  # 마지막 1000자만
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Training failed: {result.stderr}"
            )
            
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408, 
            detail="Training timed out after 30 minutes"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Training execution error: {str(e)}"
        )

@app.post("/inference")
async def trigger_inference(model_uri: str, data_path: str = "/data/test/images"):
    """추론 트리거"""
    try:
        print(f"🔮 Starting inference with model: {model_uri}")
        
        result = subprocess.run([
            'python', 'inference.py',
            '--model_uri', model_uri,
            '--data_path', data_path,
            '--output_path', '/data/inference_results.json'
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            # 추론 결과 읽기
            try:
                with open('/data/inference_results.json', 'r') as f:
                    inference_results = json.load(f)
                
                return {
                    "status": "success",
                    "message": "Inference completed successfully",
                    "results": inference_results.get('summary', {}),
                    "stdout": result.stdout[-1000:]
                }
            except Exception as e:
                return {
                    "status": "success",
                    "message": "Inference completed but couldn't read results",
                    "error": str(e),
                    "stdout": result.stdout[-1000:]
                }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {result.stderr}"
            )
            
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408,
            detail="Inference timed out after 10 minutes"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference execution error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)