import argparse
import os
import json
import torch
from PIL import Image
import numpy as np
import mlflow
import mlflow.pytorch

def load_model(model_uri):
    """MLflow에서 모델 로드"""
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    return model

def preprocess_image(image_path):
    """이미지 전처리"""
    image = Image.open(image_path).convert('RGB')
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # BCHW 형식으로
    return image

def run_inference(model_uri, data_path, output_path):
    """추론 실행"""
    print("🔮 Starting inference...")
    
    # MLflow 실험 시작
    with mlflow.start_run() as run:
        mlflow.log_param("model_uri", model_uri)
        mlflow.log_param("data_path", data_path)
        
        # 모델 로드
        model = load_model(model_uri)
        print(f"✅ Model loaded from: {model_uri}")
        
        # 클래스 매핑
        idx_to_class = {0: 'cat', 1: 'dog'}
        
        # 테스트 이미지들 처리
        results = {}
        correct_predictions = 0
        total_predictions = 0
        
        # 레이블 파일이 있는 경우 정확도 계산
        labels_file = os.path.join(os.path.dirname(data_path), 'labels.json')
        test_labels = {}
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                all_labels = json.load(f)
                test_labels = all_labels.get('test', {})
        
        # 이미지 파일들 처리
        for filename in os.listdir(data_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.rgb')):
                image_path = os.path.join(data_path, filename)
                
                try:
                    # 이미지 전처리
                    image_tensor = preprocess_image(image_path)
                    
                    # 추론
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][predicted_class_idx].item()
                    
                    predicted_class = idx_to_class[predicted_class_idx]
                    
                    results[filename] = {
                        'predicted_class': predicted_class,
                        'confidence': float(confidence),
                        'probabilities': {
                            'cat': float(probabilities[0][0]),
                            'dog': float(probabilities[0][1])
                        }
                    }
                    
                    # 정확도 계산 (라벨이 있는 경우)
                    if filename in test_labels:
                        true_label = test_labels[filename]
                        is_correct = (predicted_class == true_label)
                        results[filename]['true_label'] = true_label
                        results[filename]['correct'] = is_correct
                        
                        if is_correct:
                            correct_predictions += 1
                        total_predictions += 1
                    
                    print(f"📸 {filename}: {predicted_class} (confidence: {confidence:.3f})")
                    
                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")
                    results[filename] = {'error': str(e)}
        
        # 전체 결과 요약
        summary = {
            'total_images': len(results),
            'successful_predictions': len([r for r in results.values() if 'predicted_class' in r])
        }
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            summary['accuracy'] = float(accuracy)
            summary['correct_predictions'] = correct_predictions
            summary['total_with_labels'] = total_predictions
            mlflow.log_metric("inference_accuracy", accuracy)
            print(f"📊 Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
        
        # 결과 저장
        final_results = {
            'summary': summary,
            'predictions': results,
            'run_id': run.info.run_id
        }
        
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # MLflow에 결과 로깅
        mlflow.log_artifact(output_path, "inference_results")
        mlflow.log_param("total_images", summary['total_images'])
        
        print(f"✅ Inference completed! Results saved to: {output_path}")
        print(f"📝 Run ID: {run.info.run_id}")
        
        return run.info.run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on image classification model')
    parser.add_argument('--model_uri', type=str, required=True, help='MLflow model URI')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--output_path', type=str, default='/data/inference_results.json', help='Output file path')
    
    args = parser.parse_args()
    
    # MLflow tracking URI 설정
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    mlflow.set_experiment("image_classification")
    
    run_inference(args.model_uri, args.data_path, args.output_path)