#!/usr/bin/env python3
"""
MNIST 모델 추론 스크립트
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import numpy as np
import json
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용
import matplotlib.pyplot as plt

class MNISTNet(nn.Module):
    """MNIST를 위한 간단한 CNN 모델"""
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def visualize_predictions(model, device, test_loader, num_samples=10, save_path="predictions.png"):
    """예측 결과 시각화"""
    model.eval()
    
    # 샘플 데이터 가져오기
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # 예측
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
    
    # 시각화
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        img = images[i].cpu().squeeze()
        pred = predictions[i].cpu().item()
        true = labels[i].item()
        
        axes[i].imshow(img, cmap='gray')
        color = 'green' if pred == true else 'red'
        axes[i].set_title(f'Pred: {pred}, True: {true}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

def run_inference(model, device, test_loader):
    """전체 테스트셋에 대한 추론 실행"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            probabilities = torch.exp(outputs)  # log_softmax -> softmax
            predictions = outputs.argmax(dim=1)
            
            # 배치 결과 수집
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # 정확도 계산
            correct += predictions.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Inference progress: {batch_idx * len(data)}/{len(test_loader.dataset)} '
                      f'({100. * batch_idx / len(test_loader):.0f}%)')
    
    accuracy = 100. * correct / total
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'accuracy': accuracy,
        'total_samples': total,
        'correct_samples': correct
    }

def calculate_class_metrics(predictions, labels, num_classes=10):
    """클래스별 성능 메트릭 계산"""
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, labels=range(num_classes)
    )
    
    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
    
    class_metrics = {}
    for i in range(num_classes):
        class_metrics[f'class_{i}'] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    return class_metrics, cm.tolist()

def main():
    parser = argparse.ArgumentParser(description='MNIST Model Inference')
    parser.add_argument('--model-uri', type=str, required=True,
                        help='MLflow model URI')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='directory containing MNIST data')
    parser.add_argument('--output-dir', type=str, default='./inference_results',
                        help='directory to save inference results')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='batch size for inference')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # MLflow 설정
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    
    # 데이터 변환 정의 (훈련 시와 동일)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 테스트 데이터셋 로드
    print("Loading MNIST test dataset...")
    test_dataset = datasets.MNIST(args.data_dir, train=False, download=True,
                                  transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Test samples: {len(test_dataset)}")
    
    try:
        # MLflow에서 모델 로드
        print(f"Loading model from: {args.model_uri}")
        model = mlflow.pytorch.load_model(args.model_uri)
        model.to(device)
        
        print("✅ Model loaded successfully")
        
        # MLflow 추론 실행 시작
        with mlflow.start_run(run_name=f"mnist_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # 모델 URI 로깅
            mlflow.log_param('model_uri', args.model_uri)
            mlflow.log_param('test_samples', len(test_dataset))
            mlflow.log_param('batch_size', args.batch_size)
            
            # 추론 실행
            print("🔮 Running inference...")
            results = run_inference(model, device, test_loader)
            
            print(f"🎯 Inference completed!")
            print(f"Accuracy: {results['accuracy']:.2f}%")
            print(f"Correct: {results['correct_samples']}/{results['total_samples']}")
            
            # 클래스별 메트릭 계산
            print("📊 Calculating class-wise metrics...")
            class_metrics, confusion_matrix = calculate_class_metrics(
                results['predictions'], results['labels']
            )
            
            # 예측 시각화
            viz_path = os.path.join(args.output_dir, 'predictions_visualization.png')
            visualize_predictions(model, device, test_loader, save_path=viz_path)
            
            # 결과 저장
            inference_results = {
                'timestamp': datetime.now().isoformat(),
                'model_uri': args.model_uri,
                'overall_accuracy': results['accuracy'],
                'total_samples': results['total_samples'],
                'correct_samples': results['correct_samples'],
                'class_metrics': class_metrics,
                'confusion_matrix': confusion_matrix,
                'sample_predictions': [
                    {
                        'prediction': int(results['predictions'][i]),
                        'label': int(results['labels'][i]),
                        'confidence': float(max(results['probabilities'][i]))
                    }
                    for i in range(min(100, len(results['predictions'])))  # 처음 100개 샘플
                ]
            }
            
            # JSON으로 결과 저장
            results_path = os.path.join(args.output_dir, 'mnist_inference_results.json')
            with open(results_path, 'w') as f:
                json.dump(inference_results, f, indent=2)
            
            print(f"📁 Results saved to: {results_path}")
            print(f"🎨 Visualization saved to: {viz_path}")
            
            # MLflow에 메트릭 로깅
            mlflow.log_metrics({
                'accuracy': results['accuracy'],
                'total_samples': results['total_samples'],
                'correct_samples': results['correct_samples']
            })
            
            # 클래스별 F1 스코어 로깅
            for class_name, metrics in class_metrics.items():
                mlflow.log_metrics({
                    f'{class_name}_precision': metrics['precision'],
                    f'{class_name}_recall': metrics['recall'],
                    f'{class_name}_f1': metrics['f1_score']
                })
            
            # 아티팩트 로깅
            mlflow.log_artifact(results_path, "inference_results")
            mlflow.log_artifact(viz_path, "visualizations")
            
            print(f"✅ Results logged to MLflow")
            print(f"Run ID: {mlflow.active_run().info.run_id}")
            
            return inference_results
            
    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")
        raise

if __name__ == '__main__':
    main()