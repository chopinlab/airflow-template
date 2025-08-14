from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.datasets import Dataset
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.pytorch

# Dataset 정의
training_data_dataset = Dataset("file:///opt/airflow/data/train/")
model_dataset = Dataset("file:///opt/airflow/models/latest.pth")
inference_results_dataset = Dataset("file:///opt/airflow/data/inference_results.json")

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
    'image_classification_pipeline',
    default_args=default_args,
    description='Complete image classification pipeline - train and inference',
    schedule=timedelta(hours=6),
    catchup=False,
    tags=['pytorch', 'image-classification', 'computer-vision'],
)

class ImageDataset(TorchDataset):
    """커스텀 이미지 데이터셋"""
    
    def __init__(self, data_path, labels_dict, transform=None):
        self.data_path = data_path
        self.labels_dict = labels_dict
        self.transform = transform
        self.class_to_idx = {'cat': 0, 'dog': 1}
        
    def __len__(self):
        return len(self.labels_dict)
    
    def __getitem__(self, idx):
        filenames = list(self.labels_dict.keys())
        filename = filenames[idx]
        
        # 이미지 로드
        img_path = os.path.join(self.data_path, filename)
        image = Image.open(img_path).convert('RGB')
        
        # 텐서로 변환 (간단한 정규화)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # CHW 형식으로
        
        # 레이블
        label_name = self.labels_dict[filename]
        label = self.class_to_idx[label_name]
        
        return image, label

class SimpleCNN(nn.Module):
    """간단한 CNN 모델 (CPU 최적화)"""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # 입력: 3x64x64
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 16x64x64
        self.pool1 = nn.MaxPool2d(2, 2)              # 16x32x32
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # 32x32x32  
        self.pool2 = nn.MaxPool2d(2, 2)              # 32x16x16
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1) # 64x16x16
        self.pool3 = nn.MaxPool2d(2, 2)              # 64x8x8
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def generate_sample_data(**context):
    """샘플 데이터 생성"""
    import subprocess
    
    print("🎨 Generating sample images...")
    result = subprocess.run([
        'python', '/opt/airflow/data/generate_sample_data.py'
    ], capture_output=True, text=True, cwd='/opt/airflow')
    
    if result.returncode == 0:
        print("✅ Sample data generated successfully")
        print(result.stdout)
    else:
        print("❌ Error generating sample data")
        print(result.stderr)
        raise Exception(f"Data generation failed: {result.stderr}")
    
    return "data_generated"

def train_model(**context):
    """PyTorch 모델 훈련"""
    print("🚀 Starting model training...")
    
    # MLflow 설정
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("image_classification")
    
    with mlflow.start_run(run_name=f"training_{context['ds']}") as run:
        
        # 레이블 파일 로드
        with open('/opt/airflow/data/labels.json', 'r') as f:
            labels = json.load(f)
        
        # 데이터셋 생성
        train_dataset = ImageDataset(
            '/opt/airflow/data/train/images',
            labels['train']
        )
        
        val_dataset = ImageDataset(
            '/opt/airflow/data/val/images', 
            labels['val']
        )
        
        # 데이터 로더
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # 모델 초기화
        model = SimpleCNN(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 하이퍼파라미터 로깅
        mlflow.log_params({
            'batch_size': 4,
            'learning_rate': 0.001,
            'epochs': 10,
            'model_type': 'SimpleCNN',
            'optimizer': 'Adam'
        })
        
        print(f"📊 Training data: {len(train_dataset)} samples")
        print(f"📊 Validation data: {len(val_dataset)} samples")
        
        # 훈련 루프
        model.train()
        for epoch in range(10):  # CPU에서 빠른 훈련을 위해 10 에포크
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for batch_idx, (images, labels_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # 정확도 계산
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels_batch.size(0)
                correct_predictions += (predicted == labels_batch).sum().item()
            
            # 에포크별 결과
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct_predictions / total_predictions
            
            print(f"Epoch {epoch+1}/10 - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
            
            # MLflow 로깅
            mlflow.log_metrics({
                'train_loss': epoch_loss,
                'train_accuracy': epoch_acc
            }, step=epoch)
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels_batch in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels_batch.size(0)
                val_correct += (predicted == labels_batch).sum().item()
        
        val_accuracy = val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        print(f"🎯 Validation Results - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        # 최종 메트릭 로깅
        mlflow.log_metrics({
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'total_params': sum(p.numel() for p in model.parameters())
        })
        
        # 모델 저장 (.pth 파일)
        model_path = '/opt/airflow/models/image_classifier.pth'
        os.makedirs('/opt/airflow/models', exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': 'SimpleCNN',
            'num_classes': 2,
            'class_to_idx': {'cat': 0, 'dog': 1},
            'val_accuracy': val_accuracy,
            'epoch': 10
        }, model_path)
        
        print(f"💾 Model saved to: {model_path}")
        
        # MLflow에도 모델 등록
        mlflow.pytorch.log_model(
            model, 
            "model",
            registered_model_name="ImageClassifier"
        )
        
        # XCom에 결과 저장
        context['ti'].xcom_push(key='model_path', value=model_path)
        context['ti'].xcom_push(key='val_accuracy', value=val_accuracy)
        context['ti'].xcom_push(key='run_id', value=run.info.run_id)
        
        return {
            'model_path': model_path,
            'val_accuracy': val_accuracy,
            'run_id': run.info.run_id
        }

def run_inference(**context):
    """훈련된 모델로 추론 실행"""
    print("🔮 Running inference...")
    
    # 훈련 결과 가져오기
    model_path = context['ti'].xcom_pull(task_ids='train_model', key='model_path')
    
    if not model_path or not os.path.exists(model_path):
        raise Exception("Trained model not found!")
    
    # 레이블 파일 로드
    with open('/opt/airflow/data/labels.json', 'r') as f:
        labels = json.load(f)
    
    # 모델 로드
    checkpoint = torch.load(model_path, map_location='cpu')
    model = SimpleCNN(num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    idx_to_class = {0: 'cat', 1: 'dog'}
    
    # 테스트 데이터셋으로 추론
    test_dataset = ImageDataset(
        '/opt/airflow/data/test/images',
        labels['test']
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    predictions = []
    ground_truths = []
    
    print("🎯 Running inference on test data...")
    
    with torch.no_grad():
        for images, true_labels in test_loader:
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            pred_class = idx_to_class[predicted.item()]
            true_class = idx_to_class[true_labels.item()]
            confidence = probabilities[0][predicted].item()
            
            result = {
                'predicted_class': pred_class,
                'true_class': true_class,
                'confidence': confidence,
                'correct': pred_class == true_class
            }
            
            predictions.append(result)
            ground_truths.append(true_class == pred_class)
            
            print(f"  Predicted: {pred_class} (confidence: {confidence:.3f}) | True: {true_class} | ✅" if result['correct'] else f"  Predicted: {pred_class} (confidence: {confidence:.3f}) | True: {true_class} | ❌")
    
    # 전체 정확도 계산
    test_accuracy = sum(ground_truths) / len(ground_truths)
    
    # 결과 저장
    inference_results = {
        'test_accuracy': test_accuracy,
        'total_samples': len(predictions),
        'correct_predictions': sum(ground_truths),
        'timestamp': context['ds'],
        'model_path': model_path,
        'predictions': predictions
    }
    
    results_path = '/opt/airflow/data/inference_results.json'
    with open(results_path, 'w') as f:
        json.dump(inference_results, f, indent=2)
    
    print(f"📊 Test Accuracy: {test_accuracy:.4f}")
    print(f"📁 Results saved to: {results_path}")
    
    # MLflow에 추론 결과 로깅
    mlflow.set_tracking_uri("http://mlflow:5000")
    with mlflow.start_run(run_name=f"inference_{context['ds']}"):
        mlflow.log_metrics({
            'test_accuracy': test_accuracy,
            'total_samples': len(predictions),
            'correct_predictions': sum(ground_truths)
        })
        mlflow.log_artifact(results_path)
    
    context['ti'].xcom_push(key='test_accuracy', value=test_accuracy)
    context['ti'].xcom_push(key='results_path', value=results_path)
    
    return inference_results

def evaluate_model_performance(**context):
    """모델 성능 평가 및 보고서 생성"""
    print("📈 Evaluating model performance...")
    
    val_accuracy = context['ti'].xcom_pull(task_ids='train_model', key='val_accuracy')
    test_accuracy = context['ti'].xcom_pull(task_ids='run_inference', key='test_accuracy')
    
    print("\n" + "="*50)
    print("🎯 MODEL PERFORMANCE REPORT")
    print("="*50)
    print(f"📊 Validation Accuracy: {val_accuracy:.4f}")
    print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
    print(f"📉 Generalization Gap: {abs(val_accuracy - test_accuracy):.4f}")
    
    # 성능 평가
    if test_accuracy > 0.8:
        status = "🎉 EXCELLENT"
    elif test_accuracy > 0.6:
        status = "👍 GOOD"
    else:
        status = "⚠️  NEEDS IMPROVEMENT"
    
    print(f"🏆 Overall Performance: {status}")
    print("="*50)
    
    # 모델 승격 결정
    if test_accuracy > 0.7:
        promotion_decision = "promote_to_production"
        print("✅ Model ready for production deployment")
    else:
        promotion_decision = "retrain_needed"
        print("🔄 Model needs retraining before deployment")
    
    return {
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'promotion_decision': promotion_decision,
        'performance_status': status
    }

# Task 정의
generate_data_task = PythonOperator(
    task_id='generate_sample_data',
    python_callable=generate_sample_data,
    dag=dag,
    outlets=[training_data_dataset]
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
    outlets=[model_dataset]
)

inference_task = PythonOperator(
    task_id='run_inference', 
    python_callable=run_inference,
    dag=dag,
    outlets=[inference_results_dataset]
)

evaluate_task = PythonOperator(
    task_id='evaluate_performance',
    python_callable=evaluate_model_performance,
    dag=dag
)

# Task 의존성
generate_data_task >> train_task >> inference_task >> evaluate_task