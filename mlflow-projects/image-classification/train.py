import argparse
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

def train_model(data_path, epochs, learning_rate, batch_size):
    """PyTorch 모델 훈련"""
    print("🚀 Starting model training...")
    
    # MLflow 실험 시작
    with mlflow.start_run() as run:
        
        # 하이퍼파라미터 로깅
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        
        # 레이블 파일 로드
        with open(os.path.join(data_path, 'labels.json'), 'r') as f:
            labels = json.load(f)
        
        # 데이터셋 생성
        train_dataset = ImageDataset(
            os.path.join(data_path, 'train/images'),
            labels['train']
        )
        
        val_dataset = ImageDataset(
            os.path.join(data_path, 'val/images'), 
            labels['val']
        )
        
        # 데이터 로더
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"📊 Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        # 모델, 손실함수, 옵티마이저
        model = SimpleCNN(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 훈련 루프
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # 훈련
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * train_correct / train_total
            
            # 검증
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            # MLflow에 메트릭 로깅
            mlflow.log_metric("train_loss", train_loss/len(train_loader), step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss/len(val_loader), step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
            
            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                mlflow.log_metric("best_val_accuracy", best_val_acc)
        
        # 최종 모델 저장
        mlflow.pytorch.log_model(
            model, 
            "model",
            registered_model_name="image-classification-cnn"
        )
        
        print(f"✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        print(f"📝 Run ID: {run.info.run_id}")
        
        return run.info.run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # MLflow tracking URI 설정
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    mlflow.set_experiment("image_classification")
    
    train_model(args.data_path, args.epochs, args.learning_rate, args.batch_size)