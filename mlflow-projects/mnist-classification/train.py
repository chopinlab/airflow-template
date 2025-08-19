#!/usr/bin/env python3
"""
MNIST 손글씨 숫자 분류 모델 훈련
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import numpy as np
import os
from datetime import datetime

class MNISTNet(nn.Module):
    """MNIST를 위한 간단한 CNN 모델"""
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        # 첫 번째 컨볼루션 레이어
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # 완전연결 레이어
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

def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """한 에포크 훈련"""
    model.train()
    train_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}: [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
    
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    
    return avg_loss, accuracy

def test_epoch(model, device, test_loader, criterion):
    """테스트 평가"""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='MNIST PyTorch Training')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='directory to save MNIST data')
    parser.add_argument('--model-dir', type=str, default='./models',
                        help='directory to save models')
    parser.add_argument('--data-limit', type=int, default=None, metavar='N',
                        help='limit number of training samples (for faster training)')
    parser.add_argument('--skip-mlflow-model', action='store_true',
                        help='skip MLflow model logging (only log metrics and params)')
    
    args = parser.parse_args()
    
    # 재현가능성을 위한 시드 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 디바이스 설정 (CPU 사용)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # MLflow 설정
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    mlflow.set_experiment("mnist-classification")
    
    # 데이터 변환 정의
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 데이터 디렉토리 생성
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 데이터셋 로드
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False,
                                  transform=transform)
    
    # 데이터 제한 적용 (빠른 훈련을 위해)
    if args.data_limit and args.data_limit < len(train_dataset):
        train_dataset = torch.utils.data.Subset(train_dataset, range(args.data_limit))
        print(f"Using limited training data: {args.data_limit} samples")
    
    # 테스트 데이터도 비례해서 줄이기
    if args.data_limit and args.data_limit < len(train_dataset):
        test_limit = min(2000, len(test_dataset))  # 테스트는 최대 2000개
        test_dataset = torch.utils.data.Subset(test_dataset, range(test_limit))
        print(f"Using limited test data: {test_limit} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # MLflow 실행 시작
    with mlflow.start_run(run_name=f"mnist_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # 하이퍼파라미터 로깅
        mlflow.log_params({
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'gamma': args.gamma,
            'seed': args.seed,
            'model_type': 'CNN',
            'optimizer': 'Adadelta',
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset)
        })
        
        # 모델, 손실함수, 옵티마이저 초기화
        model = MNISTNet().to(device)
        criterion = nn.NLLLoss()
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
        
        # 모델 파라미터 수 로깅
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_params({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        })
        
        print(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
        # 훈련 루프
        best_accuracy = 0.0
        
        for epoch in range(1, args.epochs + 1):
            print(f"\n=== Epoch {epoch}/{args.epochs} ===")
            
            # 훈련
            train_loss, train_acc = train_epoch(model, device, train_loader, 
                                                optimizer, criterion, epoch)
            
            # 테스트
            test_loss, test_acc = test_epoch(model, device, test_loader, criterion)
            
            # 학습률 스케줄러 업데이트
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # 결과 출력
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # MLflow 메트릭 로깅
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'learning_rate': current_lr
            }, step=epoch)
            
            # 최고 성능 모델 저장
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_model_path = os.path.join(args.model_dir, 'best_mnist_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_accuracy': test_acc,
                    'test_loss': test_loss,
                }, best_model_path)
                
                print(f"💾 New best model saved! Accuracy: {test_acc:.2f}%")
        
        # 최종 결과
        print(f"\n🎯 Training completed!")
        print(f"Best test accuracy: {best_accuracy:.2f}%")
        
        # 최종 메트릭 로깅
        mlflow.log_metrics({
            'final_best_accuracy': best_accuracy,
            'final_train_accuracy': train_acc,
        })
        
        # 모델 MLflow에 등록 (선택적)
        if not args.skip_mlflow_model:
            try:
                mlflow.pytorch.log_model(
                    model,
                    "mnist_model",
                    registered_model_name="MNIST-CNN"
                )
                
                # 최고 성능 모델 파일도 아티팩트로 저장
                if os.path.exists(best_model_path):
                    mlflow.log_artifact(best_model_path, "models")
                    
                print("✅ Model logged to MLflow successfully!")
            except Exception as e:
                print(f"⚠️  MLflow model logging failed (but training succeeded): {e}")
        else:
            print("⚠️  Skipping MLflow model logging (--skip-mlflow-model flag)")
        
        print(f"🏃 Run ID: {mlflow.active_run().info.run_id}")

if __name__ == '__main__':
    main()