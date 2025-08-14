#!/usr/bin/env python3
"""
샘플 이미지 데이터 생성 스크립트
실제 이미지 대신 컬러 노이즈 이미지 생성
"""

import os
import json
import numpy as np
from PIL import Image
import random

def create_sample_image(class_name, width=64, height=64):
    """클래스별 특징을 가진 샘플 이미지 생성"""
    if class_name == "cat":
        # 고양이: 주황/갈색 계열
        r = np.random.randint(150, 255, (height, width))
        g = np.random.randint(100, 200, (height, width)) 
        b = np.random.randint(50, 150, (height, width))
    else:  # dog
        # 개: 갈색/베이지 계열
        r = np.random.randint(100, 200, (height, width))
        g = np.random.randint(80, 160, (height, width))
        b = np.random.randint(60, 140, (height, width))
    
    # RGB 배열 합치기
    img_array = np.stack([r, g, b], axis=-1).astype(np.uint8)
    
    # PIL Image로 변환
    img = Image.fromarray(img_array)
    return img

def generate_dataset():
    """레이블 파일 기반으로 샘플 데이터셋 생성"""
    
    # 레이블 파일 로드
    with open('/opt/airflow/data/labels.json', 'r') as f:
        labels = json.load(f)
    
    base_path = '/opt/airflow/data'
    
    # 각 split별로 이미지 생성
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_path, split, 'images')
        os.makedirs(split_path, exist_ok=True)
        
        if split in labels:
            for filename, class_name in labels[split].items():
                # 이미지 생성
                img = create_sample_image(class_name)
                
                # 저장
                img_path = os.path.join(split_path, filename)
                img.save(img_path)
                print(f"Generated: {img_path} (class: {class_name})")
    
    print(f"✅ Sample dataset generated successfully!")
    print(f"📁 Data structure:")
    print(f"   - Train: {len(labels.get('train', {}))} images")
    print(f"   - Val: {len(labels.get('val', {}))} images") 
    print(f"   - Test: {len(labels.get('test', {}))} images")

if __name__ == "__main__":
    generate_dataset()