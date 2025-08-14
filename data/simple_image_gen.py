#!/usr/bin/env python3
"""
간단한 이미지 생성 스크립트 (pandas 의존성 없음)
"""

import os
import json
import random
from PIL import Image, ImageDraw

def create_sample_image(class_name, width=64, height=64):
    """클래스별 특징을 가진 샘플 이미지 생성"""
    
    # 새 이미지 생성
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    if class_name == "cat":
        # 고양이: 주황/갈색 원들
        base_color = (255, 140, 0)  # 주황색
        for _ in range(20):
            x = random.randint(0, width-10)
            y = random.randint(0, height-10)
            radius = random.randint(3, 8)
            color_variation = (
                base_color[0] + random.randint(-50, 50),
                base_color[1] + random.randint(-50, 50), 
                base_color[2] + random.randint(-50, 50)
            )
            # 범위 제한
            color_variation = tuple(max(0, min(255, c)) for c in color_variation)
            draw.ellipse([x, y, x+radius, y+radius], fill=color_variation)
    else:  # dog
        # 개: 갈색/베이지 사각형들
        base_color = (139, 69, 19)  # 갈색
        for _ in range(15):
            x = random.randint(0, width-15)
            y = random.randint(0, height-15)
            w = random.randint(5, 12)
            h = random.randint(5, 12)
            color_variation = (
                base_color[0] + random.randint(-40, 40),
                base_color[1] + random.randint(-30, 30),
                base_color[2] + random.randint(-10, 30)
            )
            # 범위 제한
            color_variation = tuple(max(0, min(255, c)) for c in color_variation)
            draw.rectangle([x, y, x+w, y+h], fill=color_variation)
    
    return img

def generate_dataset():
    """레이블 파일 기반으로 샘플 데이터셋 생성"""
    
    # 현재 디렉토리 기준으로 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    labels_file = os.path.join(current_dir, 'labels.json')
    
    # 레이블 파일 로드
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    base_path = current_dir
    
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