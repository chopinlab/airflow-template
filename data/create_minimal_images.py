#!/usr/bin/env python3
"""
PIL 없이 최소한의 이미지 생성
"""

import os
import json

def create_minimal_ppm_image(class_name, width=64, height=64):
    """PPM 형식으로 간단한 이미지 생성"""
    
    if class_name == "cat":
        # 고양이: 주황색 (255, 140, 0)
        r, g, b = 255, 140, 0
    else:  # dog
        # 개: 갈색 (139, 69, 19)
        r, g, b = 139, 69, 19
    
    # PPM 헤더
    ppm_data = f"P3\n{width} {height}\n255\n"
    
    # 픽셀 데이터 (단색)
    for y in range(height):
        for x in range(width):
            # 약간의 변화를 위해 위치 기반 노이즈 추가
            noise_r = (x + y) % 50 - 25
            noise_g = (x * 2 + y) % 40 - 20
            noise_b = (x + y * 2) % 30 - 15
            
            final_r = max(0, min(255, r + noise_r))
            final_g = max(0, min(255, g + noise_g))
            final_b = max(0, min(255, b + noise_b))
            
            ppm_data += f"{final_r} {final_g} {final_b} "
        ppm_data += "\n"
    
    return ppm_data

def convert_ppm_to_simple_format(ppm_data, output_path):
    """PPM을 간단한 RGB 바이너리로 변환해서 저장"""
    
    # 헤더 파싱
    lines = ppm_data.strip().split('\n')
    magic = lines[0]
    dimensions = lines[1].split()
    width, height = int(dimensions[0]), int(dimensions[1])
    max_val = int(lines[2])
    
    # 픽셀 데이터 추출
    pixel_data = []
    for line in lines[3:]:
        if line.strip():
            values = line.strip().split()
            for i in range(0, len(values), 3):
                if i + 2 < len(values):
                    r = int(values[i])
                    g = int(values[i + 1]) 
                    b = int(values[i + 2])
                    pixel_data.extend([r, g, b])
    
    # 바이너리로 저장 (간단한 RGB 형식)
    with open(output_path, 'wb') as f:
        # 간단한 헤더 (width, height)
        f.write(width.to_bytes(4, 'little'))
        f.write(height.to_bytes(4, 'little'))
        # 픽셀 데이터
        f.write(bytes(pixel_data))

def generate_simple_dataset():
    """레이블 파일 기반으로 간단한 데이터셋 생성"""
    
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
                # PPM 이미지 생성
                ppm_data = create_minimal_ppm_image(class_name)
                
                # .jpg를 .rgb로 변경해서 저장
                base_name = filename.replace('.jpg', '.rgb')
                img_path = os.path.join(split_path, base_name)
                
                # 간단한 RGB 바이너리로 저장
                convert_ppm_to_simple_format(ppm_data, img_path)
                print(f"Generated: {img_path} (class: {class_name})")
    
    # 레이블 파일도 .rgb 확장자로 업데이트
    updated_labels = {}
    for split, split_labels in labels.items():
        if split == 'classes':
            updated_labels[split] = split_labels
        else:
            updated_labels[split] = {}
            for filename, class_name in split_labels.items():
                new_filename = filename.replace('.jpg', '.rgb')
                updated_labels[split][new_filename] = class_name
    
    # 업데이트된 레이블 파일 저장
    updated_labels_file = os.path.join(current_dir, 'labels_rgb.json')
    with open(updated_labels_file, 'w') as f:
        json.dump(updated_labels, f, indent=2)
    
    print(f"✅ Simple dataset generated successfully!")
    print(f"📁 Data structure:")
    print(f"   - Train: {len(labels.get('train', {}))} images")
    print(f"   - Val: {len(labels.get('val', {}))} images") 
    print(f"   - Test: {len(labels.get('test', {}))} images")
    print(f"📝 Updated labels saved to: {updated_labels_file}")

if __name__ == "__main__":
    generate_simple_dataset()