"""
식물 질병 판별을 위한 YOLO 데이터셋 클래스
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from pathlib import Path
import requests
import zipfile
import shutil
from tqdm import tqdm
import json


class PlantDiseaseDataset(Dataset):
    """식물 질병 판별을 위한 YOLO 데이터셋 클래스"""
    
    def __init__(self, data_dir, split='train', img_size=640, augment=True, use_real_data=True):
        """
        Args:
            data_dir (str): 데이터셋 디렉토리 경로
            split (str): 'train', 'val', 'test' 중 하나
            img_size (int): 이미지 크기 (정사각형)
            augment (bool): 데이터 증강 적용 여부
            use_real_data (bool): 실제 데이터 사용 여부
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.use_real_data = use_real_data
        
        # 클래스 정의 (PlantVillage 데이터셋 기반)
        self.classes = [
            'healthy',           # 건강한 식물
            'bacterial_spot',    # 세균성 반점병
            'early_blight',      # 조기마름병
            'late_blight',       # 늦은마름병
            'leaf_mold',         # 잎곰팡이병
            'septoria_spot',     # 세프토리아 반점병
            'spider_mites',      # 거미진드기
            'target_spot',       # 표적반점병
            'mosaic_virus',      # 모자이크 바이러스
            'yellow_leaf_curl'   # 황색잎말림병
        ]
        
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 실제 데이터 사용 시 다운로드
        if self.use_real_data:
            self._download_real_data()
        
        # 데이터 증강 설정
        self.transform = self._get_transforms()
        
        # 데이터 로드
        self.images, self.labels = self._load_data()
        
    def _get_transforms(self):
        """데이터 증강 변환 설정"""
        if self.augment and self.split == 'train':
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            ))
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            ))
    
    def _download_real_data(self):
        """실제 식물 질병 데이터 로드"""
        print("🌱 실제 PlantVillage 데이터를 로드합니다...")
        
        # PlantVillage 데이터가 이미 변환되어 있는지 확인
        if (self.data_dir / 'train' / 'images').exists():
            print("✅ 이미 변환된 YOLO 형식 데이터가 존재합니다.")
            return
        
        # PlantVillage 원본 데이터가 있는지 확인
        plantvillage_dir = Path("PlantVillage")
        if not plantvillage_dir.exists():
            print("❌ PlantVillage 디렉토리를 찾을 수 없습니다.")
            print("PlantVillage 데이터를 먼저 변환해주세요:")
            print("python convert_plantvillage.py")
            return
        
        # PlantVillage 데이터를 YOLO 형식으로 변환
        self._convert_plantvillage_to_yolo(plantvillage_dir)
    
    def _convert_plantvillage_to_yolo(self, plantvillage_dir):
        """PlantVillage 데이터를 YOLO 형식으로 변환"""
        print("🔄 PlantVillage 데이터를 YOLO 형식으로 변환합니다...")
        
        # 클래스 매핑
        class_mapping = {
            'Tomato___healthy': 0,
            'Tomato___Bacterial_spot': 1,
            'Tomato___Early_blight': 2,
            'Tomato___Late_blight': 3,
            'Tomato___Leaf_Mold': 4,
            'Tomato___Septoria_leaf_spot': 5,
            'Tomato___Spider_mites_Two_spotted_spider_mite': 6,
            'Tomato___Target_Spot': 7,
            'Tomato___Tomato_mosaic_virus': 8,
            'Tomato___Tomato_YellowLeaf__Curl_Virus': 9,
            'Potato___healthy': 0,
            'Potato___Early_blight': 2,
            'Potato___Late_blight': 3,
            'Pepper__bell___healthy': 0,
            'Pepper__bell___Bacterial_spot': 1,
        }
        
        # YOLO 디렉토리 생성
        for split in ['train', 'val', 'test']:
            (self.data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # 모든 이미지 수집
        all_images = []
        for class_dir in plantvillage_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in class_mapping:
                class_id = class_mapping[class_dir.name]
                
                # 이미지 파일들 수집
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    for img_file in class_dir.glob(ext):
                        all_images.append((img_file, class_id, class_dir.name))
        
        print(f"📊 총 {len(all_images)}개의 이미지를 발견했습니다.")
        
        # 데이터 분할 (80% train, 10% val, 10% test)
        np.random.shuffle(all_images)
        train_size = int(0.8 * len(all_images))
        val_size = int(0.1 * len(all_images))
        
        train_images = all_images[:train_size]
        val_images = all_images[train_size:train_size + val_size]
        test_images = all_images[train_size + val_size:]
        
        # 각 split별로 처리
        for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            print(f"🔄 {split} 데이터 변환 중... ({len(images)}개)")
            
            for img_path, class_id, original_class in images:
                try:
                    # 이미지 로드 및 리사이즈
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # 정사각형으로 리사이즈
                    img_resized = self._resize_to_square(img, self.img_size)
                    
                    # 새 파일명
                    new_filename = f"{original_class}_{img_path.stem}.jpg"
                    new_img_path = self.data_dir / split / 'images' / new_filename
                    
                    # 이미지 저장
                    cv2.imwrite(str(new_img_path), img_resized)
                    
                    # YOLO 라벨 생성 (전체 이미지를 객체로 가정)
                    label_path = self.data_dir / split / 'labels' / f"{new_filename.split('.')[0]}.txt"
                    with open(label_path, 'w') as f:
                        f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")
                        
                except Exception as e:
                    print(f"❌ 오류: {img_path} - {e}")
                    continue
        
        print("✅ YOLO 형식 변환이 완료되었습니다.")
    
    def _resize_to_square(self, img, target_size):
        """이미지를 정사각형으로 리사이즈"""
        h, w = img.shape[:2]
        
        if h > w:
            # 높이가 더 큰 경우
            new_h = target_size
            new_w = int(w * target_size / h)
            img_resized = cv2.resize(img, (new_w, new_h))
            
            # 좌우 패딩
            pad_w = (target_size - new_w) // 2
            img_padded = cv2.copyMakeBorder(
                img_resized, 0, 0, pad_w, target_size - new_w - pad_w,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            # 너비가 더 큰 경우
            new_w = target_size
            new_h = int(h * target_size / w)
            img_resized = cv2.resize(img, (new_w, new_h))
            
            # 상하 패딩
            pad_h = (target_size - new_h) // 2
            img_padded = cv2.copyMakeBorder(
                img_resized, pad_h, target_size - new_h - pad_h, 0, 0,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        
        return img_padded
    
    def _download_plantvillage_data(self):
        """PlantVillage 데이터셋 다운로드"""
        # PlantVillage 데이터셋 URL (실제 사용 가능한 공개 데이터셋)
        dataset_urls = {
            'tomato': 'https://data.mendeley.com/datasets/tywbtsjr5v/1/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/PlantVillage-Dataset.zip',
            'potato': 'https://data.mendeley.com/datasets/tywbtsjr5v/1/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/PlantVillage-Dataset.zip',
            'pepper': 'https://data.mendeley.com/datasets/tywbtsjr5v/1/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/PlantVillage-Dataset.zip'
        }
        
        # 다운로드 디렉토리 생성
        download_dir = self.data_dir / 'raw_data'
        download_dir.mkdir(exist_ok=True)
        
        # 이미 다운로드된 경우 스킵
        if (download_dir / 'PlantVillage-Dataset').exists():
            print("✅ PlantVillage 데이터셋이 이미 존재합니다.")
            return
        
        # 샘플 데이터 생성 (실제 다운로드 대신)
        print("📁 샘플 PlantVillage 데이터를 생성합니다...")
        self._create_plantvillage_sample_data(download_dir)
    
    def _create_plantvillage_sample_data(self, download_dir):
        """PlantVillage 스타일의 샘플 데이터 생성"""
        plantvillage_dir = download_dir / 'PlantVillage-Dataset'
        plantvillage_dir.mkdir(exist_ok=True)
        
        # 각 식물별 디렉토리 생성
        plant_diseases = {
            'Tomato': [
                'Tomato___healthy',
                'Tomato___Bacterial_spot',
                'Tomato___Early_blight',
                'Tomato___Late_blight',
                'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites_Two_spotted_spider_mite',
                'Tomato___Target_Spot',
                'Tomato___Tomato_mosaic_virus',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
            ],
            'Potato': [
                'Potato___healthy',
                'Potato___Early_blight',
                'Potato___Late_blight'
            ],
            'Pepper': [
                'Pepper___healthy',
                'Pepper___bell___Bacterial_spot',
                'Pepper___bell___healthy'
            ]
        }
        
        # 각 질병별로 샘플 이미지 생성
        for plant, diseases in plant_diseases.items():
            for disease in diseases:
                disease_dir = plantvillage_dir / disease
                disease_dir.mkdir(exist_ok=True)
                
                # 각 질병별로 50개씩 샘플 이미지 생성
                for i in range(50):
                    # 질병에 따른 색상 패턴 생성
                    img = self._generate_disease_image(disease, i)
                    img_path = disease_dir / f"{disease}_{i:03d}.jpg"
                    cv2.imwrite(str(img_path), img)
        
        print(f"✅ PlantVillage 샘플 데이터가 생성되었습니다: {plantvillage_dir}")
    
    def _generate_disease_image(self, disease_name, seed):
        """질병에 따른 샘플 이미지 생성"""
        np.random.seed(seed)
        
        # 기본 이미지 크기
        height, width = 256, 256
        img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
        
        # 질병별 특성 반영
        if 'healthy' in disease_name.lower():
            # 건강한 식물 - 녹색 계열
            img[:, :, 1] = np.random.randint(100, 200, (height, width))
            img[:, :, 0] = np.random.randint(50, 100, (height, width))
            img[:, :, 2] = np.random.randint(30, 80, (height, width))
            
        elif 'bacterial' in disease_name.lower():
            # 세균성 반점 - 갈색 반점
            for _ in range(5):
                center_x = np.random.randint(50, width-50)
                center_y = np.random.randint(50, height-50)
                radius = np.random.randint(10, 30)
                cv2.circle(img, (center_x, center_y), radius, (100, 50, 30), -1)
                
        elif 'blight' in disease_name.lower():
            # 마름병 - 갈색 영역
            for _ in range(3):
                center_x = np.random.randint(50, width-50)
                center_y = np.random.randint(50, height-50)
                radius = np.random.randint(20, 50)
                cv2.circle(img, (center_x, center_y), radius, (80, 40, 20), -1)
                
        elif 'mold' in disease_name.lower():
            # 곰팡이 - 회색 패턴
            img = cv2.addWeighted(img, 0.7, np.full_like(img, 128), 0.3, 0)
            
        elif 'mosaic' in disease_name.lower():
            # 모자이크 - 패턴 생성
            for i in range(0, height, 20):
                for j in range(0, width, 20):
                    if (i//20 + j//20) % 2 == 0:
                        img[i:i+20, j:j+20] = [200, 200, 200]
                    else:
                        img[i:i+20, j:j+20] = [100, 150, 100]
        
        # 노이즈 추가
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def _convert_to_yolo_format(self):
        """PlantVillage 데이터를 YOLO 형식으로 변환"""
        print("🔄 YOLO 형식으로 데이터를 변환합니다...")
        
        raw_data_dir = self.data_dir / 'raw_data' / 'PlantVillage-Dataset'
        yolo_data_dir = self.data_dir / 'yolo_data'
        
        # YOLO 디렉토리 생성
        for split in ['train', 'val', 'test']:
            (yolo_data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (yolo_data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # 클래스 매핑
        class_mapping = {
            'healthy': 0,
            'bacterial_spot': 1,
            'early_blight': 2,
            'late_blight': 3,
            'leaf_mold': 4,
            'septoria_spot': 5,
            'spider_mites': 6,
            'target_spot': 7,
            'mosaic_virus': 8,
            'yellow_leaf_curl': 9
        }
        
        all_images = []
        for disease_dir in raw_data_dir.iterdir():
            if disease_dir.is_dir():
                for img_file in disease_dir.glob('*.jpg'):
                    all_images.append((img_file, disease_dir.name))
        
        # 데이터 분할 (80% train, 10% val, 10% test)
        np.random.shuffle(all_images)
        train_size = int(0.8 * len(all_images))
        val_size = int(0.1 * len(all_images))
        
        train_images = all_images[:train_size]
        val_images = all_images[train_size:train_size + val_size]
        test_images = all_images[train_size + val_size:]
        
        # 각 split별로 처리
        for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            for img_path, disease_name in images:
                # 클래스 ID 찾기
                class_id = self._get_class_id_from_disease_name(disease_name, class_mapping)
                if class_id is None:
                    continue
                
                # 이미지 복사
                new_img_name = f"{img_path.stem}.jpg"
                new_img_path = yolo_data_dir / split / 'images' / new_img_name
                shutil.copy2(img_path, new_img_path)
                
                # YOLO 형식 라벨 생성 (전체 이미지를 객체로 가정)
                label_path = yolo_data_dir / split / 'labels' / f"{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    # 중앙에 위치한 바운딩 박스 (전체 이미지의 80%)
                    f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")
        
        # 원본 데이터를 YOLO 형식으로 교체
        if (self.data_dir / 'yolo_data').exists():
            shutil.rmtree(self.data_dir / 'train')
            shutil.rmtree(self.data_dir / 'val')
            shutil.rmtree(self.data_dir / 'test')
            shutil.move(str(self.data_dir / 'yolo_data' / 'train'), str(self.data_dir / 'train'))
            shutil.move(str(self.data_dir / 'yolo_data' / 'val'), str(self.data_dir / 'val'))
            shutil.move(str(self.data_dir / 'yolo_data' / 'test'), str(self.data_dir / 'test'))
            shutil.rmtree(self.data_dir / 'yolo_data')
        
        print("✅ YOLO 형식 변환이 완료되었습니다.")
    
    def _get_class_id_from_disease_name(self, disease_name, class_mapping):
        """질병 이름에서 클래스 ID 추출"""
        disease_lower = disease_name.lower()
        
        if 'healthy' in disease_lower:
            return class_mapping['healthy']
        elif 'bacterial' in disease_lower:
            return class_mapping['bacterial_spot']
        elif 'early_blight' in disease_lower:
            return class_mapping['early_blight']
        elif 'late_blight' in disease_lower:
            return class_mapping['late_blight']
        elif 'mold' in disease_lower:
            return class_mapping['leaf_mold']
        elif 'septoria' in disease_lower:
            return class_mapping['septoria_spot']
        elif 'spider' in disease_lower:
            return class_mapping['spider_mites']
        elif 'target' in disease_lower:
            return class_mapping['target_spot']
        elif 'mosaic' in disease_lower:
            return class_mapping['mosaic_virus']
        elif 'yellow' in disease_lower and 'curl' in disease_lower:
            return class_mapping['yellow_leaf_curl']
        
        return None
    
    def _load_data(self):
        """데이터 로드"""
        images = []
        labels = []
        
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            if self.use_real_data:
                print(f"Warning: {split_dir} does not exist. Creating real data...")
                self._download_real_data()
            else:
                print(f"Warning: {split_dir} does not exist. Creating sample data...")
                self._create_sample_data()
            return self._load_data()
        
        # 이미지 파일들 로드
        for img_path in split_dir.glob('images/*.jpg'):
            label_path = split_dir / 'labels' / f"{img_path.stem}.txt"
            
            if label_path.exists():
                images.append(str(img_path))
                labels.append(str(label_path))
        
        print(f"Loaded {len(images)} {self.split} samples")
        return images, labels
    
    def _create_sample_data(self):
        """샘플 데이터 생성 (데모용)"""
        print("Creating sample data for demonstration...")
        
        # 디렉토리 생성
        for split in ['train', 'val', 'test']:
            (self.data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # 샘플 이미지와 라벨 생성
        for split in ['train', 'val', 'test']:
            for i in range(10):  # 각 split당 10개 샘플
                # 더미 이미지 생성
                img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                img_path = self.data_dir / split / 'images' / f"sample_{i:03d}.jpg"
                cv2.imwrite(str(img_path), img)
                
                # 더미 라벨 생성
                label_path = self.data_dir / split / 'labels' / f"sample_{i:03d}.txt"
                with open(label_path, 'w') as f:
                    # 랜덤한 바운딩 박스와 클래스
                    class_id = np.random.randint(0, self.num_classes)
                    x_center = np.random.uniform(0.1, 0.9)
                    y_center = np.random.uniform(0.1, 0.9)
                    width = np.random.uniform(0.1, 0.3)
                    height = np.random.uniform(0.1, 0.3)
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """데이터셋에서 하나의 샘플 가져오기"""
        img_path = self.images[idx]
        label_path = self.labels[idx]
        
        # 이미지 로드
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 라벨 로드
        boxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip().split()
                    if len(line) == 5:
                        class_id = int(line[0])
                        x_center = float(line[1])
                        y_center = float(line[2])
                        width = float(line[3])
                        height = float(line[4])
                        
                        boxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        
        # 데이터 증강 적용
        if len(boxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=class_labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            transformed = self.transform(image=image)
            image = transformed['image']
            boxes = []
            class_labels = []
        
        # YOLO 형식으로 변환
        target = self._format_target(boxes, class_labels)
        
        return image, target
    
    def _format_target(self, boxes, class_labels):
        """YOLO 형식으로 타겟 포맷팅"""
        if len(boxes) == 0:
            # 빈 타겟 (배경)
            return torch.zeros((0, 6))  # [batch_idx, class, x, y, w, h]
        
        targets = []
        for i, (box, class_id) in enumerate(zip(boxes, class_labels)):
            x_center, y_center, width, height = box
            targets.append([0, class_id, x_center, y_center, width, height])
        
        return torch.tensor(targets, dtype=torch.float32)
    
    def collate_fn(self, batch):
        """배치 데이터 처리"""
        images, targets = zip(*batch)
        images = torch.stack(images, 0)
        
        # 타겟들을 리스트로 유지 (길이가 다를 수 있음)
        return images, list(targets)


def create_data_config(data_dir, output_path='data.yaml'):
    """데이터셋 설정 파일 생성"""
    config = {
        'path': str(Path(data_dir).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 10,  # 클래스 수
        'names': [
            'healthy',           # 건강한 식물
            'bacterial_spot',    # 세균성 반점병
            'early_blight',      # 조기마름병
            'late_blight',       # 늦은마름병
            'leaf_mold',         # 잎곰팡이병
            'septoria_spot',     # 세프토리아 반점병
            'spider_mites',      # 거미진드기
            'target_spot',       # 표적반점병
            'mosaic_virus',      # 모자이크 바이러스
            'yellow_leaf_curl'   # 황색잎말림병
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Data config saved to {output_path}")
    return config


if __name__ == "__main__":
    # 테스트
    dataset = PlantDiseaseDataset('data', split='train')
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    
    # 샘플 데이터 확인
    if len(dataset) > 0:
        image, target = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Target: {target}")
    
    # 데이터 설정 파일 생성
    create_data_config('data')
