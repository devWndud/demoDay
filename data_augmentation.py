"""
식물 질병 판별을 위한 데이터 전처리 및 증강 파이프라인
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random
from typing import List, Tuple, Dict, Any
import os


class PlantDiseaseAugmentation:
    """식물 질병 데이터 증강 클래스"""
    
    def __init__(self, img_size=640):
        self.img_size = img_size
        self.setup_transforms()
    
    def setup_transforms(self):
        """데이터 증강 변환 설정"""
        
        # 기본 전처리 (훈련/검증 공통)
        self.base_transform = A.Compose([
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
        
        # 훈련용 강화 증강
        self.train_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            
            # 기하학적 변환
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.5
            ),
            
            # 색상 변환
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.7
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            
            # 노이즈 및 블러
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.MotionBlur(blur_limit=7, p=0.2),
            
            # 날씨 효과
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=0.7,
                brightness_coefficient=0.7,
                rain_type="drizzle",
                p=0.1
            ),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.2
            ),
            
            # 잘라내기 및 패딩
            A.RandomCrop(
                height=int(self.img_size * 0.8),
                width=int(self.img_size * 0.8),
                p=0.3
            ),
            A.RandomResizedCrop(
                height=self.img_size,
                width=self.img_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.5
            ),
            
            # 정규화
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
        
        # 검증용 변환 (증강 없음)
        self.val_transform = A.Compose([
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
    
    def apply_transform(self, image, bboxes=None, class_labels=None, mode='train'):
        """변환 적용"""
        if mode == 'train':
            transform = self.train_transform
        else:
            transform = self.val_transform
        
        if bboxes is not None and class_labels is not None:
            transformed = transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            return transformed['image'], transformed['bboxes'], transformed['class_labels']
        else:
            transformed = transform(image=image)
            return transformed['image'], None, None
    
    def create_mosaic(self, images, labels, mosaic_size=640):
        """모자이크 증강 생성"""
        mosaic_image = np.zeros((mosaic_size, mosaic_size, 3), dtype=np.uint8)
        mosaic_labels = []
        
        # 4개 이미지를 2x2 그리드로 배치
        positions = [
            (0, 0),  # 좌상
            (mosaic_size//2, 0),  # 우상
            (0, mosaic_size//2),  # 좌하
            (mosaic_size//2, mosaic_size//2)  # 우하
        ]
        
        for i, (img, label) in enumerate(zip(images, labels)):
            if i >= 4:
                break
                
            # 이미지 크기 조정
            img_resized = cv2.resize(img, (mosaic_size//2, mosaic_size//2))
            
            # 위치에 이미지 배치
            x, y = positions[i]
            mosaic_image[y:y+mosaic_size//2, x:x+mosaic_size//2] = img_resized
            
            # 라벨 좌표 조정
            if label is not None:
                for bbox in label:
                    class_id, x_center, y_center, width, height = bbox
                    # 새로운 좌표 계산
                    new_x_center = (x_center * mosaic_size//2 + x) / mosaic_size
                    new_y_center = (y_center * mosaic_size//2 + y) / mosaic_size
                    new_width = width * mosaic_size//2 / mosaic_size
                    new_height = height * mosaic_size//2 / mosaic_size
                    
                    mosaic_labels.append([
                        class_id, new_x_center, new_y_center, new_width, new_height
                    ])
        
        return mosaic_image, mosaic_labels
    
    def mixup(self, image1, image2, label1, label2, alpha=0.2):
        """Mixup 증강"""
        lam = np.random.beta(alpha, alpha)
        
        # 이미지 믹스업
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_image = mixed_image.astype(np.uint8)
        
        # 라벨 결합
        mixed_labels = []
        if label1 is not None:
            mixed_labels.extend(label1)
        if label2 is not None:
            mixed_labels.extend(label2)
        
        return mixed_image, mixed_labels
    
    def cutmix(self, image1, image2, label1, label2, alpha=1.0):
        """CutMix 증강"""
        lam = np.random.beta(alpha, alpha)
        
        # 잘라낼 영역 계산
        h, w = image1.shape[:2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        # 랜덤 중심점
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # 경계 조정
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # 이미지 결합
        mixed_image = image1.copy()
        mixed_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]
        
        # 라벨 조정
        mixed_labels = []
        if label1 is not None:
            for bbox in label1:
                class_id, x_center, y_center, width, height = bbox
                # 잘라낸 영역과 겹치는지 확인
                if not self._is_bbox_in_cut_region(
                    x_center, y_center, width, height, 
                    bbx1/w, bby1/h, (bbx2-bbx1)/w, (bby2-bby1)/h
                ):
                    mixed_labels.append(bbox)
        
        if label2 is not None:
            for bbox in label2:
                class_id, x_center, y_center, width, height = bbox
                # 잘라낸 영역에 있는 라벨만 추가
                if self._is_bbox_in_cut_region(
                    x_center, y_center, width, height,
                    bbx1/w, bby1/h, (bbx2-bbx1)/w, (bby2-bby1)/h
                ):
                    # 좌표를 전체 이미지 기준으로 변환
                    new_x_center = (x_center * (bbx2-bbx1) + bbx1) / w
                    new_y_center = (y_center * (bby2-bby1) + bby1) / h
                    new_width = width * (bbx2-bbx1) / w
                    new_height = height * (bby2-bby1) / h
                    
                    mixed_labels.append([
                        class_id, new_x_center, new_y_center, new_width, new_height
                    ])
        
        return mixed_image, mixed_labels
    
    def _is_bbox_in_cut_region(self, x_center, y_center, width, height, 
                              cut_x, cut_y, cut_w, cut_h):
        """바운딩 박스가 잘라낸 영역과 겹치는지 확인"""
        bbox_x1 = x_center - width/2
        bbox_y1 = y_center - height/2
        bbox_x2 = x_center + width/2
        bbox_y2 = y_center + height/2
        
        cut_x1 = cut_x - cut_w/2
        cut_y1 = cut_y - cut_h/2
        cut_x2 = cut_x + cut_w/2
        cut_y2 = cut_y + cut_h/2
        
        return not (bbox_x2 < cut_x1 or bbox_x1 > cut_x2 or 
                   bbox_y2 < cut_y1 or bbox_y1 > cut_y2)
    
    def visualize_augmentations(self, image, bboxes=None, class_labels=None, 
                              num_samples=8, save_path=None):
        """증강 결과 시각화"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # 원본 이미지
        axes[0].imshow(image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        if bboxes is not None:
            self._draw_bboxes(axes[0], image, bboxes, class_labels)
        
        # 증강된 이미지들
        for i in range(1, num_samples):
            aug_image, aug_bboxes, aug_labels = self.apply_transform(
                image, bboxes, class_labels, mode='train'
            )
            
            # 텐서를 numpy로 변환
            if isinstance(aug_image, torch.Tensor):
                aug_image = aug_image.permute(1, 2, 0).numpy()
                # 정규화 해제
                aug_image = aug_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                aug_image = np.clip(aug_image, 0, 1)
            
            axes[i].imshow(aug_image)
            axes[i].set_title(f'Augmented {i}')
            axes[i].axis('off')
            
            if aug_bboxes is not None:
                self._draw_bboxes(axes[i], aug_image, aug_bboxes, aug_labels)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Augmentation visualization saved to: {save_path}")
        
        plt.show()
    
    def _draw_bboxes(self, ax, image, bboxes, class_labels):
        """바운딩 박스 그리기"""
        if bboxes is None or len(bboxes) == 0:
            return
        
        h, w = image.shape[:2]
        
        for bbox, class_id in zip(bboxes, class_labels):
            x_center, y_center, width, height = bbox
            
            # YOLO 형식을 절대 좌표로 변환
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # 바운딩 박스 그리기
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            
            # 클래스 라벨 표시
            ax.text(x1, y1-5, f'Class {class_id}', 
                   color='red', fontsize=8, weight='bold')


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, img_size=640):
        self.img_size = img_size
    
    def resize_image(self, image, target_size=None):
        """이미지 크기 조정"""
        if target_size is None:
            target_size = (self.img_size, self.img_size)
        
        return cv2.resize(image, target_size)
    
    def normalize_image(self, image):
        """이미지 정규화"""
        return image.astype(np.float32) / 255.0
    
    def denormalize_image(self, image):
        """정규화 해제"""
        return (image * 255).astype(np.uint8)
    
    def adjust_brightness_contrast(self, image, brightness=0, contrast=1.0):
        """밝기 및 대비 조정"""
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    
    def apply_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def remove_background(self, image, method='grabcut'):
        """배경 제거"""
        if method == 'grabcut':
            mask = np.zeros(image.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # 전체 이미지를 전경으로 초기화
            rect = (50, 50, image.shape[1]-100, image.shape[0]-100)
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            result = image * mask2[:, :, np.newaxis]
            
            return result, mask2
        
        return image, None


def create_augmentation_pipeline(config_path='augmentation_config.json'):
    """증강 파이프라인 설정 생성"""
    config = {
        "train": {
            "horizontal_flip": 0.5,
            "vertical_flip": 0.2,
            "rotation": 15,
            "brightness_contrast": 0.3,
            "hue_saturation": 20,
            "gaussian_noise": 0.3,
            "gaussian_blur": 0.2,
            "mosaic": 0.5,
            "mixup": 0.2,
            "cutmix": 0.2
        },
        "val": {
            "resize_only": True
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Augmentation config saved to: {config_path}")
    return config


if __name__ == "__main__":
    # 테스트
    aug = PlantDiseaseAugmentation()
    preprocessor = DataPreprocessor()
    
    # 샘플 이미지 생성
    sample_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    sample_bboxes = [[0.5, 0.5, 0.3, 0.3]]
    sample_labels = [0]
    
    # 증강 시각화
    aug.visualize_augmentations(
        sample_image, 
        sample_bboxes, 
        sample_labels,
        save_path='augmentation_samples.png'
    )
    
    # 설정 파일 생성
    create_augmentation_pipeline()
    
    print("Data augmentation pipeline test completed!")
