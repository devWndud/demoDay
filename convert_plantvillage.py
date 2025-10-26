"""
PlantVillage 데이터를 YOLO 형식으로 변환하는 스크립트
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import json
from tqdm import tqdm


class PlantVillageConverter:
    """PlantVillage 데이터를 YOLO 형식으로 변환하는 클래스"""
    
    def __init__(self, source_dir, output_dir):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # 클래스 매핑 (PlantVillage -> YOLO)
        self.class_mapping = {
            # Tomato classes
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
            
            # Potato classes (매핑)
            'Potato___healthy': 0,  # healthy
            'Potato___Early_blight': 2,  # early_blight
            'Potato___Late_blight': 3,  # late_blight
            
            # Pepper classes (매핑)
            'Pepper__bell___healthy': 0,  # healthy
            'Pepper__bell___Bacterial_spot': 1,  # bacterial_spot
        }
        
        # 클래스 이름 (한국어)
        self.class_names = [
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
        
        # 통계 저장
        self.stats = {
            'total_images': 0,
            'class_counts': {},
            'converted_images': 0,
            'skipped_images': 0
        }
    
    def convert_dataset(self):
        """데이터셋 변환 실행"""
        print("🌱 PlantVillage 데이터를 YOLO 형식으로 변환합니다...")
        
        # 출력 디렉토리 생성
        self._create_output_directories()
        
        # 모든 이미지 수집
        all_images = self._collect_images()
        
        if not all_images:
            print("❌ 변환할 이미지를 찾을 수 없습니다.")
            return False
        
        print(f"📊 총 {len(all_images)}개의 이미지를 발견했습니다.")
        
        # 데이터 분할 (80% train, 10% val, 10% test)
        train_size = int(0.8 * len(all_images))
        val_size = int(0.1 * len(all_images))
        
        train_images = all_images[:train_size]
        val_images = all_images[train_size:train_size + val_size]
        test_images = all_images[train_size + val_size:]
        
        print(f"📁 데이터 분할: Train({len(train_images)}), Val({len(val_images)}), Test({len(test_images)})")
        
        # 각 split별로 변환
        self._convert_split('train', train_images)
        self._convert_split('val', val_images)
        self._convert_split('test', test_images)
        
        # 데이터 설정 파일 생성
        self._create_data_config()
        
        # 통계 출력
        self._print_statistics()
        
        print("✅ 변환이 완료되었습니다!")
        return True
    
    def _create_output_directories(self):
        """출력 디렉토리 생성"""
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def _collect_images(self):
        """모든 이미지 수집"""
        all_images = []
        
        for class_dir in self.source_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in self.class_mapping:
                class_id = self.class_mapping[class_dir.name]
                class_name = self.class_names[class_id]
                
                # 이미지 파일들 수집
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(list(class_dir.glob(ext)))
                
                for img_file in image_files:
                    all_images.append({
                        'path': img_file,
                        'class_id': class_id,
                        'class_name': class_name,
                        'original_class': class_dir.name
                    })
                
                self.stats['class_counts'][class_name] = len(image_files)
                print(f"📁 {class_dir.name} -> {class_name}: {len(image_files)}개 이미지")
        
        # 랜덤 셔플
        np.random.shuffle(all_images)
        self.stats['total_images'] = len(all_images)
        
        return all_images
    
    def _convert_split(self, split_name, images):
        """특정 split 변환"""
        print(f"\n🔄 {split_name} 데이터 변환 중...")
        
        split_dir = self.output_dir / split_name
        
        for img_info in tqdm(images, desc=f"Converting {split_name}"):
            try:
                # 이미지 로드 및 전처리
                img = cv2.imread(str(img_info['path']))
                if img is None:
                    self.stats['skipped_images'] += 1
                    continue
                
                # 이미지 크기 조정 (정사각형으로)
                img_resized = self._resize_image(img)
                
                # 새 파일명 생성
                new_filename = f"{img_info['original_class']}_{img_info['path'].stem}.jpg"
                new_img_path = split_dir / 'images' / new_filename
                
                # 이미지 저장
                cv2.imwrite(str(new_img_path), img_resized)
                
                # YOLO 형식 라벨 생성
                label_path = split_dir / 'labels' / f"{new_filename.split('.')[0]}.txt"
                self._create_yolo_label(label_path, img_info['class_id'], img_resized.shape)
                
                self.stats['converted_images'] += 1
                
            except Exception as e:
                print(f"❌ 오류 발생: {img_info['path']} - {e}")
                self.stats['skipped_images'] += 1
                continue
    
    def _resize_image(self, img, target_size=640):
        """이미지를 정사각형으로 리사이즈"""
        h, w = img.shape[:2]
        
        # 정사각형으로 패딩
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
    
    def _create_yolo_label(self, label_path, class_id, img_shape):
        """YOLO 형식 라벨 파일 생성"""
        h, w = img_shape[:2]
        
        # 전체 이미지를 객체로 가정 (중앙에 80% 크기의 바운딩 박스)
        x_center = 0.5
        y_center = 0.5
        width = 0.8
        height = 0.8
        
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def _create_data_config(self):
        """데이터 설정 파일 생성"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        config_path = self.output_dir / 'data.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"📄 데이터 설정 파일 생성: {config_path}")
    
    def _print_statistics(self):
        """변환 통계 출력"""
        print("\n📊 변환 통계:")
        print(f"  총 이미지: {self.stats['total_images']}")
        print(f"  변환 성공: {self.stats['converted_images']}")
        print(f"  변환 실패: {self.stats['skipped_images']}")
        print(f"  성공률: {self.stats['converted_images']/self.stats['total_images']*100:.1f}%")
        
        print("\n🏷️ 클래스별 분포:")
        for class_name, count in self.stats['class_counts'].items():
            print(f"  {class_name}: {count}개")


def main():
    """메인 함수"""
    # 경로 설정
    source_dir = "PlantVillage"  # PlantVillage 데이터가 있는 디렉토리
    output_dir = "data"  # 변환된 데이터를 저장할 디렉토리
    
    print("🌱 PlantVillage to YOLO 변환기")
    print("=" * 50)
    
    # 변환기 초기화
    converter = PlantVillageConverter(source_dir, output_dir)
    
    # 변환 실행
    success = converter.convert_dataset()
    
    if success:
        print("\n🎉 변환이 성공적으로 완료되었습니다!")
        print(f"📁 변환된 데이터 위치: {output_dir}")
        print("\n다음 명령어로 훈련을 시작할 수 있습니다:")
        print("python train.py --data_dir data --epochs 100")
    else:
        print("\n❌ 변환에 실패했습니다.")


if __name__ == "__main__":
    main()
