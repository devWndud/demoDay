"""
실제 PlantVillage 데이터를 사용한 식물 질병 판별 시스템 실행
"""

import os
import sys
from pathlib import Path
import subprocess
import time


def run_command(command, description):
    """명령어 실행 및 결과 출력"""
    print(f"\n🔄 {description}...")
    print(f"실행 명령어: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 완료!")
        if result.stdout:
            print(f"출력: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 실패!")
        print(f"오류: {e.stderr}")
        return False


def check_plantvillage_data():
    """PlantVillage 데이터 확인"""
    plantvillage_dir = Path("PlantVillage")
    
    if not plantvillage_dir.exists():
        print("❌ PlantVillage 디렉토리를 찾을 수 없습니다.")
        print("PlantVillage 데이터를 먼저 준비해주세요.")
        return False
    
    # PlantVillage 디렉토리 내용 확인
    subdirs = [d for d in plantvillage_dir.iterdir() if d.is_dir()]
    print(f"📁 PlantVillage 디렉토리에서 {len(subdirs)}개의 클래스를 발견했습니다:")
    
    for subdir in subdirs:
        image_count = len(list(subdir.glob("*.jpg"))) + len(list(subdir.glob("*.JPG")))
        print(f"  - {subdir.name}: {image_count}개 이미지")
    
    return True


def main():
    """메인 실행 함수"""
    print("🌱 실제 PlantVillage 데이터를 사용한 식물 질병 판별 시스템")
    print("=" * 60)
    
    # 1. PlantVillage 데이터 확인
    print("\n📁 1단계: PlantVillage 데이터 확인")
    if not check_plantvillage_data():
        return
    
    # 2. 의존성 설치
    print("\n📦 2단계: 의존성 설치")
    if not run_command("pip install -r requirements.txt", "의존성 설치"):
        print("❌ 의존성 설치에 실패했습니다. 수동으로 설치해주세요.")
        return
    
    # 3. PlantVillage 데이터를 YOLO 형식으로 변환
    print("\n🔄 3단계: PlantVillage 데이터를 YOLO 형식으로 변환")
    if not run_command("python convert_plantvillage.py", "데이터 변환"):
        print("❌ 데이터 변환에 실패했습니다.")
        return
    
    # 4. 변환된 데이터 확인
    print("\n📊 4단계: 변환된 데이터 확인")
    data_dir = Path("data")
    if data_dir.exists():
        for split in ['train', 'val', 'test']:
            images_dir = data_dir / split / 'images'
            labels_dir = data_dir / split / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                image_count = len(list(images_dir.glob("*.jpg")))
                label_count = len(list(labels_dir.glob("*.txt")))
                print(f"  {split}: {image_count}개 이미지, {label_count}개 라벨")
            else:
                print(f"  {split}: 데이터 없음")
    else:
        print("❌ 변환된 데이터를 찾을 수 없습니다.")
        return
    
    # 5. 데이터셋 테스트
    print("\n🧪 5단계: 데이터셋 로드 테스트")
    if not run_command("python -c \"from dataset import PlantDiseaseDataset; ds = PlantDiseaseDataset('data', split='train', use_real_data=True); print(f'데이터셋 크기: {len(ds)}')\"", "데이터셋 테스트"):
        print("❌ 데이터셋 로드에 실패했습니다.")
        return
    
    # 6. 모델 훈련 (짧은 버전)
    print("\n🤖 6단계: 모델 훈련 (데모용 - 10 에포크)")
    train_command = "python train.py --data_dir data --epochs 10 --batch_size 8 --device cpu --pretrained"
    if not run_command(train_command, "모델 훈련"):
        print("❌ 모델 훈련에 실패했습니다.")
        return
    
    # 7. 훈련된 모델 확인
    print("\n🔍 7단계: 훈련된 모델 확인")
    model_path = "runs/train/plant_disease_detection/weights/best.pt"
    if os.path.exists(model_path):
        print(f"✅ 훈련된 모델이 생성되었습니다: {model_path}")
    else:
        print("❌ 훈련된 모델을 찾을 수 없습니다.")
        return
    
    # 8. 모델 평가
    print("\n📊 8단계: 모델 평가")
    eval_command = f"python evaluate.py --model_path {model_path} --data_config data/data.yaml"
    if not run_command(eval_command, "모델 평가"):
        print("❌ 모델 평가에 실패했습니다.")
        return
    
    # 9. 웹 애플리케이션 실행 안내
    print("\n🌐 9단계: 웹 애플리케이션 실행")
    print("웹 애플리케이션을 실행하려면 다음 명령어를 실행하세요:")
    print("streamlit run app.py")
    print("\n브라우저에서 http://localhost:8501 에 접속하세요.")
    
    # 10. 사용법 안내
    print("\n📖 사용법 안내")
    print("=" * 40)
    print("1. 웹 애플리케이션 실행:")
    print("   streamlit run app.py")
    print()
    print("2. 단일 이미지 분석:")
    print(f"   python evaluate.py --model_path {model_path} --test_images your_image.jpg")
    print()
    print("3. 배치 이미지 분석:")
    print(f"   python evaluate.py --model_path {model_path} --test_images test_images/ --output_dir results/")
    print()
    print("4. 더 많은 에포크로 모델 재훈련:")
    print("   python train.py --data_dir data --epochs 100 --batch_size 16 --device cuda")
    
    print("\n🎉 실제 PlantVillage 데이터를 사용한 시스템이 완성되었습니다!")
    print("이제 실제 식물 질병 데이터로 훈련된 모델을 사용할 수 있습니다.")


if __name__ == "__main__":
    main()
