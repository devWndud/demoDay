"""
식물 질병 판별 시스템 실행 예제
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


def main():
    """메인 실행 함수"""
    print("🌱 식물 질병 판별 시스템 실행 예제")
    print("=" * 50)
    
    # 현재 디렉토리 확인
    current_dir = Path.cwd()
    print(f"현재 디렉토리: {current_dir}")
    
    # 1. 의존성 설치
    print("\n📦 1단계: 의존성 설치")
    if not run_command("pip install -r requirements.txt", "의존성 설치"):
        print("❌ 의존성 설치에 실패했습니다. 수동으로 설치해주세요.")
        return
    
    # 2. 데이터 디렉토리 생성
    print("\n📁 2단계: 데이터 디렉토리 생성")
    data_dirs = [
        "data/train/images",
        "data/train/labels", 
        "data/val/images",
        "data/val/labels",
        "data/test/images",
        "data/test/labels"
    ]
    
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path} 디렉토리 생성")
    
    # 3. 샘플 데이터 생성
    print("\n🌱 3단계: 샘플 데이터 생성")
    if not run_command("python dataset.py", "샘플 데이터 생성"):
        print("❌ 샘플 데이터 생성에 실패했습니다.")
        return
    
    # 4. 데이터 설정 파일 생성 확인
    print("\n⚙️ 4단계: 데이터 설정 확인")
    if os.path.exists("data.yaml"):
        print("✅ data.yaml 파일이 생성되었습니다.")
    else:
        print("❌ data.yaml 파일이 없습니다.")
        return
    
    # 5. 모델 훈련 (짧은 버전)
    print("\n🤖 5단계: 모델 훈련 (데모용 - 5 에포크)")
    train_command = "python train.py --data_dir data --epochs 5 --batch_size 4 --device cpu"
    if not run_command(train_command, "모델 훈련"):
        print("❌ 모델 훈련에 실패했습니다.")
        return
    
    # 6. 훈련된 모델 확인
    print("\n🔍 6단계: 훈련된 모델 확인")
    model_path = "runs/train/plant_disease_detection/weights/best.pt"
    if os.path.exists(model_path):
        print(f"✅ 훈련된 모델이 생성되었습니다: {model_path}")
    else:
        print("❌ 훈련된 모델을 찾을 수 없습니다.")
        return
    
    # 7. 모델 평가
    print("\n📊 7단계: 모델 평가")
    eval_command = f"python evaluate.py --model_path {model_path} --data_config data.yaml"
    if not run_command(eval_command, "모델 평가"):
        print("❌ 모델 평가에 실패했습니다.")
        return
    
    # 8. 웹 애플리케이션 실행 안내
    print("\n🌐 8단계: 웹 애플리케이션 실행")
    print("웹 애플리케이션을 실행하려면 다음 명령어를 실행하세요:")
    print("streamlit run app.py")
    print("\n브라우저에서 http://localhost:8501 에 접속하세요.")
    
    # 9. 사용법 안내
    print("\n📖 사용법 안내")
    print("=" * 30)
    print("1. 웹 애플리케이션 실행:")
    print("   streamlit run app.py")
    print()
    print("2. 단일 이미지 분석:")
    print("   python evaluate.py --model_path runs/train/plant_disease_detection/weights/best.pt --test_images your_image.jpg")
    print()
    print("3. 배치 이미지 분석:")
    print("   python evaluate.py --model_path runs/train/plant_disease_detection/weights/best.pt --test_images test_images/ --output_dir results/")
    print()
    print("4. 새로운 데이터로 모델 재훈련:")
    print("   python train.py --data_dir your_data --epochs 100 --batch_size 16")
    
    print("\n🎉 모든 단계가 완료되었습니다!")
    print("이제 식물 질병 판별 시스템을 사용할 수 있습니다.")


if __name__ == "__main__":
    main()
