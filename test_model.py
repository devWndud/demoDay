"""
훈련된 모델 테스트 스크립트
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt


def test_trained_model():
    """훈련된 모델 테스트"""
    print("🧪 훈련된 모델 테스트")
    print("=" * 40)
    
    # 모델 경로 확인
    model_path = "runs/train/plant_disease_detection/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return False
    
    print(f"✅ 모델 파일 발견: {model_path}")
    
    # 모델 로드
    try:
        model = YOLO(model_path)
        print("✅ 모델 로드 성공")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return False
    
    # 클래스 정보
    class_names = [
        'healthy', 'bacterial_spot', 'early_blight', 'late_blight',
        'leaf_mold', 'septoria_spot', 'spider_mites', 'target_spot',
        'mosaic_virus', 'yellow_leaf_curl'
    ]
    
    print(f"📋 지원 클래스: {len(class_names)}개")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # 테스트 이미지 생성 (데모용)
    print("\n🖼️ 테스트 이미지 생성...")
    test_image = create_test_image()
    
    # 예측 수행
    print("🔍 예측 수행...")
    try:
        results = model.predict(
            source=test_image,
            conf=0.25,
            save=False,
            verbose=False
        )
        
        print("✅ 예측 완료!")
        
        # 결과 출력
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            print(f"\n📊 감지된 객체: {len(boxes)}개")
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                print(f"  {i+1}. {class_name}: {conf:.3f}")
        else:
            print("🔍 감지된 객체가 없습니다.")
        
        return True
        
    except Exception as e:
        print(f"❌ 예측 실패: {e}")
        return False


def create_test_image():
    """테스트용 이미지 생성"""
    # 640x640 크기의 테스트 이미지 생성
    img = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
    
    # 녹색 계열로 만들기 (식물 잎 모양)
    img[:, :, 1] = np.random.randint(100, 200, (640, 640))  # Green
    img[:, :, 0] = np.random.randint(50, 100, (640, 640))   # Blue
    img[:, :, 2] = np.random.randint(30, 80, (640, 640))    # Red
    
    # 잎 모양 패턴 추가
    center_x, center_y = 320, 320
    cv2.ellipse(img, (center_x, center_y), (200, 300), 0, 0, 360, (50, 150, 50), -1)
    
    # 노이즈 추가
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def main():
    """메인 함수"""
    print("🌱 식물 질병 판별 모델 테스트")
    print("=" * 50)
    
    # 모델 테스트
    success = test_trained_model()
    
    if success:
        print("\n🎉 모델 테스트가 성공적으로 완료되었습니다!")
        print("\n다음 명령어로 웹 애플리케이션을 실행할 수 있습니다:")
        print("streamlit run app.py")
    else:
        print("\n❌ 모델 테스트에 실패했습니다.")


if __name__ == "__main__":
    main()
