# 🌱 식물 질병 판별 YOLO 모델

AI 기반 식물 질병 자동 감지 및 분류 시스템입니다. YOLOv8 모델을 사용하여 식물 이미지에서 질병을 실시간으로 감지하고 치료 방법을 제안합니다.

## ✨ 주요 기능

- **실시간 질병 감지**: 업로드한 식물 이미지에서 질병을 즉시 감지
- **높은 정확도**: YOLOv8 기반의 정확한 질병 분류
- **10가지 질병 지원**: 주요 식물 질병들을 자동으로 감지
- **치료 방법 제안**: 감지된 질병에 대한 구체적인 치료 방법 제공
- **웹 인터페이스**: 사용자 친화적인 Streamlit 기반 웹 애플리케이션
- **배치 처리**: 여러 이미지를 동시에 분석
- **데이터 증강**: 다양한 증강 기법으로 모델 성능 향상

## 🔬 지원되는 질병 유형

1. **건강한 식물** - 질병이 없는 정상 상태
2. **세균성 반점병** - 세균에 의한 잎 반점
3. **조기마름병** - 잎 가장자리부터 시작되는 마름병
4. **늦은마름병** - 물에 젖은 듯한 반점
5. **잎곰팡이병** - 잎 뒷면의 회색 곰팡이
6. **세프토리아 반점병** - 작은 원형 갈색 반점
7. **거미진드기** - 미세한 해충 피해
8. **표적반점병** - 동심원 모양의 반점
9. **모자이크 바이러스** - 모자이크 패턴의 바이러스
10. **황색잎말림병** - 잎의 황변 및 말림

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd plant-disease-detection

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 데이터 디렉토리 구조 생성
mkdir -p data/{train,val,test}/{images,labels}

# 샘플 데이터 생성 (데모용)
python dataset.py
```

### 3. 모델 훈련

```bash
# 기본 훈련 실행
python train.py --data_dir data --epochs 100 --batch_size 16

# 커스텀 설정으로 훈련
python train.py \
    --data_dir data \
    --model_size s \
    --epochs 200 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --device cuda
```

### 4. 모델 평가

```bash
# 모델 평가
python evaluate.py --model_path runs/train/plant_disease_detection/weights/best.pt

# 테스트 이미지로 예측
python evaluate.py \
    --model_path runs/train/plant_disease_detection/weights/best.pt \
    --test_images test_images \
    --output_dir evaluation_results
```

### 5. 웹 애플리케이션 실행

```bash
# Streamlit 웹 앱 실행
streamlit run app.py

# 브라우저에서 http://localhost:8501 접속
```

## 📁 프로젝트 구조

```
plant-disease-detection/
├── app.py                    # Streamlit 웹 애플리케이션
├── train.py                  # 모델 훈련 스크립트
├── evaluate.py               # 모델 평가 및 추론
├── dataset.py                # 데이터셋 클래스 및 로더
├── data_augmentation.py      # 데이터 증강 파이프라인
├── model.yaml               # YOLO 모델 설정
├── requirements.txt         # Python 의존성
├── README.md               # 프로젝트 문서
└── data/                   # 데이터 디렉토리
    ├── train/
    │   ├── images/         # 훈련 이미지
    │   └── labels/         # 훈련 라벨 (YOLO 형식)
    ├── val/
    │   ├── images/         # 검증 이미지
    │   └── labels/         # 검증 라벨
    └── test/
        ├── images/         # 테스트 이미지
        └── labels/         # 테스트 라벨
```

## 🛠️ 사용법

### 명령행 인터페이스

#### 모델 훈련
```bash
python train.py [옵션]

옵션:
  --data_dir DATA_DIR        데이터 디렉토리 경로 (기본값: data)
  --model_size {n,s,m,l,x}   모델 크기 (기본값: n)
  --epochs EPOCHS            훈련 에포크 수 (기본값: 100)
  --batch_size BATCH_SIZE    배치 크기 (기본값: 16)
  --img_size IMG_SIZE        이미지 크기 (기본값: 640)
  --learning_rate LR         학습률 (기본값: 0.01)
  --device DEVICE            사용할 디바이스 (기본값: auto)
  --pretrained               사전 훈련된 모델 사용
```

#### 모델 평가
```bash
python evaluate.py [옵션]

옵션:
  --model_path MODEL_PATH    훈련된 모델 경로
  --data_config DATA_CONFIG  데이터 설정 파일 (기본값: data.yaml)
  --test_images TEST_IMAGES  테스트 이미지 디렉토리
  --output_dir OUTPUT_DIR    출력 디렉토리 (기본값: evaluation_results)
  --conf_threshold CONF      신뢰도 임계값 (기본값: 0.25)
  --iou_threshold IOU        IoU 임계값 (기본값: 0.5)
```

### 웹 인터페이스

1. **Streamlit 앱 실행**: `streamlit run app.py`
2. **모델 로드**: 사이드바에서 모델 선택 및 로드
3. **이미지 업로드**: 단일 이미지 또는 배치 이미지 업로드
4. **분석 실행**: 업로드된 이미지에 대한 질병 감지 및 분석
5. **결과 확인**: 감지된 질병, 신뢰도, 치료 방법 확인

## 🔧 고급 설정

### 데이터 증강 설정

`data_augmentation.py`에서 다양한 증강 기법을 설정할 수 있습니다:

```python
# 증강 설정 예시
augmentation_config = {
    "train": {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.2,
        "rotation": 15,
        "brightness_contrast": 0.3,
        "hue_saturation": 20,
        "gaussian_noise": 0.3,
        "mosaic": 0.5,
        "mixup": 0.2,
        "cutmix": 0.2
    }
}
```

### 모델 커스터마이징

`model.yaml`에서 YOLO 모델 구조를 커스터마이징할 수 있습니다:

```yaml
# 백본 네트워크 설정
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  # ... 더 많은 레이어

# 헤드 설정
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]
  # ... 더 많은 레이어
```

## 📊 성능 지표

모델의 성능은 다음과 같은 지표로 평가됩니다:

- **mAP@0.5**: IoU 0.5에서의 평균 정밀도
- **mAP@0.5:0.95**: IoU 0.5-0.95에서의 평균 정밀도
- **Precision**: 정밀도
- **Recall**: 재현율
- **F1-Score**: F1 점수

## 🐛 문제 해결

### 일반적인 문제

1. **CUDA 메모리 부족**
   ```bash
   # 배치 크기 줄이기
   python train.py --batch_size 8
   ```

2. **모델 로드 실패**
   ```bash
   # 모델 파일 경로 확인
   ls -la runs/train/plant_disease_detection/weights/
   ```

3. **데이터 로드 오류**
   ```bash
   # 데이터 구조 확인
   python dataset.py
   ```

### 로그 확인

훈련 및 평가 과정의 로그는 다음 위치에서 확인할 수 있습니다:

- 훈련 로그: `runs/train/plant_disease_detection/`
- 평가 로그: `evaluation_results/`

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의

프로젝트에 대한 문의사항이나 제안사항이 있으시면 언제든지 연락해주세요.

- 이메일: [your-email@example.com]
- 프로젝트 링크: [https://github.com/your-username/plant-disease-detection](https://github.com/your-username/plant-disease-detection)

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트들의 도움을 받았습니다:

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [Albumentations](https://albumentations.ai/)

---

**🌱 식물을 건강하게, AI와 함께! 🌱**
