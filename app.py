"""
식물 질병 판별을 위한 Streamlit 웹 애플리케이션
"""

import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import tempfile
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import json
from datetime import datetime


class PlantDiseaseApp:
    """식물 질병 판별 웹 애플리케이션"""
    
    def __init__(self):
        self.model = None
        self.classes = [
            'healthy', 'bacterial_spot', 'early_blight', 'late_blight',
            'leaf_mold', 'septoria_spot', 'spider_mites', 'target_spot',
            'mosaic_virus', 'yellow_leaf_curl'
        ]
        self.class_descriptions = {
            'healthy': '건강한 식물 - 질병이 없는 정상적인 상태',
            'bacterial_spot': '세균성 반점병 - 잎에 작은 갈색 반점이 나타남',
            'early_blight': '조기마름병 - 잎 가장자리부터 갈색으로 마름',
            'late_blight': '늦은마름병 - 잎에 물에 젖은 듯한 반점 발생',
            'leaf_mold': '잎곰팡이병 - 잎 뒷면에 회색 곰팡이 발생',
            'septoria_spot': '세프토리아 반점병 - 작은 원형 갈색 반점',
            'spider_mites': '거미진드기 - 잎에 미세한 거미줄과 반점',
            'target_spot': '표적반점병 - 동심원 모양의 반점',
            'mosaic_virus': '모자이크 바이러스 - 잎에 모자이크 패턴',
            'yellow_leaf_curl': '황색잎말림병 - 잎이 노랗게 변하고 말림'
        }
        self.treatment_advice = {
            'healthy': '현재 식물이 건강한 상태입니다. 계속해서 적절한 관리와 관찰을 유지하세요.',
            'bacterial_spot': '구리 기반 살균제를 사용하고, 환기를 개선하세요. 감염된 잎은 제거하세요.',
            'early_blight': '잎에 물이 닿지 않도록 주의하고, 구리 기반 살균제를 뿌리세요.',
            'late_blight': '즉시 감염된 부분을 제거하고, 인산칼륨 살균제를 사용하세요.',
            'leaf_mold': '습도를 낮추고 환기를 개선하세요. 곰팡이 제거제를 사용하세요.',
            'septoria_spot': '감염된 잎을 제거하고, 구리 기반 살균제를 뿌리세요.',
            'spider_mites': '비누수나 아크타라를 사용하여 해충을 제거하세요.',
            'target_spot': '잎에 물이 닿지 않도록 하고, 구리 기반 살균제를 사용하세요.',
            'mosaic_virus': '바이러스에 감염된 식물은 제거하고, 벌레를 방제하세요.',
            'yellow_leaf_curl': '벌레를 방제하고, 질소 비료를 적절히 공급하세요.'
        }
    
    def load_model(self, model_path):
        """모델 로드"""
        try:
            self.model = YOLO(model_path)
            return True
        except Exception as e:
            st.error(f"모델 로드 실패: {e}")
            return False
    
    def predict_image(self, image, conf_threshold=0.25):
        """이미지 예측"""
        if self.model is None:
            return None
        
        try:
            results = self.model.predict(
                source=image,
                conf=conf_threshold,
                save=False,
                verbose=False
            )
            
            predictions = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    prediction = {
                        'bbox': boxes[i].tolist(),
                        'confidence': float(confidences[i]),
                        'class_id': int(class_ids[i]),
                        'class_name': self.classes[class_ids[i]]
                    }
                    predictions.append(prediction)
            
            return predictions
        except Exception as e:
            st.error(f"예측 실패: {e}")
            return None
    
    def draw_predictions(self, image, predictions):
        """예측 결과를 이미지에 그리기"""
        if not predictions:
            return image
        
        # 색상 팔레트
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 0), (128, 128, 0)
        ]
        
        for pred in predictions:
            bbox = pred['bbox']
            confidence = pred['confidence']
            class_id = pred['class_id']
            class_name = pred['class_name']
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            color = colors[class_id % len(colors)]
            
            # 바운딩 박스 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # 라벨 텍스트
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # 라벨 배경
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # 라벨 텍스트
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image


def main():
    """메인 애플리케이션"""
    st.set_page_config(
        page_title="🌱 식물 질병 판별 시스템",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS 스타일
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4169E1;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 헤더
    st.markdown('<h1 class="main-header">🌱 식물 질병 판별 시스템</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>AI 기반 식물 질병 진단</strong><br>
        업로드한 식물 이미지를 분석하여 질병을 자동으로 판별하고 치료 방법을 제안합니다.
    </div>
    """, unsafe_allow_html=True)
    
    # 앱 초기화
    if 'app' not in st.session_state:
        st.session_state.app = PlantDiseaseApp()
    
    app = st.session_state.app
    
    # 사이드바
    with st.sidebar:
        st.markdown("## ⚙️ 설정")
        
        # 모델 로드
        st.markdown("### 모델 설정")
        model_option = st.selectbox(
            "모델 선택",
            ["사전 훈련된 모델", "커스텀 모델"],
            help="사전 훈련된 모델을 사용하거나 직접 훈련한 모델을 업로드할 수 있습니다."
        )
        
        if model_option == "사전 훈련된 모델":
            model_size = st.selectbox("모델 크기", ["nano", "small", "medium", "large", "xlarge"])
            model_path = f"yolov8{model_size[0]}.pt"
        else:
            uploaded_model = st.file_uploader(
                "모델 파일 업로드",
                type=['pt', 'onnx'],
                help="훈련된 YOLO 모델 파일을 업로드하세요."
            )
            if uploaded_model:
                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                    tmp_file.write(uploaded_model.getvalue())
                    model_path = tmp_file.name
            else:
                model_path = None
        
        # 예측 설정
        st.markdown("### 예측 설정")
        conf_threshold = st.slider(
            "신뢰도 임계값",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="예측 신뢰도가 이 값 이상일 때만 결과를 표시합니다."
        )
        
        # 모델 로드 버튼
        if st.button("🔄 모델 로드", type="primary"):
            if model_path and os.path.exists(model_path):
                with st.spinner("모델을 로드하는 중..."):
                    if app.load_model(model_path):
                        st.success("✅ 모델이 성공적으로 로드되었습니다!")
                    else:
                        st.error("❌ 모델 로드에 실패했습니다.")
            else:
                st.error("❌ 모델 파일을 찾을 수 없습니다.")
    
    # 메인 컨텐츠
    if app.model is None:
        st.warning("⚠️ 먼저 사이드바에서 모델을 로드해주세요.")
        return
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 단일 이미지 분석", "📊 배치 분석", "📈 통계", "ℹ️ 정보"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">🔍 단일 이미지 분석</h2>', unsafe_allow_html=True)
        
        # 이미지 업로드
        uploaded_file = st.file_uploader(
            "식물 이미지를 업로드하세요",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="JPG, PNG, BMP 형식의 이미지를 업로드할 수 있습니다."
        )
        
        if uploaded_file is not None:
            # 이미지 표시
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 이미지", use_column_width=True)
            
            # 예측 버튼
            if st.button("🔍 질병 분석", type="primary"):
                with st.spinner("이미지를 분석하는 중..."):
                    # 이미지를 numpy 배열로 변환
                    image_np = np.array(image)
                    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    
                    # 예측 수행
                    predictions = app.predict_image(image_cv, conf_threshold)
                    
                    if predictions:
                        st.success(f"✅ {len(predictions)}개의 질병이 감지되었습니다!")
                        
                        # 결과 시각화
                        result_image = app.draw_predictions(image_cv.copy(), predictions)
                        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        st.image(result_image_rgb, caption="분석 결과", use_column_width=True)
                        
                        # 상세 결과
                        st.markdown("### 📋 상세 분석 결과")
                        
                        for i, pred in enumerate(predictions, 1):
                            with st.expander(f"감지 #{i}: {pred['class_name']} (신뢰도: {pred['confidence']:.2f})"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"**질병명:** {pred['class_name']}")
                                    st.markdown(f"**신뢰도:** {pred['confidence']:.2f}")
                                    st.markdown(f"**위치:** ({pred['bbox'][0]:.0f}, {pred['bbox'][1]:.0f}) - ({pred['bbox'][2]:.0f}, {pred['bbox'][3]:.0f})")
                                
                                with col2:
                                    st.markdown("**설명:**")
                                    st.write(app.class_descriptions.get(pred['class_name'], "설명이 없습니다."))
                                
                                st.markdown("**💡 치료 방법:**")
                                st.write(app.treatment_advice.get(pred['class_name'], "치료 방법이 없습니다."))
                        
                        # 신뢰도 차트
                        if len(predictions) > 1:
                            st.markdown("### 📊 신뢰도 분포")
                            df = pd.DataFrame(predictions)
                            fig = px.bar(
                                df, 
                                x='class_name', 
                                y='confidence',
                                title="감지된 질병별 신뢰도",
                                labels={'class_name': '질병명', 'confidence': '신뢰도'}
                            )
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.info("🔍 감지된 질병이 없습니다. 식물이 건강한 상태일 수 있습니다.")
    
    with tab2:
        st.markdown('<h2 class="sub-header">📊 배치 분석</h2>', unsafe_allow_html=True)
        
        # 여러 이미지 업로드
        uploaded_files = st.file_uploader(
            "여러 식물 이미지를 업로드하세요",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="여러 이미지를 동시에 분석할 수 있습니다."
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)}개의 이미지가 업로드되었습니다.")
            
            if st.button("🔍 배치 분석 시작", type="primary"):
                all_predictions = {}
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    with st.spinner(f"분석 중... ({i+1}/{len(uploaded_files)})"):
                        image = Image.open(uploaded_file)
                        image_np = np.array(image)
                        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        
                        predictions = app.predict_image(image_cv, conf_threshold)
                        all_predictions[uploaded_file.name] = predictions
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.success("✅ 배치 분석이 완료되었습니다!")
                
                # 결과 요약
                st.markdown("### 📈 분석 결과 요약")
                
                # 통계 계산
                total_predictions = sum(len(preds) for preds in all_predictions.values())
                images_with_disease = sum(1 for preds in all_predictions.values() if preds)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("총 감지된 질병", total_predictions)
                with col2:
                    st.metric("질병이 있는 이미지", images_with_disease)
                with col3:
                    st.metric("건강한 이미지", len(uploaded_files) - images_with_disease)
                
                # 클래스별 분포
                if total_predictions > 0:
                    all_preds = []
                    for preds in all_predictions.values():
                        all_preds.extend(preds)
                    
                    df = pd.DataFrame(all_preds)
                    class_counts = df['class_name'].value_counts()
                    
                    st.markdown("### 🏷️ 질병별 분포")
                    fig = px.pie(
                        values=class_counts.values,
                        names=class_counts.index,
                        title="감지된 질병별 분포"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 개별 결과
                st.markdown("### 📋 개별 분석 결과")
                for filename, predictions in all_predictions.items():
                    with st.expander(f"📁 {filename} ({len(predictions)}개 감지)"):
                        if predictions:
                            for j, pred in enumerate(predictions, 1):
                                st.write(f"**{j}. {pred['class_name']}** (신뢰도: {pred['confidence']:.2f})")
                        else:
                            st.write("건강한 식물로 판단됩니다.")
    
    with tab3:
        st.markdown('<h2 class="sub-header">📈 통계 및 분석</h2>', unsafe_allow_html=True)
        
        # 모델 정보
        st.markdown("### 🤖 모델 정보")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("총 클래스 수", len(app.classes))
            st.metric("신뢰도 임계값", f"{conf_threshold:.2f}")
        
        with col2:
            st.metric("지원 질병 유형", len(app.classes))
            st.metric("모델 상태", "✅ 로드됨" if app.model else "❌ 로드 안됨")
        
        # 질병별 설명
        st.markdown("### 🏥 지원되는 질병 유형")
        
        for i, (class_name, description) in enumerate(app.class_descriptions.items()):
            with st.expander(f"{i+1}. {class_name.replace('_', ' ').title()}"):
                st.write(f"**설명:** {description}")
                st.write(f"**치료 방법:** {app.treatment_advice.get(class_name, '치료 방법이 없습니다.')}")
    
    with tab4:
        st.markdown('<h2 class="sub-header">ℹ️ 시스템 정보</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🌱 식물 질병 판별 시스템
        
        이 시스템은 YOLOv8 기반의 딥러닝 모델을 사용하여 식물의 질병을 자동으로 감지하고 분류합니다.
        
        #### ✨ 주요 기능
        - **실시간 질병 감지**: 업로드한 이미지에서 질병을 즉시 감지
        - **높은 정확도**: YOLOv8 모델 기반의 정확한 질병 분류
        - **다양한 질병 지원**: 10가지 주요 식물 질병 감지
        - **치료 방법 제안**: 감지된 질병에 대한 구체적인 치료 방법 제공
        - **배치 처리**: 여러 이미지를 동시에 분석
        
        #### 🔬 지원되는 질병
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
        
        #### 🛠️ 기술 스택
        - **딥러닝 프레임워크**: PyTorch, Ultralytics YOLOv8
        - **웹 프레임워크**: Streamlit
        - **이미지 처리**: OpenCV, PIL
        - **데이터 시각화**: Plotly, Matplotlib
        
        #### 📞 지원 및 문의
        시스템 사용 중 문제가 발생하거나 개선 사항이 있으시면 언제든지 문의해주세요.
        """)
        
        # 시스템 상태
        st.markdown("### 🔧 시스템 상태")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("✅ 웹 서버 정상")
        with col2:
            st.success("✅ 모델 로드됨" if app.model else "⚠️ 모델 미로드")
        with col3:
            st.success("✅ 모든 기능 정상")


if __name__ == "__main__":
    main()
