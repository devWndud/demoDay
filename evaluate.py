"""
식물 질병 판별 모델 평가 및 추론 스크립트
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import json
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from typing import List, Dict, Tuple, Any
import argparse
from dataset import PlantDiseaseDataset


class PlantDiseaseEvaluator:
    """식물 질병 판별 모델 평가 클래스"""
    
    def __init__(self, model_path, data_config_path='data.yaml'):
        self.model_path = model_path
        self.data_config_path = data_config_path
        self.model = None
        self.classes = [
            'healthy', 'bacterial_spot', 'early_blight', 'late_blight',
            'leaf_mold', 'septoria_spot', 'spider_mites', 'target_spot',
            'mosaic_virus', 'yellow_leaf_curl'
        ]
        self.num_classes = len(self.classes)
        
    def load_model(self):
        """모델 로드"""
        self.model = YOLO(self.model_path)
        print(f"Model loaded from: {self.model_path}")
    
    def evaluate_model(self, split='val', conf_threshold=0.25, iou_threshold=0.5):
        """모델 평가"""
        if self.model is None:
            self.load_model()
        
        print(f"Evaluating model on {split} set...")
        
        # YOLO 내장 평가
        results = self.model.val(
            data=self.data_config_path,
            split=split,
            conf=conf_threshold,
            iou=iou_threshold,
            save_json=True,
            save_hybrid=True,
            plots=True
        )
        
        # 평가 결과 출력
        print("\n=== Evaluation Results ===")
        print(f"mAP@0.5: {results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall: {results.box.mr:.4f}")
        
        return results
    
    def predict_single_image(self, image_path, conf_threshold=0.25, save_result=True):
        """단일 이미지 예측"""
        if self.model is None:
            self.load_model()
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 예측 수행
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        # 결과 처리
        predictions = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
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
        
        # 결과 시각화
        if save_result:
            self._visualize_prediction(image_rgb, predictions, image_path)
        
        return predictions
    
    def predict_batch(self, image_dir, output_dir='predictions', conf_threshold=0.25):
        """배치 예측"""
        if self.model is None:
            self.load_model()
        
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 이미지 파일들 찾기
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f'*{ext}')))
            image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
        
        print(f"Found {len(image_files)} images to process")
        
        all_predictions = {}
        
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            try:
                predictions = self.predict_single_image(
                    str(image_path), 
                    conf_threshold=conf_threshold,
                    save_result=False
                )
                all_predictions[image_path.name] = predictions
                
                # 결과 시각화 및 저장
                image = cv2.imread(str(image_path))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self._visualize_prediction(
                    image_rgb, 
                    predictions, 
                    str(image_path),
                    save_path=output_dir / f"pred_{image_path.stem}.jpg"
                )
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # 결과를 JSON으로 저장
        results_file = output_dir / 'predictions.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_predictions, f, indent=2, ensure_ascii=False)
        
        print(f"Batch prediction completed. Results saved to: {output_dir}")
        return all_predictions
    
    def _visualize_prediction(self, image, predictions, image_path, save_path=None):
        """예측 결과 시각화"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # 색상 팔레트
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_classes))
        
        for pred in predictions:
            bbox = pred['bbox']
            confidence = pred['confidence']
            class_id = pred['class_id']
            class_name = pred['class_name']
            
            # 바운딩 박스 그리기
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            rect = plt.Rectangle(
                (x1, y1), width, height,
                fill=False, 
                edgecolor=colors[class_id], 
                linewidth=3
            )
            ax.add_patch(rect)
            
            # 라벨 텍스트
            label = f"{class_name}: {confidence:.2f}"
            ax.text(
                x1, y1-10, label,
                color=colors[class_id],
                fontsize=12,
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
        
        ax.set_title(f"Plant Disease Detection Results", fontsize=16)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_predictions(self, predictions_dict, save_path='analysis_results.png'):
        """예측 결과 분석"""
        all_predictions = []
        for image_name, predictions in predictions_dict.items():
            for pred in predictions:
                pred['image_name'] = image_name
                all_predictions.append(pred)
        
        if not all_predictions:
            print("No predictions to analyze")
            return
        
        df = pd.DataFrame(all_predictions)
        
        # 분석 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 클래스별 예측 분포
        class_counts = df['class_name'].value_counts()
        axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Prediction Distribution by Class')
        
        # 신뢰도 분포
        axes[0, 1].hist(df['confidence'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Confidence Score Distribution')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Count')
        
        # 클래스별 평균 신뢰도
        class_confidence = df.groupby('class_name')['confidence'].mean().sort_values(ascending=False)
        axes[1, 0].bar(range(len(class_confidence)), class_confidence.values)
        axes[1, 0].set_title('Average Confidence by Class')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Average Confidence')
        axes[1, 0].set_xticks(range(len(class_confidence)))
        axes[1, 0].set_xticklabels(class_confidence.index, rotation=45)
        
        # 이미지별 예측 수
        image_counts = df['image_name'].value_counts()
        axes[1, 1].hist(image_counts.values, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Predictions per Image')
        axes[1, 1].set_xlabel('Number of Predictions')
        axes[1, 1].set_ylabel('Number of Images')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 통계 출력
        print("\n=== Prediction Analysis ===")
        print(f"Total predictions: {len(df)}")
        print(f"Average confidence: {df['confidence'].mean():.3f}")
        print(f"Most common class: {class_counts.index[0]} ({class_counts.iloc[0]} predictions)")
        print(f"Images with predictions: {len(image_counts)}")
        print(f"Average predictions per image: {len(df) / len(image_counts):.2f}")
    
    def compare_models(self, model_paths, test_data_path, save_path='model_comparison.png'):
        """여러 모델 비교"""
        results = {}
        
        for model_path in model_paths:
            print(f"Evaluating model: {model_path}")
            self.model_path = model_path
            self.load_model()
            
            eval_results = self.evaluate_model()
            results[os.path.basename(model_path)] = {
                'mAP@0.5': eval_results.box.map50,
                'mAP@0.5:0.95': eval_results.box.map,
                'Precision': eval_results.box.mp,
                'Recall': eval_results.box.mr
            }
        
        # 비교 시각화
        df = pd.DataFrame(results).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
        titles = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            df[metric].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f'{title} Comparison')
            ax.set_ylabel(title)
            ax.set_xlabel('Model')
            ax.tick_params(axis='x', rotation=45)
            
            # 값 표시
            for j, v in enumerate(df[metric]):
                ax.text(j, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    
    def generate_report(self, predictions_dict, output_path='evaluation_report.html'):
        """평가 보고서 생성"""
        all_predictions = []
        for image_name, predictions in predictions_dict.items():
            for pred in predictions:
                pred['image_name'] = image_name
                all_predictions.append(pred)
        
        df = pd.DataFrame(all_predictions)
        
        # HTML 보고서 생성
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Plant Disease Detection Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e7f3ff; padding: 10px; margin: 10px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌱 Plant Disease Detection Model Evaluation Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Summary Statistics</h2>
                <div class="metric">
                    <strong>Total Predictions:</strong> {len(df)}
                </div>
                <div class="metric">
                    <strong>Average Confidence:</strong> {df['confidence'].mean():.3f}
                </div>
                <div class="metric">
                    <strong>Images Processed:</strong> {len(set(df['image_name']))}
                </div>
            </div>
            
            <div class="section">
                <h2>🏷️ Class Distribution</h2>
                <table>
                    <tr><th>Class</th><th>Count</th><th>Percentage</th></tr>
        """
        
        class_counts = df['class_name'].value_counts()
        for class_name, count in class_counts.items():
            percentage = (count / len(df)) * 100
            html_content += f"<tr><td>{class_name}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>📈 Confidence Analysis</h2>
                <div class="metric">
                    <strong>High Confidence (>0.8):</strong> {len(df[df['confidence'] > 0.8])} predictions
                </div>
                <div class="metric">
                    <strong>Medium Confidence (0.5-0.8):</strong> {len(df[(df['confidence'] >= 0.5) & (df['confidence'] <= 0.8)])} predictions
                </div>
                <div class="metric">
                    <strong>Low Confidence (<0.5):</strong> {len(df[df['confidence'] < 0.5])} predictions
                </div>
            </div>
        </body>
        </html>
        """.format(
            len(df[df['confidence'] > 0.8]),
            len(df[(df['confidence'] >= 0.5) & (df['confidence'] <= 0.8)]),
            len(df[df['confidence'] < 0.5])
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Evaluation report saved to: {output_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Plant Disease Detection Model Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_config', type=str, default='data.yaml', help='Data configuration file')
    parser.add_argument('--test_images', type=str, help='Path to test images directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 평가기 초기화
    evaluator = PlantDiseaseEvaluator(args.model_path, args.data_config)
    
    # 모델 평가
    print("=== Model Evaluation ===")
    eval_results = evaluator.evaluate_model(
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    # 테스트 이미지가 제공된 경우 예측 수행
    if args.test_images:
        print("\n=== Batch Prediction ===")
        predictions = evaluator.predict_batch(
            args.test_images,
            output_dir=os.path.join(args.output_dir, 'predictions'),
            conf_threshold=args.conf_threshold
        )
        
        # 결과 분석
        print("\n=== Prediction Analysis ===")
        evaluator.analyze_predictions(
            predictions,
            save_path=os.path.join(args.output_dir, 'analysis.png')
        )
        
        # 보고서 생성
        evaluator.generate_report(
            predictions,
            output_path=os.path.join(args.output_dir, 'report.html')
        )
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
