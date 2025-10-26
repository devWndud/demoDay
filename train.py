"""
식물 질병 판별을 위한 YOLOv8 모델 훈련 스크립트
"""

import os
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from dataset import PlantDiseaseDataset, create_data_config
import argparse


class PlantDiseaseTrainer:
    """식물 질병 판별 모델 훈련 클래스"""
    
    def __init__(self, config_path='model.yaml', data_config_path='data.yaml'):
        self.config_path = config_path
        self.data_config_path = data_config_path
        self.model = None
        self.results = None
        
    def setup_model(self, model_size='n', pretrained=True):
        """모델 설정"""
        if pretrained:
            # 사전 훈련된 YOLOv8 모델 로드
            model_name = f'yolov8{model_size}.pt'
            self.model = YOLO(model_name)
            print(f"Loaded pretrained model: {model_name}")
        else:
            # 커스텀 모델 로드
            self.model = YOLO(self.config_path)
            print(f"Loaded custom model: {self.config_path}")
    
    def train(self, 
              epochs=100,
              batch_size=16,
              img_size=640,
              learning_rate=0.01,
              device='auto',
              save_dir='runs/train',
              patience=50,
              save_period=10):
        """모델 훈련"""
        
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        # 훈련 설정
        train_args = {
            'data': self.data_config_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'lr0': learning_rate,
            'device': device,
            'project': save_dir,
            'name': 'plant_disease_detection',
            'patience': patience,
            'save_period': save_period,
            'save': True,
            'save_txt': True,
            'save_conf': True,
            'plots': True,
            'val': True,
            'verbose': True,
            # 데이터 증강 설정
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            # 최적화 설정
            'optimizer': 'AdamW',
            'weight_decay': 0.0005,
            'momentum': 0.937,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
        }
        
        print("Starting training...")
        print(f"Training arguments: {train_args}")
        
        # 훈련 실행
        self.results = self.model.train(**train_args)
        
        print("Training completed!")
        return self.results
    
    def validate(self, data_path=None):
        """모델 검증"""
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        if data_path is None:
            data_path = self.data_config_path
        
        print("Starting validation...")
        val_results = self.model.val(data=data_path)
        
        print("Validation completed!")
        return val_results
    
    def predict(self, source, save_dir='runs/predict', conf_threshold=0.25):
        """예측 수행"""
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        print(f"Running prediction on: {source}")
        results = self.model.predict(
            source=source,
            save=True,
            save_txt=True,
            save_conf=True,
            conf=conf_threshold,
            project=save_dir,
            name='plant_disease_prediction'
        )
        
        return results
    
    def export_model(self, format='onnx', save_dir='exports'):
        """모델 내보내기"""
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Exporting model to {format} format...")
        exported_path = self.model.export(format=format)
        
        # 내보낸 모델을 지정된 디렉토리로 이동
        if os.path.exists(exported_path):
            new_path = os.path.join(save_dir, os.path.basename(exported_path))
            os.rename(exported_path, new_path)
            exported_path = new_path
        
        print(f"Model exported to: {exported_path}")
        return exported_path
    
    def plot_training_results(self, save_path='training_results.png'):
        """훈련 결과 시각화"""
        if self.results is None:
            print("No training results available.")
            return
        
        # 훈련 메트릭 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.results.results_dict['train/box_loss'], label='Train Box Loss')
        axes[0, 0].plot(self.results.results_dict['val/box_loss'], label='Val Box Loss')
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.results.results_dict['train/cls_loss'], label='Train Class Loss')
        axes[0, 1].plot(self.results.results_dict['val/cls_loss'], label='Val Class Loss')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # mAP curves
        axes[1, 0].plot(self.results.results_dict['metrics/mAP50(B)'], label='mAP@0.5')
        axes[1, 0].plot(self.results.results_dict['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
        axes[1, 0].set_title('Mean Average Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Precision and Recall
        axes[1, 1].plot(self.results.results_dict['metrics/precision(B)'], label='Precision')
        axes[1, 1].plot(self.results.results_dict['metrics/recall(B)'], label='Recall')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training results plot saved to: {save_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Plant Disease Detection Training')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='Model size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, default=640, help='Image size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    
    args = parser.parse_args()
    
    # 데이터 설정 파일 생성
    create_data_config(args.data_dir)
    
    # 훈련기 초기화
    trainer = PlantDiseaseTrainer()
    
    # 모델 설정
    trainer.setup_model(model_size=args.model_size, pretrained=args.pretrained)
    
    # 훈련 실행
    results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # 검증
    val_results = trainer.validate()
    
    # 결과 시각화
    trainer.plot_training_results()
    
    # 모델 내보내기
    trainer.export_model(format='onnx')
    
    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
