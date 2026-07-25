# YOLOv8 Traffic Object Detection

교통 영상 데이터셋을 활용하여 YOLOv8 기반 객체 탐지 모델을 학습하고,
데이터 분할, 학습 조건, 증강 기법 및 모델 크기에 따른 성능을 비교한 프로젝트입니다.

## 1. Project Overview

- Task: Traffic object detection
- Model: YOLOv8
- Dataset source: Roboflow Universe
- Number of classes: 8

### Classes

- Bicycle
- E-bike
- Jeepney
- Motorcycle
- Pedestrian
- Tricycle
- Truck
- Vehicle

## 2. Project Structure

```text
week3_object_detection/
├── data_download.py
├── object_detection.py
├── result.py
├── README.md
└── results/
```

- data_download.py: 데이터 다운로드 및 데이터 분할 정리
- object_detection.py: YOLOv8 모델 학습과 실험
- result.py: 모델 평가, 성능 비교 및 결과 시각화
- results/: 실험 결과 이미지와 표

## 3. Dataset Preparation

원본 데이터에서 인접 영상 프레임이 서로 다른 데이터 분할에 포함되는
데이터 누수 가능성을 확인했습니다.

보다 신뢰할 수 있는 평가를 위해 영상 단위로 데이터를 다시 분할한
cleaned dataset을 생성했습니다.

Roboflow API key는 코드에 직접 저장하지 않고 실행 시 입력받습니다.

```python
from getpass import getpass

api_key = getpass("Roboflow API Key: ")
```

## 4. Experiments

| Experiment | Model   | Dataset split |   Epochs | Main setting                  |
| ---------- | ------- | ------------- | -------: | ----------------------------- |
| Exp1       | YOLOv8n | Original      | Baseline | Reference experiment          |
| Exp2       | YOLOv8n | Cleaned       | Baseline | Cleaned split baseline        |
| Exp3       | YOLOv8n | Cleaned       |       50 | Increased epochs              |
| Exp4       | YOLOv8n | Cleaned       | Baseline | Traffic-oriented augmentation |
| Exp5       | YOLOv8s | Cleaned       | Baseline | Larger model                  |

Exp1은 원본 validation split을 사용했기 때문에 Exp2–Exp5와 직접적인
성능 비교에는 사용하지 않고 참고 결과로만 활용했습니다.

## 5. Resualts
| Experiment | Precision | Recall   | mAP50    | mAP50-95 |
| ---------- | --------: | -------: | -------: | -------: |
| Exp2       | 0.641455  | 0.649349 | 0.663383 | 0.434827 |
| Exp3       | 0.656824  | 0.668734 | 0.690718 | 0.454752 |
| Exp4       | 0.610167  | 0.641229 | 0.643051 | 0.413946 |
| Exp5       | 0.650607  | 0.675032 | 0.684143 | 0.452981 |


## 6. Notes
- Dataset files are not included in this repository.
- Model weights (.pt) and training output directories are excluded.
- A Roboflow API key is required to download the dataset.
