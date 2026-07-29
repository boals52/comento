# Image Preprocessing using OpenCV

## 개발 환경

- Python 3.10
- OpenCV
- NumPy
- Matplotlib
- Open3D
- Hugging Face Datasets

---

# Week 1. 이미지 전처리 및 이상치 탐지

## 데이터셋

- Dataset : Food101
- Source : https://huggingface.co/datasets/ethz/food101

---

## 전처리 과정

다음과 같은 순서로 이미지를 전처리하였습니다.

1. 이미지 크기 조정 (224 × 224)
2. 좌우 반전
3. 이미지 회전 (15°)
4. 밝기 증가
5. Grayscale 변환
6. Gaussian Blur 적용
7. Normalize

---

## 이상치 탐지

밝기 이상치를 먼저 제거한 후 남은 이미지에 객체 크기 필터를 순차적으로 적용하였습니다.

### 1. 어두운 이미지 제거

처음 100장의 평균 밝기를 계산한 뒤 이를 임계값 후보를 비교하였습니다.

| 임계값 기준 | 제거된 이미지 |
|---|---:|
| 평균 밝기 | 51 / 100 |
| 평균 밝기 - 10 | 40 / 100 |
| 평균 밝기 - 1 표준편차 | 16 / 100 |

임계값을 평균 - 1표준편차로 설정하였을 때 데이터 유실이 완화되었으며, 제거 이미지의 시각적 확인을 병행항 최종 기준으로 선택하였습니다.

### 2. 객체 크기 필터링

Food101 데이터셋에는 객체의 Bounding Box나 Segmentation Mask가 제공되지 않습니다.

따라서 다음 과정으로 전경을 근사하였습니다. 

1. Grayscale 변환
2. Gaussian Blur 적용
3. Otsu Threshold 적용
4. 전체 픽셀 중 전경 비율 계산

샘플의 전경 비율이 약 0.3~0.8 범위임을 확인하고, 전경 비율이 0.3 미만인 이미지를 객체가 작은 이상치로 판단하였습니다.


---

## Week 1 결과

- Food101 데이터셋 전처리 수행
- 원본 이미지/전처리된 이미지 저장
- 밝기 기반 이상치 제거
- 객체 크기 기반 이상치 제거

---

# Week2. 기본 3D Vision 구현

## 전체 처리 과정

단일 이미지에서 다음 순서로 기본적인 3D 표현을 생성하였습니다.

Original Image
    ↓
Grayscale Depth Map
    ↓
Colorized Depth Map
    ↓
Point Cloud
    ↓
Voxel Grid
    ↓
Height-field Mesh

### 1. Depth Map 생성

입력 이미지를 Grayscale로 변환한 뒤, 픽셀 밝기값을 가성의 깊이값으로 사용하였습니다.

본 프로젝트에서는 실제 깊이 추정 모델을 사용하지 않았으므로,
이 값은 실제 거리 정보가 아닌 psudo-depth입니다.

### 2. Colorized Depth Map 생성

깊이 차이를 시각적으로 확인하기 위해 OpenCV의 Color Map을 적용하였습니다.
Color Depth Map은 시각화를 위한 결과이며, Point Cloud 생성에는 원본 Grayscale Depth Map을 사용하였습니다.

### 3. Point Cloud 생성 및 시각화

Depth Map의 각 픽셀 위치를 x, y좌표로 사용하고,
픽셀 밝기값을 z 좌표로 사용하여 Point Cloud를 생성하였습니다.

Open3D의 PointCloud 객체로 변환하여 3차원으로 시각화하였습니다. 시각화 속도를 개선하기 위해 일정 간격으로 포인트를 샘플링하였습니다.
각 포인트의 z값을 정규화한 뒤 Jet Color Map을 적용하여 상대적인 깊이 차이를 색상으로 표현하였습니다.

### 4. Voxel Grid 생성
Point Cloud를 일정 크기의 3차원 격자로 양자화하여 Voxel Grid를 생성하였습니다.

Voxel Grid는 연속적인 Point Cloud를 공간 단위로 단순화하여 전체 구조를 보다 명확하게 확인할 수 있도록 합니다.
Voxel의 색상은 입력 Point Cloud의 색상 정보를 기반으로 생성됩니다.

### 5. Height-field Mesh 생성

Depth Map의 규칙적인 픽셀 격자를 이용하여 Mesh를 생성하였습니다.

인접한 네 개의 픽셀을 하나의 사각형으로 보고,
각 사각형을 두 개의 삼각형으로 분할하여 표면을 구성하였습니다.

top-left ------- top-right
    |            / |
    |          /   |
    |        /     |
bottom-left ---- bottom-right

또한 Mesh 생성 시 stride를 적용하여 사용할 정점의 수를 조절하였습니다.
생성된 Mesh에는 깊이 기반 정점 색상을 적용했습니다.

## Unit Test
pytest를 이용하여 주요 함수의 정상 동작과 예외 처리를 확인하였습니다.

### 테스트 항목
generate_depth_map()
- 출력 크기가 입력 이미지의 높이와 너비가 같은지 확인
- 출력값이 NumPy 배열인지 확인
- 입력 이미지가 None일 때 ValueError가 발생하는지 확인

## Week 2 결과

- Grayscale Depth Map
- Colorized Depth Map
- Depth-colored Point Cloud
- Voxel Grid
- Height-field Mesh

# Week3. YOLOv8 Traffic Object Detection

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

## 6. 추가 코드 수정 및 결과 분석
- 예측 Bounding Box에 class confidence score 표시
- 클래스별 인스턴스 수 분석
- 클래스별 평균/중앙값 객체 크기 분석
- small object 비율과 AP의 관계 분석

클래스별 인스턴스 수는 약 1,460~1,666개로 비교적 균형 있게 분포하여,
클래스 불균형이 성능 저하의 주요 원인이라고 보기는 어려웠다.

Pedestrian과 Motorcycle 클래스는 small object 비율이 각각 91.92%, 84.85%로 높았으며,
CCTV 영상의 원거리 촬영과 주변 객체에 의한 가림으로 인해 낮은 AP를 보인 것으로 판단하였다.

반면 Vehicle 클래스는 small object 비율이 57.16%로 상대적으로 낮음에도
exp3 기준 AP가 26.15로 가장 낮았다.
따라서 Vehicle의 낮은 성능은 객체 크기보다는 Jeepney, Truck 등 다른 차량 계열 클래스와의
시각적 특징 중복 또는 클래스 정의의 모호성으로 인한 오분류 가능성과 관련된 것으로 추정하였다.

정확한 원인 확인을 위해서는 향후 confusion matrix 및 클래스별 오탐·미탐 분석이 필요하다.

## 7. Notes
- Dataset files are not included in this repository.
- Model weights (.pt) and training output directories are excluded.
- A Roboflow API key is required to download the dataset.
