# Image Preprocessing using OpenCV

## Week 1
- Image preprocessing
- Outlier detection
- OpenCV

## 코드 소개

Hugging Face의 Food101 데이터셋을 활용하여 이미지 전처리와 이상치 탐지를 수행하고,  
Grayscale 이미지를 가상의 깊이 정보로 변환하여 기본적인 3D 시각화를 구현한 프로젝트입니다.

OpenCV와 NumPy를 이용하여 이미지 전처리 및 깊이 맵 변환을 수행하였으며,  
Open3D를 활용하여 Point Cloud, Voxel Grid, Height-field Mesh를 시각화하였습니다.

또한 pytest를 이용하여 주요 함수의 출력 형태와 예외 처리를 검증하였습니다.

---

## 개발 환경

- Python 3.10
- OpenCV
- NumPy
- Matplotlib
- Open3D
- Hugging Face Datasets

---

## 데이터셋

- Dataset : Food101
- Source : https://huggingface.co/datasets/ethz/food101

---

# Week 1. 이미지 전처리 및 이상치 탐지

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