# Image Preprocessing using OpenCV

## 코드 소개

Hugging Face의 Food101 데이터셋을 이용하여 AI 학습을 위한 이미지 전처리를 수행하는 것을 목표로 합니다.

OpenCV를 활용하여 이미지 전처리를 수행하고, Git을 이용하여 프로젝트를 관리하였습니다.

---

## 개발 환경

- Python 3.10
- OpenCV
- NumPy
- Hugging Face Datasets

---

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

### 1. 어두운 이미지 제거

- 처음 100장의 평균 밝기를 계산
- 평균 밝기보다 낮은 이미지를 어두운 이미지로 판단

### 2. 객체 크기 필터링

Food101 데이터셋에는 객체의 Bounding Box가 제공되지 않으므로, Otsu Threshold를 이용하여 전경(Foreground)을 추출하였습니다.

Foreground Pixel 비율을 객체 크기의 근사값으로 사용하였으며, 전경 비율이 0.4 미만인 이미지를 객체가 작은 이상치로 판단하였습니다.


---

## 결과

- Food101 데이터셋 전처리 수행
- 전처리된 이미지 저장
- 밝기 기반 이상치 제거
- 객체 크기 기반 이상치 제거