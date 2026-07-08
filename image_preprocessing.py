from datasets import load_dataset
import numpy as np
import cv2
import os

# 데이터셋 로드
dataset = load_dataset("ethz/food101")

# 저장 폴더 생성
# os.makedirs("preprocessed_samples", exist_ok = True)

def resize_image(image, size = (224, 224)) :
    # 이미지를 지정한 크기로 변경
    return cv2.resize(image, size)

def convert_to_grayscale(image) :
    # 컬러 이미지를 Grayscale 이미지로 변환
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_blur(image, kernel_size = (5, 5)) :
    # Gaussian Blur를 적용하여 노이즈를 제거
    return cv2.GaussianBlur(image, kernel_size, 0)

def horizontal_flip(image) :
    # 데이터 증강_이미지 좌우 반전
    return cv2.flip(image, 1)

def rotate_image(image, angle = 15) :
    # 데이터 증강_이미지를 지정한 각도만큼 회전
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    
    return rotated

def change_brightness(image, beta = 30) :
    # 데이터 증강_이미지의 밝기 증가
    return cv2.convertScaleAbs(image, alpha = 1.0, beta = beta)

def normalize_image(image) :
    # 픽셀 값을 0~1 범위로 정규화
    return image.astype(np.float32) / 255.0

def preprocess_for_save(image) :
    # 저장을 위한 기본 전처리 수행
    # 크기 조정 -> 데이터 증강 -> Grayscale -> Blur
    image = resize_image(image)
    image = horizontal_flip(image)
    image = rotate_image(image)
    image = change_brightness(image)
    image = convert_to_grayscale(image)
    image = apply_blur(image)
    return image

# 처음 5장만 처리
def main() : 
    for i, sample in enumerate(dataset["train"]):
        if i>= 5 :
            break
        # Hugging Face는 PIL(Pillow) 형식으로 이미지를 제공함
        # openCV는 numpy.ndarray를 이용하기 때문에 변환이 필요함
        image = np.array(sample["image"])
    
        # PIL은 RGB 순서를 이용하지만 openCV는 BGR 순서를 사용함
        # RGB -> BGR로 변환 필요
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
        processed_image = preprocess_for_save(image)
        file_name = f"preprocessed_samples/sample_{i+1}.jpg"
        cv2.imwrite(file_name,
                    processed_image)
        
        print(f"saved : {file_name}")
    
        normalized_image = normalize_image(processed_image)
        
        

if __name__ == "__main__" :
    main()

# # 첫 번째 데이터로 점검하기
# sample = dataset["train"][0]
# image = np.array(sample["image"])
# print(type(sample["image"]))
# print(type(image))

# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# print(image.shape)

# cv2.imwrite(
#     "preprocessed_samples/original.jpg",
#     image
# )

# resized = cv2.resize(image, (224, 224))
# print(resized.shape)