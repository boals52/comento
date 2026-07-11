from datasets import load_dataset
import numpy as np
import cv2
import os

# 데이터셋 로드
dataset = load_dataset("ethz/food101")

# 저장 폴더 생성
os.makedirs("preprocessed_samples", exist_ok = True)

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

def calculate_brightness_threshold(dataset, sample_size = 100) :
    # 이미지들의 평균 밝기를 계산하는 함수
    brightness_list = []
    for i, sample in enumerate(dataset["train"]):
        if i>= sample_size :
            break
        image = np.array(sample["image"])
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 밝기 필터링을 위해 평균 밝기를 계산
        brightness = np.mean(gray)
        brightness_list.append(brightness)
        
    threshold = np.mean(brightness_list)
    
    return threshold

def is_dark_image(image, threshold) :
    # 평균 밝기가 threshold보다 낮으면 True를 반환
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < threshold

def is_small_object(image, foreground_ratio_threshold) :
    # 전경 비율을 이용하여 객체의 크기가 작은지를 판별
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    foreground_pixels = np.sum(binary == 255)
    total_pixels = binary.shape[0] * binary.shape[1]
    
    foreground_ratio = foreground_pixels / total_pixels
    
    print(f"Foreground Ratio : {foreground_ratio:.3f}")
    
    return foreground_ratio < foreground_ratio_threshold


# 처음 100장만 처리
def main() :
    threshold = calculate_brightness_threshold(dataset)
    print(f"brightness threshold : {threshold : .2f}")
    
    removed_d = 0
    removed_s = 0

    for i, sample in enumerate(dataset["train"]):
        if i>= 100 :
            break
        # Hugging Face는 PIL(Pillow) 형식으로 이미지를 제공함
        # openCV는 numpy.ndarray를 이용하기 때문에 변환이 필요함
        image = np.array(sample["image"])
    
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if is_dark_image(image, threshold) :
            removed_d += 1
            continue
        
        if is_small_object(image, foreground_ratio_threshold = 0.4) :
            removed_s += 1
            continue
        
        processed_image = preprocess_for_save(image)
        
        file_name = f"preprocessed_samples/sample_{i+1}.jpg"
        cv2.imwrite(file_name,
                    processed_image)
        
        print(f"saved : {file_name}")
    
        normalized_image = normalize_image(processed_image)
    
    print(f"removed {removed_d} dark images")
    print(f"removed {removed_s} small object images")
        
    

if __name__ == "__main__" :
    main()