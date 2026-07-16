import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from datasets import load_dataset

def generate_depth_map(image) :
    # 입력 이미지를 Grayscale로 변환한 후 컬러맵을 적용하여 
    # 밝기 값을 가상의 깊이 값으로 사용한 깊이 맵을 생성
    if image is None :
        raise ValueError("입력된 이미지가 없습니다.")
    depth_map = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return depth_map

def colorize_depth_map(depth_map : np.ndarray) -> np.ndarray :
    # 단일 채널 깊이 맵에 JET 컬러맵을 적용
    
    if depth_map is None :
        raise ValueError("입력된 깊이 맵이 없습니다.")
    return cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

def depth_map_to_point_cloud(depth_map:np.ndarray) -> np.ndarray :
    # 깊이 맵의 각 픽셀을 (x, y, z) 좌표로 변환
    
    if depth_map is None :
        raise ValueError("입력된 깊이 맵이 없습니다.")
    
    if depth_map.ndim != 2 :
        raise ValueError("깊이 맵은 2차원 단일 채널이어야 합니다.")
    
    height, width = depth_map.shape
    
    x_coordinates, y_coordinates = np.meshgrid(
        np.arange(width),
        np.arange(height)
    )
    
    z_coordinates = depth_map.astype(np.float32)
    
    point_cloud = np.column_stack((
        x_coordinates.ravel(),
        y_coordinates.ravel(),
        z_coordinates.ravel()
    ))
    
    return point_cloud

def visualize_point_cloud(point_cloud) :
    # Point Cloud를 3D Scatter Plot으로 시각화함
    
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111, projection = "3d")
    
    ax.scatter(
        point_cloud[:,0],  # x
        point_cloud[:,1],  # y
        point_cloud[:, 2], # z
        c = point_cloud[:,2], # 깊이에 따라 색상 지정
        cmap = "jet",
        s = 1 # 점 크기
    )
    
    ax.invert_yaxis()
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Depth")
    
    ax.set_title("3D Point Cloud")
    
    plt.show()
    
def main() :
    # 결과 저장 폴더 생성
    os.makedirs("depth_results", exist_ok=True)
    
    # Food101 데이터셋 로드
    dataset = load_dataset("ethz/food101")
    
    # 첫번째 이미지 사용
    sample = dataset["train"][0]
    image = np.array(sample["image"])
    
    # PIL RGB -> Opencv BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 계산용 단일 채널 깊이 맵
    depth_map = generate_depth_map(image)
    
    # 시각화용 컬러 깊이 맵
    color_depth_map = colorize_depth_map(depth_map)
    
    # 포인트 클라우드 생성
    point_cloud = depth_map_to_point_cloud(depth_map)
    
    # 포인트 클라우드 시각화
    visualize_point_cloud(point_cloud)
    
    # # 결과 저장
    # cv2.imwrite("depth_results/original.jpg", image)
    # cv2.imwrite("depth_results/depth_map_gray.jpg", depth_map)
    # cv2.imwrite("depth_results/depth_map_color.jpg", color_depth_map)
    
    # np.save("depth_results/point_cloud.npy", point_cloud)
    np.save("depth_results/depth_map.npy", depth_map)
    
    print("Original image shape :", image.shape)
    print("Depth map shape : ", depth_map.shape)
    print("Point cloud shape : ", point_cloud.shape)
    print("Depth map results saved")
    
if __name__ == "__main__" :
    main()