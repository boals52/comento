import numpy as np
import open3d as o3d

def load_point_cloud(file_path : str) -> np.ndarray :
    # 넘파이 형식의 포인트 클라우드를 불러옴
    point_cloud = np.load(file_path)
    
    if point_cloud.ndim != 2 or point_cloud.shape[1] !=3 :
        raise ValueError("포인트 클라우드의 shape은 (N, 3)이어야 합니다.")
    
    return point_cloud

def convert_to_open3d(point_cloud : np.ndarray) -> o3d.geometry.PointCloud:
    # Numpy 배열을 Open3D PointCloud 객체로 변환
    open3d_point_cloud = o3d.geometry.PointCloud()
    
    open3d_point_cloud.points = o3d.utility.Vector3dVector(
        point_cloud.astype(np.float64)
    )
    
    return open3d_point_cloud

def main() :
    point_cloud = load_point_cloud(
        "depth_results/point_cloud.npy"
    )
    
    point_cloud = point_cloud[::20]
    
    print("Point cloud shape : ", point_cloud.shape)
    print("First five points:")
    print(point_cloud[:5])
    
    open3d_point_cloud = convert_to_open3d(point_cloud)
    
    o3d.visualization.draw_geometries(
        [open3d_point_cloud],
        window_name = "Point Cloud Visualization",
        width = 1000,
        height = 700
    )
    
if __name__ == "__main__" :
    main()