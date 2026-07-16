import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def load_point_cloud(file_path : str) -> np.ndarray :
    # 넘파이 형식의 포인트 클라우드를 불러옴
    point_cloud = np.load(file_path)
    
    if point_cloud.ndim != 2 or point_cloud.shape[1] !=3 :
        raise ValueError("포인트 클라우드의 shape은 (N, 3)이어야 합니다.")
    
    return point_cloud

def convert_to_open3d(
    point_cloud : np.ndarray,
    colors : np.ndarray | None = None
    ) -> o3d.geometry.PointCloud:
    # Numpy 배열을 Open3D PointCloud 객체로 변환
    open3d_point_cloud = o3d.geometry.PointCloud()
    
    open3d_point_cloud.points = o3d.utility.Vector3dVector(
        point_cloud.astype(np.float64)
    )
    
    if colors is not None :
        if colors.shape != point_cloud.shape :
            raise ValueError(
                "색상 배열의 shape은 포인트 클라우드와 같아야 합니다."
            )
        
        open3d_point_cloud.colors = o3d.utility.Vector3dVector(
            colors.astype(np.float64)
        )
    
    return open3d_point_cloud

def visualize_voxel_grid(
    open3d_point_cloud : o3d.geometry.PointCloud,
    voxel_size : float = 10.0
) -> None :
    # 포인트 클라우드가 차지하는 공간을 일정한 크기의 복셀로 변환해 시각화
    
    if open3d_point_cloud.is_empty() :
        raise ValueError("입력된 포인트 클라우드가 비어 있습니다.")
    
    if voxel_size <= 0 :
        raise ValueError("voxel_size는 0보다 커야 합니다.")
    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        open3d_point_cloud,
        voxel_size = voxel_size
    )
    
    print("Number of voxels :", len(voxel_grid.get_voxels()))
    
    o3d.visualization.draw_geometries(
        [voxel_grid],
        window_name = "Voxel Grid Visualization",
        width=1000,
        height=700
    )
    
def depth_map_to_mesh(
    depth_map : np.ndarray,
    stride : int = 4,
    z_scale : float = 1.0
) -> o3d.geometry.TriangleMesh :
    # 깊이 맵의 규칙적인 픽셀 격자를 삼각형으로 연결하여 Height-field 형태의 Mesh를 생성
    if depth_map is None :
        raise ValueError("입력된 깊이 맵이 없습니다.")
    if depth_map.ndim != 2 :
        raise ValueError("깊이 맵은 2차원 단일 채널이어야 합니다.")
    if stride <= 0 :
        raise ValueError("stride는 0보다 커야 합니다.")
    
    sampled_depth = depth_map[::stride, ::stride].astype(
        np.float64
    )
    
    height, width = sampled_depth.shape
    
    x_coordinates, y_coordinates = np.meshgrid(
        np.arange(width) * stride,
        np.arange(height) * stride
    )
    
    vertices = np.column_stack((
        x_coordinates.ravel(),
        y_coordinates.ravel(),
        sampled_depth.ravel() * z_scale
    ))
    
    triangles = []
    
    for row in range(height - 1) :
        for col in range(width - 1) :
            top_left = row * width + col
            top_right = top_left + 1
            bottom_left = (row + 1) * width + col
            bottom_right = bottom_left + 1
            
            # 하나의 사각형을 두 개의 삼각형으로 분할
            triangles.append([
                top_left,
                bottom_left,
                top_right
            ])
            
            triangles.append([
                top_right,
                bottom_left,
                bottom_right
            ])
    
    triangles = np.asarray(
        triangles,
        dtype = np.int32
    )
    
    mesh = o3d.geometry.TriangleMesh()
    
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # 깊이값을 기준으로 정점별 색상 생성
    vertex_colors = apply_depth_colors(vertices)
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        vertex_colors
    )
    
    # 표면의 조명 표현을 위한 법선 계선
    mesh.compute_vertex_normals()
    
    return mesh

def visualize_mesh(
    mesh : o3d.geometry.TriangleMesh
) -> None :
    # Triangle Mesh를 시각화
    
    if mesh.is_empty() :
        raise ValueError("입력된 Mesh가 비어 있습니다.")
    
    print("Number of vertices", len(mesh.vertices))
    print("Number of triangles :", len(mesh.triangles))
    
    o3d.visualization.draw_geometries(
        [mesh],
        window_name = "Triangle Mesh Visualization",
        width = 1000,
        height = 700,
        mesh_show_back_face = True
    )
    
def apply_depth_colors(
    point_cloud : np.ndarray
) -> np.ndarray :
    # 포인트 클라우드의 z값을 0~1로 정규화한 뒤 JET 컬러맵을 적용하여 RGB 색상을 정함
    
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3 :
        raise ValueError("포인트 클라우드의 shape는 (N, 3)이어야 합니다.")
    if len(point_cloud) == 0 :
        raise ValueError("포인트 클라우드가 비어 있습니다.")
    
    depth = point_cloud[:,2]
    
    depth_min = depth.min()
    depth_max = depth.max()
    
    if depth_max == depth_min :
        normalized_depth = np.zeros_like(depth, dtype=np.float64)
    
    else :
        normalized_depth = (
        (depth - depth_min) / (depth_max - depth_min)
        )
    
    colors = plt.colormaps["jet"](normalized_depth)[:, :3]
    
    return colors

def main() :
    point_cloud = load_point_cloud(
        "depth_results/point_cloud.npy"
    )
    
    sampled_point_cloud = point_cloud[::50]
    colors = apply_depth_colors(sampled_point_cloud)
    
    open3d_point_cloud = convert_to_open3d(
        sampled_point_cloud,
        colors
    )
    
    print("Sampled point cloud shape : ", sampled_point_cloud.shape)
    print("Colors shape :", colors.shape)

    
    # # Point Cloud
    # o3d.visualization.draw_geometries(
    #     [open3d_point_cloud],
    #     window_name = "Point Cloud Visualization",
    #     width = 1000,
    #     height = 700
    # )
    
    # # Voxel Grid
    # visualize_voxel_grid(
    #     open3d_point_cloud,
    #     voxel_size=10.0
    # )
    
    # Triangle Mesh
    depth_map = np.load(
        "depth_results/depth_map.npy"
    )
    
    mesh = depth_map_to_mesh(
        depth_map,
        stride = 2,
        z_scale = 0.5
    )
    
    visualize_mesh(mesh)
    
if __name__ == "__main__" :
    main()