import numpy as np
import pytest

from image_to_3d import (
    generate_depth_map,
    colorize_depth_map,
    depth_map_to_point_cloud,
)

def test_generate_depth_map_output_shape() :
    # 출력 깊이 맵의 크기가 입력 이미지와 같은지 확인
    image = np.zeros((100, 100, 3), dtype = np.uint8)
    
    depth_map = generate_depth_map(image)

    assert depth_map.shape == image.shape[:2], \
        "Depth Map의 크기가 입력 이미지와 다릅니다."

def test_generate_depth_map_output_type() :
    # 출력값이 numpy 배열인지 확인
    image = np.zeros((100, 100, 3), dtype = np.uint8)
    
    depth_map = generate_depth_map(image)
    
    assert isinstance(depth_map, np.ndarray), \
        "Depth Map의 자료형이 numpy.ndarray가 아닙니다."
    
def test_generate_depth_map_none_input() :
    # 입력이미지가 None이면 ValueError가 발생하는지 확인
    
    with pytest.raises(ValueError, match = "입력된 이미지가 없습니다.") :
        generate_depth_map(None)
        
if __name__ == "__main__" :
    pytest.main(["-v"])