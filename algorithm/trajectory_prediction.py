import numpy as np
from scipy.interpolate import interp1d

def trajectory_prediction(target, uav):
    estimated_target_position = predict_next_position(uav.view_target_trajectory)
    if estimated_target_position is None:
        direction_probs = np.array([0.5, 0.5, 0.5, 0.5])
        direction_probs[uav.uav_orientation] += 1.0
        direction_probs /= direction_probs.sum()  # Normalize to probabilities
        return direction_probs
    difference = estimated_target_position - uav.uav_position
    probs = np.array([max(0, difference[1]), max(0, -difference[1]),
                      max(0, -difference[0]), max(0, difference[0])])
    probs = probs / probs.sum()
    return probs

def predict_next_position(positions):
    valid_positions = [(i, pos) for i, pos in enumerate(positions) if pos is not None]
    if len(valid_positions) < 2:
        return None
    indices, coords = zip(*valid_positions)

    # 将索引和对应的x, y坐标分别提取出来
    indices = np.array(indices)
    x_coords = np.array([coord[0] for coord in coords])
    y_coords = np.array([coord[1] for coord in coords])

    # 创建线性插值函数
    x_interp = interp1d(indices, x_coords, fill_value="extrapolate")
    y_interp = interp1d(indices, y_coords, fill_value="extrapolate")

    # 预测下一点位置
    next_index = len(positions)
    next_x = x_interp(next_index)
    next_y = y_interp(next_index)

    return np.array([next_x, next_y])
