# target_movement.py

import random
import numpy as np
from params import step_size

def get_farthest_direction(target_position, uav_position):
    """计算目标远离无人机的最佳方向"""
    potential_moves = {
        'up': np.array([0, step_size]),
        'down': np.array([0, -step_size]),
        'left': np.array([-step_size, 0]),
        'right': np.array([step_size, 0])
    }

    farthest_direction = None
    max_distance = -np.inf
    for direction, move in potential_moves.items():
        new_position = target_position + move
        distance = np.linalg.norm(new_position - uav_position)
        if distance > max_distance:
            max_distance = distance
            farthest_direction = direction

    return farthest_direction

def check_boundary_and_adjust(target_position):
    """检查边界并调整方向"""
    x, y = target_position
    if x < 1000:
        return 'right'
    elif x > 9000:
        return 'left'
    elif y < 1000:
        return 'up'
    elif y > 9000:
        return 'down'
    else:
        return None

def adversarial_movement(lock_steps, target_position, uav_position, adversarial_direction):
    if lock_steps <= 0:
        # 重新选择对抗方向
        if np.random.rand() < 0.5:
            adversarial_direction = get_farthest_direction(target_position, uav_position)
        else:
            adversarial_direction = np.random.choice(['up', 'down', 'left', 'right'])
        # 锁定对抗方向几时间步
        lock_steps = random.randint(10, 30)
    else:
        lock_steps -= 1

    boundary_direction = check_boundary_and_adjust(target_position)
    if boundary_direction:
        adversarial_direction = boundary_direction
        lock_steps = random.randint(10, 50)

        # 根据锁定的对抗方向移动目标
    if adversarial_direction == 'up':
        target_position[1] += step_size
    elif adversarial_direction == 'down':
        target_position[1] -= step_size
    elif adversarial_direction == 'left':
        target_position[0] -= step_size
    elif adversarial_direction == 'right':
        target_position[0] += step_size
    target_position = np.clip(target_position, 0, 10000)

    return adversarial_direction, target_position, lock_steps