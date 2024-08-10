import numpy as np

from algorithm.particle_filter import particle_filter, ParticleFilter
from algorithm.trajectory_interpolation import trajectory_interpolation
from animation import animation_video
from bandit.bandit_algorithm import Exp4IX
from environment.generate_target_trajectory import generate_smooth_trajectory, generate_first_control_point_near
from environment.target_movement_adversarial import get_farthest_direction, check_boundary_and_adjust, \
    adversarial_movement
from params import step_size, frame_num  # 引入调参文件中的参数

np.random.seed(0)

def simulation_mode(mode='tracking'):
    # 初始化无人机位置和朝向为随机点和随机方向（0: up, 1: right, 2: down, 3: left）
    uav_position = np.random.uniform(2000, 8000, size=(2,))
    uav_orientation = np.random.choice([0, 1, 2, 3])  # 方向: 0: up, 1: right, 2: down, 3: left

    # 生成目标的平滑轨迹
    if mode == 'tracking':
        target_positions = generate_smooth_trajectory(uav_position, num_control_points=8, num_points=1000)
    else:  # mode == 'adversarial'
        lock_steps = 0  # 锁定方向次数
        target_position = generate_first_control_point_near(uav_position)
        target_positions = [target_position.copy()]
        adversarial_direction = None
    uav_positions = [uav_position.copy()]
    uav_orientations = [uav_orientation]

    # 粒子滤波算法
    pf = ParticleFilter(num_particles=1000, uav_position=uav_position, uav_orientation=uav_orientation)


    # 模拟无人机追踪目标的过程，迭代frame_num轮
    epoch = min(frame_num, len(target_positions)) if mode == 'tracking' else frame_num
    exp4ix = Exp4IX(n=epoch, k=4, M=2, delta=0.1)

    for i in range(epoch):
        if mode == 'tracking':
            target_position = target_positions[i]
        else:
            adversarial_direction, target_position, lock_steps = adversarial_movement(lock_steps, target_positions[-1], uav_position, adversarial_direction)
            target_positions.append(target_position.copy())

        probs_interpolation = trajectory_interpolation(target_position, uav_position, uav_orientation)
        probs_particle_filter = particle_filter(target_position, uav_position, uav_orientation, pf)

        # 将两个结果合并为一个列表
        expert_advice = np.vstack((probs_interpolation, probs_particle_filter))
        probs = exp4ix.get_probs(expert_advice)

        # 随机选择一个方向移动
        move = np.random.choice(['up', 'down', 'left', 'right'], p=probs)

        if move == 'up':
            uav_position[1] += step_size
            uav_orientation = 0
        elif move == 'down':
            uav_position[1] -= step_size
            uav_orientation = 2
        elif move == 'left':
            uav_position[0] -= step_size
            uav_orientation = 3
        elif move == 'right':
            uav_position[0] += step_size
            uav_orientation = 1

        # 保持无人机位置在[0, 1000]范围内
        uav_position = np.clip(uav_position, 0, 10000)

        # 记录无人机位置和朝向
        uav_positions.append(uav_position.copy())
        uav_orientations.append(uav_orientation)

        exp4ix.update(expert_advice, uav_position, uav_orientation, target_position, move)

    uav_positions = np.array(uav_positions)
    uav_orientations = np.array(uav_orientations)
    target_positions = np.array(target_positions)

    animation_video(target_positions, uav_positions, uav_orientations)

# Example of how to call the simulation with different modes:
# simulation_mode(mode='tracking')  # 选择“tracking”模式运行追踪模拟
simulation_mode(mode='adversarial')  # 选择“adversarial”模式运行对抗模拟