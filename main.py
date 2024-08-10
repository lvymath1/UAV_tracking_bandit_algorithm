import numpy as np

from algorithm.particle_filter import particle_filter, ParticleFilter
from algorithm.trajectory_interpolation import trajectory_interpolation
from plot.animation import animation_video
from bandit.bandit_algorithm import Exp4IX
from environment.generate_target_trajectory import generate_smooth_trajectory, generate_first_control_point_near
from environment.target_movement_adversarial import adversarial_movement
from params import step_size, frame_num  # Import parameters from the tuning file

np.random.seed(2)

def simulation_mode(mode='tracking'):
    # Initialize UAV position and orientation to a random point and random direction (0: up, 1: right, 2: down, 3: left)
    uav_position = np.random.uniform(2000, 8000, size=(2,))
    uav_orientation = np.random.choice([0, 1, 2, 3])  # Directions: 0: up, 1: right, 2: down, 3: left

    # Generate the smooth trajectory for the target
    if mode == 'tracking':
        target_positions = generate_smooth_trajectory(uav_position, num_control_points=8, num_points=1000)
    else:  # mode == 'adversarial'
        lock_steps = 0  # Number of times the direction is locked
        target_position = generate_first_control_point_near(uav_position)
        target_positions = [target_position.copy()]
        adversarial_direction = None
    uav_positions = [uav_position.copy()]
    uav_orientations = [uav_orientation]

    # Particle filter algorithm
    pf = ParticleFilter(num_particles=1000, uav_position=uav_position, uav_orientation=uav_orientation)

    # Simulate the UAV tracking the target, iterating for `frame_num` rounds
    epoch = min(frame_num, len(target_positions)) if mode == 'tracking' else frame_num
    exp4ix = Exp4IX(n=epoch, k=4, M=2, delta=0.01)

    for i in range(epoch):
        if mode == 'tracking':
            target_position = target_positions[i]
        else:
            adversarial_direction, lock_steps, target_position = adversarial_movement(adversarial_direction, lock_steps, target_positions[-1], uav_position)
            target_positions.append(target_position.copy())

        probs_interpolation = trajectory_interpolation(target_position, uav_position, uav_orientation)
        probs_particle_filter = particle_filter(target_position, uav_position, uav_orientation, pf)

        # Combine both results into one list
        expert_advice = np.vstack((probs_interpolation, probs_particle_filter))
        probs = exp4ix.get_probs(expert_advice)

        # Randomly select a direction to move
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

        # Ensure UAV position remains within the [0, 10000] range
        uav_position = np.clip(uav_position, 0, 10000)

        # Record UAV position and orientation
        uav_positions.append(uav_position.copy())
        uav_orientations.append(uav_orientation)

        exp4ix.update(expert_advice, uav_position, uav_orientation, target_position, move)

    uav_positions = np.array(uav_positions)
    uav_orientations = np.array(uav_orientations)
    target_positions = np.array(target_positions)

    animation_video(target_positions, uav_positions, uav_orientations, mode)

# Example of how to call the simulation with different modes:
simulation_mode(mode='tracking')  # Run tracking simulation in "tracking" mode
# simulation_mode(mode='adversarial')  # Run adversarial simulation in "adversarial" mode
