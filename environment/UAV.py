import numpy as np

from params import step_size


class UAV():
    def __init__(self):
        self.uav_position = np.random.uniform(2000, 8000, size=(2,))
        self.uav_orientation = np.random.choice([0, 1, 2, 3])
        self.uav_positions = [self.uav_position.copy()]
        self.uav_orientations = [self.uav_orientation]
        self.view_target_trajectory = []

    def update(self, target):
        if target.is_target_in_view(target, self):
            self.view_target_trajectory.append(target.target_position)
        else:
            self.view_target_trajectory.append(None)

    def move(self, probs):
        move = np.random.choice(['up', 'down', 'left', 'right'], p=probs)

        if move == 'up':
            self.uav_position[1] += step_size
            self.uav_orientation = 0
        elif move == 'down':
            self.uav_position[1] -= step_size
            self.uav_orientation = 2
        elif move == 'left':
            self.uav_position[0] -= step_size
            self.uav_orientation = 3
        elif move == 'right':
            self.uav_position[0] += step_size
            self.uav_orientation = 1

        # Ensure UAV position remains within the [0, 10000] range
        self.uav_position = np.clip(self.uav_position, 0, 10000)

        self.uav_positions.append(self.uav_position.copy())
        self.uav_orientations.append(self.uav_orientation)