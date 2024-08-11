import random

import numpy as np

from params import step_size


def generate_first_control_point_near(uav_position):
    point = np.random.uniform(
        low=np.array(uav_position) - [200, 200],
        high=np.array(uav_position) + [200, 200]
    )
    return np.clip(point, [2000, 2000], [8000, 8000])



class Target_adversarial:
    def __init__(self, uav_position):
        self.target_position = generate_first_control_point_near(uav_position)
        self.target_positions = [self.target_position.copy()]
        self.lock_steps = 0  # Number of times the direction is locked
        self.adversarial_direction = 'up'

    def update(self, uav_position):
        def get_farthest_direction(uav_position):
            """Calculate the direction for the target to move farthest away from the UAV."""
            potential_moves = {
                'up': np.array([0, step_size]),
                'down': np.array([0, -step_size]),
                'left': np.array([-step_size, 0]),
                'right': np.array([step_size, 0])
            }

            farthest_direction = None
            max_distance = -np.inf
            for direction, move in potential_moves.items():
                new_position = self.target_position + move
                distance = np.linalg.norm(new_position - uav_position)
                if distance > max_distance:
                    max_distance = distance
                    farthest_direction = direction

            return farthest_direction

        def check_boundary_and_adjust(target_position):
            """Check the boundaries and adjust the direction if necessary."""
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

        if self.lock_steps <= 0:
            # Re-select the adversarial direction
            if np.random.rand() < 0.5:
                self.adversarial_direction = get_farthest_direction(uav_position)
            else:
                self.adversarial_direction = np.random.choice(['up', 'down', 'left', 'right'])
            # Lock the adversarial direction for a number of steps
            self.lock_steps = random.randint(10, 30)
        else:
            self.lock_steps -= 1

        boundary_direction = check_boundary_and_adjust(self.target_position)
        if boundary_direction:
            self.adversarial_direction = boundary_direction
            self.lock_steps = random.randint(10, 30)

        # Move the target based on the locked adversarial direction
        if self.adversarial_direction == 'up':
            self.target_position[1] += step_size
        elif self.adversarial_direction == 'down':
            self.target_position[1] -= step_size
        elif self.adversarial_direction == 'left':
            self.target_position[0] -= step_size
        elif self.adversarial_direction == 'right':
            self.target_position[0] += step_size
        self.target_position = np.clip(self.target_position, 0, 10000)
        self.target_positions.append(self.target_position.copy())
        return

    def is_target_in_view(self, target, uav):
        difference = target.target_position - uav.uav_position

        if uav.uav_orientation == 0:  # Up
            return (difference[1] > 0) and (abs(difference[0]) <= 800) and (difference[1] <= 1000)
        elif uav.uav_orientation == 1:  # Right
            return (difference[0] > 0) and (abs(difference[1]) <= 800) and (difference[0] <= 1000)
        elif uav.uav_orientation == 2:  # Down
            return (difference[1] < 0) and (abs(difference[0]) <= 800) and (abs(difference[1]) <= 1000)
        elif uav.uav_orientation == 3:  # Left
            return (difference[0] < 0) and (abs(difference[1]) <= 800) and (abs(difference[0]) <= 1000)
        else:
            return False
